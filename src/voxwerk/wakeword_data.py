"""Host-side storage for VoxWerk wakeword projects and training records."""

from __future__ import annotations

import json
import re
import shutil
import time
import threading
import uuid
from pathlib import Path
from typing import Any


SLUG_PATTERN = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,50}[a-z0-9])?\Z")
SAMPLE_ID_PATTERN = re.compile(r"[a-f0-9]{32}\Z")
AUDIO_SUFFIXES = {".wav", ".webm", ".ogg", ".mp3", ".flac"}
MAX_SAMPLE_BYTES = 40 * 1024 * 1024
MAX_RECORDS = 1000
STORE_LOCK = threading.RLock()


def validate_slug(value: object) -> str:
    if not isinstance(value, str) or not SLUG_PATTERN.fullmatch(value):
        raise ValueError("Ungültige ID; erlaubt sind Kleinbuchstaben, Zahlen und Bindestriche")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Datensatz kann nicht gelesen werden: {path.name}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Ungültiger Datensatz: {path.name}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _require_name(payload: dict[str, Any]) -> str:
    name = payload.get("name")
    if not isinstance(name, str) or not name.strip() or len(name.strip()) > 100:
        raise ValueError("Name fehlt oder ist länger als 100 Zeichen")
    return name.strip()


class WakewordStore:
    def __init__(self, root: Path):
        self.root = root / "wakeword"
        self.profiles_dir = self.root / "profiles"
        self.wakewords_dir = self.root / "wakewords"
        self.runs_dir = self.root / "runs"
        self.shared_negatives_dir = self.root / "shared_negatives"

    def list_profiles(self) -> list[dict[str, Any]]:
        if not self.profiles_dir.is_dir():
            return []
        return [
            _read_json(path)
            for path in sorted(self.profiles_dir.glob("*.json"))
            if path.is_file() and not path.is_symlink()
        ]

    def save_profile(self, payload: dict[str, Any], profile_id: object = None) -> dict[str, Any]:
        profile_id = validate_slug(profile_id) if profile_id else validate_slug(_slugify(_require_name(payload)))
        existing_path = self.profiles_dir / f"{profile_id}.json"
        current = _read_json(existing_path) if existing_path.exists() else {}
        profile = {
            **current,
            "id": profile_id,
            "name": _require_name(payload),
            "engine": _limited_text(payload.get("engine", current.get("engine", "microwakeword")), "Engine", 60),
            "target": _limited_text(payload.get("target", current.get("target", "ESPHome / ESP32")), "Zielplattform", 100),
            "sample_rate": _bounded_int(payload.get("sample_rate", current.get("sample_rate", 16000)), "Samplerate", 8000, 48000),
            "notes": _limited_text(payload.get("notes", current.get("notes", "")), "Notizen", 2000),
            "updated_at": int(time.time()),
        }
        if not current:
            profile["created_at"] = profile["updated_at"]
            profile["advanced"] = {}
        _write_json(existing_path, profile)
        return profile

    def delete_profile(self, profile_id: object) -> None:
        profile_id = validate_slug(profile_id)
        profile_path = self.profiles_dir / f"{profile_id}.json"
        if not profile_path.is_file() or profile_path.is_symlink():
            raise ValueError("Trainingsprofil nicht gefunden")
        if any(project.get("profile_id") == profile_id for project in self.list_wakewords()):
            raise ValueError("Profil wird noch von einem Wakeword verwendet")
        profile_path.unlink()

    def list_wakewords(self) -> list[dict[str, Any]]:
        if not self.wakewords_dir.is_dir():
            return []
        projects = []
        for project_dir in sorted(self.wakewords_dir.iterdir()):
            if not project_dir.is_dir() or project_dir.is_symlink():
                continue
            manifest = project_dir / "wakeword.json"
            if manifest.is_file() and not manifest.is_symlink():
                projects.append(_read_json(manifest))
        return projects

    def save_wakeword(self, payload: dict[str, Any], wakeword_id: object = None) -> dict[str, Any]:
        name = _require_name(payload)
        wakeword_id = validate_slug(wakeword_id) if wakeword_id else validate_slug(_slugify(name))
        project_dir = self.wakewords_dir / wakeword_id
        if project_dir.is_symlink():
            raise ValueError("Wakeword-Pfad ist ungültig")
        manifest_path = project_dir / "wakeword.json"
        current = _read_json(manifest_path) if manifest_path.exists() else {}
        phrase = payload.get("phrase", current.get("phrase", ""))
        if not isinstance(phrase, str) or len(phrase.strip()) > 200:
            raise ValueError("Wakeword-Phrase fehlt oder ist zu lang")
        profile_id = payload.get("profile_id", current.get("profile_id")) or None
        if profile_id is not None:
            profile_id = validate_slug(profile_id)
            if not (self.profiles_dir / f"{profile_id}.json").is_file():
                raise ValueError("Ausgewähltes Trainingsprofil existiert nicht")
        variants = payload.get("variants", current.get("variants", []))
        if not isinstance(variants, list) or len(variants) > 100 or any(not isinstance(item, str) or len(item) > 200 for item in variants):
            raise ValueError("Aussprachevarianten müssen eine Liste kurzer Texte sein")
        negative_phrases = payload.get("negative_phrases", current.get("negative_phrases", []))
        if not isinstance(negative_phrases, list) or len(negative_phrases) > 200 or any(not isinstance(item, str) or len(item) > 300 for item in negative_phrases):
            raise ValueError("Negative Sätze müssen eine Liste kurzer Texte sein")
        now = int(time.time())
        project = {
            **current,
            "id": wakeword_id,
            "name": name,
            "phrase": phrase.strip(),
            "language": _limited_text(payload.get("language", current.get("language", "de-DE")), "Sprache", 30),
            "profile_id": profile_id,
            "variants": [item.strip() for item in variants if item.strip()],
            "negative_phrases": [item.strip() for item in negative_phrases if item.strip()],
            "created_at": current.get("created_at", now),
            "updated_at": now,
        }
        (project_dir / "samples" / "positive").mkdir(parents=True, exist_ok=True)
        (project_dir / "samples" / "negative").mkdir(parents=True, exist_ok=True)
        (project_dir / "datasets").mkdir(exist_ok=True)
        (project_dir / "runs").mkdir(exist_ok=True)
        _write_json(manifest_path, project)
        return project

    def delete_wakeword(self, wakeword_id: object) -> None:
        wakeword_id = validate_slug(wakeword_id)
        with STORE_LOCK:
            project_dir = self.wakewords_dir / wakeword_id
            if project_dir.is_symlink() or not (project_dir / "wakeword.json").is_file():
                raise ValueError("Wakeword nicht gefunden")
            shutil.rmtree(project_dir)

    def list_samples(self, wakeword_id: object) -> list[dict[str, Any]]:
        project_dir = self._project_dir(wakeword_id)
        manifest = _read_json(project_dir / "wakeword.json")
        samples = manifest.get("samples", [])
        if not isinstance(samples, list):
            return []
        return samples

    def save_sample(
        self,
        wakeword_id: object,
        kind: object,
        filename: object,
        content: bytes,
        title: object = None,
        *,
        source: str = "recorded",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with STORE_LOCK:
            project_dir = self._project_dir(wakeword_id)
            if kind not in {"positive", "negative"}:
                raise ValueError("Sampletyp muss positiv oder negativ sein")
            if not isinstance(filename, str):
                raise ValueError("Audiodatei fehlt")
            suffix = Path(filename).suffix.lower()
            if suffix not in AUDIO_SUFFIXES:
                raise ValueError("Audioformat nicht unterstützt")
            if not content or len(content) > MAX_SAMPLE_BYTES:
                raise ValueError("Audiodatei ist leer oder größer als 40 MB")
            manifest_path = project_dir / "wakeword.json"
            manifest = _read_json(manifest_path)
            samples = manifest.setdefault("samples", [])
            if not isinstance(samples, list) or len(samples) >= MAX_RECORDS:
                raise ValueError("Maximale Anzahl von 1000 Samples erreicht")
            sample_id = uuid.uuid4().hex
            stored_name = f"{sample_id}{suffix}"
            (project_dir / "samples" / str(kind) / stored_name).write_bytes(content)
            sample = {
                "id": sample_id,
                "kind": kind,
                "filename": stored_name,
                "title": _limited_text(title or Path(filename).stem, "Titel", 100),
                "source": source,
                "created_at": int(time.time()),
            }
            if metadata:
                sample.update(metadata)
            samples.append(sample)
            _write_json(manifest_path, manifest)
            return sample

    def update_sample(self, wakeword_id: object, sample_id: object, payload: dict[str, Any]) -> dict[str, Any]:
        project_dir = self._project_dir(wakeword_id)
        sample_id = _validate_sample_id(sample_id)
        manifest_path = project_dir / "wakeword.json"
        manifest = _read_json(manifest_path)
        samples = manifest.get("samples", [])
        sample = next((item for item in samples if item.get("id") == sample_id), None)
        if sample is None:
            raise ValueError("Aufnahme nicht gefunden")
        if "title" in payload:
            sample["title"] = _limited_text(payload["title"], "Titel", 100)
        if "transcript" in payload:
            sample["transcript"] = _limited_text(payload["transcript"], "Transkript", 300)
        if "kind" in payload and payload["kind"] != sample["kind"]:
            new_kind = payload["kind"]
            if new_kind not in {"positive", "negative"}:
                raise ValueError("Sampletyp muss positiv oder negativ sein")
            old_path = project_dir / "samples" / sample["kind"] / sample["filename"]
            new_path = project_dir / "samples" / new_kind / sample["filename"]
            if old_path.is_symlink() or not old_path.is_file():
                raise ValueError("Audiodatei nicht gefunden")
            new_path.parent.mkdir(parents=True, exist_ok=True)
            old_path.replace(new_path)
            sample["kind"] = new_kind
        _write_json(manifest_path, manifest)
        return sample

    def delete_sample(self, wakeword_id: object, sample_id: object) -> None:
        project_dir = self._project_dir(wakeword_id)
        sample_id = _validate_sample_id(sample_id)
        manifest_path = project_dir / "wakeword.json"
        manifest = _read_json(manifest_path)
        samples = manifest.get("samples", [])
        sample = next((item for item in samples if item.get("id") == sample_id), None)
        if sample is None:
            raise ValueError("Aufnahme nicht gefunden")
        audio_path = project_dir / "samples" / sample["kind"] / sample["filename"]
        if audio_path.is_symlink():
            raise ValueError("Audiodateipfad ist ungültig")
        audio_path.unlink(missing_ok=True)
        manifest["samples"] = [item for item in samples if item.get("id") != sample_id]
        _write_json(manifest_path, manifest)

    def delete_samples(self, wakeword_id: object, sample_ids: object) -> list[dict[str, Any]]:
        project_dir = self._project_dir(wakeword_id)
        if not isinstance(sample_ids, list) or not sample_ids or len(sample_ids) > MAX_RECORDS:
            raise ValueError("Wähle mindestens eine Aufnahme zum Löschen aus")
        validated_ids = [_validate_sample_id(sample_id) for sample_id in sample_ids]
        if len(set(validated_ids)) != len(validated_ids):
            raise ValueError("Die Auswahl enthält doppelte Aufnahmen")
        with STORE_LOCK:
            manifest_path = project_dir / "wakeword.json"
            manifest = _read_json(manifest_path)
            samples = manifest.get("samples", [])
            selected = {sample["id"]: sample for sample in samples if sample.get("id") in validated_ids}
            if len(selected) != len(validated_ids):
                raise ValueError("Mindestens eine ausgewählte Aufnahme wurde nicht gefunden")
            paths = []
            for sample in selected.values():
                audio_path = project_dir / "samples" / sample["kind"] / sample["filename"]
                if audio_path.is_symlink():
                    raise ValueError("Audiodateipfad ist ungültig")
                paths.append(audio_path)
            for audio_path in paths:
                audio_path.unlink(missing_ok=True)
            manifest["samples"] = [sample for sample in samples if sample.get("id") not in selected]
            _write_json(manifest_path, manifest)
            return list(selected.values())

    def create_run(self, wakeword_id: object, payload: dict[str, Any]) -> dict[str, Any]:
        project_dir = self._project_dir(wakeword_id)
        project = _read_json(project_dir / "wakeword.json")
        profile_id = payload.get("profile_id", project.get("profile_id"))
        if not profile_id:
            raise ValueError("Für einen Trainingslauf muss ein Trainingsprofil ausgewählt sein")
        profile_id = validate_slug(profile_id)
        profile_path = self.profiles_dir / f"{profile_id}.json"
        if not profile_path.is_file():
            raise ValueError("Trainingsprofil nicht gefunden")
        selected = payload.get("sample_ids", [])
        if not isinstance(selected, list) or any(not isinstance(item, str) for item in selected):
            raise ValueError("Sampleauswahl ist ungültig")
        samples = self.list_samples(wakeword_id)
        available = {sample["id"]: sample for sample in samples}
        if len(set(selected)) != len(selected) or any(item not in available for item in selected):
            raise ValueError("Die Sampleauswahl enthält unbekannte oder doppelte Einträge")
        if not any(available[item]["kind"] == "positive" for item in selected):
            raise ValueError("Mindestens ein positives Sample muss ausgewählt sein")
        if not any(available[item]["kind"] == "negative" for item in selected):
            raise ValueError("Mindestens ein negatives Sample muss ausgewählt sein")
        run_id = time.strftime("run-%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
        run_dir = project_dir / "runs" / run_id
        run_samples = []
        for item in selected:
            sample = dict(available[item])
            source_path = project_dir / "samples" / sample["kind"] / sample["filename"]
            if source_path.is_symlink() or not source_path.is_file():
                raise ValueError(f"Audiodatei für Sample {sample['title']} fehlt")
            snapshot_filename = f"{sample['id']}{source_path.suffix.lower()}"
            (run_dir / "samples").mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, run_dir / "samples" / snapshot_filename)
            sample["snapshot_file"] = f"samples/{snapshot_filename}"
            run_samples.append(sample)
        run = {
            "id": run_id,
            "wakeword_id": project["id"],
            "created_at": int(time.time()),
            "status": "prepared",
            "profile": _read_json(profile_path),
            "wakeword": project,
            "sample_ids": selected,
            "samples": run_samples,
            "tests": [],
        }
        _write_json(run_dir / "run.json", run)
        return run

    def list_runs(self, wakeword_id: object) -> list[dict[str, Any]]:
        run_dir = self._project_dir(wakeword_id) / "runs"
        if not run_dir.is_dir():
            return []
        return [_read_json(path) for path in sorted(run_dir.glob("*/run.json"), reverse=True) if path.is_file() and not path.is_symlink()]

    def delete_run(self, wakeword_id: object, run_id: object) -> None:
        wakeword_id = validate_slug(wakeword_id)
        if not isinstance(run_id, str) or not re.fullmatch(r"run-[0-9]{8}-[0-9]{6}-[a-f0-9]{8}", run_id):
            raise ValueError("Ungültige Trainingslauf-ID")
        run_dir = self.wakewords_dir / wakeword_id / "runs" / run_id
        run_path = run_dir / "run.json"
        if run_dir.is_symlink() or run_path.is_symlink() or not run_path.is_file():
            raise ValueError("Trainingslauf nicht gefunden")
        shutil.rmtree(run_dir)

    def _project_dir(self, wakeword_id: object) -> Path:
        wakeword_id = validate_slug(wakeword_id)
        project_dir = self.wakewords_dir / wakeword_id
        manifest = project_dir / "wakeword.json"
        if project_dir.is_symlink() or manifest.is_symlink() or not manifest.is_file():
            raise ValueError("Wakeword nicht gefunden")
        return project_dir


def _slugify(text: str) -> str:
    import unicodedata

    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_text).strip("-").lower()[:52].strip("-")
    return slug or "wakeword"


def _limited_text(value: object, label: str, maximum: int) -> str:
    if not isinstance(value, str) or len(value.strip()) > maximum:
        raise ValueError(f"{label} ist ungültig oder zu lang")
    return value.strip()


def _bounded_int(value: object, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{label} muss zwischen {minimum} und {maximum} liegen")
    return value


def _validate_sample_id(value: object) -> str:
    if not isinstance(value, str) or not SAMPLE_ID_PATTERN.fullmatch(value):
        raise ValueError("Ungültige Sample-ID")
    return value