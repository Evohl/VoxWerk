"""Local-first Piper voice studio web application."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import random
import re
import shutil
import threading
import time
import unicodedata
import uuid
import wave
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from urllib.error import URLError
from urllib.parse import parse_qs, quote, unquote, urlsplit
from urllib.request import urlopen

from importlib.resources import files
from .wakeword_data import MAX_RECORDS, MAX_SAMPLE_BYTES, WakewordStore
from .wakeword_training import TrainingStopped, check_wakeword_integration, run_training_pipeline, training_environment, validate_training_parameters, _wav_samples
from .wakeword_testing import list_test_models, score_test_audio, score_test_pcm_frame
from .storage import data_dir as configured_data_dir, data_directory_info, save_data_directory
from .wakeword_install import cancel_install, get_install_job, start_install


APP_NAME = "voxwerk"
CATALOG_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/voices.json"
MODEL_BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
MAX_CATALOG_BYTES = 12 * 1024 * 1024
MAX_JSON_BYTES = 32_768
MAX_TEXT_LENGTH = 2500
MAX_MODEL_BYTES = 500 * 1024 * 1024
VOICE_ID_PATTERN = re.compile(r"[^/\\]{1,100}\Z")
SYNTHESIS_LOCK = threading.Lock()
GENERATION_JOBS: dict[str, dict[str, object]] = {}
GENERATION_JOBS_LOCK = threading.Lock()
TRAINING_JOBS: dict[str, dict[str, object]] = {}
TRAINING_JOBS_LOCK = threading.Lock()
TRAINING_STOP_EVENTS: dict[str, threading.Event] = {}
LOGGER = logging.getLogger(APP_NAME)


def user_data_dir() -> Path:
    return Path.home() / ".local" / "share" / APP_NAME


def user_cache_dir() -> Path:
    return Path.home() / ".cache" / APP_NAME


def xdg_path(variable: str, fallback: Path) -> Path:
    configured = os.environ.get(variable)
    return Path(configured).expanduser() / APP_NAME if configured else fallback


def data_dir() -> Path:
    return configured_data_dir()


def cache_dir() -> Path:
    return xdg_path("XDG_CACHE_HOME", user_cache_dir())


def validate_voice_id(value: object) -> str:
    if not isinstance(value, str) or not VOICE_ID_PATTERN.fullmatch(value) or not all(character.isalnum() or character in "_-" for character in value):
        raise ValueError("Ungültige Stimmen-ID")
    return value


def _api_client_allowed(client_address: str, bind_address: str) -> bool:
    import ipaddress

    try:
        client_ip = ipaddress.ip_address(client_address)
        bind_ip = ipaddress.ip_address(bind_address)
    except ValueError:
        return False
    return client_ip.is_loopback or not bind_ip.is_loopback


def _run_wakeword_training(
    wakeword_id: str,
    run_id: str,
    environment: dict[str, object],
    parameters: dict[str, object],
) -> None:
    store = WakewordStore(data_dir())
    run_dir = store._project_dir(wakeword_id) / "runs" / run_id

    stop_event = TRAINING_STOP_EVENTS.get(run_id, threading.Event())

    def set_stage(stage: str, progress: int) -> None:
        store.update_run(wakeword_id, run_id, status="running", training={"stage": stage, "progress": progress})
        with TRAINING_JOBS_LOCK:
            TRAINING_JOBS[run_id] = {"status": "running", "stage": stage, "progress": progress, "wakeword_id": wakeword_id}

    try:
        store.update_run(wakeword_id, run_id, status="running", training={"stage": "Training wird vorbereitet", "progress": 0})
        run = next(item for item in store.list_runs(wakeword_id) if item["id"] == run_id)
        model_dir = run_training_pipeline(
            Path(environment["workspace"]),
            Path(environment["python"]),
            run_dir,
            run,
            parameters,
            set_stage,
            stop_event.is_set,
        )
        store.update_run(
            wakeword_id,
            run_id,
            status="completed",
            training={"stage": "Training abgeschlossen", "progress": 100, "model_directory": str(model_dir.relative_to(run_dir)), "finished_at": int(time.time())},
        )
        with TRAINING_JOBS_LOCK:
            TRAINING_JOBS[run_id] = {"status": "completed", "progress": 100, "wakeword_id": wakeword_id}
    except TrainingStopped:
        store.update_run(
            wakeword_id,
            run_id,
            status="stopped",
            training={"stage": "Training abgebrochen", "finished_at": int(time.time())},
        )
        with TRAINING_JOBS_LOCK:
            TRAINING_JOBS[run_id] = {"status": "stopped", "wakeword_id": wakeword_id}
    except Exception as error:
        LOGGER.exception("Wakeword training failed for run %s", run_id)
        try:
            store.update_run(
                wakeword_id,
                run_id,
                status="failed",
                training={"stage": "Training fehlgeschlagen", "error": str(error), "finished_at": int(time.time())},
            )
        except (OSError, ValueError):
            LOGGER.exception("Could not persist failure status for run %s", run_id)
        with TRAINING_JOBS_LOCK:
            TRAINING_JOBS[run_id] = {"status": "failed", "wakeword_id": wakeword_id}
    finally:
        TRAINING_STOP_EVENTS.pop(run_id, None)


def language_display_name(name: str, code: str) -> str:
    normalized_code = code.replace("-", "_")
    if name.casefold() == "english" and normalized_code.casefold().startswith("en_"):
        return f"{name} ({normalized_code.rsplit('_', 1)[-1].upper()})"
    return name


def catalog_entries(raw_catalog: object) -> list[dict[str, object]]:
    if not isinstance(raw_catalog, dict):
        raise ValueError("Der Piper-Katalog hat ein ungültiges Format")
    entries: list[dict[str, object]] = []
    for voice_id, raw_voice in raw_catalog.items():
        if not isinstance(raw_voice, dict):
            continue
        try:
            validate_voice_id(voice_id)
        except ValueError:
            continue
        language = raw_voice.get("language")
        language = language if isinstance(language, dict) else {}
        file_map = raw_voice.get("files")
        file_map = file_map if isinstance(file_map, dict) else {}
        model_path = next(
            (path for path in file_map if isinstance(path, str) and path.endswith(f"/{voice_id}.onnx")),
            None,
        )
        metadata_path = next(
            (path for path in file_map if isinstance(path, str) and path.endswith(f"/{voice_id}.onnx.json")),
            None,
        )
        if not model_path or not metadata_path:
            continue
        try:
            safe_model_path = PurePosixPath(model_path)
            safe_metadata_path = PurePosixPath(metadata_path)
            if safe_model_path.is_absolute() or ".." in safe_model_path.parts or safe_metadata_path.is_absolute() or ".." in safe_metadata_path.parts:
                continue
            model_size = int(file_map[model_path].get("size_bytes", 0))
            model_md5 = str(file_map[model_path].get("md5_digest", ""))
            metadata_md5 = str(file_map[metadata_path].get("md5_digest", ""))
        except (AttributeError, TypeError, ValueError):
            continue
        language_code = str(language.get("code") or "")
        language_name = str(language.get("name_english") or language.get("name_native") or language_code or "Unknown")
        entries.append({
            "id": voice_id,
            "name": str(raw_voice.get("name") or voice_id),
            "language": language_display_name(language_name, language_code),
            "language_code": language_code,
            "quality": str(raw_voice.get("quality") or "unknown"),
            "speakers": int(raw_voice.get("num_speakers") or 1),
            "size_bytes": model_size,
            "model_path": model_path,
            "metadata_path": metadata_path,
            "model_md5": model_md5,
            "metadata_md5": metadata_md5,
            "license_url": f"https://huggingface.co/rhasspy/piper-voices/tree/main/{safe_model_path.parent}",
        })
    entries.sort(key=lambda item: (str(item["language"]).lower(), str(item["name"]).lower(), str(item["quality"])))
    return entries


def filter_catalog(entries: list[dict[str, object]], query: str = "", language: str = "", quality: str = "") -> list[dict[str, object]]:
    words = query.casefold().split()
    return [
        entry for entry in entries
        if (not language or entry["language_code"] == language)
        and (not quality or entry["quality"] == quality)
        and all(word in " ".join(str(entry[key]) for key in ("id", "name", "language", "language_code", "quality")).casefold() for word in words)
    ]


def _download_file(relative_path: str, expected_md5: str, destination: Path, expected_size: int = 0) -> None:
    path = PurePosixPath(relative_path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Ungültiger Modellpfad im Katalog")
    if not re.fullmatch(r"[0-9a-fA-F]{32}", expected_md5):
        raise ValueError("Der Katalog enthält keine gültige MD5-Prüfsumme")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{threading.get_ident()}.part")
    digest = hashlib.md5()
    size = 0
    try:
        download_url = MODEL_BASE_URL + quote(str(path), safe="/._-") + "?download=true"
        with urlopen(download_url, timeout=45) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_MODEL_BYTES or (expected_size and size > expected_size):
                    raise ValueError("Der Download überschreitet die erwartete Dateigröße")
                digest.update(chunk)
                output.write(chunk)
        if expected_size and size != expected_size:
            raise ValueError("Der Download hat eine unerwartete Dateigröße")
        if digest.hexdigest().lower() != expected_md5.lower():
            raise ValueError("Die Prüfsumme der heruntergeladenen Datei stimmt nicht")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def installed_voices() -> list[dict[str, object]]:
    voices: list[dict[str, object]] = []
    root = data_dir() / "voices"
    if not root.is_dir():
        return voices
    for model_path in sorted(root.glob("*/*.onnx")):
        voice_id = model_path.stem
        sidecar = model_path.with_name(f"{voice_id}.onnx.json")
        if not sidecar.is_file():
            continue
        try:
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            language = metadata.get("language", {})
        except (OSError, json.JSONDecodeError):
            continue
        language_code = str(language.get("code") or "")
        language_name = str(language.get("name_english") or language_code or "Unknown")
        voices.append({
            "id": voice_id,
            "name": str(metadata.get("dataset") or voice_id),
            "language": language_display_name(language_name, language_code),
            "path": str(model_path),
        })
    return voices


def remove_voice(voice_id: object) -> None:
    voice_id = validate_voice_id(voice_id)
    voices_root = data_dir() / "voices"
    voice_dir = voices_root / voice_id
    if voice_dir.is_symlink() or not voice_dir.is_dir():
        raise ValueError("Stimme ist nicht installiert")
    with SYNTHESIS_LOCK:
        load_voice.cache_clear()
        shutil.rmtree(voice_dir)


@lru_cache(maxsize=2)
def load_voice(model_path: str):
    from piper import PiperVoice

    LOGGER.info("Loading Piper model %s", model_path)
    return PiperVoice.load(model_path, use_cuda=False)


def safe_slug(text: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-zA-Z0-9]+", "-", ascii_text).strip("-").lower()[:52] or "speech"


def allocate_weighted_samples(total: int, weights: dict[str, int]) -> dict[str, int]:
    if isinstance(total, bool) or not isinstance(total, int) or not 1 <= total <= 500:
        raise ValueError("Samplezahl muss zwischen 1 und 500 liegen")
    if any(isinstance(weight, bool) or not isinstance(weight, int) or not 0 <= weight <= 100 for weight in weights.values()):
        raise ValueError("Mindestens eine Stimme mit Gewicht 1 bis 100 auswählen")
    selected = {voice_id: weight for voice_id, weight in weights.items() if weight > 0}
    if not selected:
        raise ValueError("Mindestens eine Stimme mit Gewicht 1 bis 100 auswählen")
    weight_total = sum(selected.values())
    exact = {voice_id: total * weight / weight_total for voice_id, weight in selected.items()}
    allocation = {voice_id: int(count) for voice_id, count in exact.items()}
    remainder = total - sum(allocation.values())
    priority = sorted(selected, key=lambda voice_id: (-(exact[voice_id] - allocation[voice_id]), voice_id))
    for voice_id in priority[:remainder]:
        allocation[voice_id] += 1
    return allocation


def synthesize_wav(text: str, voice_id: str, rate: float = 1.0) -> bytes:
    voice = next((item for item in installed_voices() if item["id"] == voice_id), None)
    if voice is None:
        raise ValueError(f"Stimme {voice_id} ist nicht installiert")
    output = io.BytesIO()
    with SYNTHESIS_LOCK:
        from piper import SynthesisConfig

        model = load_voice(str(voice["path"]))
        with wave.open(output, "wb") as wav_file:
            model.synthesize_wav(text, wav_file=wav_file, syn_config=SynthesisConfig(length_scale=1.0 / rate))
    return output.getvalue()


def build_sample_plan(texts: list[str], total: int, tempo_variation: int = 20, seed: int | None = None) -> list[dict[str, object]]:
    if any(not isinstance(text, str) for text in texts):
        raise ValueError("Mindestens eine gültige Textvariante wird benötigt")
    texts = [text.strip() for text in texts if text.strip()]
    if not texts:
        raise ValueError("Mindestens eine gültige Textvariante wird benötigt")
    if isinstance(tempo_variation, bool) or not isinstance(tempo_variation, int) or not 0 <= tempo_variation <= 30:
        raise ValueError("Tempo-Streuung muss zwischen 0 und 30 Prozent liegen")
    spread = tempo_variation / 100
    length_scales = [1.0] if spread == 0 else [1.0 - spread, 1.0 - spread / 2, 1.0, 1.0 + spread / 2, 1.0 + spread]
    plan = [
        {"text": texts[index % len(texts)].strip(), "length_scale": length_scales[index % len(length_scales)]}
        for index in range(total)
    ]
    random.Random(seed).shuffle(plan)
    return plan


def _run_sample_generation(
    job_id: str,
    wakeword_id: str,
    allocation: dict[str, int],
    sample_kind: str = "positive",
    tempo_variation: int = 20,
    relative_weights: dict[str, int] | None = None,
    seed: int | None = None,
) -> None:
    store = WakewordStore(data_dir())
    try:
        project = next((item for item in store.list_wakewords() if item["id"] == wakeword_id), None)
        if project is None:
            raise ValueError("Wakeword-Projekt nicht gefunden")
        texts = (
            [project.get("phrase", ""), *project.get("variants", [])]
            if sample_kind == "positive"
            else project.get("negative_phrases", [])
        )
        texts = [text.strip() for text in texts if isinstance(text, str) and text.strip()]
        if not texts:
            raise ValueError("Im Wakeword-Projekt ist keine Phrase hinterlegt")
        voices = {str(item["id"]): item for item in installed_voices()}
        total = sum(allocation.values())
        with GENERATION_JOBS_LOCK:
            GENERATION_JOBS[job_id].update({"status": "running", "total": total, "completed": 0, "allocation": allocation, "tempo_variation": tempo_variation})
        completed = 0
        for voice_id, count in allocation.items():
            voice = voices[voice_id]
            plan = build_sample_plan(texts, count, tempo_variation, seed=(seed or 0) + sum(voice_id.encode("utf-8")))
            for index, sample_plan in enumerate(plan):
                text = str(sample_plan["text"])
                length_scale = float(sample_plan["length_scale"])
                audio = synthesize_wav(text, voice_id, rate=1.0 / length_scale)
                store.save_sample(
                    wakeword_id,
                    sample_kind,
                    f"{voice_id}-{index + 1}.wav",
                    audio,
                    f"{voice['name']} · {index + 1}",
                    source="generated",
                    metadata={
                        "voice_id": voice_id,
                        "voice_name": voice["name"],
                        "relative_weight": (relative_weights or {}).get(voice_id, 0),
                        "transcript": text,
                        "length_scale": round(length_scale, 3),
                        "generator": "piper",
                    },
                )
                completed += 1
                with GENERATION_JOBS_LOCK:
                    GENERATION_JOBS[job_id]["completed"] = completed
        with GENERATION_JOBS_LOCK:
            GENERATION_JOBS[job_id]["status"] = "completed"
    except Exception as error:
        LOGGER.exception("Wakeword sample generation failed for %s", wakeword_id)
        with GENERATION_JOBS_LOCK:
            job = GENERATION_JOBS.get(job_id)
            if job is not None:
                job.update({"status": "failed", "error": str(error)})


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "VoxWerk/0.1"

    def log_message(self, format_string: str, *args: object) -> None:
        LOGGER.info("%s - %s", self.address_string(), format_string % args)

    def send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, payload: object, status: int = 200) -> None:
        self.send_bytes(json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", status)

    def is_api_client_allowed(self) -> bool:
        return _api_client_allowed(self.client_address[0], self.server.server_address[0])

    def _read_json(self) -> dict[str, object] | None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_json({"error": "Ungültige Request-Größe"}, 400)
            return None
        if not 0 < length <= MAX_JSON_BYTES:
            self.send_json({"error": "Request ist leer oder zu groß"}, 413)
            return None
        try:
            payload = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self.send_json({"error": "Request muss gültiges JSON enthalten"}, 400)
            return None
        if not isinstance(payload, dict):
            self.send_json({"error": "Request muss ein JSON-Objekt sein"}, 400)
            return None
        return payload

    def _wakeword_store(self) -> WakewordStore:
        return WakewordStore(data_dir())

    def do_GET(self) -> None:
        request = urlsplit(self.path)
        if request.path.startswith("/api/") and not self.is_api_client_allowed():
            self.send_json({"error": "Remote-Zugriff erfordert einen nicht-lokalen --host-Wert beim Start von VoxWerk"}, 403)
            return
        if request.path == "/":
            page = files("voxwerk").joinpath("web/index.html").read_bytes()
            self.send_bytes(page, "text/html; charset=utf-8")
            return
        if request.path == "/api/voices":
            self.send_json({"voices": installed_voices()})
            return
        if request.path == "/api/wakeword":
            store = self._wakeword_store()
            self.send_json({"profiles": store.list_profiles(), "wakewords": store.list_wakewords()})
            return
        if request.path == "/api/wakeword/training/options":
            environment = training_environment()
            self.send_json({key: value for key, value in environment.items() if key not in {"workspace", "python"}})
            return
        if request.path.startswith("/api/wakeword/install/"):
            job_id = request.path.rsplit("/", 1)[-1]
            job = get_install_job(job_id)
            self.send_json({"job": job} if job else {"error": "Installationsauftrag nicht gefunden"}, 200 if job else 404)
            return
        if request.path == "/api/settings/data-directory":
            info = data_directory_info()
            self.send_json({"path": str(info["path"]), "source": info["source"], "locked": info["locked"]})
            return
        if request.path == "/api/wakeword/package":
            self.send_json(check_wakeword_integration())
            return
        if request.path == "/api/wakeword/test/models":
            import ipaddress

            if not ipaddress.ip_address(self.client_address[0]).is_loopback:
                self.send_json({"error": "Wakeword-Tests sind nur über localhost verfügbar"}, 403)
                return
            environment = training_environment()
            try:
                models = list_test_models(Path(environment["workspace"]), self._wakeword_store())
                self.send_json({"models": models, "ready": environment["ready"], "message": environment["message"]})
            except (OSError, ValueError) as error:
                self.send_json({"error": str(error)}, 500)
            return
        if request.path.startswith("/api/wakeword/"):
            parts = request.path.strip("/").split("/")
            store = self._wakeword_store()
            try:
                if len(parts) == 4 and parts[2] == "generation":
                    with GENERATION_JOBS_LOCK:
                        job = GENERATION_JOBS.get(parts[3])
                        if job is None:
                            self.send_json({"error": "Generierungsauftrag nicht gefunden"}, 404)
                        else:
                            self.send_json({"job": dict(job)})
                    return
                if len(parts) == 4 and parts[2] == "runs":
                    self.send_json({"runs": store.list_runs(parts[3])})
                    return
                if len(parts) == 4 and parts[2] == "samples":
                    self.send_json({"samples": store.list_samples(parts[3])})
                    return
                if len(parts) == 5 and parts[2] == "audio":
                    sample_id = parts[4]
                    sample = next((item for item in store.list_samples(parts[3]) if item["id"] == sample_id), None)
                    if sample is None:
                        raise ValueError("Aufnahme nicht gefunden")
                    audio_path = store._project_dir(parts[3]) / "samples" / sample["kind"] / sample["filename"]
                    content_types = {".wav": "audio/wav", ".webm": "audio/webm", ".ogg": "audio/ogg", ".mp3": "audio/mpeg", ".flac": "audio/flac"}
                    self.send_bytes(audio_path.read_bytes(), content_types.get(audio_path.suffix.lower(), "application/octet-stream"))
                    return
            except (ValueError, OSError) as error:
                self.send_json({"error": str(error)}, 404)
                return
        if request.path == "/api/recordings":
            recording_dir = data_dir() / "recordings"
            recordings = []
            if recording_dir.is_dir():
                for audio_path in sorted(recording_dir.glob("*.wav"), key=lambda path: path.stat().st_mtime, reverse=True)[:100]:
                    recordings.append({
                        "filename": audio_path.name,
                        "title": audio_path.stem.split("_", 2)[-1].replace("-", " "),
                        "url": f"/api/audio/{audio_path.name}",
                    })
            self.send_json({"recordings": recordings})
            return
        if request.path == "/api/catalog":
            try:
                cache_path = cache_dir() / "voices.json"
                if "refresh=1" in request.query or not cache_path.is_file():
                    with urlopen(CATALOG_URL, timeout=20) as response:
                        catalog_bytes = response.read(MAX_CATALOG_BYTES + 1)
                    if len(catalog_bytes) > MAX_CATALOG_BYTES:
                        raise ValueError("Der Piper-Katalog überschreitet die erlaubte Größe")
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_bytes(catalog_bytes)
                raw_catalog = json.loads(cache_path.read_text(encoding="utf-8"))
                entries = catalog_entries(raw_catalog)
                params = parse_qs(request.query)
                filtered = filter_catalog(entries, params.get("q", [""])[0], params.get("language", [""])[0], params.get("quality", [""])[0])
                self.send_json({"voices": filtered, "count": len(filtered), "total": len(entries)})
            except (OSError, URLError, TimeoutError, ValueError, json.JSONDecodeError) as error:
                LOGGER.warning("Could not load Piper catalog: %s", error)
                self.send_json({"error": f"Stimmenkatalog nicht verfügbar: {error}"}, 503)
            return
        if request.path.startswith("/api/audio/"):
            filename = unquote(request.path.removeprefix("/api/audio/"))
            if Path(filename).name != filename or not filename.endswith(".wav"):
                self.send_json({"error": "Ungültiger Audiodateiname"}, 400)
                return
            try:
                body = (data_dir() / "recordings" / filename).read_bytes()
            except OSError:
                self.send_json({"error": "Aufnahme nicht gefunden"}, 404)
                return
            self.send_bytes(body, "audio/wav")
            return
        self.send_json({"error": "Nicht gefunden"}, 404)

    def do_POST(self) -> None:
        request = urlsplit(self.path)
        if not self.is_api_client_allowed():
            self.send_json({"error": "Remote-Zugriff erfordert einen nicht-lokalen --host-Wert beim Start von VoxWerk"}, 403)
            return
        if request.path == "/api/wakeword/install/start":
            payload = self._read_json()
            if payload is None:
                return
            environment = training_environment()
            if environment["package_locked"]:
                self.send_json({"error": "VOXWERK_WAKEWORD_DIR ist gesetzt; Installation über die Oberfläche ist gesperrt"}, 409)
                return
            try:
                with TRAINING_JOBS_LOCK:
                    training_active = any(job.get("status") in {"queued", "running"} for job in TRAINING_JOBS.values())
                if training_active:
                    raise ValueError("Wakeword-Umgebung kann während eines Trainings nicht installiert werden")
                job = start_install()
                self.send_json({"job": job}, 202)
            except (OSError, ValueError) as error:
                self.send_json({"error": str(error)}, 400)
            return
        if request.path.startswith("/api/wakeword/install/") and request.path.endswith("/cancel"):
            job_id = request.path.split("/")[-2]
            job = cancel_install(job_id)
            self.send_json({"job": job} if job else {"error": "Kein aktiver Installationsauftrag gefunden"}, 200 if job else 404)
            return
        if request.path == "/api/settings/data-directory":
            import ipaddress

            if not ipaddress.ip_address(self.client_address[0]).is_loopback:
                self.send_json({"error": "Datenordner können nur über localhost geändert werden"}, 403)
                return
            payload = self._read_json()
            if payload is None:
                return
            with TRAINING_JOBS_LOCK, GENERATION_JOBS_LOCK:
                jobs_active = any(job.get("status") in {"queued", "running"} for job in TRAINING_JOBS.values()) or any(
                    job.get("status") in {"queued", "running"} for job in GENERATION_JOBS.values()
                )
            if jobs_active:
                self.send_json({"error": "Datenordner kann während eines Trainings oder einer Samplegenerierung nicht verschoben werden"}, 409)
                return
            try:
                info = save_data_directory(payload.get("path"))
                self.send_json({"path": str(info["path"]), "source": info["source"], "locked": info["locked"], "copied": info["copied"]})
            except (OSError, ValueError) as error:
                self.send_json({"error": str(error)}, 400)
            return
        if request.path == "/api/wakeword/test/score":
            import ipaddress

            if not ipaddress.ip_address(self.client_address[0]).is_loopback:
                self.send_json({"error": "Wakeword-Tests sind nur über localhost verfügbar"}, 403)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self.send_json({"error": "Ungültige Audiogröße"}, 400)
                return
            if not 0 < length <= 50 * 1024 * 1024:
                self.send_json({"error": "Audio fehlt oder überschreitet 50 MB"}, 413)
                return
            params = parse_qs(request.query)
            model_id = params.get("model", [""])[0]
            cutoff = params.get("cutoff", ["0.5"])[0]
            live = params.get("live", ["0"])[0] == "1"
            reset_worker = params.get("reset", ["0"])[0] == "1"
            try:
                environment = training_environment()
                result = score_test_audio(
                    self.rfile.read(length),
                    model_id,
                    cutoff,
                    Path(environment["workspace"]),
                    self._wakeword_store(),
                    live=live,
                    reset_worker=reset_worker,
                )
                self.send_json(result)
            except ValueError as error:
                self.send_json({"error": str(error)}, 400)
            except RuntimeError as error:
                self.send_json({"error": str(error)}, 409)
            except OSError as error:
                self.send_json({"error": str(error)}, 500)
            return
        if request.path == "/api/wakeword/test/stream":
            import ipaddress

            if not ipaddress.ip_address(self.client_address[0]).is_loopback:
                self.send_json({"error": "Wakeword-Tests sind nur über localhost verfügbar"}, 403)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self.send_json({"error": "Ungültige PCM-Frame-Größe"}, 400)
                return
            if not 0 < length <= 128 * 1024:
                self.send_json({"error": "PCM-Frame fehlt oder überschreitet 4 Sekunden"}, 413)
                return
            params = parse_qs(request.query)
            try:
                result = score_test_pcm_frame(
                    self.rfile.read(length),
                    params.get("model", [""])[0],
                    params.get("cutoff", ["0.5"])[0],
                    Path(training_environment()["workspace"]),
                    self._wakeword_store(),
                    reset_worker=params.get("reset", ["0"])[0] == "1",
                )
                self.send_json(result)
            except ValueError as error:
                self.send_json({"error": str(error)}, 400)
            except RuntimeError as error:
                self.send_json({"error": str(error)}, 409)
            except OSError as error:
                self.send_json({"error": str(error)}, 500)
            return
        if request.path.startswith("/api/wakeword/"):
            self._handle_wakeword_post(request)
            return
        if request.path not in {"/api/download", "/api/remove", "/api/synthesize"}:
            self.send_json({"error": "Nicht gefunden"}, 404)
            return
        payload = self._read_json()
        if payload is None:
            return
        try:
            if request.path == "/api/download":
                self._download_voice(payload)
            elif request.path == "/api/remove":
                voice_id = validate_voice_id(payload.get("id"))
                remove_voice(voice_id)
                self.send_json({"voices": installed_voices()})
            else:
                self._synthesize(payload)
        except (ValueError, OSError, URLError, TimeoutError) as error:
            self.send_json({"error": str(error)}, 400)
        except Exception as error:
            LOGGER.exception("Request failed for %s", request.path)
            self.send_json({"error": f"VoxWerk konnte die Anfrage nicht abschließen: {error}"}, 500)

    def _handle_wakeword_post(self, request) -> None:
        store = self._wakeword_store()
        parts = request.path.strip("/").split("/")
        try:
            if parts == ["api", "wakeword", "samples", "upload"]:
                params = parse_qs(request.query)
                wakeword_id = params.get("wakeword", [""])[0]
                kind = params.get("kind", [""])[0]
                filename = params.get("filename", [""])[0]
                title = params.get("title", [""])[0]
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    raise ValueError("Ungültige Upload-Größe") from None
                if not 0 < length <= MAX_SAMPLE_BYTES:
                    self.send_json({"error": "Audio muss zwischen 1 Byte und 40 MB groß sein"}, 413)
                    return
                sample = store.save_sample(wakeword_id, kind, filename, self.rfile.read(length), title)
                self.send_json({"sample": sample}, 201)
                return
            payload = self._read_json()
            if payload is None:
                return
            if parts == ["api", "wakeword", "generate"]:
                wakeword_id = payload.get("wakeword_id")
                project = next((item for item in store.list_wakewords() if item["id"] == wakeword_id), None)
                if project is None:
                    raise ValueError("Wakeword-Projekt nicht gefunden")
                voices = payload.get("voices")
                total = payload.get("total_samples")
                sample_kind = payload.get("kind", "positive")
                if sample_kind not in {"positive", "negative"}:
                    raise ValueError("Sampletyp muss positiv oder negativ sein")
                if not isinstance(voices, dict) or any(not isinstance(key, str) for key in voices):
                    raise ValueError("Stimmenauswahl ist ungültig")
                installed = {str(item["id"]): item for item in installed_voices()}
                if any(voice_id not in installed for voice_id in voices):
                    raise ValueError("Die Auswahl enthält nicht installierte Stimmen")
                allocation = allocate_weighted_samples(total, voices)
                if len(store.list_samples(wakeword_id)) + total > MAX_RECORDS:
                    raise ValueError("Die maximale Projektgröße von 1000 Samples würde überschritten")
                tempo_variation = payload.get("tempo_variation", 20)
                texts = (
                    [project.get("phrase", ""), *project.get("variants", [])]
                    if sample_kind == "positive"
                    else project.get("negative_phrases", [])
                )
                build_sample_plan(texts, total, tempo_variation)
                job_id = uuid.uuid4().hex
                with GENERATION_JOBS_LOCK:
                    if any(job.get("wakeword_id") == wakeword_id and job.get("status") in {"queued", "running"} for job in GENERATION_JOBS.values()):
                        raise ValueError("Für dieses Wakeword läuft bereits eine Samplegenerierung")
                    for existing_id, existing_job in list(GENERATION_JOBS.items()):
                        if len(GENERATION_JOBS) < 64:
                            break
                        if existing_job.get("status") in {"completed", "failed"}:
                            GENERATION_JOBS.pop(existing_id, None)
                    GENERATION_JOBS[job_id] = {
                        "id": job_id,
                        "wakeword_id": wakeword_id,
                        "status": "queued",
                        "kind": sample_kind,
                        "completed": 0,
                        "total": total,
                        "tempo_variation": tempo_variation,
                        "allocation": allocation,
                    }
                    job = dict(GENERATION_JOBS[job_id])
                threading.Thread(
                    target=_run_sample_generation,
                    args=(job_id, wakeword_id, allocation, sample_kind, tempo_variation, voices, int(job_id[:12], 16)),
                    daemon=True,
                ).start()
                self.send_json({"job": job}, 202)
            elif parts == ["api", "wakeword", "profiles", "save"]:
                profile = store.save_profile(payload, payload.get("id"))
                self.send_json({"profile": profile})
            elif parts == ["api", "wakeword", "profiles", "delete"]:
                store.delete_profile(payload.get("id"))
                self.send_json({"profiles": store.list_profiles()})
            elif parts == ["api", "wakeword", "save"]:
                project = store.save_wakeword(payload, payload.get("id"))
                self.send_json({"wakeword": project})
            elif parts == ["api", "wakeword", "delete"]:
                wakeword_id = payload.get("id")
                with TRAINING_JOBS_LOCK:
                    if any(job.get("wakeword_id") == wakeword_id and job.get("status") in {"queued", "running"} for job in TRAINING_JOBS.values()):
                        raise ValueError("Wakeword kann während eines Trainings nicht gelöscht werden")
                store.delete_wakeword(wakeword_id)
                self.send_json({"wakewords": store.list_wakewords()})
            elif parts == ["api", "wakeword", "training", "start"]:
                self._start_wakeword_training(store, payload)
            elif parts == ["api", "wakeword", "training", "stop"]:
                run_id = payload.get("run_id")
                with TRAINING_JOBS_LOCK:
                    job = TRAINING_JOBS.get(str(run_id))
                    stop_event = TRAINING_STOP_EVENTS.get(str(run_id))
                    if job is None or job.get("status") not in {"queued", "running"} or stop_event is None:
                        raise ValueError("Für diesen Trainingslauf ist kein aktiver Abbruch möglich")
                    stop_event.set()
                    job.update({"status": "running", "stage": "Abbruch wird ausgeführt"})
                self.send_json({"run_id": str(run_id), "status": "stopping"}, 202)
            elif len(parts) == 5 and parts[2] == "samples" and parts[4] == "edit":
                sample = store.update_sample(parts[3], payload.get("sample_id"), payload)
                self.send_json({"sample": sample})
            elif len(parts) == 5 and parts[2] == "samples" and parts[4] == "delete-many":
                deleted = store.delete_samples(parts[3], payload.get("sample_ids"))
                self.send_json({"deleted": len(deleted), "samples": store.list_samples(parts[3])})
            elif len(parts) == 5 and parts[2] == "samples" and parts[4] == "delete":
                store.delete_sample(parts[3], payload.get("sample_id"))
                self.send_json({"samples": store.list_samples(parts[3])})
            elif len(parts) == 5 and parts[2] == "runs" and parts[4] == "create":
                run = store.create_run(parts[3], payload)
                self.send_json({"run": run}, 201)
            elif len(parts) == 6 and parts[2] == "runs" and parts[5] == "delete":
                with TRAINING_JOBS_LOCK:
                    if TRAINING_JOBS.get(parts[4], {}).get("status") in {"queued", "running"}:
                        raise ValueError("Trainingslauf kann während der Ausführung nicht gelöscht werden")
                store.delete_run(parts[3], parts[4])
                self.send_json({"runs": store.list_runs(parts[3])})
            else:
                self.send_json({"error": "Nicht gefunden"}, 404)
        except (ValueError, OSError) as error:
            self.send_json({"error": str(error)}, 400)
        except Exception as error:
            LOGGER.exception("Wakeword request failed for %s", request.path)
            self.send_json({"error": f"VoxWerk konnte die Wakeword-Anfrage nicht abschließen: {error}"}, 500)

    def _start_wakeword_training(self, store: WakewordStore, payload: dict[str, object]) -> None:
        wakeword_id = payload.get("wakeword_id")
        project = next((item for item in store.list_wakewords() if item["id"] == wakeword_id), None)
        if project is None:
            raise ValueError("Wakeword-Projekt nicht gefunden")
        run_id = payload.get("run_id")
        run = next((item for item in store.list_runs(str(wakeword_id)) if item["id"] == run_id), None)
        if run is None:
            raise ValueError("Datensatz-Snapshot nicht gefunden")
        if run.get("status") != "prepared" or run.get("training"):
            raise ValueError("Training kann nur aus einem vorbereiteten Snapshot gestartet werden")
        profile = run.get("profile")
        if profile is None or profile.get("engine") != "microwakeword":
            raise ValueError("Das Snapshot-Profil muss die Engine microWakeWord verwenden")

        environment = training_environment()
        if not environment["ready"]:
            raise ValueError(str(environment["message"]))
        parameters = validate_training_parameters(payload)
        available = {item["id"] for item in environment["datasets"] if item["available"]}
        if any(weight > 0 and dataset_id not in available for dataset_id, weight in parameters["dataset_weights"].items()):
            raise ValueError("Ein aktivierter Feature-Datensatz fehlt in der microWakeWord-Umgebung")

        run_dir = store._project_dir(str(wakeword_id)) / "runs" / run["id"]
        _wav_samples(run_dir, run)

        training = {
            "stage": "Wartet auf Start",
            "progress": 0,
            "parameters": parameters,
            "log_file": "training.log",
            "model_directory": "model",
            "created_at": int(time.time()),
        }
        with TRAINING_JOBS_LOCK:
            if any(job.get("status") in {"queued", "running"} for job in TRAINING_JOBS.values()):
                    raise ValueError("Es läuft bereits ein microWakeWord-Training")
            for job_id, job in list(TRAINING_JOBS.items()):
                if len(TRAINING_JOBS) < 32:
                    break
                if job.get("status") in {"completed", "failed"}:
                    TRAINING_JOBS.pop(job_id, None)
            run = store.update_run(str(wakeword_id), run["id"], status="queued", training=training)
            TRAINING_JOBS[run["id"]] = {"status": "queued", "wakeword_id": str(wakeword_id)}
            TRAINING_STOP_EVENTS[run["id"]] = threading.Event()
        thread = threading.Thread(
            target=_run_wakeword_training,
            args=(str(wakeword_id), run["id"], environment, parameters),
            daemon=True,
        )
        thread.start()
        self.send_json({"run": run}, 202)

    def _download_voice(self, payload: dict[str, object]) -> None:
        voice_id = validate_voice_id(payload.get("id"))
        with urlopen(CATALOG_URL, timeout=20) as response:
            catalog_bytes = response.read(MAX_CATALOG_BYTES + 1)
        if len(catalog_bytes) > MAX_CATALOG_BYTES:
            raise ValueError("Der Piper-Katalog überschreitet die erlaubte Größe")
        raw_catalog = json.loads(catalog_bytes)
        entry = next((item for item in catalog_entries(raw_catalog) if item["id"] == voice_id), None)
        if entry is None:
            raise ValueError("Stimme ist nicht im offiziellen Piper-Katalog enthalten")
        voice_dir = data_dir() / "voices" / voice_id
        model_path = voice_dir / f"{voice_id}.onnx"
        metadata_path = voice_dir / f"{voice_id}.onnx.json"
        if not model_path.is_file():
            _download_file(str(entry["model_path"]), str(entry["model_md5"]), model_path, int(entry["size_bytes"]))
        if not metadata_path.is_file():
            metadata_info = next(item for item in raw_catalog[voice_id]["files"].items() if item[0] == entry["metadata_path"])[1]
            _download_file(str(entry["metadata_path"]), str(entry["metadata_md5"]), metadata_path, int(metadata_info.get("size_bytes", 0)))
        self.send_json({"voice": next((item for item in installed_voices() if item["id"] == voice_id), None)})

    def _synthesize(self, payload: dict[str, object]) -> None:
        text = payload.get("text")
        if not isinstance(text, str) or not text.strip() or len(text.strip()) > MAX_TEXT_LENGTH:
            raise ValueError("Text fehlt oder überschreitet 2500 Zeichen")
        voice_id = validate_voice_id(payload.get("voice"))
        try:
            rate = min(1.5, max(0.65, float(payload.get("rate", 1.0))))
        except (TypeError, ValueError):
            rate = 1.0
        voice = next((item for item in installed_voices() if item["id"] == voice_id), None)
        if voice is None:
            raise ValueError("Stimme ist nicht installiert")
        audio = synthesize_wav(text.strip(), voice_id, rate)
        if payload.get("save") is True:
            recording_dir = data_dir() / "recordings"
            recording_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns() % 1_000_000:06d}_{safe_slug(str(voice['name']))}_{safe_slug(text)}.wav"
            (recording_dir / filename).write_bytes(audio)
            self.send_json({"filename": filename, "audio_url": f"/api/audio/{filename}"})
            return
        self.send_bytes(audio, "audio/wav")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: localhost only)")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), RequestHandler)
    LOGGER.info("VoxWerk is ready at http://%s:%d", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("Shutting down")
    finally:
        server.server_close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()