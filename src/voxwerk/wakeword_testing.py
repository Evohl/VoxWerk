from __future__ import annotations

import atexit
import base64
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import tempfile
import time
import wave
from pathlib import Path, PurePosixPath
from typing import Any

from .wakeword_data import WakewordStore, validate_slug
from .wakeword_training import training_environment

MAX_TEST_AUDIO_BYTES = 50 * 1024 * 1024
MAX_TEST_AUDIO_SECONDS = 30
MAX_PCM_FRAME_BYTES = 128 * 1024
RUN_ID_PATTERN = re.compile(r"run-[0-9]{8}-[0-9]{6}-[a-f0-9]{8}\Z")
TEST_WORKER_LOCK = threading.RLock()
TEST_WORKER: subprocess.Popen[str] | None = None


def _stop_test_worker() -> None:
    global TEST_WORKER
    with TEST_WORKER_LOCK:
        if TEST_WORKER and TEST_WORKER.poll() is None:
            TEST_WORKER.terminate()
            try:
                TEST_WORKER.wait(timeout=2)
            except subprocess.TimeoutExpired:
                TEST_WORKER.kill()
        TEST_WORKER = None


atexit.register(_stop_test_worker)


def _model_cutoff(model_path: Path) -> float:
    manifest_path = model_path.with_suffix(".json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cutoff = float(manifest.get("micro", {}).get("probability_cutoff", 0.5))
        return min(1.0, max(0.0, cutoff)) if math.isfinite(cutoff) else 0.5
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return 0.5


def list_test_models(workspace: Path, store: WakewordStore) -> list[dict[str, Any]]:
    workspace = workspace.resolve()
    models: list[dict[str, Any]] = []
    external_roots = ("export", "trained_models", "custom_wakewords")
    for root_name in external_roots:
        root = workspace / root_name
        if not root.is_dir() or root.is_symlink():
            continue
        for model_path in sorted(root.rglob("*.tflite")):
            if model_path.is_symlink() or not model_path.is_file():
                continue
            relative_path = model_path.relative_to(workspace).as_posix()
            label = model_path.stem.replace("_", " ")
            parts = model_path.relative_to(root).parts
            if root_name in {"trained_models", "custom_wakewords"} and parts:
                label = f"{parts[0].replace('_', ' ')} · {label}"
            manifest_path = model_path.with_suffix(".json")
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                label = str(manifest.get("wake_word") or label)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass
            models.append({
                "id": f"wakeword/{relative_path}",
                "label": label,
                "filename": model_path.name,
                "source": root_name,
                "cutoff": _model_cutoff(model_path),
            })

    for project in store.list_wakewords():
        for run in store.list_runs(project["id"]):
            if run.get("status") != "completed":
                continue
            run_id = run.get("id", "")
            if not isinstance(run_id, str) or not RUN_ID_PATTERN.fullmatch(run_id):
                continue
            run_dir = store._project_dir(project["id"]) / "runs" / run_id
            model_directory = run.get("training", {}).get("model_directory", "model")
            relative_directory = PurePosixPath(str(model_directory))
            if relative_directory.is_absolute() or ".." in relative_directory.parts:
                continue
            model_root = run_dir.joinpath(*relative_directory.parts)
            if model_root.is_symlink() or not model_root.is_dir():
                continue
            for model_path in sorted(model_root.rglob("*.tflite")):
                if model_path.is_symlink() or not model_path.is_file():
                    continue
                relative_model = model_path.relative_to(model_root).as_posix()
                models.append({
                    "id": f"voxwerk/{project['id']}/{run_id}/{relative_model}",
                    "label": str(project.get("name") or project["id"]),
                    "filename": model_path.name,
                    "source": "VoxWerk",
                    "cutoff": _model_cutoff(model_path),
                })

    return sorted(models, key=lambda item: (item["source"] != "export", item["label"].casefold(), item["id"]))


def _resolve_model(model_id: object, workspace: Path, store: WakewordStore) -> Path:
    if not isinstance(model_id, str):
        raise ValueError("Testmodell fehlt")
    parts = PurePosixPath(model_id).parts
    if not parts or PurePosixPath(model_id).is_absolute() or ".." in parts or "\\" in model_id:
        raise ValueError("Ungültige Testmodell-ID")

    if parts[0] == "wakeword" and len(parts) >= 2:
        root = workspace.resolve()
        model_path = root.joinpath(*parts[1:])
        allowed_roots = [(root / name).resolve() for name in ("export", "trained_models", "custom_wakewords")]
        resolved = model_path.resolve()
        if not any(resolved.is_relative_to(allowed_root) for allowed_root in allowed_roots):
            raise ValueError("Testmodell liegt außerhalb der freigegebenen Wakeword-Ordner")
    elif parts[0] == "voxwerk" and len(parts) >= 4:
        wakeword_id = validate_slug(parts[1])
        run_id = parts[2]
        if not RUN_ID_PATTERN.fullmatch(run_id):
            raise ValueError("Ungültiger Trainingslauf")
        run = next((item for item in store.list_runs(wakeword_id) if item.get("id") == run_id), None)
        if run is None or run.get("status") != "completed":
            raise ValueError("Das VoxWerk-Modell ist noch nicht fertig")
        run_dir = store._project_dir(wakeword_id) / "runs" / run_id
        model_directory = PurePosixPath(str(run.get("training", {}).get("model_directory", "model")))
        if model_directory.is_absolute() or ".." in model_directory.parts:
            raise ValueError("Ungültiger Modellpfad im Trainingslauf")
        model_root = run_dir.joinpath(*model_directory.parts).resolve()
        resolved = run_dir.joinpath(*model_directory.parts, *parts[3:]).resolve()
        if not resolved.is_relative_to(model_root):
            raise ValueError("Testmodell liegt außerhalb des Trainingslaufs")
    else:
        raise ValueError("Unbekanntes Testmodell")

    if resolved.suffix.lower() != ".tflite" or not resolved.is_file():
        raise ValueError("TFLite-Testmodell nicht gefunden")
    return resolved


def _model_label(model_id: str, model_path: Path, store: WakewordStore) -> str:
    parts = PurePosixPath(model_id).parts
    if parts[0] == "voxwerk" and len(parts) > 1:
        project = next((item for item in store.list_wakewords() if item["id"] == parts[1]), None)
        if project:
            return str(project.get("name") or project["id"])
    if parts[0] == "wakeword":
        try:
            manifest = json.loads(model_path.with_suffix(".json").read_text(encoding="utf-8"))
            label = manifest.get("wake_word")
            if isinstance(label, str) and label.strip():
                return label.strip()
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    return model_path.stem.replace("_", " ")


def _score_with_worker(python: Path, workspace: Path, model_path: Path, wav_path: Path, reset: bool) -> float:
    global TEST_WORKER
    source_root = str(Path(__file__).resolve().parent.parent)
    environment = os.environ.copy()
    existing_path = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(item for item in (source_root, existing_path) if item)
    with TEST_WORKER_LOCK:
        if TEST_WORKER is None or TEST_WORKER.poll() is not None:
            TEST_WORKER = subprocess.Popen(
                [str(python), "-u", "-m", "voxwerk.wakeword_test_worker", str(workspace)],
                cwd=workspace,
                env=environment,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        if TEST_WORKER.stdin is None or TEST_WORKER.stdout is None:
            raise RuntimeError("Wakeword-Testworker konnte nicht gestartet werden")
        TEST_WORKER.stdin.write(json.dumps({"model_path": str(model_path), "audio_path": str(wav_path), "reset": reset}) + "\n")
        TEST_WORKER.stdin.flush()
        response = TEST_WORKER.stdout.readline()
        if not response:
            _stop_test_worker()
            raise RuntimeError("Wakeword-Testworker wurde unerwartet beendet")
        result = json.loads(response)
        if "error" in result:
            raise RuntimeError(f"Modell konnte das Audio nicht auswerten: {result['error']}")
        return float(result["score"])


def score_test_audio(
    audio_bytes: bytes,
    model_id: object,
    cutoff: object,
    workspace: Path,
    store: WakewordStore,
    live: bool = False,
    reset_worker: bool | None = None,
) -> dict[str, Any]:
    if not audio_bytes or len(audio_bytes) > MAX_TEST_AUDIO_BYTES:
        raise ValueError("Audio fehlt oder überschreitet 50 MB")
    if isinstance(cutoff, bool):
        raise ValueError("Cutoff muss zwischen 0 und 1 liegen")
    try:
        cutoff_value = float(cutoff)
    except (TypeError, ValueError):
        raise ValueError("Cutoff muss zwischen 0 und 1 liegen") from None
    if not math.isfinite(cutoff_value) or not 0 <= cutoff_value <= 1:
        raise ValueError("Cutoff muss zwischen 0 und 1 liegen")

    environment = training_environment()
    if not environment["ready"]:
        raise ValueError(str(environment["message"]))
    model_path = _resolve_model(model_id, Path(environment["workspace"]), store)
    model_label = _model_label(str(model_id), model_path, store)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg ist erforderlich, um Browser- und Audiodateien zu testen")

    with tempfile.TemporaryDirectory(prefix="voxwerk-wakeword-test-") as temporary_directory:
        temporary_root = Path(temporary_directory)
        input_path = temporary_root / "input.audio"
        wav_path = temporary_root / "input.wav"
        input_path.write_bytes(audio_bytes)
        converted = subprocess.run(
            [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(input_path), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav_path)],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if converted.returncode != 0:
            raise ValueError("Audioformat konnte nicht gelesen werden: " + converted.stderr[-400:])
        try:
            with wave.open(str(wav_path), "rb") as audio:
                duration = audio.getnframes() / audio.getframerate()
        except (OSError, wave.Error):
            raise ValueError("Audio konnte nicht in WAV umgewandelt werden") from None
        if not 0 < duration <= MAX_TEST_AUDIO_SECONDS:
            raise ValueError("Wakeword-Testaudio muss höchstens 30 Sekunden lang sein")

        reset = not live if reset_worker is None else reset_worker
        score = _score_with_worker(Path(environment["python"]), Path(environment["workspace"]), model_path, wav_path, reset=reset)

    return {
        "score": score,
        "cutoff": cutoff_value,
        "detected": score >= cutoff_value,
        "model_id": model_id,
        "model_label": model_label,
        "duration": round(duration, 3),
        "evaluated_at": int(time.time()),
    }


def score_test_pcm_frame(
    pcm_bytes: bytes,
    model_id: object,
    cutoff: object,
    workspace: Path,
    store: WakewordStore,
    reset_worker: bool = False,
) -> dict[str, Any]:
    if not pcm_bytes or len(pcm_bytes) > MAX_PCM_FRAME_BYTES or len(pcm_bytes) % 2:
        raise ValueError("PCM-Frame ist leer, unvollständig oder größer als 4 Sekunden")
    if isinstance(cutoff, bool):
        raise ValueError("Cutoff muss zwischen 0 und 1 liegen")
    try:
        cutoff_value = float(cutoff)
    except (TypeError, ValueError):
        raise ValueError("Cutoff muss zwischen 0 und 1 liegen") from None
    if not math.isfinite(cutoff_value) or not 0 <= cutoff_value <= 1:
        raise ValueError("Cutoff muss zwischen 0 und 1 liegen")

    environment = training_environment()
    if not environment["ready"]:
        raise ValueError(str(environment["message"]))
    model_path = _resolve_model(model_id, Path(environment["workspace"]), store)
    score, prediction_count, detection_ready = _score_pcm_with_worker(
        Path(environment["python"]),
        Path(environment["workspace"]),
        model_path,
        pcm_bytes,
        reset_worker,
    )
    return {
        "score": score,
        "cutoff": cutoff_value,
        "detected": detection_ready and score > cutoff_value,
        "detection_ready": detection_ready,
        "model_id": model_id,
        "model_label": _model_label(str(model_id), model_path, store),
        "duration": round(len(pcm_bytes) / 2 / 16000, 3),
        "predictions": prediction_count,
        "evaluated_at": int(time.time()),
    }


def _score_pcm_with_worker(
    python: Path,
    workspace: Path,
    model_path: Path,
    pcm_bytes: bytes,
    reset: bool,
) -> tuple[float, int, bool]:
    global TEST_WORKER
    source_root = str(Path(__file__).resolve().parent.parent)
    environment = os.environ.copy()
    existing_path = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(item for item in (source_root, existing_path) if item)
    with TEST_WORKER_LOCK:
        if TEST_WORKER is None or TEST_WORKER.poll() is not None:
            TEST_WORKER = subprocess.Popen(
                [str(python), "-u", "-m", "voxwerk.wakeword_test_worker", str(workspace)],
                cwd=workspace,
                env=environment,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        if TEST_WORKER.stdin is None or TEST_WORKER.stdout is None:
            raise RuntimeError("Wakeword-Testworker konnte nicht gestartet werden")
        TEST_WORKER.stdin.write(json.dumps({
            "mode": "pcm",
            "model_path": str(model_path),
            "pcm": base64.b64encode(pcm_bytes).decode("ascii"),
            "reset": reset,
        }) + "\n")
        TEST_WORKER.stdin.flush()
        response = TEST_WORKER.stdout.readline()
        if not response:
            _stop_test_worker()
            raise RuntimeError("Wakeword-Testworker wurde unerwartet beendet")
        result = json.loads(response)
        if "error" in result:
            raise RuntimeError(f"Modell konnte den Audiostream nicht auswerten: {result['error']}")
        return float(result["score"]), int(result["predictions"]), bool(result["detection_ready"])
