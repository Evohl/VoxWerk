from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import time
import wave
from pathlib import Path, PurePosixPath
from typing import Any, Callable


TRAINING_DATASETS = {
    "speech": {"label": "Alltagssprache", "default_weight": 10, "truncation_strategy": "random"},
    "dinner_party": {"label": "Gespräche", "default_weight": 10, "truncation_strategy": "random"},
    "no_speech": {"label": "Umgebungsgeräusche", "default_weight": 5, "truncation_strategy": "random"},
    "dinner_party_eval": {"label": "Ambient-Auswertung", "default_weight": 0, "truncation_strategy": "split", "required": True},
}


class TrainingStopped(Exception):
    pass


def wakeword_package_dir() -> Path:
    return Path.home() / ".venvs" / "wakeword"


def training_environment() -> dict[str, Any]:
    configured_root = os.environ.get("VOXWERK_WAKEWORD_DIR")
    root = (Path(configured_root).expanduser() if configured_root else wakeword_package_dir()).resolve()
    configured_python = os.environ.get("VOXWERK_WAKEWORD_PYTHON")
    if configured_python:
        python = Path(configured_python).expanduser().resolve()
    else:
        candidates = (root / "bin" / "python", root / ".venv" / "bin" / "python", root / "Scripts" / "python.exe", root / ".venv" / "Scripts" / "python.exe")
        python = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])

    script = root / "scripts" / "generate_positive_features.py"
    trainer = root / "microWakeWord" / "microwakeword" / "model_train_eval.py"
    datasets = []
    for dataset_id, metadata in TRAINING_DATASETS.items():
        dataset_path = root / "negative_datasets" / dataset_id
        splits = ("validation_ambient", "testing_ambient") if metadata.get("required") else ("training",)
        available = dataset_path.is_dir() and any(
            next((dataset_path / split).glob("**/*_mmap"), None) is not None
            for split in splits
        )
        datasets.append({"id": dataset_id, **metadata, "available": available})

    required_eval = root / "negative_datasets" / "dinner_party_eval"
    has_ambient_features = any(
        next((required_eval / split).glob("**/*_mmap"), None) is not None
        for split in ("validation_ambient", "testing_ambient")
    )
    ready = python.is_file() and script.is_file() and trainer.is_file() and has_ambient_features
    message = ""
    if not ready:
        message = "Das Wakeword-Paket ist nicht installiert oder unvollständig. Installiere es unter Einstellungen → Wakeword-Paket."
    return {
        "ready": ready,
        "message": message,
        "datasets": datasets,
        "workspace": root,
        "python": python,
        "package_path": str(root),
        "package_source": "environment" if configured_root else "default",
        "package_locked": bool(configured_root),
    }


def check_wakeword_integration() -> dict[str, Any]:
    environment = training_environment()
    root = Path(environment["workspace"])
    python = Path(environment["python"])
    checks = [
        {"id": "package_directory", "label": "Paketordner", "ok": root.is_dir(), "detail": str(root)},
        {"id": "python", "label": "Python-Umgebung", "ok": python.is_file(), "detail": str(python)},
        {"id": "feature_script", "label": "Feature-Skript", "ok": (root / "scripts" / "generate_positive_features.py").is_file(), "detail": "scripts/generate_positive_features.py"},
        {"id": "trainer", "label": "microWakeWord-Trainer", "ok": (root / "microWakeWord" / "microwakeword" / "model_train_eval.py").is_file(), "detail": "microWakeWord/microwakeword/model_train_eval.py"},
    ]
    ambient = root / "negative_datasets" / "dinner_party_eval"
    has_ambient = any(next((ambient / split).glob("**/*_mmap"), None) is not None for split in ("validation_ambient", "testing_ambient"))
    checks.append({"id": "ambient", "label": "Ambient-Auswertung", "ok": has_ambient, "detail": "dinner_party_eval"})

    imports_ok = False
    import_detail = "Python-Umgebung fehlt"
    if python.is_file():
        try:
            result = subprocess.run(
                [str(python), "-c", "import numpy, tensorflow, ai_edge_litert, pymicro_features, mmap_ninja; from microwakeword.inference import Model"],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=45,
                check=False,
            )
            imports_ok = result.returncode == 0
            import_detail = "Alle Laufzeitmodule importierbar" if imports_ok else (result.stderr.strip().splitlines()[-1] if result.stderr.strip() else f"Importtest fehlgeschlagen ({result.returncode})")
        except (OSError, subprocess.TimeoutExpired) as error:
            import_detail = str(error) or "Importtest-Zeitüberschreitung"
    checks.append({"id": "runtime_imports", "label": "Laufzeit-Abhängigkeiten", "ok": imports_ok, "detail": import_detail})
    ready = all(check["ok"] for check in checks)
    return {
        "package_path": environment["package_path"],
        "package_source": environment["package_source"],
        "package_locked": environment["package_locked"],
        "ready": ready,
        "message": "Wakeword-Paket ist vollständig integriert." if ready else "Mindestens eine Integrationsprüfung ist fehlgeschlagen.",
        "checks": checks,
    }


def validate_training_parameters(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Trainingsparameter sind ungültig")

    def bounded_int(name: str, minimum: int, maximum: int, default: int) -> int:
        value = payload.get(name, default)
        if isinstance(value, bool):
            raise ValueError(f"{name} muss zwischen {minimum} und {maximum} liegen")
        try:
            result = int(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} muss zwischen {minimum} und {maximum} liegen") from None
        if result != value and not (isinstance(value, str) and str(result) == value.strip()):
            raise ValueError(f"{name} muss eine ganze Zahl sein")
        if not minimum <= result <= maximum:
            raise ValueError(f"{name} muss zwischen {minimum} und {maximum} liegen")
        return result

    raw_learning_rate = payload.get("learning_rate", 0.00101)
    if isinstance(raw_learning_rate, str):
        raw_learning_rate = raw_learning_rate.strip().replace(",", ".")
        try:
            learning_rate = float(raw_learning_rate)
        except (TypeError, ValueError):
            raise ValueError("Lernrate muss zwischen 0.00001 und 0.01 liegen") from None
    else:
        try:
            learning_rate = float(raw_learning_rate)
        except (TypeError, ValueError):
            raise ValueError("Lernrate muss zwischen 0.00001 und 0.01 liegen") from None
    if not math.isfinite(learning_rate) or not 0.00001 <= learning_rate <= 0.01:
        raise ValueError("Lernrate muss zwischen 0.00001 und 0.01 liegen")
    learning_rate_steps = (learning_rate - 0.00001) / 0.0001
    if not math.isclose(learning_rate_steps, round(learning_rate_steps), rel_tol=0, abs_tol=1e-8):
        raise ValueError("Lernrate muss in Schritten von 0.0001 ab 0.00001 angegeben werden")

    dataset_weights = payload.get("dataset_weights", {})
    if not isinstance(dataset_weights, dict) or any(dataset_id not in TRAINING_DATASETS for dataset_id in dataset_weights):
        raise ValueError("Datensatz-Auswahl ist ungültig")
    normalized_weights = {}
    for dataset_id, metadata in TRAINING_DATASETS.items():
        weight = dataset_weights.get(dataset_id, metadata["default_weight"])
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or not 0 <= weight <= 50:
            raise ValueError("Datensatz-Gewichte müssen zwischen 0 und 50 liegen")
        if metadata.get("required") and weight != 0:
            raise ValueError("Ambient-Auswertung muss das Gewicht 0 behalten")
        normalized_weights[dataset_id] = float(weight)

    return {
        "training_steps": bounded_int("training_steps", 1000, 50000, 10000),
        "batch_size": bounded_int("batch_size", 16, 512, 128),
        "negative_class_weight": bounded_int("negative_class_weight", 1, 50, 15),
        "learning_rate": learning_rate,
        "dataset_weights": normalized_weights,
    }


def _wav_samples(run_dir: Path, run: dict[str, Any]) -> dict[str, list[Path]]:
    grouped = {"positive": [], "negative": []}
    for sample in run.get("samples", []):
        kind = sample.get("kind")
        if kind not in grouped:
            continue
        relative_path = PurePosixPath(str(sample.get("snapshot_file", "")))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("Snapshot enthält einen ungültigen Dateipfad")
        source = run_dir.joinpath(*relative_path.parts)
        if source.suffix.lower() != ".wav" or source.is_symlink() or not source.is_file():
            raise ValueError("Training unterstützt derzeit nur WAV-Samples aus dem Run-Snapshot")
        try:
            with wave.open(str(source), "rb") as audio:
                duration = audio.getnframes() / audio.getframerate()
                if audio.getnchannels() != 1 or audio.getsampwidth() != 2 or audio.getframerate() < 8000:
                    raise ValueError("Training erwartet WAV in Mono und 16-Bit-PCM")
                minimum, maximum = (0.5, 1.5) if kind == "positive" else (0.3, 3.0)
                if not minimum < duration < maximum:
                    raise ValueError(f"{kind.title()}-WAV muss länger als {minimum} s und kürzer als {maximum} s sein")
        except (wave.Error, OSError):
            raise ValueError("Ein ausgewähltes WAV-Sample ist beschädigt") from None
        grouped[kind].append(source)
    if len(grouped["positive"]) < 5 or len(grouped["negative"]) < 5:
        raise ValueError("Wähle mindestens fünf positive und fünf negative WAV-Samples in passender Länge aus")
    return grouped


def run_training_pipeline(
    workspace: Path,
    python: Path,
    run_dir: Path,
    run: dict[str, Any],
    parameters: dict[str, Any],
    set_stage: Callable[[str, int], None],
    should_stop: Callable[[], bool] | None = None,
) -> Path:
    grouped = _wav_samples(run_dir, run)
    work_dir = run_dir / "training-work"
    sample_dirs = {kind: work_dir / "samples" / kind for kind in grouped}
    feature_dirs = {kind: work_dir / "features" / kind for kind in grouped}
    for kind in grouped:
        sample_dirs[kind].mkdir(parents=True, exist_ok=True)
        feature_dirs[kind].mkdir(parents=True, exist_ok=True)
        for index, source in enumerate(grouped[kind], start=1):
            shutil.copyfile(source, sample_dirs[kind] / f"{index:05d}.wav")

    feature_script = workspace / "scripts" / "generate_positive_features.py"
    log_path = run_dir / "training.log"

    def run_stage(stage: str, command: list[str], start_progress: int, end_progress: int) -> None:
        set_stage(stage, start_progress)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n$ {' '.join(command)}\n")
            log.flush()
            process = subprocess.Popen(command, cwd=workspace, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                if should_stop and should_stop():
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise TrainingStopped("Training wurde abgebrochen")
                if stage == "Modell wird trainiert und quantisiert":
                    try:
                        with log_path.open("rb") as progress_log:
                            progress_log.seek(0, os.SEEK_END)
                            progress_log.seek(max(0, progress_log.tell() - 8192))
                            recent = progress_log.read().decode("utf-8", errors="replace")
                        steps = re.findall(r"Step\s+#(\d+):", recent)
                        if steps:
                            ratio = min(1.0, int(steps[-1]) / max(1, int(parameters["training_steps"])))
                            set_stage(stage, min(94, start_progress + int(ratio * (end_progress - start_progress))))
                    except OSError:
                        pass
                time.sleep(0.2)
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, command)
        if should_stop and should_stop():
            raise TrainingStopped("Training wurde abgebrochen")

    for kind in ("positive", "negative"):
        command = [str(python), str(feature_script), "--input-dir", str(sample_dirs[kind]), "--features-dir", str(feature_dirs[kind])]
        if kind == "negative":
            command.extend(["--min-duration", "0.3", "--max-duration", "3.0"])
        run_stage(f"{kind.title()}-Features werden berechnet", command, 1 if kind == "positive" else 3, 3 if kind == "positive" else 5)

    available_datasets = {item["id"]: item for item in training_environment()["datasets"] if item["available"]}
    if "dinner_party_eval" not in available_datasets:
        raise ValueError("Ambient-Auswertungsdaten fehlen in der microWakeWord-Umgebung")
    features = [
        {"features_dir": str(feature_dirs["positive"]), "sampling_weight": 2.0, "penalty_weight": 1.0, "truth": True, "truncation_strategy": "truncate_start", "type": "mmap"},
        {"features_dir": str(feature_dirs["negative"]), "sampling_weight": 4.0, "penalty_weight": 1.0, "truth": False, "truncation_strategy": "random", "type": "mmap"},
    ]
    for dataset_id, metadata in TRAINING_DATASETS.items():
        if dataset_id not in available_datasets:
            continue
        weight = parameters["dataset_weights"][dataset_id]
        if metadata.get("required") or weight > 0:
            features.append({
                "features_dir": str(workspace / "negative_datasets" / dataset_id),
                "sampling_weight": weight,
                "penalty_weight": 1.0,
                "truth": False,
                "truncation_strategy": metadata["truncation_strategy"],
                "type": "mmap",
            })

    model_dir = run_dir / "model"
    config_path = run_dir / "training_config.yaml"
    config = {
        "window_step_ms": 10,
        "train_dir": str(model_dir),
        "features": features,
        "training_steps": [parameters["training_steps"]],
        "positive_class_weight": [1],
        "negative_class_weight": [parameters["negative_class_weight"]],
        "learning_rates": [parameters["learning_rate"]],
        "batch_size": parameters["batch_size"],
        "time_mask_max_size": [10],
        "time_mask_count": [2],
        "freq_mask_max_size": [6],
        "freq_mask_count": [2],
        "eval_step_interval": min(500, parameters["training_steps"]),
        "clip_duration_ms": 1800,
        "target_minimization": 0.9,
        "minimization_metric": "ambient_false_positives_per_hour",
        "maximization_metric": "average_viable_recall",
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    command = [
        str(python), "-u", "-m", "microwakeword.model_train_eval",
        f"--training_config={config_path}", "--train", "1", "--restore_checkpoint", "0",
        "--test_tf_nonstreaming", "0", "--test_tflite_nonstreaming", "0",
        "--test_tflite_nonstreaming_quantized", "0", "--test_tflite_streaming", "0",
        "--test_tflite_streaming_quantized", "1", "--use_weights", "best_weights",
        "mixednet", "--pointwise_filters", "64,64,64,64", "--repeat_in_block", "1,1,1,1",
        "--mixconv_kernel_sizes", "[5],[7,11],[9,15],[23]", "--residual_connection", "0,0,0,0",
        "--first_conv_filters", "32", "--first_conv_kernel_size", "5", "--stride", "3",
    ]
    run_stage("Modell wird trainiert und quantisiert", command, 5, 95)
    return model_dir