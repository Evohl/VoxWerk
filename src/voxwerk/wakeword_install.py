from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from .wakeword_feature_generator import __file__ as feature_generator_path
from .wakeword_training import wakeword_package_dir

UPSTREAM_REPOSITORY = "https://github.com/kahrendt/microWakeWord.git"
DATASET_BASE_URL = "https://huggingface.co/datasets/kahrendt/microwakeword/resolve/main/"
DATASET_ARCHIVES = (
    ("dinner_party_eval.zip", "dinner_party_eval"),
    ("dinner_party.zip", "dinner_party"),
    ("no_speech.zip", "no_speech"),
    ("speech.zip", "speech"),
)
MINIMUM_FREE_BYTES = 40 * 1024**3
INSTALL_MARKER = ".voxwerk-wakeword-install.json"

INSTALL_LOCK = threading.RLock()
INSTALL_JOB: dict[str, Any] | None = None
INSTALL_CANCEL: threading.Event | None = None
LOGGER = logging.getLogger("voxwerk.wakeword_install")


class InstallCancelled(Exception):
    pass


def _set_job(job_id: str, **changes: Any) -> None:
    with INSTALL_LOCK:
        if INSTALL_JOB and INSTALL_JOB["id"] == job_id:
            INSTALL_JOB.update(changes, updated_at=time.time())


def get_install_job(job_id: object) -> dict[str, Any] | None:
    with INSTALL_LOCK:
        if INSTALL_JOB and INSTALL_JOB["id"] == job_id:
            return dict(INSTALL_JOB)
    return None


def _python_for_venv() -> Path:
    for name in ("python3.11", "python3.12", "python3.10"):
        executable = shutil.which(name)
        if executable:
            return Path(executable)
    if (3, 10) <= sys.version_info[:2] <= (3, 12):
        return Path(sys.executable)
    raise RuntimeError("Für TensorFlow wird Python 3.10 bis 3.12 benötigt; installiere Python 3.11 oder 3.12 und starte VoxWerk erneut.")


def _check_cancel(cancel: threading.Event) -> None:
    if cancel.is_set():
        raise InstallCancelled("Installation abgebrochen")


def _run_command(command: list[str], cwd: Path, log_path: Path, stage: str, cancel: threading.Event, job_id: str) -> None:
    LOGGER.info("Wakeword-Paket: %s", stage)
    _set_job(job_id, stage=stage, progress=0, message=stage)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n$ " + " ".join(command) + "\n")
        log.flush()
        process = subprocess.Popen(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            if cancel.wait(0.25):
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise InstallCancelled("Installation abgebrochen")
        if process.returncode:
            raise RuntimeError(f"{stage} fehlgeschlagen (Exit-Code {process.returncode}); Details: {log_path}")


def _safe_extract(archive: Path, target_root: Path, cancel: threading.Event, job_id: str, base_progress: int, progress_span: int) -> None:
    with zipfile.ZipFile(archive) as zip_file:
        members = zip_file.infolist()
        total = max(1, len(members))
        root = target_root.resolve()
        for index, member in enumerate(members):
            _check_cancel(cancel)
            relative = PurePosixPath(member.filename)
            if relative.is_absolute() or ".." in relative.parts or "\\" in member.filename:
                raise RuntimeError("Datensatzarchiv enthält einen ungültigen Pfad")
            destination = (target_root / Path(*relative.parts)).resolve()
            if not destination.is_relative_to(root):
                raise RuntimeError("Datensatzarchiv verlässt den Zielordner")
            mode = member.external_attr >> 16
            if mode & 0o170000 == 0o120000:
                raise RuntimeError("Datensatzarchiv enthält einen symbolischen Link")
            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with zip_file.open(member) as source, destination.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
            if index % 32 == 0 or index + 1 == total:
                progress = base_progress + int((index + 1) / total * progress_span)
                _set_job(job_id, progress=progress, message=f"Entpacke {archive.stem}: {index + 1}/{total} Dateien")


def _download_dataset(archive_name: str, dataset_name: str, workspace: Path, cancel: threading.Event, job_id: str, index: int) -> None:
    datasets_dir = workspace / "negative_datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    destination = datasets_dir / dataset_name
    if destination.is_dir() and any(destination.rglob("*_mmap")):
        _set_job(job_id, progress=45 + index * 13, message=f"{dataset_name} ist bereits installiert")
        return

    partial = datasets_dir / f".{archive_name}.part"
    existing_size = partial.stat().st_size if partial.exists() else 0
    if not zipfile.is_zipfile(partial):
        request = urllib.request.Request(DATASET_BASE_URL + archive_name, headers={"Range": f"bytes={existing_size}-"} if existing_size else {})
        with urllib.request.urlopen(request, timeout=60) as response:
            append = existing_size > 0 and response.status == 206
            if not append:
                existing_size = 0
            total_bytes = int(response.headers.get("Content-Length", "0")) + existing_size
            mode = "ab" if append else "wb"
            downloaded = existing_size
            with partial.open(mode) as output:
                while True:
                    _check_cancel(cancel)
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    downloaded += len(chunk)
                    if total_bytes:
                        percent = min(12, int(downloaded / total_bytes * 12))
                        _set_job(job_id, progress=45 + index * 13 + percent, message=f"Lade {dataset_name}: {downloaded // (1024 * 1024)} MB")
    _check_cancel(cancel)
    if not zipfile.is_zipfile(partial):
        raise RuntimeError(f"Download von {archive_name} ist kein gültiges ZIP; Teil-Download bleibt zur Wiederaufnahme erhalten")
    extraction_root = datasets_dir / f".{dataset_name}.extracting"
    if extraction_root.exists():
        shutil.rmtree(extraction_root)
    extraction_root.mkdir()
    _safe_extract(partial, extraction_root, cancel, job_id, 45 + index * 13, 12)
    extracted_dataset = extraction_root / dataset_name
    if not extracted_dataset.is_dir():
        raise RuntimeError(f"Archiv {archive_name} enthält den erwarteten Ordner {dataset_name} nicht")
    if destination.exists():
        if any(destination.iterdir()):
            raise RuntimeError(f"{destination} existiert bereits und wird nicht überschrieben")
        destination.rmdir()
    extracted_dataset.rename(destination)
    shutil.rmtree(extraction_root)
    partial.unlink(missing_ok=True)


def _install_package(job_id: str, cancel: threading.Event) -> None:
    path = wakeword_package_dir().resolve()
    marker_path = path / INSTALL_MARKER
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"{path} ist kein Ordner")
        if not marker_path.is_file() and any(path.iterdir()):
            raise ValueError(f"{path} enthält fremde Dateien; sie werden nicht überschrieben. Entferne oder verschiebe den Ordner und starte die Installation erneut.")

    parent = path
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    if shutil.disk_usage(parent).free < MINIMUM_FREE_BYTES:
        raise ValueError("Am Installationsort sind weniger als 40 GB frei; die entpackten Datensätze belegen rund 25 GB, dazu kommen Python-Umgebung und temporärer Download.")

    path.mkdir(parents=True, exist_ok=True)
    if not marker_path.is_file():
        marker_path.write_text(json.dumps({"managed_by": "voxwerk", "version": 1}) + "\n", encoding="utf-8")

    _check_cancel(cancel)
    python = _python_for_venv()
    micro_wake_word = path / "microWakeWord"
    clone_tmp = path / ".microWakeWord-clone"
    log_path = path / "voxwerk-install.log"
    log_path.touch(exist_ok=True)
    if not micro_wake_word.is_dir():
        if clone_tmp.exists():
            shutil.rmtree(clone_tmp)
        _run_command(["git", "clone", "--depth", "1", UPSTREAM_REPOSITORY, str(clone_tmp)], path, log_path, "Lade microWakeWord-Code", cancel, job_id)
        clone_tmp.rename(micro_wake_word)

    _check_cancel(cancel)
    venv_python = path / "bin" / "python"
    if not venv_python.is_file():
        _run_command([str(python), "-m", "venv", str(path)], path, log_path, "Erstelle isolierte Python-Umgebung", cancel, job_id)
    _run_command([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"], path, log_path, "Aktualisiere pip", cancel, job_id)
    _run_command([str(venv_python), "-m", "pip", "install", "-e", str(micro_wake_word)], path, log_path, "Installiere microWakeWord und TensorFlow", cancel, job_id)

    scripts = path / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    generator_target = scripts / "generate_positive_features.py"
    if not generator_target.exists():
        shutil.copyfile(Path(feature_generator_path), generator_target)

    for index, (archive_name, dataset_name) in enumerate(DATASET_ARCHIVES):
        _check_cancel(cancel)
        LOGGER.info("Wakeword-Datensatz: %s", dataset_name)
        _download_dataset(archive_name, dataset_name, path, cancel, job_id, index)

    ambient_data = path / "negative_datasets" / "dinner_party_eval"
    if not any(ambient_data.glob("**/*_mmap")):
        raise RuntimeError("Ambient-Auswertungsarchiv enthält keine verwendbaren microWakeWord-Features")

    _check_cancel(cancel)
    _set_job(job_id, status="completed", stage="Abgeschlossen", progress=100, message="Das Wakeword-Paket ist installiert.", error="")


def start_install() -> dict[str, Any]:
    global INSTALL_JOB, INSTALL_CANCEL
    path = wakeword_package_dir()
    with INSTALL_LOCK:
        if INSTALL_JOB and INSTALL_JOB.get("status") in {"queued", "running", "cancelling"}:
            raise ValueError("Eine Wakeword-Installation läuft bereits.")
        job_id = f"install-{int(time.time())}-{os.urandom(4).hex()}"
        INSTALL_CANCEL = threading.Event()
        INSTALL_JOB = {
            "id": job_id,
            "path": str(path.resolve()),
            "status": "running",
            "stage": "Wird vorbereitet",
            "progress": 0,
            "message": "Installation wird vorbereitet.",
            "error": "",
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        cancel = INSTALL_CANCEL

    LOGGER.info("Wakeword-Paket-Installation/Update gestartet: %s", path)

    def worker() -> None:
        try:
            _install_package(job_id, cancel)
            LOGGER.info("Wakeword-Paket-Installation/Update abgeschlossen: %s", path)
        except InstallCancelled as error:
            _set_job(job_id, status="cancelled", stage="Abgebrochen", message=str(error), error="")
            LOGGER.info("Wakeword-Paket-Installation/Update abgebrochen: %s", path)
        except Exception as error:
            _set_job(job_id, status="failed", stage="Fehlgeschlagen", message=str(error), error=str(error))
            LOGGER.error("Wakeword-Paket-Installation/Update fehlgeschlagen: %s: %s", path, error)

    threading.Thread(target=worker, name="voxwerk-wakeword-install", daemon=True).start()
    return get_install_job(job_id) or {}


def cancel_install(job_id: object) -> dict[str, Any] | None:
    with INSTALL_LOCK:
        if not INSTALL_JOB or INSTALL_JOB.get("id") != job_id or INSTALL_JOB.get("status") != "running":
            return None
        if INSTALL_CANCEL:
            INSTALL_CANCEL.set()
        INSTALL_JOB.update(status="cancelling", message="Installation wird abgebrochen.", updated_at=time.time())
        return dict(INSTALL_JOB)
