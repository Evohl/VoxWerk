"""Persistent location for user-created VoxWerk data."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


APP_NAME = "voxwerk"
DATA_DIRECTORY_KEY = "data_directory"


def default_data_dir() -> Path:
    configured = os.environ.get("XDG_DATA_HOME")
    base = Path(configured).expanduser() if configured else Path.home() / ".local" / "share"
    return base / APP_NAME


def settings_path() -> Path:
    configured = os.environ.get("XDG_CONFIG_HOME")
    base = Path(configured).expanduser() if configured else Path.home() / ".config"
    return base / APP_NAME / "settings.json"


def _read_settings() -> dict[str, Any]:
    try:
        value = json.loads(settings_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def data_directory_info() -> dict[str, Any]:
    configured = os.environ.get("VOXWERK_DATA_DIR")
    saved = _read_settings().get(DATA_DIRECTORY_KEY)
    if configured:
        path = Path(configured).expanduser().resolve()
        source = "environment"
    elif isinstance(saved, str) and saved.strip():
        path = Path(saved).expanduser().resolve()
        source = "settings"
    else:
        path = default_data_dir().resolve()
        source = "default"
    return {
        "path": path,
        "source": source,
        "locked": bool(configured),
    }


def data_dir() -> Path:
    return data_directory_info()["path"]


def _write_settings(value: dict[str, Any]) -> None:
    target = settings_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, prefix=".settings-", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        temporary.replace(target)
    finally:
        if temporary and temporary.exists():
            temporary.unlink()


def save_data_directory(value: object) -> dict[str, Any]:
    if os.environ.get("VOXWERK_DATA_DIR"):
        raise ValueError("VOXWERK_DATA_DIR ist gesetzt und überschreibt die Datenordner-Auswahl")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Bitte einen absoluten Datenordner angeben")

    candidate = Path(value.strip()).expanduser()
    if not candidate.is_absolute():
        raise ValueError("Bitte einen absoluten Datenordner angeben")
    destination = candidate.resolve()
    source = data_dir().resolve()
    if destination == source:
        return {**data_directory_info(), "copied": False}
    if destination == Path(destination.anchor) or destination in source.parents or source in destination.parents:
        raise ValueError("Datenordner dürfen sich nicht überschneiden")

    if destination.exists():
        if not destination.is_dir():
            raise ValueError("Der ausgewählte Datenpfad ist kein Ordner")
        marker = destination / ".voxwerk-data-root"
        if marker.is_file() and marker.read_text(encoding="utf-8") == "voxwerk-user-data-v1\n":
            _write_settings({**_read_settings(), DATA_DIRECTORY_KEY: str(destination)})
            return {**data_directory_info(), "copied": False}
        if any(destination.iterdir()):
            raise ValueError("Der Zielordner muss leer sein; bestehende Dateien werden nicht überschrieben")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.voxwerk-migration-", dir=destination.parent))
    try:
        if source.is_dir():
            for child in source.iterdir():
                target = staging / child.name
                if child.is_dir() and not child.is_symlink():
                    shutil.copytree(child, target, symlinks=True)
                elif child.is_symlink():
                    target.symlink_to(os.readlink(child))
                else:
                    shutil.copy2(child, target)
        (staging / ".voxwerk-data-root").write_text("voxwerk-user-data-v1\n", encoding="utf-8")
        if destination.exists():
            destination.rmdir()
        staging.rename(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    _write_settings({**_read_settings(), DATA_DIRECTORY_KEY: str(destination)})
    return {**data_directory_info(), "copied": True}