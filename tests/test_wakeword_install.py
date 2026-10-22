from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path
from threading import Event
from unittest.mock import patch

from voxwerk.wakeword_install import _install_package, _safe_extract, start_install


class WakewordPackageInstallTests(unittest.TestCase):
    def test_start_install_logs_install_or_update(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / ".venv" / "wakeword"
            with (
                patch("voxwerk.wakeword_install.wakeword_package_dir", return_value=package),
                patch("voxwerk.wakeword_install.threading.Thread") as thread,
                patch("voxwerk.wakeword_install.INSTALL_JOB", None),
                patch("voxwerk.wakeword_install.INSTALL_CANCEL", None),
                self.assertLogs("voxwerk.wakeword_install", level="INFO") as captured,
            ):
                start_install()
        thread.return_value.start.assert_called_once()
        self.assertIn("Wakeword-Paket-Installation/Update gestartet", captured.output[0])

    def test_refuses_to_install_into_a_folder_with_foreign_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / ".venv" / "wakeword"
            package.mkdir(parents=True)
            (package / "user-file.txt").write_text("keep", encoding="utf-8")
            with patch("voxwerk.wakeword_install.wakeword_package_dir", return_value=package):
                with self.assertRaisesRegex(ValueError, "fremde Dateien"):
                    _install_package("install-test", Event())
            self.assertEqual((package / "user-file.txt").read_text(encoding="utf-8"), "keep")


class WakewordArchiveExtractionTests(unittest.TestCase):
    def test_extracts_dataset_members_under_target_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "dataset.zip"
            with zipfile.ZipFile(archive, "w") as zip_file:
                zip_file.writestr("dataset/training/example_mmap/metadata.json", "{}")
            target = root / "extracted"
            target.mkdir()
            _safe_extract(archive, target, Event(), "install-test", 0, 100)
            self.assertEqual((target / "dataset/training/example_mmap/metadata.json").read_text(), "{}")

    def test_rejects_archive_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "malicious.zip"
            with zipfile.ZipFile(archive, "w") as zip_file:
                zip_file.writestr("../../escape.txt", "outside")
            target = root / "extracted"
            target.mkdir()
            with self.assertRaisesRegex(RuntimeError, "ungültigen Pfad"):
                _safe_extract(archive, target, Event(), "install-test", 0, 100)
            self.assertFalse((root.parent / "escape.txt").exists())


if __name__ == "__main__":
    unittest.main()
