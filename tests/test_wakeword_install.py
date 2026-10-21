from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path
from threading import Event

from voxwerk.wakeword_install import _safe_extract


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
