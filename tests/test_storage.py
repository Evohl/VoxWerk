from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from voxwerk.storage import data_dir, data_directory_info, save_data_directory


class VoxWerkDataDirectoryTests(unittest.TestCase):
    def test_defaults_to_xdg_data_home(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", {"XDG_DATA_HOME": directory, "XDG_CONFIG_HOME": str(Path(directory) / "config"), "VOXWERK_DATA_DIR": ""}, clear=False):
            self.assertEqual(data_dir(), Path(directory) / "voxwerk")

    def test_migrates_existing_data_and_persists_selected_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old_data = root / "old" / "voxwerk"
            old_data.mkdir(parents=True)
            (old_data / "recordings").mkdir()
            (old_data / "recordings" / "take.wav").write_bytes(b"audio")
            sample_directory = old_data / "wakeword" / "wakewords" / "nova" / "samples" / "positive"
            sample_directory.mkdir(parents=True)
            (sample_directory / "sample.wav").write_bytes(b"sample")
            destination = root / "external" / "voxwerk-data"
            config_home = root / "config"
            environment = {
                "XDG_DATA_HOME": str(root / "old"),
                "XDG_CONFIG_HOME": str(config_home),
                "VOXWERK_DATA_DIR": "",
            }
            with patch.dict("os.environ", environment, clear=False):
                result = save_data_directory(str(destination))
                self.assertTrue(result["copied"])
                self.assertEqual(data_dir(), destination)
                self.assertEqual((destination / "recordings" / "take.wav").read_bytes(), b"audio")
                self.assertEqual((destination / "wakeword" / "wakewords" / "nova" / "samples" / "positive" / "sample.wav").read_bytes(), b"sample")
                self.assertEqual((old_data / "recordings" / "take.wav").read_bytes(), b"audio")
                saved = json.loads((config_home / "voxwerk" / "settings.json").read_text(encoding="utf-8"))
                self.assertEqual(saved["data_directory"], str(destination))

    def test_rejects_nonempty_overlapping_or_relative_destination(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", {"XDG_DATA_HOME": directory, "VOXWERK_DATA_DIR": ""}, clear=False):
            root = Path(directory) / "voxwerk"
            nonempty = Path(directory) / "not-empty"
            nonempty.mkdir()
            (nonempty / "user-file").touch()
            for target in ("relative", str(nonempty), str(root / "nested"), str(Path(directory))):
                with self.subTest(target=target), self.assertRaises(ValueError):
                    save_data_directory(target)

    def test_environment_override_locks_data_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            override = Path(directory) / "override"
            with patch.dict("os.environ", {"VOXWERK_DATA_DIR": str(override)}, clear=False):
                self.assertEqual(data_directory_info()["source"], "environment")
                self.assertTrue(data_directory_info()["locked"])
                with self.assertRaisesRegex(ValueError, "überschreibt"):
                    save_data_directory(str(Path(directory) / "other"))


if __name__ == "__main__":
    unittest.main()
