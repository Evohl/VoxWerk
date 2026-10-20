from __future__ import annotations

import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from voxwerk.app import _api_client_allowed, _download_file, catalog_entries, filter_catalog, validate_voice_id


class VoiceCatalogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = {
            "de_DE-test-medium": {
                "name": "test",
                "language": {"code": "de_DE", "name_english": "German"},
                "quality": "medium",
                "num_speakers": 1,
                "files": {
                    "de/de_DE/test/medium/de_DE-test-medium.onnx": {"size_bytes": 123, "md5_digest": "a" * 32},
                    "de/de_DE/test/medium/de_DE-test-medium.onnx.json": {"size_bytes": 45, "md5_digest": "b" * 32},
                },
            },
            "bad/voice": {"files": {}},
        }

    def test_catalog_entries_include_only_downloadable_voice_models(self) -> None:
        entries = catalog_entries(self.catalog)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["id"], "de_DE-test-medium")
        self.assertEqual(entries[0]["size_bytes"], 123)

    def test_catalog_can_filter_by_language_quality_and_words(self) -> None:
        entries = catalog_entries(self.catalog)
        self.assertEqual(len(filter_catalog(entries, "test german", "de_DE", "medium")), 1)
        self.assertEqual(filter_catalog(entries, language="en_US"), [])

    def test_voice_ids_reject_paths_and_unexpected_characters(self) -> None:
        self.assertEqual(validate_voice_id("de_DE-test-medium"), "de_DE-test-medium")
        self.assertEqual(validate_voice_id("pt_PT-tugão-medium"), "pt_PT-tugão-medium")
        for value in ("../voice", "voice/../other", "", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_voice_id(value)

    def test_download_activates_only_checksum_verified_file(self) -> None:
        body = b"verified voice model"
        checksum = hashlib.md5(body).hexdigest()
        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "voice.onnx"
            with patch("voxwerk.app.urlopen", return_value=io.BytesIO(body)):
                _download_file("de/de_DE/voice/medium/voice.onnx", checksum, destination, len(body))
            self.assertEqual(destination.read_bytes(), body)

    def test_download_rejects_checksum_mismatch_without_final_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "voice.onnx"
            with patch("voxwerk.app.urlopen", return_value=io.BytesIO(b"unverified")):
                with self.assertRaisesRegex(ValueError, "Prüfsumme"):
                    _download_file("de/de_DE/voice/medium/voice.onnx", "a" * 32, destination)
            self.assertFalse(destination.exists())


class ApiAccessTests(unittest.TestCase):
    def test_remote_client_requires_non_loopback_bind_address(self) -> None:
        self.assertFalse(_api_client_allowed("10.10.0.42", "127.0.0.1"))
        self.assertTrue(_api_client_allowed("10.10.0.42", "0.0.0.0"))

    def test_loopback_client_remains_allowed(self) -> None:
        self.assertTrue(_api_client_allowed("127.0.0.1", "127.0.0.1"))

    def test_invalid_addresses_are_rejected(self) -> None:
        self.assertFalse(_api_client_allowed("not-an-ip", "0.0.0.0"))


if __name__ == "__main__":
    unittest.main()