from __future__ import annotations

import io
import json
import tempfile
import threading
import unittest
import wave
import zipfile
from pathlib import Path
from unittest.mock import patch

from voxwerk.app import GENERATION_JOBS, GENERATION_JOBS_LOCK, _run_sample_generation, _wakeword_run_export
from voxwerk.wakeword_data import WakewordStore, validate_slug


class WakewordStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.store = WakewordStore(Path(self.temporary_directory.name))
        self.profile = self.store.save_profile({"name": "ESP32 Standard"})
        self.project = self.store.save_wakeword({
            "name": "Hey Nova",
            "phrase": "Hey Nova",
            "profile_id": self.profile["id"],
            "variants": ["Hei Nova"],
            "negative_phrases": ["Guten Morgen", "Licht bitte an"],
        })

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    @staticmethod
    def make_wav(duration: float) -> bytes:
        output = io.BytesIO()
        with wave.open(output, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(b"\x00\x00" * round(16000 * duration))
        return output.getvalue()

    def test_profile_can_be_edited_but_not_deleted_while_in_use(self) -> None:
        updated = self.store.save_profile({"name": "ESP32 Updated", "sample_rate": 22050}, self.profile["id"])
        self.assertEqual(updated["id"], self.profile["id"])
        self.assertEqual(updated["sample_rate"], 22050)
        with self.assertRaisesRegex(ValueError, "verwendet"):
            self.store.delete_profile(self.profile["id"])

    def test_wakeword_negative_phrases_are_saved_and_edited(self) -> None:
        project = self.store.save_wakeword({
            "name": "Hey Negative",
            "phrase": "Hey Negative",
            "negative_phrases": ["hey commando", "hey commoner"],
        })
        self.assertEqual(project["negative_phrases"], ["hey commando", "hey commoner"])
        updated = self.store.save_wakeword({
            "id": project["id"],
            "name": project["name"],
            "phrase": project["phrase"],
            "negative_phrases": ["hey command"],
        }, project["id"])
        self.assertEqual(updated["negative_phrases"], ["hey command"])

    def test_wakeword_keeps_negative_phrases_separate_from_pronunciation_variants(self) -> None:
        updated = self.store.save_wakeword({
            "name": self.project["name"],
            "phrase": self.project["phrase"],
            "negative_phrases": ["  Guten Morgen  ", "Licht bitte an"],
        }, self.project["id"])
        self.assertEqual(updated["variants"], ["Hei Nova"])
        self.assertEqual(updated["negative_phrases"], ["Guten Morgen", "Licht bitte an"])

    def test_sample_can_be_added_edited_reclassified_and_deleted(self) -> None:
        positive = self.store.save_sample(self.project["id"], "positive", "recording.wav", b"positive", "Take one")
        negative = self.store.save_sample(self.project["id"], "negative", "noise.wav", b"negative", "Room noise")
        updated = self.store.update_sample(self.project["id"], positive["id"], {
            "title": "Clean take",
            "transcript": "Hey Nova",
        })
        self.assertEqual(updated["title"], "Clean take")
        self.assertEqual(updated["transcript"], "Hey Nova")
        reclassified = self.store.update_sample(self.project["id"], positive["id"], {"kind": "negative"})
        self.assertEqual(reclassified["kind"], "negative")
        self.assertEqual(len(self.store.list_samples(self.project["id"])), 2)
        self.store.delete_sample(self.project["id"], negative["id"])
        self.assertEqual([sample["id"] for sample in self.store.list_samples(self.project["id"])], [positive["id"]])

    def test_multiple_samples_can_be_deleted_as_one_validated_batch(self) -> None:
        first = self.store.save_sample(self.project["id"], "positive", "first.wav", b"first")
        second = self.store.save_sample(self.project["id"], "positive", "second.wav", b"second")
        negative = self.store.save_sample(self.project["id"], "negative", "negative.wav", b"negative")
        with self.assertRaisesRegex(ValueError, "nicht gefunden"):
            self.store.delete_samples(self.project["id"], [first["id"], "0" * 32])
        self.assertEqual(len(self.store.list_samples(self.project["id"])), 3)
        deleted = self.store.delete_samples(self.project["id"], [first["id"], second["id"]])
        self.assertEqual({sample["id"] for sample in deleted}, {first["id"], second["id"]})
        self.assertEqual([sample["id"] for sample in self.store.list_samples(self.project["id"])], [negative["id"]])

    def test_run_captures_profile_wakeword_and_selected_samples(self) -> None:
        positive = self.store.save_sample(self.project["id"], "positive", "positive.wav", b"positive")
        negative = self.store.save_sample(self.project["id"], "negative", "negative.wav", b"negative")
        run = self.store.create_run(self.project["id"], {"sample_ids": [positive["id"], negative["id"]]})
        self.assertEqual(run["status"], "prepared")
        self.assertEqual(run["profile"]["id"], self.profile["id"])
        snapshot_path = self.store.wakewords_dir / self.project["id"] / "runs" / run["id"] / run["samples"][0]["snapshot_file"]
        self.assertTrue(snapshot_path.is_file())
        self.assertEqual(snapshot_path.read_bytes(), b"positive")
        self.assertEqual(len(self.store.list_runs(self.project["id"])), 1)
        self.store.delete_run(self.project["id"], run["id"])
        self.assertEqual(self.store.list_runs(self.project["id"]), [])
        self.assertFalse(snapshot_path.parent.parent.exists())

    def test_run_requires_positive_and_negative_samples(self) -> None:
        positive = self.store.save_sample(self.project["id"], "positive", "positive.wav", b"positive")
        with self.assertRaisesRegex(ValueError, "negatives"):
            self.store.create_run(self.project["id"], {"sample_ids": [positive["id"]]})

    def test_run_status_and_training_parameters_are_persisted(self) -> None:
        positive = self.store.save_sample(self.project["id"], "positive", "positive.wav", b"positive")
        negative = self.store.save_sample(self.project["id"], "negative", "negative.wav", b"negative")
        run = self.store.create_run(self.project["id"], {"sample_ids": [positive["id"], negative["id"]]})
        updated = self.store.update_run(
            self.project["id"],
            run["id"],
            status="queued",
            training={"stage": "Wartet auf Start", "parameters": {"training_steps": 10000}},
        )
        self.assertEqual(updated["status"], "queued")
        self.assertEqual(self.store.list_runs(self.project["id"])[0]["training"]["parameters"]["training_steps"], 10000)

    def test_completed_wakeword_run_can_be_downloaded_as_model_archive(self) -> None:
        positive = self.store.save_sample(self.project["id"], "positive", "positive.wav", b"positive")
        negative = self.store.save_sample(self.project["id"], "negative", "negative.wav", b"negative")
        run = self.store.create_run(self.project["id"], {"sample_ids": [positive["id"], negative["id"]]})
        run_dir = self.store._project_dir(self.project["id"]) / "runs" / run["id"]
        model_dir = run_dir / "model"
        model_dir.mkdir()
        (model_dir / "streaming.tflite").write_bytes(b"model bytes")
        (model_dir / "streaming.json").write_text('{"wake_word": "Hey Nova"}', encoding="utf-8")
        self.store.update_run(self.project["id"], run["id"], status="completed", training={"model_directory": "model"})

        archive_bytes, filename = _wakeword_run_export(self.store, self.project["id"], run["id"])

        self.assertEqual(filename, "hey-nova-wakeword.zip")
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            self.assertEqual(set(archive.namelist()), {"models/hey-nova/hey-nova.tflite", "models/hey-nova/hey-nova.json"})
            self.assertEqual(archive.read("models/hey-nova/hey-nova.tflite"), b"model bytes")
            manifest = json.loads(archive.read("models/hey-nova/hey-nova.json"))
            self.assertEqual(manifest["wake_word"], "Hey Nova")
            self.assertEqual(manifest["model"], "hey-nova.tflite")
            self.assertEqual(manifest["micro"]["probability_cutoff"], 0.5)

    def test_wakeword_run_export_rejects_unfinished_run(self) -> None:
        positive = self.store.save_sample(self.project["id"], "positive", "positive.wav", b"positive")
        negative = self.store.save_sample(self.project["id"], "negative", "negative.wav", b"negative")
        run = self.store.create_run(self.project["id"], {"sample_ids": [positive["id"], negative["id"]]})
        with self.assertRaisesRegex(ValueError, "abgeschlossene"):
            _wakeword_run_export(self.store, self.project["id"], run["id"])

    def test_deleting_wakeword_removes_only_its_project_data(self) -> None:
        other = self.store.save_wakeword({"name": "Hey Other", "phrase": "Hey Other"})
        self.store.delete_wakeword(self.project["id"])
        remaining = self.store.list_wakewords()
        self.assertEqual([project["id"] for project in remaining], [other["id"]])

    def test_invalid_slugs_and_sample_formats_are_rejected(self) -> None:
        for value in ("../other", "UPPER", "with/slash", ""):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_slug(value)
        with self.assertRaisesRegex(ValueError, "Audioformat"):
            self.store.save_sample(self.project["id"], "positive", "voice.exe", b"audio")

    def test_sample_generator_saves_voice_and_transcript_provenance(self) -> None:
        with GENERATION_JOBS_LOCK:
            GENERATION_JOBS["test-job"] = {"id": "test-job", "status": "queued", "completed": 0, "total": 2}
        try:
            with (
                patch("voxwerk.app.data_dir", return_value=Path(self.temporary_directory.name)),
                patch("voxwerk.app.installed_voices", return_value=[{"id": "voice-a", "name": "Voice A", "path": "/voice-a.onnx"}]),
                patch("voxwerk.app.synthesize_wav", return_value=self.make_wav(1.0)),
            ):
                _run_sample_generation("test-job", self.project["id"], {"voice-a": 2}, tempo_variation=10, relative_weights={"voice-a": 75}, seed=24)
            samples = self.store.list_samples(self.project["id"])
            self.assertEqual(len(samples), 2)
            self.assertTrue(all(sample["source"] == "generated" for sample in samples))
            self.assertTrue(all(sample["voice_id"] == "voice-a" for sample in samples))
            self.assertTrue(all(sample["voice_name"] == "Voice A" for sample in samples))
            self.assertTrue(all(sample["relative_weight"] == 75 for sample in samples))
            self.assertTrue(all(any(abs(sample["length_scale"] - expected) < 0.001 for expected in (0.9, 0.95)) for sample in samples))
            self.assertTrue(all(sample["transcript"] in {"Hey Nova", "Hei Nova"} for sample in samples))
            self.assertEqual(GENERATION_JOBS["test-job"]["status"], "completed")
        finally:
            with GENERATION_JOBS_LOCK:
                GENERATION_JOBS.pop("test-job", None)

    def test_sample_generator_can_create_negative_phrases(self) -> None:
        with GENERATION_JOBS_LOCK:
            GENERATION_JOBS["negative-job"] = {"id": "negative-job", "status": "queued", "completed": 0, "total": 2}
        try:
            with (
                patch("voxwerk.app.data_dir", return_value=Path(self.temporary_directory.name)),
                patch("voxwerk.app.installed_voices", return_value=[{"id": "voice-a", "name": "Voice A", "path": "/voice-a.onnx"}]),
                patch("voxwerk.app.synthesize_wav", return_value=self.make_wav(1.0)),
            ):
                _run_sample_generation("negative-job", self.project["id"], {"voice-a": 2}, sample_kind="negative", tempo_variation=0)
            samples = self.store.list_samples(self.project["id"])
            self.assertEqual(len(samples), 2)
            self.assertTrue(all(sample["kind"] == "negative" for sample in samples))
            self.assertTrue(all(sample["transcript"] in {"Guten Morgen", "Licht bitte an"} for sample in samples))
            self.assertEqual(GENERATION_JOBS["negative-job"]["status"], "completed")
        finally:
            with GENERATION_JOBS_LOCK:
                GENERATION_JOBS.pop("negative-job", None)

    def test_positive_sample_generation_adjusts_duration_into_training_window(self) -> None:
        with GENERATION_JOBS_LOCK:
            GENERATION_JOBS["duration-job"] = {"id": "duration-job", "status": "queued", "completed": 0, "total": 1}
        durations = iter((0.4, 0.55))
        synthesized_scales: list[float] = []

        def synthesize(text: str, voice_id: str, rate: float = 1.0) -> bytes:
            synthesized_scales.append(1.0 / rate)
            return self.make_wav(next(durations))

        try:
            with (
                patch("voxwerk.app.data_dir", return_value=Path(self.temporary_directory.name)),
                patch("voxwerk.app.installed_voices", return_value=[{"id": "voice-a", "name": "Voice A", "path": "/voice-a.onnx"}]),
                patch("voxwerk.app.synthesize_wav", side_effect=synthesize),
            ):
                _run_sample_generation("duration-job", self.project["id"], {"voice-a": 1}, tempo_variation=0)
            sample = self.store.list_samples(self.project["id"])[0]
            self.assertEqual(synthesized_scales, [1.0, 1.375])
            self.assertEqual(sample["length_scale"], 1.375)
            self.assertEqual(GENERATION_JOBS["duration-job"]["status"], "completed")
        finally:
            with GENERATION_JOBS_LOCK:
                GENERATION_JOBS.pop("duration-job", None)

    def test_negative_sample_generation_adjusts_duration_into_training_window(self) -> None:
        with GENERATION_JOBS_LOCK:
            GENERATION_JOBS["negative-duration-job"] = {"id": "negative-duration-job", "status": "queued", "completed": 0, "total": 1}
        durations = iter((0.2, 0.35))
        synthesized_scales: list[float] = []

        def synthesize(text: str, voice_id: str, rate: float = 1.0) -> bytes:
            synthesized_scales.append(1.0 / rate)
            return self.make_wav(next(durations))

        try:
            with (
                patch("voxwerk.app.data_dir", return_value=Path(self.temporary_directory.name)),
                patch("voxwerk.app.installed_voices", return_value=[{"id": "voice-a", "name": "Voice A", "path": "/voice-a.onnx"}]),
                patch("voxwerk.app.synthesize_wav", side_effect=synthesize),
            ):
                _run_sample_generation("negative-duration-job", self.project["id"], {"voice-a": 1}, sample_kind="negative", tempo_variation=0)
            sample = self.store.list_samples(self.project["id"])[0]
            self.assertEqual(synthesized_scales, [1.0, 2.175])
            self.assertEqual(sample["length_scale"], 2.175)
            self.assertEqual(GENERATION_JOBS["negative-duration-job"]["status"], "completed")
        finally:
            with GENERATION_JOBS_LOCK:
                GENERATION_JOBS.pop("negative-duration-job", None)

    def test_sample_generator_does_not_recreate_deleted_project(self) -> None:
        with GENERATION_JOBS_LOCK:
            GENERATION_JOBS["deleted-project-job"] = {"id": "deleted-project-job", "status": "queued", "completed": 0, "total": 1}
        try:
            def delete_project_during_synthesis(*args: object, **kwargs: object) -> bytes:
                self.store.delete_wakeword(self.project["id"])
                return b"generated wav"

            with (
                patch("voxwerk.app.data_dir", return_value=Path(self.temporary_directory.name)),
                patch("voxwerk.app.installed_voices", return_value=[{"id": "voice-a", "name": "Voice A", "path": "/voice-a.onnx"}]),
                patch("voxwerk.app.synthesize_wav", side_effect=delete_project_during_synthesis),
                patch("voxwerk.app.LOGGER.exception"),
            ):
                _run_sample_generation("deleted-project-job", self.project["id"], {"voice-a": 1})
            self.assertFalse((self.store.wakewords_dir / self.project["id"]).exists())
            self.assertEqual(GENERATION_JOBS["deleted-project-job"]["status"], "failed")
        finally:
            with GENERATION_JOBS_LOCK:
                GENERATION_JOBS.pop("deleted-project-job", None)

    def test_wakeword_deletion_waits_for_sample_commit(self) -> None:
        writing_manifest = threading.Event()
        release_manifest = threading.Event()
        delete_started = threading.Event()
        delete_finished = threading.Event()
        manifest_path = self.store.wakewords_dir / self.project["id"] / "wakeword.json"
        original_write_json = __import__("voxwerk.wakeword_data", fromlist=["_write_json"])._write_json

        def blocked_write_json(path: Path, value: dict[str, object]) -> None:
            if path == manifest_path:
                writing_manifest.set()
                if not release_manifest.wait(timeout=2):
                    raise TimeoutError("Test writer was not released")
            original_write_json(path, value)

        def save_sample() -> None:
            self.store.save_sample(self.project["id"], "positive", "test.wav", b"sample")

        def delete_project() -> None:
            delete_started.set()
            self.store.delete_wakeword(self.project["id"])
            delete_finished.set()

        with patch("voxwerk.wakeword_data._write_json", side_effect=blocked_write_json):
            writer = threading.Thread(target=save_sample)
            writer.start()
            self.assertTrue(writing_manifest.wait(timeout=2))
            deleter = threading.Thread(target=delete_project)
            deleter.start()
            self.assertTrue(delete_started.wait(timeout=2))
            self.assertFalse(delete_finished.wait(timeout=0.05))
            release_manifest.set()
            writer.join(timeout=2)
            deleter.join(timeout=2)

        self.assertFalse(writer.is_alive())
        self.assertFalse(deleter.is_alive())
        self.assertTrue(delete_finished.is_set())
        self.assertFalse((self.store.wakewords_dir / self.project["id"]).exists())


if __name__ == "__main__":
    unittest.main()