from __future__ import annotations

import json
import io
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from voxwerk.wakeword_data import WakewordStore
from voxwerk.wakeword_testing import _resolve_model, list_test_models, score_test_audio, score_test_pcm_frame
from voxwerk.wakeword_test_worker import complete_pcm_frames, streaming_score


class WakewordModelDiscoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.store = WakewordStore(self.root / "data")
        self.profile = self.store.save_profile({"name": "Testprofil", "engine": "microwakeword"})

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_discovers_external_models_and_completed_voxwerk_runs(self) -> None:
        export_model = self.root / "export" / "commander.tflite"
        export_model.parent.mkdir(parents=True)
        export_model.write_bytes(b"model")
        export_model.with_suffix(".json").write_text(
            json.dumps({"wake_word": "Commander", "micro": {"probability_cutoff": 0.3}}),
            encoding="utf-8",
        )

        project = self.store.save_wakeword({"name": "Hey Nova", "phrase": "Hey Nova", "profile_id": self.profile["id"]})
        positive = self.store.save_sample(project["id"], "positive", "positive.wav", b"positive")
        negative = self.store.save_sample(project["id"], "negative", "negative.wav", b"negative")
        run = self.store.create_run(project["id"], {"sample_ids": [positive["id"], negative["id"]]})
        run_dir = self.store._project_dir(project["id"]) / "runs" / run["id"]
        model_path = run_dir / "model" / "streaming.tflite"
        model_path.parent.mkdir(parents=True)
        model_path.write_bytes(b"model")
        self.store.update_run(
            project["id"],
            run["id"],
            status="completed",
            training={"model_directory": "model"},
        )

        models = list_test_models(self.root, self.store)
        self.assertEqual([item["id"] for item in models], ["wakeword/export/commander.tflite", f"voxwerk/{project['id']}/{run['id']}/streaming.tflite"])
        self.assertEqual(models[0]["label"], "Commander")
        self.assertEqual(models[0]["cutoff"], 0.3)
        self.assertEqual(models[1]["label"], "Hey Nova")

    def test_does_not_list_models_from_incomplete_runs(self) -> None:
        project = self.store.save_wakeword({"name": "Hey Nova", "phrase": "Hey Nova", "profile_id": self.profile["id"]})
        positive = self.store.save_sample(project["id"], "positive", "positive.wav", b"positive")
        negative = self.store.save_sample(project["id"], "negative", "negative.wav", b"negative")
        run = self.store.create_run(project["id"], {"sample_ids": [positive["id"], negative["id"]]})
        model_path = self.store._project_dir(project["id"]) / "runs" / run["id"] / "model" / "unfinished.tflite"
        model_path.parent.mkdir(parents=True)
        model_path.write_bytes(b"model")
        self.assertEqual(list_test_models(self.root, self.store), [])

    def test_model_resolution_rejects_paths_outside_allowed_roots(self) -> None:
        outside_model = self.root / "secret.tflite"
        outside_model.write_bytes(b"model")
        with self.assertRaisesRegex(ValueError, "Ungültige Testmodell-ID"):
            _resolve_model("../../secret.tflite", self.root, self.store)
        with self.assertRaisesRegex(ValueError, "Unbekanntes Testmodell"):
            _resolve_model("secret.tflite", self.root, self.store)

    def test_voxwerk_model_resolution_requires_completed_run(self) -> None:
        project = self.store.save_wakeword({"name": "Hey Nova", "phrase": "Hey Nova", "profile_id": self.profile["id"]})
        positive = self.store.save_sample(project["id"], "positive", "positive.wav", b"positive")
        negative = self.store.save_sample(project["id"], "negative", "negative.wav", b"negative")
        run = self.store.create_run(project["id"], {"sample_ids": [positive["id"], negative["id"]]})
        model_path = self.store._project_dir(project["id"]) / "runs" / run["id"] / "model" / "unfinished.tflite"
        model_path.parent.mkdir(parents=True)
        model_path.write_bytes(b"model")
        model_id = f"voxwerk/{project['id']}/{run['id']}/unfinished.tflite"
        with self.assertRaisesRegex(ValueError, "noch nicht fertig"):
            _resolve_model(model_id, self.root, self.store)

    def test_audio_test_converts_and_scores_without_persisting_recording(self) -> None:
        model_path = self.root / "export" / "commander.tflite"
        model_path.parent.mkdir(parents=True)
        model_path.write_bytes(b"model")
        wave_buffer = io.BytesIO()
        with wave.open(wave_buffer, "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(16000)
            audio.writeframes(b"\0\0" * 16000)

        def fake_convert(command: list[str], **kwargs: object) -> object:
            Path(command[-1]).write_bytes(wave_buffer.getvalue())
            return type("Completed", (), {"returncode": 0, "stderr": ""})()

        environment = {"ready": True, "workspace": self.root, "python": Path("/usr/bin/python3")}
        with (
            patch("voxwerk.wakeword_testing.training_environment", return_value=environment),
            patch("voxwerk.wakeword_testing.shutil.which", return_value="/usr/bin/ffmpeg"),
            patch("voxwerk.wakeword_testing.subprocess.run", side_effect=fake_convert),
            patch("voxwerk.wakeword_testing._score_with_worker", return_value=0.8) as score_worker,
        ):
            result = score_test_audio(
                b"browser audio",
                "wakeword/export/commander.tflite",
                0.75,
                self.root,
                self.store,
            )

        self.assertEqual(result["score"], 0.8)
        self.assertTrue(result["detected"])
        self.assertEqual(result["duration"], 1.0)
        self.assertEqual(list(self.root.rglob("*.wav")), [])
        score_worker.assert_called_once()
        self.assertTrue(score_worker.call_args.kwargs["reset"])


class PcmFrameAssemblyTests(unittest.TestCase):
    def test_streaming_score_uses_five_zero_initialized_slots(self) -> None:
        warming = streaming_score([0.0, 0.0, 0.0, 0.0, 0.72], False)
        self.assertAlmostEqual(warming["score"], 0.144)
        self.assertFalse(warming["detection_ready"])

        ready = streaming_score([0.72] * 5, True)
        self.assertEqual(ready, {"score": 0.72, "detection_ready": True})

    def test_preserves_partial_frontend_frames_across_transport_packets(self) -> None:
        first_packet = b"a" * 500
        first_frames, remainder = complete_pcm_frames(first_packet)
        self.assertEqual([len(frame) for frame in first_frames], [320])
        self.assertEqual(len(remainder), 180)

        second_frames, final_remainder = complete_pcm_frames(b"b" * 460, remainder)
        self.assertEqual([len(frame) for frame in second_frames], [320, 320])
        self.assertEqual(final_remainder, b"")

    def test_pcm_score_uses_persistent_worker_and_rejects_oversized_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "export" / "commander.tflite"
            model_path.parent.mkdir(parents=True)
            model_path.write_bytes(b"model")
            environment = {"ready": True, "workspace": root, "python": Path("/usr/bin/python3")}
            with (
                patch("voxwerk.wakeword_testing.training_environment", return_value=environment),
                patch("voxwerk.wakeword_testing._score_pcm_with_worker", return_value=(0.72, 4, True)) as score_worker,
            ):
                result = score_test_pcm_frame(b"\0\0" * 2048, "wakeword/export/commander.tflite", 0.7, root, WakewordStore(root / "data"), reset_worker=True)
            self.assertEqual(result["score"], 0.72)
            self.assertTrue(result["detected"])
            self.assertEqual(result["predictions"], 4)
            self.assertTrue(score_worker.call_args.args[-1])
            with self.assertRaisesRegex(ValueError, "4 Sekunden"):
                score_test_pcm_frame(b"\0\0" * 66000, "wakeword/export/commander.tflite", 0.5, root, WakewordStore(root / "data"))


if __name__ == "__main__":
    unittest.main()
