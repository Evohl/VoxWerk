from __future__ import annotations

import tempfile
import wave
import unittest
from pathlib import Path
from unittest.mock import patch

from voxwerk.app import RequestHandler, TRAINING_JOBS, TRAINING_JOBS_LOCK, TRAINING_STOP_EVENTS, _run_wakeword_training
from voxwerk.wakeword_data import WakewordStore
from voxwerk.wakeword_training import TrainingStopped, _wav_samples, check_wakeword_integration, run_training_pipeline, training_environment, validate_training_parameters


class TrainingParameterTests(unittest.TestCase):
    def test_training_parameters_have_bounded_defaults(self) -> None:
        parameters = validate_training_parameters({})
        self.assertEqual(parameters["training_steps"], 10000)
        self.assertEqual(parameters["batch_size"], 128)
        self.assertEqual(parameters["negative_class_weight"], 15)
        self.assertEqual(parameters["learning_rate"], 0.00101)
        self.assertEqual(parameters["dataset_weights"]["dinner_party_eval"], 0)

    def test_learning_rate_accepts_precise_values_and_decimal_comma(self) -> None:
        for value, expected in (("0,00101", 0.00101), ("0,00191", 0.00191), ("0.00191", 0.00191)):
            with self.subTest(value=value):
                self.assertEqual(validate_training_parameters({"learning_rate": value})["learning_rate"], expected)

    def test_learning_rate_rejects_values_outside_step_grid(self) -> None:
        with self.assertRaisesRegex(ValueError, "Schritten von 0.0001"):
            validate_training_parameters({"learning_rate": "0,00192"})

    def test_training_parameters_reject_invalid_ranges_and_datasets(self) -> None:
        for payload in (
            {"training_steps": 999},
            {"batch_size": 513},
            {"learning_rate": float("inf")},
            {"dataset_weights": {"unknown": 1}},
            {"dataset_weights": {"dinner_party_eval": 1}},
        ):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                validate_training_parameters(payload)


class TrainingEnvironmentTests(unittest.TestCase):
    def test_integration_check_reports_missing_components(self) -> None:
        environment = {
            "workspace": Path("/missing/wakeword"),
            "python": Path("/missing/wakeword/bin/python"),
            "package_path": "/missing/wakeword",
            "package_source": "default",
            "package_locked": False,
        }
        with patch("voxwerk.wakeword_training.training_environment", return_value=environment):
            result = check_wakeword_integration()
        self.assertFalse(result["ready"])
        self.assertEqual({check["id"] for check in result["checks"] if not check["ok"]}, {
            "package_directory", "python", "feature_script", "trainer", "ambient", "runtime_imports"
        })

    def test_integration_check_reports_success_when_runtime_imports_and_files_exist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python = root / "bin" / "python"
            feature_script = root / "scripts" / "generate_positive_features.py"
            trainer = root / "microWakeWord" / "microwakeword" / "model_train_eval.py"
            ambient = root / "negative_datasets" / "dinner_party_eval" / "testing_ambient" / "features_mmap"
            for path in (python, feature_script, trainer, ambient):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            environment = {
                "workspace": root,
                "python": python,
                "package_path": str(root),
                "package_source": "default",
                "package_locked": False,
            }
            with (
                patch("voxwerk.wakeword_training.training_environment", return_value=environment),
                patch("voxwerk.wakeword_training.subprocess.run", return_value=type("Result", (), {"returncode": 0, "stderr": ""})()),
            ):
                result = check_wakeword_integration()
        self.assertTrue(result["ready"])
        self.assertTrue(all(check["ok"] for check in result["checks"]))

    def test_environment_requires_trainer_and_ambient_features(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python = root / ".venv" / "bin" / "python"
            script = root / "scripts" / "generate_positive_features.py"
            trainer = root / "microWakeWord" / "microwakeword" / "model_train_eval.py"
            ambient = root / "negative_datasets" / "dinner_party_eval" / "testing_ambient" / "features_mmap"
            for path in (python, script, trainer, ambient):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            with patch.dict("os.environ", {"VOXWERK_WAKEWORD_DIR": directory}, clear=False):
                environment = training_environment()
            self.assertTrue(environment["ready"])
            self.assertEqual(environment["workspace"], root)

            ambient.unlink()
            with patch.dict("os.environ", {"VOXWERK_WAKEWORD_DIR": directory}, clear=False):
                environment = training_environment()
            self.assertFalse(environment["ready"])

    def test_package_is_installed_at_a_fixed_location_under_home(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            package = home / ".venv" / "wakeword"
            for relative_path in (
                "bin/python",
                "scripts/generate_positive_features.py",
                "microWakeWord/microwakeword/model_train_eval.py",
                "negative_datasets/dinner_party_eval/testing_ambient/features_mmap",
            ):
                target = package / relative_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.touch()
            with patch.dict("os.environ", {"HOME": directory, "VOXWERK_WAKEWORD_DIR": "", "VOXWERK_WAKEWORD_PYTHON": ""}, clear=False):
                environment = training_environment()
            self.assertTrue(environment["ready"])
            self.assertEqual(environment["workspace"], package.resolve())
            self.assertEqual(environment["python"], package.resolve() / "bin" / "python")
            self.assertEqual(environment["package_source"], "default")
            self.assertFalse(environment["package_locked"])

    def test_missing_package_reports_installation_hint(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", {"HOME": directory, "VOXWERK_WAKEWORD_DIR": ""}, clear=False):
            environment = training_environment()
        self.assertFalse(environment["ready"])
        self.assertIn("Wakeword-Paket", environment["message"])

    def test_environment_override_locks_package_location(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", {"VOXWERK_WAKEWORD_DIR": directory}, clear=False):
            environment = training_environment()
        self.assertEqual(environment["workspace"], Path(directory).resolve())
        self.assertEqual(environment["package_source"], "environment")
        self.assertTrue(environment["package_locked"])

    def test_python_override_preserves_virtual_environment_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            interpreter = root / "base" / "python"
            interpreter.parent.mkdir()
            interpreter.touch()
            venv_python = root / "venv" / "bin" / "python"
            venv_python.parent.mkdir(parents=True)
            venv_python.symlink_to(interpreter)
            with patch.dict("os.environ", {"VOXWERK_WAKEWORD_DIR": directory, "VOXWERK_WAKEWORD_PYTHON": str(venv_python)}, clear=False):
                environment = training_environment()
        self.assertEqual(environment["python"], venv_python)


class TrainingSnapshotTests(unittest.TestCase):
    def test_training_accepts_wav_snapshots_and_rejects_other_audio(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            samples = []
            for kind in ("positive", "negative"):
                for index in range(5):
                    filename = f"{kind}-{index}.wav"
                    with wave.open(str(run_dir / filename), "wb") as audio:
                        audio.setnchannels(1)
                        audio.setsampwidth(2)
                        audio.setframerate(16000)
                        audio.writeframes(b"\0\0" * 9600)
                    samples.append({"kind": kind, "snapshot_file": filename})
            grouped = _wav_samples(run_dir, {"samples": samples})
            self.assertEqual(len(grouped["positive"]), 5)
            samples[0]["snapshot_file"] = "recording.webm"
            with self.assertRaisesRegex(ValueError, "nur WAV"):
                _wav_samples(run_dir, {"samples": samples})

    def test_training_pipeline_terminates_active_subprocess_when_stopped(self) -> None:
        class Process:
            returncode = None

            def __init__(self) -> None:
                self.terminated = False

            def poll(self) -> int | None:
                return self.returncode

            def terminate(self) -> None:
                self.terminated = True
                self.returncode = -15

            def wait(self, timeout: int | None = None) -> int:
                return self.returncode or 0

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            (workspace / "scripts").mkdir(parents=True)
            (workspace / "scripts" / "generate_positive_features.py").touch()
            run_dir = root / "run"
            samples = []
            for kind in ("positive", "negative"):
                for index in range(5):
                    relative_path = Path("samples") / f"{kind}-{index}.wav"
                    audio_path = run_dir / relative_path
                    audio_path.parent.mkdir(parents=True, exist_ok=True)
                    with wave.open(str(audio_path), "wb") as audio:
                        audio.setnchannels(1)
                        audio.setsampwidth(2)
                        audio.setframerate(16000)
                        audio.writeframes(b"\0\0" * 9600)
                    samples.append({"kind": kind, "snapshot_file": relative_path.as_posix()})
            process = Process()
            stages = []
            with patch("voxwerk.wakeword_training.subprocess.Popen", return_value=process):
                with self.assertRaises(TrainingStopped):
                    run_training_pipeline(
                        workspace,
                        root / "python",
                        run_dir,
                        {"samples": samples},
                        {"training_steps": 10000},
                        lambda stage, progress: stages.append((stage, progress)),
                        lambda: True,
                    )
            self.assertTrue(process.terminated)
            self.assertTrue(stages)

    def test_training_starts_existing_snapshot_without_creating_another_run(self) -> None:
        class Response:
            def send_json(self, payload: dict[str, object], status: int = 200) -> None:
                self.payload = payload
                self.status = status

        with tempfile.TemporaryDirectory() as directory:
            store = WakewordStore(Path(directory))
            profile = store.save_profile({"name": "microWakeWord", "engine": "microwakeword"})
            project = store.save_wakeword({"name": "Hey Nova", "phrase": "Hey Nova", "profile_id": profile["id"]})
            sample_ids = []
            for kind in ("positive", "negative"):
                for index in range(5):
                    audio_buffer = Path(directory) / f"{kind}-{index}.wav"
                    with wave.open(str(audio_buffer), "wb") as audio:
                        audio.setnchannels(1)
                        audio.setsampwidth(2)
                        audio.setframerate(16000)
                        audio.writeframes(b"\0\0" * 9600)
                    sample = store.save_sample(project["id"], kind, audio_buffer.name, audio_buffer.read_bytes())
                    sample_ids.append(sample["id"])
            run = store.create_run(project["id"], {"sample_ids": sample_ids})
            environment = {
                "ready": True,
                "message": "",
                "workspace": Path(directory),
                "python": Path(directory) / "python",
                "datasets": [{"id": dataset_id, "available": True} for dataset_id in ("speech", "dinner_party", "no_speech", "dinner_party_eval")],
            }
            response = Response()
            with (
                patch("voxwerk.app.training_environment", return_value=environment),
                patch("voxwerk.app.threading.Thread") as thread,
                self.assertLogs("voxwerk", level="INFO") as captured,
            ):
                RequestHandler._start_wakeword_training(response, store, {"wakeword_id": project["id"], "run_id": run["id"]})

            updated_runs = store.list_runs(project["id"])
            self.assertEqual(response.status, 202)
            self.assertEqual(response.payload["run"]["id"], run["id"])
            self.assertEqual(len(updated_runs), 1)
            self.assertEqual(updated_runs[0]["status"], "queued")
            thread.return_value.start.assert_called_once()
            self.assertIn("Wakeword-Training startet", captured.output[0])
            with TRAINING_JOBS_LOCK:
                TRAINING_JOBS.pop(run["id"], None)

    def test_failed_snapshot_can_be_retried_and_partial_artifacts_are_removed(self) -> None:
        class Response:
            def send_json(self, payload: dict[str, object], status: int = 200) -> None:
                self.payload = payload
                self.status = status

        with tempfile.TemporaryDirectory() as directory:
            store = WakewordStore(Path(directory))
            profile = store.save_profile({"name": "microWakeWord", "engine": "microwakeword"})
            project = store.save_wakeword({"name": "Hey Nova", "phrase": "Hey Nova", "profile_id": profile["id"]})
            sample_ids = []
            for kind in ("positive", "negative"):
                for index in range(5):
                    audio_path = Path(directory) / f"{kind}-{index}.wav"
                    with wave.open(str(audio_path), "wb") as audio:
                        audio.setnchannels(1)
                        audio.setsampwidth(2)
                        audio.setframerate(16000)
                        audio.writeframes(b"\0\0" * 9600)
                    sample = store.save_sample(project["id"], kind, audio_path.name, audio_path.read_bytes())
                    sample_ids.append(sample["id"])
            run = store.create_run(project["id"], {"sample_ids": sample_ids})
            run = store.update_run(project["id"], run["id"], status="failed", training={"stage": "Training fehlgeschlagen", "error": "torchcodec fehlt"})
            run_dir = store._project_dir(project["id"]) / "runs" / run["id"]
            partial_feature = run_dir / "training-work" / "features" / "positive" / "partial.bin"
            partial_feature.parent.mkdir(parents=True)
            partial_feature.write_bytes(b"incomplete")
            partial_model = run_dir / "model" / "checkpoint.bin"
            partial_model.parent.mkdir()
            partial_model.write_bytes(b"incomplete")
            config_path = run_dir / "training_config.yaml"
            config_path.write_text("partial", encoding="utf-8")
            log_path = run_dir / "training.log"
            log_path.write_text("previous failure", encoding="utf-8")
            environment = {
                "ready": True,
                "message": "",
                "workspace": Path(directory),
                "python": Path(directory) / "python",
                "datasets": [{"id": dataset_id, "available": True} for dataset_id in ("speech", "dinner_party", "no_speech", "dinner_party_eval")],
            }
            response = Response()
            try:
                with (
                    patch("voxwerk.app.training_environment", return_value=environment),
                    patch("voxwerk.app.threading.Thread") as thread,
                ):
                    RequestHandler._start_wakeword_training(response, store, {"wakeword_id": project["id"], "run_id": run["id"]})

                retried = store.list_runs(project["id"])[0]
                self.assertEqual(response.status, 202)
                self.assertEqual(retried["status"], "queued")
                self.assertIsNone(retried["training"]["error"])
                self.assertFalse((run_dir / "training-work").exists())
                self.assertFalse((run_dir / "model").exists())
                self.assertFalse(config_path.exists())
                self.assertEqual(log_path.read_text(encoding="utf-8"), "previous failure")
                thread.return_value.start.assert_called_once()
            finally:
                with TRAINING_JOBS_LOCK:
                    TRAINING_JOBS.pop(run["id"], None)
                    TRAINING_STOP_EVENTS.pop(run["id"], None)

    def test_stopped_training_persists_stopped_run_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = WakewordStore(Path(directory))
            profile = store.save_profile({"name": "microWakeWord", "engine": "microwakeword"})
            project = store.save_wakeword({"name": "Hey Nova", "phrase": "Hey Nova", "profile_id": profile["id"]})
            sample_ids = []
            for kind in ("positive", "negative"):
                sample = store.save_sample(project["id"], kind, f"{kind}.wav", b"audio")
                sample_ids.append(sample["id"])
            run = store.create_run(project["id"], {"sample_ids": sample_ids})
            run_id = run["id"]
            with TRAINING_JOBS_LOCK:
                TRAINING_JOBS[run_id] = {"status": "running", "wakeword_id": project["id"]}
                TRAINING_STOP_EVENTS[run_id] = __import__("threading").Event()
            try:
                with (
                    patch("voxwerk.app.data_dir", return_value=Path(directory)),
                    patch("voxwerk.app.run_training_pipeline", side_effect=TrainingStopped("Training wurde abgebrochen")),
                ):
                    _run_wakeword_training(project["id"], run_id, {"workspace": Path(directory), "python": Path(directory) / "python"}, {})
                stopped_run = store.list_runs(project["id"])[0]
                self.assertEqual(stopped_run["status"], "stopped")
                self.assertEqual(stopped_run["training"]["stage"], "Training abgebrochen")
                with TRAINING_JOBS_LOCK:
                    self.assertEqual(TRAINING_JOBS[run_id]["status"], "stopped")
                    self.assertNotIn(run_id, TRAINING_STOP_EVENTS)
            finally:
                with TRAINING_JOBS_LOCK:
                    TRAINING_JOBS.pop(run_id, None)
                    TRAINING_STOP_EVENTS.pop(run_id, None)


if __name__ == "__main__":
    unittest.main()