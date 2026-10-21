from __future__ import annotations

import json
import base64
import math
import sys
import wave
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from pymicro_features import MicroFrontend

CONTEXT_SECONDS = 3.2
TAIL_SECONDS = 0.2
ROOM_TONE_RMS = 10


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as audio_file:
        if audio_file.getnchannels() != 1 or audio_file.getsampwidth() != 2:
            raise ValueError("Testaudio muss Mono mit 16 Bit sein")
        rate = audio_file.getframerate()
        audio = np.frombuffer(audio_file.readframes(audio_file.getnframes()), dtype=np.int16)
    if rate != 16000:
        from scipy.signal import resample_poly

        divisor = math.gcd(16000, rate)
        audio = np.clip(resample_poly(audio.astype(np.float32), 16000 // divisor, rate // divisor), -32768, 32767).astype(np.int16)
    return audio


def embed_in_context(audio: np.ndarray) -> np.ndarray:
    # The preprocessor needs a lead-in before noise reduction and gain control settle.
    total, tail = int(CONTEXT_SECONDS * 16000), int(TAIL_SECONDS * 16000)
    output = np.random.default_rng(0).normal(0, ROOM_TONE_RMS, total).astype(np.int16)
    audio = audio[-(total - tail):]
    output[total - tail - len(audio): total - tail] = audio
    return output


def complete_pcm_frames(pcm: bytes, remainder: bytes = b"", frame_bytes: int = 320) -> tuple[list[bytes], bytes]:
    combined = remainder + pcm
    complete_bytes = len(combined) - len(combined) % frame_bytes
    frames = [combined[offset:offset + frame_bytes] for offset in range(0, complete_bytes, frame_bytes)]
    return frames, combined[complete_bytes:]


def streaming_score(probabilities: list[float], ready: bool) -> dict[str, Any]:
    score = sum(probabilities) / 5
    return {"score": float(score), "detection_ready": ready}


def _serve(workspace: Path) -> None:
    micro_wake_word_dir = workspace / "microWakeWord"
    sys.path.insert(0, str(micro_wake_word_dir))

    from microwakeword.inference import Model

    active_path = ""
    model = None
    recent_predictions: deque[float] = deque([0.0] * 5, maxlen=5)
    feature_window: deque[np.ndarray] = deque()
    frontend = None
    feature_steps = 0
    pcm_remainder = b""

    for line in sys.stdin:
        try:
            request = json.loads(line)
            model_path = str(Path(request["model_path"]).resolve())
            if request.get("reset") or model_path != active_path or model is None:
                model = Model(model_path, stride=3)
                active_path = model_path
                recent_predictions = deque([0.0] * 5, maxlen=5)
                feature_window = deque(maxlen=int(model.input_feature_slices))
                frontend = MicroFrontend()
                feature_steps = 0
                pcm_remainder = b""
            if request.get("mode") == "pcm":
                pcm = base64.b64decode(request["pcm"], validate=True)
                if not pcm or len(pcm) % 2:
                    raise ValueError("PCM-Frame ist leer oder unvollständig")
                predictions = []
                pcm_frames, pcm_remainder = complete_pcm_frames(pcm, pcm_remainder)
                for pcm_frame in pcm_frames:
                    frontend_result = frontend.process_samples(pcm_frame)
                    if not frontend_result.features:
                        continue
                    feature_window.append(np.asarray(frontend_result.features, dtype=np.float32))
                    feature_steps += 1
                    if len(feature_window) == model.input_feature_slices and feature_steps % model.stride == 0:
                        prediction = model.predict_spectrogram(np.stack(feature_window))
                        predictions.extend(np.asarray(prediction, dtype=np.float32).reshape(-1).tolist())
                recent_predictions.extend(float(value) for value in predictions)
                result = streaming_score(list(recent_predictions), feature_steps >= 100)
                result["predictions"] = len(predictions)
                sys.stdout.write(json.dumps(result) + "\n")
                sys.stdout.flush()
                continue
            audio_path = Path(request["audio_path"])
            audio = read_wav(audio_path)
            if request.get("reset"):
                audio = embed_in_context(audio)
            predictions = np.asarray(model.predict_clip(audio, step_ms=10), dtype=np.float32)
            combined = [*recent_predictions, *predictions.tolist()]
            if len(combined) >= 5:
                first_window = max(0, len(combined) - len(predictions) - 4)
                window_scores = [sum(combined[index:index + 5]) / 5 for index in range(first_window, len(combined) - 4)]
                score = max(window_scores, default=0.0)
            else:
                score = 0.0
            recent_predictions.extend(float(value) for value in predictions)
            result: dict[str, Any] = {"score": float(score)}
        except Exception as error:
            result = {"error": str(error)}
        sys.stdout.write(json.dumps(result) + "\n")
        sys.stdout.flush()


def main() -> None:
    _serve(Path(sys.argv[1]).resolve())


if __name__ == "__main__":
    main()
