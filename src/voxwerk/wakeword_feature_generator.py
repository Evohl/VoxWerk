"""Workspace script for turning VoxWerk WAV samples into microWakeWord features."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def generate_features(input_directory: Path, features_directory: Path, minimum: float, maximum: float) -> None:
    workspace = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(workspace / "microWakeWord"))

    from mmap_ninja.ragged import RaggedMmap
    from microwakeword.audio.augmentation import Augmentation
    from microwakeword.audio.clips import Clips
    from microwakeword.audio.spectrograms import SpectrogramGeneration

    if not input_directory.is_dir() or not any(input_directory.glob("*.wav")):
        raise ValueError(f"Keine WAV-Dateien gefunden: {input_directory}")
    features_directory.mkdir(parents=True, exist_ok=True)

    clips = Clips(
        input_directory=str(input_directory),
        file_pattern="*.wav",
        min_clip_duration_s=minimum,
        max_clip_duration_s=maximum,
        remove_silence=False,
        random_split_seed=10,
        split_count=0.1,
    )
    augmenter = Augmentation(
        augmentation_duration_s=3.2,
        augmentation_probabilities={
            "Gain": 1.0,
            "PitchShift": 0.2,
            "BandStopFilter": 0.1,
            "TanhDistortion": 0.1,
            "AddColorNoise": 0.2,
        },
        min_gain_db=-32.0,
        max_gain_db=4.0,
        min_jitter_s=0.195,
        max_jitter_s=0.205,
    )

    for directory_name, split_name, repetitions, slide_frames in (
        ("training", "train", 3, 10),
        ("validation", "validation", 1, 10),
        ("testing", "test", 1, 1),
    ):
        output = features_directory / directory_name / "wakeword_mmap"
        if output.exists():
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        generator = SpectrogramGeneration(
            clips=clips,
            augmenter=augmenter if split_name != "test" else None,
            slide_frames=slide_frames,
            step_ms=10,
        )
        RaggedMmap.from_generator(
            out_dir=str(output),
            sample_generator=generator.spectrogram_generator(split=split_name, repeat=repetitions),
            batch_size=100,
            verbose=True,
        )


def main() -> None:
    workspace = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=workspace / "generated_samples")
    parser.add_argument("--features-dir", type=Path, default=workspace / "generated_augmented_features")
    parser.add_argument("--min-duration", type=float, default=0.5)
    parser.add_argument("--max-duration", type=float, default=1.5)
    arguments = parser.parse_args()
    generate_features(arguments.input_dir.resolve(), arguments.features_dir.resolve(), arguments.min_duration, arguments.max_duration)


if __name__ == "__main__":
    main()