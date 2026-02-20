"""Offline audio feature extraction using the same AudioAnalyzer + BeatDetector pipeline.

Processes an audio file in chunks through the real-time analysis pipeline,
producing a list of AudioFrames. Same beat detection code runs live and offline.
"""

import numpy as np
import soundfile as sf

from .analyzer import AudioAnalyzer
from .beat import BeatDetector
from .frame import AudioFrame


def analyze_file(
    audio_path: str,
    block_size: int = 1024,
    smooth_tau: float = 0.05,
    onset_threshold_mult: float = 1.4,
) -> tuple[list[AudioFrame], int]:
    """Analyze an audio file through the full AudioAnalyzer + BeatDetector pipeline.

    Args:
        audio_path: Path to audio file (WAV/FLAC/OGG).
        block_size: FFT block size (same as real-time).
        smooth_tau: Analyzer EMA time constant.
        onset_threshold_mult: Beat detector onset sensitivity.

    Returns:
        (frames, sample_rate): List of AudioFrames and the file's sample rate.
    """
    audio, sr = sf.read(audio_path, dtype="float32")

    # Mix to mono
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    analyzer = AudioAnalyzer(sample_rate=sr, block_size=block_size, smooth_tau=smooth_tau)
    beat_detector = BeatDetector(sample_rate=sr, onset_threshold_mult=onset_threshold_mult)

    frames: list[AudioFrame] = []
    num_blocks = len(audio) // block_size

    for i in range(num_blocks):
        start = i * block_size
        chunk = audio[start:start + block_size]
        timestamp = start / sr

        frame = analyzer.analyze(chunk, timestamp=timestamp)
        frame = beat_detector.process(frame)
        frames.append(frame)

    return frames, sr
