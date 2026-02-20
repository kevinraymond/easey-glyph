"""Audio capture, analysis, and beat detection for EASEy-GLYPH."""

from .frame import AudioFrame
from .analyzer import AudioAnalyzer
from .beat import BeatDetector

__all__ = ["AudioFrame", "AudioAnalyzer", "BeatDetector", "AudioCapture"]


def __getattr__(name: str):
    if name == "AudioCapture":
        from .capture import AudioCapture
        return AudioCapture
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
