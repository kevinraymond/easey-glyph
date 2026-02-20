"""Video output senders for high-resolution output and video routing.

Supports NDI (network), Syphon (macOS), Spout (Windows), and PNG recording.
All senders implement the VideoOutputSender interface.
"""

from abc import ABC, abstractmethod

import numpy as np


class VideoOutputSender(ABC):
    """Abstract base class for video output senders."""

    @abstractmethod
    def start(self, width: int, height: int, fps: float, name: str = "EASEy-GLYPH") -> bool:
        """Initialize the sender. Returns True on success."""
        ...

    @abstractmethod
    def send_frame(self, frame: np.ndarray) -> bool:
        """Send a frame. Input: [H, W, 4] RGBA uint8. Returns True on success."""
        ...

    @abstractmethod
    def stop(self):
        """Clean up resources."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for status display."""
        ...


# Optional sender imports with availability flags
HAS_NDI = False
HAS_SYPHON = False
HAS_SPOUT = False

try:
    from easey_glyph.output.ndi_sender import NDISender
    HAS_NDI = True
except ImportError:
    pass

try:
    from easey_glyph.output.syphon_sender import SyphonSender
    HAS_SYPHON = True
except ImportError:
    pass

try:
    from easey_glyph.output.spout_sender import SpoutSender
    HAS_SPOUT = True
except ImportError:
    pass

# Always available
from easey_glyph.output.png_recorder import PNGRecorder
