"""Spout video output sender (Windows only)."""

import numpy as np

from easey_glyph.output import VideoOutputSender

import SpoutGL


class SpoutSender(VideoOutputSender):
    """Send frames via Spout (Windows OpenGL texture sharing)."""

    def __init__(self):
        self._sender = None
        self._width = 0
        self._height = 0
        self._name_str = "EASEy-GLYPH"

    def start(self, width: int, height: int, fps: float, name: str = "EASEy-GLYPH") -> bool:
        self._name_str = name
        self._width = width
        self._height = height
        try:
            self._sender = SpoutGL.SpoutSender()
            self._sender.setSenderName(name)
            print(f"Spout: started sender '{name}' at {width}x{height}")
            return True
        except Exception as e:
            print(f"Spout: failed to start — {e}")
            self._sender = None
            return False

    def send_frame(self, frame: np.ndarray) -> bool:
        if self._sender is None:
            return False
        try:
            # Spout expects RGBA, which is what we have
            self._sender.sendImage(
                frame.tobytes(),
                self._width,
                self._height,
                SpoutGL.GL_RGBA,
                False,
            )
            return True
        except Exception as e:
            print(f"Spout: send error — {e}")
            return False

    def stop(self):
        if self._sender is not None:
            try:
                self._sender.releaseSender()
            except Exception:
                pass
            self._sender = None
            print("Spout: stopped")

    @property
    def name(self) -> str:
        return "Spout"
