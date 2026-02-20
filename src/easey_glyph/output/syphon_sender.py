"""Syphon video output sender (macOS only).

Shares frames with VJ apps (Resolume, VDMX, MadMapper, etc.) via Metal textures.
Requires: uv pip install syphon-python
"""

import numpy as np

from easey_glyph.output import VideoOutputSender

import syphon
from syphon.utils.raw import create_mtl_texture
from syphon.utils.numpy import copy_image_to_mtl_texture


class SyphonSender(VideoOutputSender):
    """Send frames via Syphon (macOS Metal GPU texture sharing)."""

    def __init__(self):
        self._server = None
        self._texture = None
        self._width = 0
        self._height = 0
        self._name_str = "EASEy-GLYPH"

    def start(self, width: int, height: int, fps: float, name: str = "EASEy-GLYPH") -> bool:
        self._name_str = name
        self._width = width
        self._height = height
        try:
            self._server = syphon.SyphonMetalServer(name)
            self._texture = create_mtl_texture(self._server.device, width, height)
            print(f"Syphon: started server '{name}' at {width}x{height}")
            return True
        except Exception as e:
            print(f"Syphon: failed to start — {e}")
            self._server = None
            return False

    def send_frame(self, frame: np.ndarray) -> bool:
        if self._server is None:
            return False
        try:
            # Flip vertically (Metal origin is bottom-left)
            flipped = frame[::-1]
            copy_image_to_mtl_texture(flipped, self._texture)
            self._server.publish_frame_texture(self._texture)
            return True
        except Exception as e:
            print(f"Syphon: send error — {e}")
            return False

    def stop(self):
        if self._server is not None:
            try:
                self._server.stop()
            except Exception:
                pass
            self._server = None
            print("Syphon: stopped")

    @property
    def name(self) -> str:
        return "Syphon"
