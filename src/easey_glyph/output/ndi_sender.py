"""NDI video output sender using cyndilib."""

from fractions import Fraction

import numpy as np

from easey_glyph.output import VideoOutputSender

from cyndilib.sender import Sender
from cyndilib.video_frame import VideoSendFrame
from cyndilib.wrapper.ndi_structs import FourCC


class NDISender(VideoOutputSender):
    """Send frames over NDI (Network Device Interface) using cyndilib."""

    def __init__(self):
        self._sender: Sender | None = None
        self._video_frame: VideoSendFrame | None = None
        self._width = 0
        self._height = 0
        self._name_str = "EASEy-GLYPH"

    def start(self, width: int, height: int, fps: float, name: str = "EASEy-GLYPH") -> bool:
        self._name_str = name
        self._width = width
        self._height = height
        try:
            self._video_frame = VideoSendFrame()
            self._video_frame.set_resolution(width, height)
            self._video_frame.set_frame_rate(Fraction(int(fps), 1))
            self._video_frame.set_fourcc(FourCC.RGBA)

            self._sender = Sender(ndi_name=name)
            self._sender.set_video_frame(self._video_frame)
            self._sender.open()
            print(f"NDI: started sender '{name}' at {width}x{height} (RGBA)")
            return True
        except Exception as e:
            print(f"NDI: failed to start — {e}")
            self._sender = None
            self._video_frame = None
            return False

    def send_frame(self, frame: np.ndarray) -> bool:
        sender = self._sender  # local snapshot — guards against concurrent stop()
        if sender is None:
            return False
        try:
            sender.write_video_async(frame.ravel())
            return True
        except RuntimeError as e:
            if "no write frame available" in str(e):
                return False  # normal backpressure — drop frame silently
            print(f"NDI: send error — {e}")
            return False
        except Exception as e:
            print(f"NDI: send error — {e}")
            return False

    def stop(self):
        if self._sender is not None:
            try:
                self._sender.close()
            except Exception:
                pass
            self._sender = None
            self._video_frame = None
            print("NDI: stopped")

    @property
    def name(self) -> str:
        return f"NDI ({self._name_str})"
