"""PNG sequence recorder — writes frames as numbered PNGs for later encoding."""

from pathlib import Path

import numpy as np
from PIL import Image

from easey_glyph.output import VideoOutputSender


class PNGRecorder(VideoOutputSender):
    """Write RGBA frames as numbered PNGs to a directory."""

    def __init__(self):
        self._dir: Path | None = None
        self._frame_count: int = 0
        self._active: bool = False

    def start(self, width: int, height: int, fps: float, name: str = "EASEy-GLYPH") -> bool:
        # Directory is set via set_output_dir before start
        if self._dir is None:
            return False
        self._dir.mkdir(parents=True, exist_ok=True)
        self._frame_count = 0
        self._active = True
        print(f"PNGRecorder: recording to {self._dir}")
        return True

    def set_output_dir(self, path: str):
        self._dir = Path(path)

    def send_frame(self, frame: np.ndarray) -> bool:
        if not self._active or self._dir is None:
            return False
        img = Image.fromarray(frame, "RGBA")
        filename = self._dir / f"frame_{self._frame_count:06d}.png"
        img.save(filename, compress_level=1)
        self._frame_count += 1
        return True

    def stop(self):
        if self._active:
            print(f"PNGRecorder: stopped after {self._frame_count} frames -> {self._dir}")
        self._active = False

    @property
    def name(self) -> str:
        return "Recording"

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def active(self) -> bool:
        return self._active
