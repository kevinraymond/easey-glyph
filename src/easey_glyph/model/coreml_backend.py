"""CoreML inference backends for macOS.

Drop-in replacements for FlowUNet and GlyphSuperRes that run via CoreML
instead of PyTorch. Used automatically when .mlpackage files are available
on MPS devices.
"""

import numpy as np


class CoreMLFlowModel:
    """Drop-in replacement for FlowUNet on macOS via CoreML.

    The .mlpackage contains a MultiStepWrapper that unrolls the full
    Euler ODE solve, so a single predict() call produces a complete grid.
    """

    def __init__(self, model_path: str):
        import coremltools as ct
        self.model = ct.models.MLModel(model_path)

    def generate_grid(self, noise: np.ndarray, audio: np.ndarray | None = None) -> np.ndarray:
        """Generate a glyph grid from noise.

        Args:
            noise: [1, 16, 32, 32] float32
            audio: [1, 12] float32 or None (zeros if None)

        Returns:
            [1, 16, 32, 32] float32 grid
        """
        if audio is None:
            audio = np.zeros((1, 12), dtype=np.float32)
        pred = self.model.predict({"grid": noise, "audio": audio})
        return pred["output"]


class CoreMLSuperRes:
    """Drop-in replacement for GlyphSuperRes on macOS via CoreML."""

    def __init__(self, model_path: str):
        import coremltools as ct
        self.model = ct.models.MLModel(model_path)

    def upscale(self, rgb: np.ndarray) -> np.ndarray:
        """Upscale a 32x32 RGB image to 256x256.

        Args:
            rgb: [1, 3, 32, 32] float32 in [0, 1]

        Returns:
            [1, 3, 256, 256] float32 in [0, 1]
        """
        pred = self.model.predict({"rgb": rgb})
        return pred["output"]
