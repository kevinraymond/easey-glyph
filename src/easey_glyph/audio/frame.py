"""AudioFrame: standardized audio features, all normalized 0.0-1.0."""

from dataclasses import dataclass, field


@dataclass
class AudioFrame:
    """Audio features computed per analysis window.

    Band layout (7 bands):
        0: sub_bass    (20-60 Hz)
        1: bass        (60-250 Hz)
        2: low_mid     (250-500 Hz)
        3: mid         (500-2000 Hz)
        4: upper_mid   (2000-4000 Hz)
        5: presence    (4000-6000 Hz)
        6: brilliance  (6000-20000 Hz)
    """

    timestamp: float = 0.0

    # Energy
    rms: float = 0.0
    peak: float = 0.0
    energy_bands: list[float] = field(default_factory=lambda: [0.0] * 7)

    @property
    def bass(self) -> float:
        return (self.energy_bands[0] + self.energy_bands[1]) / 2 if len(self.energy_bands) >= 2 else 0.0

    @property
    def mid(self) -> float:
        if len(self.energy_bands) >= 5:
            return (self.energy_bands[2] + self.energy_bands[3] + self.energy_bands[4]) / 3
        return 0.0

    @property
    def treble(self) -> float:
        if len(self.energy_bands) >= 7:
            return (self.energy_bands[5] + self.energy_bands[6]) / 2
        return 0.0

    # Kick-band spectral flux (30-120Hz, normalized 0-1) for beat detection
    raw_bass: float = 0.0

    # Rhythm
    is_onset: bool = False
    onset_strength: float = 0.0
    beat_phase: float = 0.0
    bpm: float = 120.0
    is_beat: bool = False

    # Timbre
    spectral_centroid: float = 0.0
    spectral_flux: float = 0.0
    spectral_flatness: float = 0.0
    spectral_rolloff: float = 0.0
    spectral_bandwidth: float = 0.0
    zero_crossing_rate: float = 0.0
