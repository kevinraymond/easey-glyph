"""Beat detection: onset via kick-band spectral flux, BPM via inter-onset intervals.

Uses normalized spectral flux in 30-120Hz (from analyzer) as onset signal.
Spectral flux measures sudden spectral change — kicks produce sharp spikes
while sustained bass (guitar, synths) produces near-zero flux.

Works identically for real-time and offline — uses frame.timestamp for timing.
"""

import math

import numpy as np

from .frame import AudioFrame


class BeatDetector:
    """Real-time beat and onset detection from audio features."""

    def __init__(
        self,
        sample_rate: int = 44100,
        onset_threshold_mult: float = 1.4,
        onset_cooldown: float = 0.05,
        beat_cooldown: float = 0.25,
        bpm_smooth: float = 0.7,
        bpm_window: float = 10.0,
        onset_window: int = 43,
    ):
        """
        Args:
            sample_rate: Audio sample rate.
            onset_threshold_mult: Bass delta must exceed mean + mult * std to trigger.
            onset_cooldown: Minimum seconds between onsets.
            beat_cooldown: Minimum seconds between beats.
            bpm_smooth: EMA factor for BPM (lower = faster response).
            bpm_window: Seconds of onset history for BPM estimation.
            onset_window: Frames of bass delta history for adaptive threshold (~1s).
        """
        self.sample_rate = sample_rate
        self._onset_threshold_mult = onset_threshold_mult
        self._onset_cooldown = onset_cooldown
        self._beat_cooldown = beat_cooldown
        self._bpm_smooth = bpm_smooth
        self._bpm_window = bpm_window
        self._onset_window = onset_window

        # State
        self._flux_history: list[float] = []
        self._last_onset_time = 0.0
        self._last_timestamp = 0.0
        self._onset_times: list[float] = []
        self._current_bpm = 120.0
        self._last_beat_time = 0.0
        self._beat_interval = 0.5  # 120 BPM
        self._held_onset = 0.0          # onset strength with hold + decay
        self._onset_decay_tau = 0.20    # ~200ms exponential decay

    def set_onset_threshold(self, mult: float):
        """Update onset detection threshold multiplier."""
        self._onset_threshold_mult = mult

    def process(self, frame: AudioFrame) -> AudioFrame:
        """Augment AudioFrame with beat/onset detection results."""
        now = frame.timestamp
        dt = max(now - self._last_timestamp, 0.0) if self._last_timestamp > 0 else 0.0
        self._last_timestamp = now

        # --- Onset detection via kick-band spectral flux ---
        # raw_bass is normalized spectral flux in 30-120Hz (already a change signal).
        flux = frame.raw_bass

        self._flux_history.append(flux)
        if len(self._flux_history) > self._onset_window:
            self._flux_history.pop(0)

        is_onset = False
        onset_strength = 0.0
        if len(self._flux_history) >= 5:
            arr = np.array(self._flux_history)
            mean_val = arr.mean()
            std_val = arr.std() + 1e-6
            threshold = mean_val + self._onset_threshold_mult * std_val

            if flux > threshold and (now - self._last_onset_time) > self._onset_cooldown:
                is_onset = True
                onset_strength = min((flux - mean_val) / (std_val * 1.5 + 1e-6), 1.0)
                self._last_onset_time = now
                self._onset_times.append(now)

        # --- Onset hold + decay (instant attack, exponential release) ---
        if onset_strength > self._held_onset:
            self._held_onset = onset_strength
        elif dt > 0:
            decay = math.exp(-dt / self._onset_decay_tau)
            self._held_onset *= decay

        # --- BPM estimation via inter-onset intervals ---
        cutoff = now - self._bpm_window
        self._onset_times = [t for t in self._onset_times if t > cutoff]

        if len(self._onset_times) >= 4:
            intervals = np.diff(self._onset_times)
            valid = intervals[(intervals > 0.2) & (intervals < 1.5)]
            if len(valid) >= 2:
                median_interval = float(np.median(valid))
                raw_bpm = 60.0 / median_interval
                self._current_bpm = (
                    self._bpm_smooth * self._current_bpm
                    + (1 - self._bpm_smooth) * raw_bpm
                )
                self._beat_interval = 60.0 / self._current_bpm

        # --- Beat phase ---
        elapsed = now - self._last_beat_time
        beat_phase = (elapsed % self._beat_interval) / self._beat_interval

        is_beat = False
        if is_onset and elapsed >= self._beat_cooldown:
            self._last_beat_time = now
            is_beat = True
            beat_phase = 0.0

        # Update frame
        frame.is_onset = is_onset
        frame.onset_strength = self._held_onset
        frame.beat_phase = beat_phase
        frame.bpm = self._current_bpm
        frame.is_beat = is_beat

        return frame
