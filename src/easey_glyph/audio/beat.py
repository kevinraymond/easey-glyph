"""3-stage beat detection pipeline: OnsetDetector → TempoEstimator → BeatScheduler.

Ported from EASE's TypeScript implementation, adapted for Python/numpy with
float magnitude spectra (instead of uint8 Web Audio API data).

Stage 1 — OnsetDetector: Multi-band spectral flux with adaptive threshold (median + k*MAD).
Stage 2 — TempoEstimator: Autocorrelation with harmonic enhancement.
Stage 3 — BeatScheduler: Predictive beat tracking with phase correction.

Works identically for real-time and offline — uses frame.timestamp for timing.
"""

from __future__ import annotations

import logging
import math

import numpy as np

logger = logging.getLogger(__name__)

from .frame import AudioFrame


# ---------------------------------------------------------------------------
# Circular buffer (numpy-backed, O(1) push, O(n) stats)
# ---------------------------------------------------------------------------

class CircularBuffer:
    """Fixed-size ring buffer with statistical methods for signal processing."""

    __slots__ = ("_buf", "_cap", "_write", "_count")

    def __init__(self, capacity: int):
        self._cap = capacity
        self._buf = np.zeros(capacity, dtype=np.float64)
        self._write = 0
        self._count = 0

    def push(self, value: float) -> None:
        self._buf[self._write] = value
        self._write = (self._write + 1) % self._cap
        if self._count < self._cap:
            self._count += 1

    @property
    def length(self) -> int:
        return self._count

    def clear(self) -> None:
        self._write = 0
        self._count = 0
        self._buf[:] = 0.0

    def _values(self) -> np.ndarray:
        """Return view of stored values (oldest first). Returns a copy."""
        if self._count == 0:
            return np.array([], dtype=np.float64)
        if self._count < self._cap:
            return self._buf[:self._count].copy()
        # Full: wrap around
        start = self._write  # oldest
        return np.concatenate([self._buf[start:], self._buf[:start]])

    def to_array(self) -> np.ndarray:
        return self._values()

    def get_last(self) -> float:
        if self._count == 0:
            return 0.0
        return float(self._buf[(self._write - 1) % self._cap])

    def median(self) -> float:
        if self._count == 0:
            return 0.0
        return float(np.median(self._values()))

    def mad(self) -> float:
        """Median Absolute Deviation."""
        if self._count == 0:
            return 0.0
        vals = self._values()
        med = np.median(vals)
        return float(np.median(np.abs(vals - med)))

    def max(self) -> float:
        if self._count == 0:
            return 0.0
        if self._count < self._cap:
            return float(self._buf[:self._count].max())
        return float(self._buf.max())

    def mean(self) -> float:
        if self._count == 0:
            return 0.0
        if self._count < self._cap:
            return float(self._buf[:self._count].mean())
        return float(self._buf.mean())


# ---------------------------------------------------------------------------
# Stage 1: Multi-band onset detection
# ---------------------------------------------------------------------------

# Frequency bands for onset detection (Hz range, weight)
_ONSET_BANDS = [
    (20, 80, 0.4),     # sub-bass (kick drums)
    (80, 250, 0.3),    # bass (bass guitar/synth)
    (500, 2000, 0.2),  # mid (snares/vocals)
    (2000, 4000, 0.1), # high-mid (hi-hats/cymbals)
]


class OnsetDetector:
    """Multi-band spectral flux onset detection with adaptive threshold.

    Uses half-wave rectified spectral flux across 4 frequency bands,
    combined with perceptual weights emphasizing low frequencies.
    Threshold uses median + k*MAD (robust to outliers).
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        threshold_mult: float = 2.0,
        history_size: int = 22,          # ~0.5s at 43fps
        long_term_size: int = 172,       # ~4s at 43fps
        threshold_ceiling: float = 0.5,
        silence_threshold: float = 0.002,
    ):
        self._sr = sample_rate
        self._threshold_mult = threshold_mult
        self._silence_threshold = silence_threshold
        self._threshold_ceiling = threshold_ceiling

        # Per-band state
        self._band_masks: dict[int, list[tuple[int, int]]] = {}  # fft_size → [(lo_bin, hi_bin), ...]
        self._prev_mags: list[np.ndarray | None] = [None] * len(_ONSET_BANDS)
        self._band_flux_history = [CircularBuffer(history_size) for _ in _ONSET_BANDS]

        # Combined onset function
        self._onset_history = CircularBuffer(history_size)
        self._long_term_history = CircularBuffer(long_term_size)

        # Silence tracking
        self._silent_frames = 0

    def _get_bin_range(self, lo_hz: float, hi_hz: float, fft_size: int) -> tuple[int, int]:
        """Convert Hz range to FFT bin range."""
        bin_width = self._sr / fft_size
        lo_bin = max(0, int(round(lo_hz / bin_width)))
        hi_bin = min(fft_size // 2, int(round(hi_hz / bin_width)))
        return lo_bin, hi_bin

    def process(
        self,
        bass_spectrum: np.ndarray,   # 4096-pt FFT magnitudes
        mid_spectrum: np.ndarray,    # 1024-pt FFT magnitudes
        high_spectrum: np.ndarray,   # 512-pt FFT magnitudes
        rms: float,
    ) -> tuple[bool, float, float]:
        """Detect onsets from multi-resolution spectra.

        Args:
            bass_spectrum: |FFT| from 4096-pt window (sub-bass + bass bands)
            mid_spectrum: |FFT| from 1024-pt window (mid band)
            high_spectrum: |FFT| from 512-pt window (high-mid band)
            rms: Current RMS energy (for silence gating)

        Returns:
            (is_onset, onset_strength, combined_flux)
        """
        # Silence gate
        if rms < self._silence_threshold:
            self._silent_frames += 1
            return False, 0.0, 0.0
        self._silent_frames = 0

        # Map bands to spectra: sub-bass & bass → 4096, mid → 1024, high-mid → 512
        spectra = [bass_spectrum, bass_spectrum, mid_spectrum, high_spectrum]
        fft_sizes = [len(bass_spectrum) * 2 - 2, len(bass_spectrum) * 2 - 2,
                     len(mid_spectrum) * 2 - 2, len(high_spectrum) * 2 - 2]

        band_flux = np.zeros(len(_ONSET_BANDS))

        for i, (lo_hz, hi_hz, weight) in enumerate(_ONSET_BANDS):
            spectrum = spectra[i]
            fft_size = fft_sizes[i]
            lo_bin, hi_bin = self._get_bin_range(lo_hz, hi_hz, fft_size)

            if hi_bin <= lo_bin:
                continue

            current_mags = spectrum[lo_bin:hi_bin + 1].astype(np.float64)

            if self._prev_mags[i] is not None and len(self._prev_mags[i]) == len(current_mags):
                # Half-wave rectified spectral flux (only increases)
                diff = current_mags - self._prev_mags[i]
                flux = float(np.sum(np.maximum(diff, 0))) / len(current_mags)
            else:
                flux = 0.0

            self._prev_mags[i] = current_mags.copy()
            band_flux[i] = flux
            self._band_flux_history[i].push(flux)

        # Weighted combination
        weights = np.array([b[2] for b in _ONSET_BANDS])
        combined_flux = float(np.sum(band_flux * weights) / weights.sum())

        self._onset_history.push(combined_flux)
        self._long_term_history.push(combined_flux)

        # Adaptive threshold: median + k * MAD
        threshold = self._compute_threshold()
        is_onset = combined_flux > threshold

        return is_onset, combined_flux, combined_flux

    def _compute_threshold(self) -> float:
        """Compute adaptive threshold using median + k * MAD with ceiling."""
        median = self._onset_history.median()
        mad = self._onset_history.mad()
        base_threshold = median + self._threshold_mult * mad

        min_threshold = 0.001

        # Cap at proportion of long-term max
        max_threshold = float("inf")
        if self._long_term_history.length > self._onset_history.length:
            lt_max = self._long_term_history.max()
            max_threshold = lt_max * self._threshold_ceiling

        # Also cap at 80% of recent max
        recent_max = self._onset_history.max()
        recent_ceiling = recent_max * 0.8

        capped = min(base_threshold, max_threshold, recent_ceiling)
        return max(min_threshold, capped)

    def get_onset_function_history(self) -> np.ndarray:
        """Return onset function history for tempo estimator."""
        return self._onset_history.to_array()

    @property
    def is_sustained_silence(self) -> bool:
        return self._silent_frames >= 30

    def set_threshold(self, mult: float) -> None:
        self._threshold_mult = mult


# ---------------------------------------------------------------------------
# Stage 2: Autocorrelation-based tempo estimation
# ---------------------------------------------------------------------------

class TempoEstimator:
    """Estimates tempo via autocorrelation of the onset detection function.

    Uses harmonic enhancement to avoid half/double tempo errors:
    Enhanced(lag) = R(lag) + 0.5*R(2*lag) + 0.33*R(3*lag) + 0.25*R(4*lag)
    """

    def __init__(
        self,
        history_seconds: float = 4.0,
        bpm_range: tuple[float, float] = (40, 300),
        smoothing_factor: float = 0.15,
        frame_rate: float = 43.0,
    ):
        self._bpm_range = bpm_range
        self._smoothing = smoothing_factor

        # History (seconds * fps frames)
        history_size = int(math.ceil(history_seconds * frame_rate))
        self._onset_history = CircularBuffer(history_size)
        self._frame_time_history = CircularBuffer(30)

        self._frame_rate = frame_rate
        self._frame_time = 1.0 / frame_rate  # seconds per frame
        self._initial_frame_time = self._frame_time
        self._last_time = 0.0
        self._frame_count = 0

        # Current estimate
        self._current_bpm = 0.0
        self._current_confidence = 0.0
        self._current_period_frames = 0

        # Stable tempo (locked when consistent)
        self._stable_bpm = 0.0
        self._stable_period_frames = 0
        self._stability_counter = 0

    def update(self, onset_value: float, timestamp: float) -> tuple[float, float, float]:
        """Update tempo estimate.

        Args:
            onset_value: Current onset function value (combined flux)
            timestamp: Time in seconds

        Returns:
            (bpm, confidence, period_seconds)
        """
        # Track frame timing
        if self._last_time > 0:
            dt = timestamp - self._last_time
            if 0 < dt < 0.1:  # reject outliers
                self._frame_time_history.push(dt)
                if self._frame_time_history.length >= 10:
                    measured = self._frame_time_history.mean()
                    # Clamp to ±15% of initial — prevents wild BPM swings
                    # from timestamp jitter (e.g. batched frame delivery)
                    lo = self._initial_frame_time * 0.85
                    hi = self._initial_frame_time * 1.15
                    self._frame_time = max(lo, min(hi, measured))
                    self._frame_rate = 1.0 / self._frame_time
        self._last_time = timestamp

        self._onset_history.push(onset_value)
        self._frame_count += 1

        # Need enough history (at least 2s)
        min_frames = int(math.ceil(2 * self._frame_rate))
        if self._onset_history.length < min_frames:
            return 0.0, 0.0, 0.0

        # Compute autocorrelation every ~6 frames (~10Hz update rate)
        if self._frame_count % 6 != 0:
            period_s = self._current_period_frames * self._frame_time
            return self._current_bpm, self._current_confidence, period_s

        raw_bpm, confidence, period_frames = self._compute_tempo()

        # Smooth BPM
        prev_bpm = self._current_bpm
        if self._current_bpm > 0 and raw_bpm > 0:
            self._current_bpm = self._smooth_bpm(self._current_bpm, raw_bpm, confidence)
        elif raw_bpm > 0:
            self._current_bpm = raw_bpm

        if raw_bpm > 0:
            logger.debug(
                "TEMPO raw=%.1f conf=%.2f smooth=%.1f→%.1f stable=%.1f(n=%d) ft=%.4f",
                raw_bpm, confidence, prev_bpm, self._current_bpm,
                self._stable_bpm, self._stability_counter, self._frame_time,
            )

        self._current_confidence = confidence
        self._current_period_frames = period_frames

        # Track stability
        if confidence > 0.5 and self._stable_bpm > 0:
            bpm_diff = abs(self._current_bpm - self._stable_bpm) / self._stable_bpm
            if bpm_diff < 0.08:
                self._stability_counter += 1
            elif bpm_diff > 0.3 and self._stability_counter < 60:
                logger.debug(
                    "STABLE reset %.1f→%.1f (diff=%.2f, counter=%d)",
                    self._stable_bpm, self._current_bpm, bpm_diff, self._stability_counter,
                )
                self._stable_bpm = self._current_bpm
                self._stable_period_frames = period_frames
                self._stability_counter = 0
        elif confidence > 0.5 and self._current_bpm > 0:
            self._stable_bpm = self._current_bpm
            self._stable_period_frames = period_frames
            self._stability_counter = 1

        # Use stable tempo when current is erratic
        is_jumping = (self._stable_bpm > 0 and
                      abs(self._current_bpm - self._stable_bpm) / max(self._stable_bpm, 1) > 0.15)
        if (confidence < 0.5 or is_jumping) and self._stable_bpm > 0 and self._stability_counter > 60:
            logger.debug(
                "STABLE override %.1f→%.1f (jumping=%s, conf=%.2f)",
                self._current_bpm, self._stable_bpm, is_jumping, confidence,
            )
            self._current_bpm = self._stable_bpm
            self._current_period_frames = self._stable_period_frames

        period_s = self._current_period_frames * self._frame_time
        return self._current_bpm, self._current_confidence, period_s

    def _compute_tempo(self) -> tuple[float, float, float]:
        """Compute tempo via windowed autocorrelation with harmonic enhancement."""
        history = self._onset_history.to_array()
        n = len(history)

        # Convert BPM range to lag range (frames)
        # lag = 60 / (bpm * frame_time) = frame_rate * 60 / bpm
        max_lag = min(
            int(60.0 / (self._bpm_range[0] * self._frame_time)),
            n // 2,
        )
        min_lag = max(
            int(60.0 / (self._bpm_range[1] * self._frame_time)),
            1,
        )

        if max_lag <= min_lag:
            return 0.0, 0.0, 0

        # Autocorrelation
        autocorr = np.zeros(max_lag + 1)

        # Energy at zero lag (for normalization)
        energy = float(np.sum(history * history))
        autocorr[0] = energy

        for lag in range(min_lag, max_lag + 1):
            autocorr[lag] = float(np.sum(history[:n - lag] * history[lag:]))

        # Harmonic enhancement
        enhanced = np.zeros(max_lag + 1)
        for lag in range(min_lag, max_lag + 1):
            val = autocorr[lag]
            if lag * 2 <= max_lag:
                val += 0.5 * autocorr[lag * 2]
            if lag * 3 <= max_lag:
                val += 0.33 * autocorr[lag * 3]
            if lag * 4 <= max_lag:
                val += 0.25 * autocorr[lag * 4]
            enhanced[lag] = val

        # Find peak
        best_lag = min_lag + int(np.argmax(enhanced[min_lag:max_lag + 1]))
        best_value = enhanced[best_lag]

        # Octave correction: prefer the longest period (lowest tempo) among
        # harmonically related candidates.  Without this, lag N often beats
        # lag 2N because harmonic enhancement adds 0.5*R(2N) to lag N's
        # score — giving the double-tempo candidate free credit from the
        # fundamental's autocorrelation.
        #
        # Search a window of ±2 lags around 2*best_lag to handle frame
        # quantization — e.g. lag 9 (284 BPM) should find lag 16 (160 BPM)
        # not lag 18 (142 BPM).
        while best_lag * 2 <= max_lag:
            center = best_lag * 2
            search_lo = max(min_lag, center - 2)
            search_hi = min(max_lag, center + 2)

            best_candidate = None
            best_candidate_val = 0.0
            for dl in range(search_lo, search_hi + 1):
                dv = enhanced[dl]
                # Must be a local peak (not just a random point on the slope)
                if dl > min_lag and dv < enhanced[dl - 1]:
                    continue
                if dl < max_lag and dv < enhanced[dl + 1]:
                    continue
                if dv > best_candidate_val:
                    best_candidate = dl
                    best_candidate_val = dv

            if best_candidate is not None and best_candidate_val >= best_value * 0.8:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "OCTAVE correct lag %d (%.0f bpm) → %d (%.0f bpm)  "
                        "vals %.0f vs %.0f (%.0f%%)",
                        best_lag, 60.0 / (best_lag * self._frame_time),
                        best_candidate, 60.0 / (best_candidate * self._frame_time),
                        best_value, best_candidate_val,
                        100.0 * best_candidate_val / max(best_value, 1),
                    )
                best_lag = best_candidate
                best_value = best_candidate_val
            else:
                break

        # Parabolic interpolation for sub-frame precision.
        # At high BPMs (e.g. 287 = lag 9), adjacent integer lags span ~30 BPM.
        # Without interpolation, the peak jitters between e.g. lag 9 (287) and
        # lag 10 (258), causing the smoothed BPM to drift cyclically.
        refined_lag = float(best_lag)
        if min_lag < best_lag < max_lag:
            alpha = enhanced[best_lag - 1]
            beta = enhanced[best_lag]
            gamma = enhanced[best_lag + 1]
            denom = 2.0 * (2.0 * beta - alpha - gamma)
            if abs(denom) > 1e-10:
                p = (alpha - gamma) / denom
                # Clamp to ±0.5 (don't jump past neighboring bins)
                refined_lag = best_lag + max(-0.5, min(0.5, p))

        # Convert to BPM using refined (fractional) lag
        period_s = refined_lag * self._frame_time
        bpm = 60.0 / period_s if period_s > 0 else 0.0

        # Confidence
        raw_conf = min(1.0, best_value / energy) if energy > 0 else 0.0
        confidence = max(0.15, raw_conf) if raw_conf > 0.05 else raw_conf

        # Log top 3 peaks for diagnostics
        if logger.isEnabledFor(logging.DEBUG):
            peak_lags = np.argsort(enhanced[min_lag:max_lag + 1])[::-1][:3] + min_lag
            peaks_str = ", ".join(
                f"lag{l}={60.0 / (l * self._frame_time):.0f}bpm({enhanced[l]:.2f})"
                for l in peak_lags
            )
            logger.debug(
                "AUTOCORR lags=[%d,%d] best=%d refined=%.2f bpm=%.1f conf=%.2f | %s",
                min_lag, max_lag, best_lag, refined_lag, bpm, confidence, peaks_str,
            )

        if bpm < self._bpm_range[0] or bpm > self._bpm_range[1]:
            return 0.0, 0.0, 0

        # Return unrounded BPM — smoothing operates on continuous values,
        # rounding happens downstream for display only.
        return bpm, confidence, refined_lag

    def _smooth_bpm(self, current: float, new: float, confidence: float) -> float:
        """Smooth BPM with octave error rejection."""
        ratio = new / current if current > 0 else 1.0
        is_half = abs(ratio - 0.5) < 0.15
        is_double = abs(ratio - 2.0) < 0.15
        change = abs(new - current) / max(current, 1)

        # Octave shifts: snap quickly when confident, reject when not
        if is_half or is_double:
            if confidence >= 0.7:
                effective = 0.5
                result = current + effective * (new - current)
                logger.debug("SMOOTH octave snap: cur=%.1f new=%.1f conf=%.2f → %.1f", current, new, confidence, result)
                return result
            else:
                logger.debug("SMOOTH reject octave: cur=%.1f new=%.1f ratio=%.2f conf=%.2f", current, new, ratio, confidence)
                return current

        if change > 0.25 and confidence < 0.7:
            logger.debug("SMOOTH reject big jump: cur=%.1f new=%.1f change=%.2f conf=%.2f", current, new, change, confidence)
            return current

        effective = self._smoothing
        if change > 0.1 and confidence < 0.5:
            effective *= 0.3
        elif confidence > 0.7 and change < 0.05:
            effective *= 1.5

        result = current + effective * (new - current)
        if abs(new - current) > 5:
            logger.debug("SMOOTH accept: cur=%.1f new=%.1f conf=%.2f eff=%.3f → %.1f", current, new, confidence, effective, result)
        return result

    @property
    def period_seconds(self) -> float:
        return self._current_period_frames * self._frame_time

    @property
    def bpm(self) -> float:
        return self._current_bpm

    @property
    def confidence(self) -> float:
        return self._current_confidence


# ---------------------------------------------------------------------------
# Stage 3: Beat scheduler (prediction + confirmation)
# ---------------------------------------------------------------------------

class BeatScheduler:
    """Predictive beat tracking with state machine.

    States: WAITING → EXPECTING → CONFIRMED/MISSED → EXPECTING → ...

    When tempo confidence is high, predicts beats and confirms with onsets.
    When confidence is low, falls back to onset-only detection.
    Phase correction gradually aligns prediction to actual rhythm.
    """

    def __init__(
        self,
        beat_window: float = 0.08,       # ±80ms confirmation window
        refractory: float = 0.15,         # 150ms min between beats (400 BPM max)
        min_tempo_confidence: float = 0.4,
        phase_correction_rate: float = 0.3,
        max_consecutive_misses: int = 4,
        beat_timeout: float = 3.0,        # seconds
    ):
        self._beat_window = beat_window
        self._refractory = refractory
        self._min_confidence = min_tempo_confidence
        self._phase_correction = phase_correction_rate
        self._max_misses = max_consecutive_misses
        self._beat_timeout = beat_timeout

        # State
        self._state = "WAITING"  # WAITING | EXPECTING | CONFIRMED | MISSED
        self._last_beat_time = 0.0
        self._next_predicted = 0.0
        self._phase = 0.0

        # Tempo info (from TempoEstimator)
        self._bpm = 0.0
        self._period = 0.0   # seconds
        self._tempo_confidence = 0.0

        # Tracking
        self._beat_strength = 0.0
        self._tracking_confidence = 0.0
        self._consecutive_misses = 0
        self._last_fired_time = 0.0
        self._stored_period = 0.0  # backup for prediction during low confidence

    def update_tempo(self, bpm: float, period: float, confidence: float) -> None:
        """Update with latest tempo estimate."""
        self._bpm = bpm
        self._period = period
        self._tempo_confidence = confidence
        if confidence >= self._min_confidence and period > 0:
            self._stored_period = period

    def process(
        self,
        is_onset: bool,
        onset_strength: float,
        timestamp: float,
        is_silence: bool = False,
    ) -> tuple[bool, float, float]:
        """Main beat decision method.

        Args:
            is_onset: Whether an onset was detected this frame
            onset_strength: Strength of onset
            timestamp: Current time in seconds
            is_silence: Sustained silence flag

        Returns:
            (is_beat, beat_phase, smoothed_bpm)
        """
        time_since_beat = timestamp - self._last_beat_time if self._last_beat_time > 0 else float("inf")
        in_refractory = time_since_beat < self._refractory

        if is_onset:
            self._beat_strength = onset_strength

        # Update phase and count missed predicted beats
        missed = 0
        if self._period > 0:
            missed = self._update_phase(timestamp)

        # Beat timeout recovery
        time_since_fired = timestamp - self._last_fired_time if self._last_fired_time > 0 else 0.0
        if time_since_fired > self._beat_timeout and self._last_fired_time > 0:
            self._tracking_confidence *= 0.3
            self._consecutive_misses = 0
            self._state = "WAITING"
            self._last_fired_time = timestamp

        # Silence → pause beat detection
        if is_silence:
            if self._state != "WAITING":
                self._state = "WAITING"
                self._consecutive_misses = 0
            return False, 0.0, self._bpm

        # Determine if we should fire a beat
        is_beat = False

        if missed > 0 and self._tempo_confidence >= self._min_confidence:
            # Predicted beat that wasn't confirmed by onset
            is_beat = True
            self._beat_strength = 0.5
            self._state = "MISSED"
        elif self._tempo_confidence < self._min_confidence or self._period == 0:
            # Low confidence: onset-only mode
            is_beat = self._onset_only(is_onset, in_refractory)

            # Backup prediction: use stored period if onset-only hasn't fired
            if not is_beat and self._stored_period > 0 and self._last_fired_time > 0:
                if time_since_fired >= self._stored_period * 0.9 and not in_refractory:
                    is_beat = True
                    self._beat_strength = 0.5
        else:
            # High confidence: predictive mode
            is_beat = self._predictive(is_onset, timestamp, in_refractory)

        if is_beat:
            self._last_beat_time = timestamp
            self._last_fired_time = timestamp
            self._phase = 0.0
            self._consecutive_misses = 0
            if self._period > 0:
                self._next_predicted = timestamp + self._period
            self._tracking_confidence = min(1.0, self._tracking_confidence + 0.15)
        else:
            self._tracking_confidence = max(0.0, self._tracking_confidence - 0.0005)

        return is_beat, self._phase, self._bpm

    def _onset_only(self, is_onset: bool, in_refractory: bool) -> bool:
        """Fallback: trigger on detected onsets."""
        self._state = "WAITING"
        if is_onset and not in_refractory:
            self._state = "CONFIRMED"
            return True
        return False

    def _predictive(self, is_onset: bool, timestamp: float, in_refractory: bool) -> bool:
        """Predictive mode: confirmation window + phase correction."""
        dist = timestamp - self._next_predicted if self._next_predicted > 0 else float("inf")
        in_window = abs(dist) <= self._beat_window
        window_expired = dist > self._beat_window

        if self._state == "WAITING":
            if self._next_predicted == 0:
                if is_onset and not in_refractory:
                    self._next_predicted = timestamp + self._period
                    self._state = "CONFIRMED"
                    return True
            else:
                self._state = "EXPECTING"
            return False

        elif self._state == "EXPECTING":
            if is_onset and in_window and not in_refractory:
                self._state = "CONFIRMED"
                if self._period > 0:
                    phase_error = dist / self._period
                    self._apply_phase_correction(phase_error)
                return True
            elif window_expired:
                self._state = "MISSED"
                return True
            return False

        else:  # CONFIRMED or MISSED
            self._state = "EXPECTING"
            if is_onset and in_window and not in_refractory:
                self._state = "CONFIRMED"
                return True
            return False

    def _update_phase(self, timestamp: float) -> int:
        """Update beat phase. Returns count of missed predicted beats."""
        if self._last_beat_time == 0 or self._period == 0:
            return 0

        elapsed = timestamp - self._last_beat_time
        self._phase = (elapsed / self._period) % 1.0

        missed = 0
        while self._next_predicted > 0 and timestamp > self._next_predicted + self._beat_window:
            self._consecutive_misses += 1
            missed += 1
            self._next_predicted += self._period
            if self._consecutive_misses == self._max_misses:
                self._tracking_confidence *= 0.7
        return missed

    def _apply_phase_correction(self, phase_error: float) -> None:
        """Gradually adjust prediction to align with detected onsets."""
        clamped = max(-0.5, min(0.5, phase_error))
        correction = clamped * self._period * self._phase_correction
        self._next_predicted += correction


# ---------------------------------------------------------------------------
# Main BeatDetector facade (replaces old BeatDetector class)
# ---------------------------------------------------------------------------

class BeatDetector:
    """3-stage beat detection pipeline.

    Drop-in replacement for the original single-stage BeatDetector.
    Same public API: process(frame) → augmented AudioFrame.

    Requires spectra from AudioAnalyzer — call set_spectra() before process(),
    or use process_with_spectra() directly.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        onset_threshold_mult: float = 2.0,
        onset_cooldown: float = 0.05,
        beat_cooldown: float = 0.25,
        bpm_smooth: float = 0.7,
        bpm_window: float = 10.0,
        onset_window: int = 43,
    ):
        frame_rate = sample_rate / 1024  # approximate fps (~43 at 44100/1024)

        self._onset_detector = OnsetDetector(
            sample_rate=sample_rate,
            threshold_mult=onset_threshold_mult,
            history_size=max(10, int(0.5 * frame_rate)),   # ~0.5s
            long_term_size=max(20, int(4.0 * frame_rate)),  # ~4s
        )
        self._tempo_estimator = TempoEstimator(
            history_seconds=4.0,
            bpm_range=(40, 300),
            smoothing_factor=0.15,
            frame_rate=frame_rate,
        )
        self._beat_scheduler = BeatScheduler()

        # Onset hold+decay (200ms exponential release — smooth visual pulses)
        self._held_onset = 0.0
        self._onset_decay_tau = 0.20
        self._last_timestamp = 0.0

        # Onset cooldown
        self._onset_cooldown = onset_cooldown
        self._last_onset_time = 0.0

        # Pending spectra (set by analyzer before process)
        self._bass_spectrum: np.ndarray | None = None
        self._mid_spectrum: np.ndarray | None = None
        self._high_spectrum: np.ndarray | None = None

    def set_onset_threshold(self, mult: float) -> None:
        """Update onset detection threshold multiplier."""
        self._onset_detector.set_threshold(mult)

    def set_spectra(
        self,
        bass_spectrum: np.ndarray,
        mid_spectrum: np.ndarray,
        high_spectrum: np.ndarray,
    ) -> None:
        """Set spectra for next process() call."""
        self._bass_spectrum = bass_spectrum
        self._mid_spectrum = mid_spectrum
        self._high_spectrum = high_spectrum

    def process(self, frame: AudioFrame) -> AudioFrame:
        """Augment AudioFrame with beat/onset detection.

        Spectra must be set via set_spectra() before calling this.
        Falls back to raw_bass-only detection if spectra aren't available.
        """
        if self._bass_spectrum is not None:
            return self.process_with_spectra(
                frame,
                self._bass_spectrum,
                self._mid_spectrum,
                self._high_spectrum,
            )
        # Fallback: create fake spectra from raw_bass (backward compat)
        return self._process_fallback(frame)

    def process_with_spectra(
        self,
        frame: AudioFrame,
        bass_spectrum: np.ndarray,
        mid_spectrum: np.ndarray,
        high_spectrum: np.ndarray,
    ) -> AudioFrame:
        """Full 3-stage pipeline with multi-resolution spectra."""
        now = frame.timestamp
        dt = max(now - self._last_timestamp, 0.0) if self._last_timestamp > 0 else 0.0
        self._last_timestamp = now

        # Stage 1: Onset detection
        is_onset, onset_strength, combined_flux = self._onset_detector.process(
            bass_spectrum, mid_spectrum, high_spectrum, frame.rms,
        )

        # Apply onset cooldown
        if is_onset and (now - self._last_onset_time) < self._onset_cooldown:
            is_onset = False
        if is_onset:
            self._last_onset_time = now

        # Stage 2: Tempo estimation
        bpm, confidence, period_s = self._tempo_estimator.update(combined_flux, now)

        # Stage 3: Beat scheduling
        self._beat_scheduler.update_tempo(bpm, period_s, confidence)
        is_beat, beat_phase, smoothed_bpm = self._beat_scheduler.process(
            is_onset, onset_strength, now,
            is_silence=self._onset_detector.is_sustained_silence,
        )

        # Onset hold+decay (instant attack, 200ms exponential release)
        if onset_strength > self._held_onset:
            self._held_onset = onset_strength
        elif dt > 0:
            self._held_onset *= math.exp(-dt / self._onset_decay_tau)

        # Freeze phase at 0 during silence
        if frame.rms < 1e-4:
            beat_phase = 0.0

        # Update frame (same fields as before — downstream unchanged)
        frame.is_onset = is_onset
        frame.onset_strength = self._held_onset
        frame.beat_phase = beat_phase
        frame.bpm = smoothed_bpm if smoothed_bpm > 0 else 120.0
        frame.is_beat = is_beat

        return frame

    def _process_fallback(self, frame: AudioFrame) -> AudioFrame:
        """Legacy fallback when spectra aren't provided."""
        now = frame.timestamp
        dt = max(now - self._last_timestamp, 0.0) if self._last_timestamp > 0 else 0.0
        self._last_timestamp = now

        # Use raw_bass as a simple onset signal
        flux = frame.raw_bass
        is_onset = flux > 0.5 and (now - self._last_onset_time) > self._onset_cooldown
        if is_onset:
            self._last_onset_time = now

        onset_strength = min(flux, 1.0) if is_onset else 0.0

        # Onset hold+decay
        if onset_strength > self._held_onset:
            self._held_onset = onset_strength
        elif dt > 0:
            self._held_onset *= math.exp(-dt / self._onset_decay_tau)

        # Simple beat phase from raw_bass timing
        if frame.rms < 1e-4:
            beat_phase = 0.0
        else:
            beat_phase = frame.beat_phase

        frame.is_onset = is_onset
        frame.onset_strength = self._held_onset
        frame.is_beat = is_onset

        return frame
