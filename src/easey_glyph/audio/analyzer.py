"""Audio analysis: multi-resolution FFT, dB-scale energy, asymmetric smoothing.

Tier 1 SOTA improvements over original:
- Asymmetric attack/release EMA (fast rise, slow decay)
- Multi-resolution FFT: 4096 for bass, 1024 for mids, 512 for highs
- dB-scale band energy with 80dB range normalization
- Adaptive min/max normalization (tracks current dynamic context)
- 4 new spectral features: flatness, rolloff, bandwidth, ZCR
- Optional mel-scale filterbank
"""

import math
import time

import numpy as np

from .frame import AudioFrame

# 7-band frequency boundaries (Hz)
BAND_EDGES = [20, 60, 250, 500, 2000, 4000, 6000, 20000]

# dB floor — silence threshold
DB_FLOOR = -80.0
DB_RANGE = 80.0  # 0 dB to -80 dB


def _build_mel_filterbank(n_mels: int, n_fft: int, sr: int,
                          fmin: float = 20.0, fmax: float = 20000.0) -> np.ndarray:
    """Build a mel-scale triangular filterbank matrix.

    Returns:
        [n_mels, n_fft//2+1] float64 array.
    """
    # Mel scale conversions
    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    n_bins = n_fft // 2 + 1
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    freqs = np.linspace(0, sr / 2, n_bins)
    filterbank = np.zeros((n_mels, n_bins), dtype=np.float64)

    for i in range(n_mels):
        lo, center, hi = hz_points[i], hz_points[i + 1], hz_points[i + 2]
        # Rising slope
        up = (freqs - lo) / max(center - lo, 1e-10)
        # Falling slope
        down = (hi - freqs) / max(hi - center, 1e-10)
        filterbank[i] = np.maximum(0, np.minimum(up, down))
        # Slaney normalization: area = 1
        area = filterbank[i].sum()
        if area > 0:
            filterbank[i] /= area

    return filterbank


class AudioAnalyzer:
    """Computes audio features from raw PCM samples.

    Multi-resolution FFT with asymmetric attack/release smoothing,
    dB-scale energy, and adaptive normalization.
    """

    def __init__(self, sample_rate: int = 44100, block_size: int = 1024,
                 smooth_tau: float | None = None,
                 attack_tau: float = 0.005, release_tau: float = 0.08,
                 band_decay: float = 0.99,
                 use_mel: bool = False, n_mels: int = 7):
        """
        Args:
            sample_rate: Audio sample rate in Hz.
            block_size: Base block size (used for capture chunk sizing).
            smooth_tau: Legacy compat — if set, maps to attack=tau*0.1, release=tau*1.6.
            attack_tau: EMA time constant for rising signals (seconds). Lower = snappier.
            release_tau: EMA time constant for falling signals (seconds). Higher = smoother decay.
            band_decay: Per-frame decay for adaptive min/max (0.99-0.999).
            use_mel: Use mel-scale filterbank instead of linear bands.
            n_mels: Number of mel bands (only when use_mel=True).
        """
        self.sample_rate = sample_rate
        self.block_size = block_size

        # Backward compat: smooth_tau overrides attack/release
        if smooth_tau is not None:
            self.attack_tau = smooth_tau * 0.1
            self.release_tau = smooth_tau * 1.6
        else:
            self.attack_tau = attack_tau
            self.release_tau = release_tau

        self.band_decay = band_decay
        self.use_mel = use_mel

        # Multi-resolution FFT sizes
        self._fft_bass = 4096    # ~10.8 Hz resolution for bass
        self._fft_mid = 1024     # ~43 Hz resolution for mids
        self._fft_high = 512     # ~86 Hz resolution for highs (fast transient)

        # Ring buffer for accumulating samples (needs 4096 for bass FFT)
        self._ring_buf = np.zeros(self._fft_bass, dtype=np.float64)
        self._ring_pos = 0       # how many total samples written (wraps)
        self._ring_filled = 0    # how many valid samples in buffer

        # Windows for each FFT size (precomputed)
        self._win_bass = np.hanning(self._fft_bass)
        self._win_mid = np.hanning(self._fft_mid)
        self._win_high = np.hanning(self._fft_high)

        # Band masks for each FFT resolution
        self._bass_band_masks = self._make_band_masks(self._fft_bass, [0, 1])  # sub_bass, bass
        self._mid_band_masks = self._make_band_masks(self._fft_mid, [2, 3, 4])  # low_mid, mid, upper_mid
        self._high_band_masks = self._make_band_masks(self._fft_high, [5, 6])  # presence, brilliance

        # Kick detection: spectral flux in 30-120Hz via 4096-point FFT.
        # High resolution (~10.8Hz) isolates kick fundamentals from bass guitar/synths.
        bass_freqs = np.fft.rfftfreq(self._fft_bass, 1.0 / sample_rate)
        self._kick_mask = (bass_freqs >= 30) & (bass_freqs <= 120)
        self._prev_kick_spectrum: np.ndarray | None = None

        # State: smoothed values
        self._smoothed_bands = np.zeros(7, dtype=np.float64)
        self._smoothed_centroid = 0.5
        self._smoothed_flux = 0.0
        self._smoothed_flatness = 0.0
        self._smoothed_rolloff = 0.5
        self._smoothed_bandwidth = 0.0
        self._smoothed_zcr = 0.0
        self._smoothed_rms = 0.0

        self._prev_spectrum = None
        self._last_time = None

        # Adaptive min/max normalization (tracks current dynamic context)
        self._band_running_min = np.full(7, 0.0, dtype=np.float64)
        self._band_running_max = np.full(7, 0.01, dtype=np.float64)
        self._norm_decay = 0.005  # slow adaptation to current level

        # Per-feature adaptive min/max (same approach as band energies)
        self._feat_min = np.zeros(6, dtype=np.float64)   # centroid, flux, flat, roll, bw, zcr
        self._feat_max = np.full(6, 0.01, dtype=np.float64)

        # Auto-gain RMS
        self._rms_running_min = 0.0
        self._rms_running_max = 0.01

        # Kick flux running max for normalization
        self._kick_flux_max = 0.01

        # Mel filterbank (optional)
        if use_mel:
            self._mel_filterbank = _build_mel_filterbank(n_mels, self._fft_mid, sample_rate)
            self._n_mel_bands = n_mels
        else:
            self._mel_filterbank = None

    def _make_band_masks(self, fft_size: int, band_indices: list[int]) -> list[tuple[int, np.ndarray]]:
        """Create (band_index, mask) pairs for specific bands at given FFT resolution."""
        freqs = np.fft.rfftfreq(fft_size, 1.0 / self.sample_rate)
        masks = []
        for i in band_indices:
            mask = (freqs >= BAND_EDGES[i]) & (freqs < BAND_EDGES[i + 1])
            masks.append((i, mask))
        return masks

    def _asymmetric_smooth(self, current: float, target: float, dt: float) -> float:
        """Asymmetric EMA: fast attack, slow release."""
        if target > current:
            tau = self.attack_tau
        else:
            tau = self.release_tau
        alpha = 1.0 - math.exp(-dt / max(tau, 1e-6))
        return current + alpha * (target - current)

    def _asymmetric_smooth_vec(self, current: np.ndarray, target: np.ndarray, dt: float) -> np.ndarray:
        """Vectorized asymmetric EMA for band arrays."""
        attack_alpha = 1.0 - math.exp(-dt / max(self.attack_tau, 1e-6))
        release_alpha = 1.0 - math.exp(-dt / max(self.release_tau, 1e-6))
        alpha = np.where(target > current, attack_alpha, release_alpha)
        return current + alpha * (target - current)

    def _adaptive_normalize(self, raw: np.ndarray) -> np.ndarray:
        """Adaptive min/max normalization tracking current dynamic context."""
        # Min rises slowly toward signal, max decays slowly toward signal
        self._band_running_min += self._norm_decay * (raw - self._band_running_min)
        self._band_running_max += self._norm_decay * (raw - self._band_running_max)

        # Ensure max >= raw (instant jump up), min <= raw (instant jump down)
        self._band_running_max = np.maximum(self._band_running_max, raw)
        self._band_running_min = np.minimum(self._band_running_min, raw)

        # Floor to prevent division by zero
        span = np.maximum(self._band_running_max - self._band_running_min, 0.01)
        return np.clip((raw - self._band_running_min) / span, 0, 1)

    def _adaptive_normalize_scalar(self, raw: float, running_min: float, running_max: float) -> tuple[float, float, float]:
        """Scalar adaptive min/max for RMS etc. Returns (normalized, new_min, new_max)."""
        new_min = running_min + self._norm_decay * (raw - running_min)
        new_max = running_max + self._norm_decay * (raw - running_max)
        new_max = max(new_max, raw)
        new_min = min(new_min, raw)
        span = max(new_max - new_min, 0.01)
        return min(max((raw - new_min) / span, 0.0), 1.0), new_min, new_max

    def _linear_to_db_norm(self, linear_rms: float) -> float:
        """Convert linear RMS to dB-normalized [0, 1] over 80dB range."""
        if linear_rms < 1e-10:
            return 0.0
        db = 20.0 * math.log10(linear_rms)
        return max(0.0, min(1.0, (db - DB_FLOOR) / DB_RANGE))

    def _push_ring(self, samples: np.ndarray):
        """Push samples into the ring buffer."""
        n = len(samples)
        buf_len = len(self._ring_buf)

        if n >= buf_len:
            # More samples than buffer — just take the last buf_len
            self._ring_buf[:] = samples[-buf_len:]
            self._ring_pos = buf_len
            self._ring_filled = buf_len
        else:
            pos = self._ring_pos % buf_len
            end = pos + n
            if end <= buf_len:
                self._ring_buf[pos:end] = samples
            else:
                first = buf_len - pos
                self._ring_buf[pos:] = samples[:first]
                self._ring_buf[:n - first] = samples[first:]
            self._ring_pos += n
            self._ring_filled = min(self._ring_filled + n, buf_len)

    def _get_ring_tail(self, n: int) -> np.ndarray:
        """Get the last n samples from the ring buffer (zero-padded if not enough)."""
        buf_len = len(self._ring_buf)
        available = min(self._ring_filled, n)
        if available == 0:
            return np.zeros(n, dtype=np.float64)

        pos = self._ring_pos % buf_len
        result = np.zeros(n, dtype=np.float64)

        # Copy from ring buffer tail
        start = (pos - available) % buf_len
        if start + available <= buf_len:
            result[n - available:] = self._ring_buf[start:start + available]
        else:
            first = buf_len - start
            result[n - available:n - available + first] = self._ring_buf[start:]
            result[n - available + first:] = self._ring_buf[:available - first]

        return result

    def analyze(self, samples: np.ndarray, timestamp: float | None = None) -> AudioFrame:
        """Analyze a block of mono float32 samples, return AudioFrame.

        Args:
            samples: Mono float32 audio samples.
            timestamp: Optional explicit timestamp (for offline processing).
                       If None, uses wall-clock time.
        """
        if timestamp is not None:
            now = timestamp
        else:
            now = time.time()

        if self._last_time is None:
            dt = self.block_size / self.sample_rate
        else:
            dt = max(now - self._last_time, 0.0)
        self._last_time = now

        # Ensure mono
        if samples.ndim > 1:
            samples = samples.mean(axis=1)

        # Convert to float64 for computation
        samples = samples.astype(np.float64)

        # Push into ring buffer for multi-resolution FFT
        self._push_ring(samples)

        # RMS and peak (on raw input block)
        rms_linear = float(np.sqrt(np.mean(samples ** 2)))
        peak = float(np.max(np.abs(samples)))

        # Zero Crossing Rate (computed on raw samples before windowing)
        if len(samples) > 1:
            sign_changes = np.sum(np.abs(np.diff(np.sign(samples))) > 0)
            zcr_raw = float(sign_changes) / len(samples)
        else:
            zcr_raw = 0.0

        # Noise gate
        if rms_linear < 0.002:
            # Decay toward zero using release tau
            release_alpha = 1.0 - math.exp(-dt / max(self.release_tau, 1e-6))
            self._smoothed_bands *= (1.0 - release_alpha)
            self._smoothed_centroid += release_alpha * (0.5 - self._smoothed_centroid)
            self._smoothed_flux *= (1.0 - release_alpha)
            self._smoothed_flatness *= (1.0 - release_alpha)
            self._smoothed_rolloff += release_alpha * (0.5 - self._smoothed_rolloff)
            self._smoothed_bandwidth *= (1.0 - release_alpha)
            self._smoothed_zcr *= (1.0 - release_alpha)
            self._smoothed_rms *= (1.0 - release_alpha)
            return AudioFrame(
                timestamp=now,
                rms=float(max(self._smoothed_rms, 0.0)),
                peak=0.0,
                energy_bands=self._smoothed_bands.clip(0, 1).tolist(),
                spectral_centroid=float(self._smoothed_centroid),
                spectral_flux=float(max(self._smoothed_flux, 0.0)),
                spectral_flatness=float(max(self._smoothed_flatness, 0.0)),
                spectral_rolloff=float(self._smoothed_rolloff),
                spectral_bandwidth=float(max(self._smoothed_bandwidth, 0.0)),
                zero_crossing_rate=float(max(self._smoothed_zcr, 0.0)),
            )

        # --- Multi-resolution FFT ---
        # Bass bands: 4096-point FFT (10.8 Hz resolution)
        bass_samples = self._get_ring_tail(self._fft_bass)
        bass_windowed = bass_samples * self._win_bass
        bass_spectrum = np.abs(np.fft.rfft(bass_windowed))
        bass_power = bass_spectrum ** 2

        # Mid bands: 1024-point FFT (43 Hz resolution)
        mid_samples = self._get_ring_tail(self._fft_mid)
        mid_windowed = mid_samples * self._win_mid
        mid_spectrum = np.abs(np.fft.rfft(mid_windowed))
        mid_power = mid_spectrum ** 2

        # High bands: 512-point FFT (86 Hz resolution)
        high_samples = self._get_ring_tail(self._fft_high)
        high_windowed = high_samples * self._win_high
        high_spectrum = np.abs(np.fft.rfft(high_windowed))
        high_power = high_spectrum ** 2

        # --- Band energy (linear for bass/mid, dB for high) ---
        raw_bands = np.zeros(7, dtype=np.float64)

        # Bass bands: linear RMS — dB compresses kick dynamics too much
        for band_idx, mask in self._bass_band_masks:
            if mask.any():
                raw_bands[band_idx] = float(np.sqrt(np.mean(bass_power[mask])))

        for band_idx, mask in self._mid_band_masks:
            if mask.any():
                raw_bands[band_idx] = float(np.sqrt(np.mean(mid_power[mask])))

        for band_idx, mask in self._high_band_masks:
            if mask.any():
                band_rms = float(np.sqrt(np.mean(high_power[mask])))
                raw_bands[band_idx] = self._linear_to_db_norm(band_rms)

        # Kick detection: spectral flux in 30-120Hz band (4096-point FFT).
        # Half-wave rectified spectral difference — only increases count.
        # This directly measures sudden spectral change (kick transients) while
        # ignoring sustained bass energy (bass guitar, synth pads).
        kick_spectrum = bass_spectrum[self._kick_mask]
        if self._prev_kick_spectrum is not None:
            diff = kick_spectrum - self._prev_kick_spectrum
            kick_flux = float(np.sum(np.maximum(diff, 0)))
        else:
            kick_flux = 0.0
        self._prev_kick_spectrum = kick_spectrum.copy()
        # Running max normalization (slow decay)
        self._kick_flux_max = max(self._kick_flux_max * self.band_decay, kick_flux, 0.01)
        raw_bass = kick_flux / self._kick_flux_max

        # Adaptive min/max normalization
        norm_bands = self._adaptive_normalize(raw_bands)

        # Asymmetric smoothing (fast attack, slow release)
        self._smoothed_bands = self._asymmetric_smooth_vec(self._smoothed_bands, norm_bands, dt)
        energy_bands = self._smoothed_bands.clip(0, 1).tolist()

        # --- Spectral features (from mid-resolution spectrum, widest usable range) ---
        mid_freqs = np.fft.rfftfreq(self._fft_mid, 1.0 / self.sample_rate)
        total_energy = mid_spectrum.sum()

        # --- Compute raw spectral features ---
        # Spectral centroid
        if total_energy > 1e-10:
            centroid_hz = float(np.sum(mid_freqs * mid_spectrum) / total_energy)
            centroid_raw = float(np.clip((centroid_hz - 200) / (8000 - 200), 0, 1))
        else:
            centroid_raw = 0.5

        # Spectral flux (half-wave rectified, normalized by spectral energy)
        if self._prev_spectrum is not None:
            diff = mid_spectrum - self._prev_spectrum
            flux_sum = float(np.sum(np.maximum(diff, 0)))
            flux_raw = flux_sum / (total_energy + 1e-10)
        else:
            flux_raw = 0.0
        self._prev_spectrum = mid_spectrum.copy()

        # Spectral flatness (geometric mean / arithmetic mean of power)
        mid_power_pos = mid_power[1:]  # skip DC
        if mid_power_pos.sum() > 1e-20:
            log_mean = np.mean(np.log(mid_power_pos + 1e-20))
            geo_mean = np.exp(log_mean)
            arith_mean = np.mean(mid_power_pos)
            flatness_raw = float(np.clip(geo_mean / (arith_mean + 1e-20), 0, 1))
        else:
            flatness_raw = 0.0

        # Spectral rolloff (frequency below which 85% of energy resides)
        if total_energy > 1e-10:
            cumsum = np.cumsum(mid_spectrum)
            rolloff_idx = np.searchsorted(cumsum, 0.85 * total_energy)
            rolloff_hz = mid_freqs[min(rolloff_idx, len(mid_freqs) - 1)]
            rolloff_raw = float(np.clip((rolloff_hz - 200) / (12000 - 200), 0, 1))
        else:
            rolloff_raw = 0.5

        # Spectral bandwidth (std dev of frequency around centroid)
        if total_energy > 1e-10:
            centroid_val = np.sum(mid_freqs * mid_spectrum) / total_energy
            deviation = mid_freqs - centroid_val
            bandwidth_hz = float(np.sqrt(np.sum(mid_spectrum * deviation ** 2) / total_energy))
            bandwidth_raw = float(np.clip(bandwidth_hz / 5000.0, 0, 1))
        else:
            bandwidth_raw = 0.0

        # ZCR (typical range 0-0.5 for audio)
        zcr_raw_norm = float(np.clip(zcr_raw / 0.5, 0, 1))

        # --- Adaptive min/max normalization for all 6 features ---
        raw_feats = np.array([centroid_raw, flux_raw, flatness_raw,
                              rolloff_raw, bandwidth_raw, zcr_raw_norm])
        self._feat_min += self._norm_decay * (raw_feats - self._feat_min)
        self._feat_max += self._norm_decay * (raw_feats - self._feat_max)
        self._feat_max = np.maximum(self._feat_max, raw_feats)
        self._feat_min = np.minimum(self._feat_min, raw_feats)
        feat_span = np.maximum(self._feat_max - self._feat_min, 0.001)
        norm_feats = np.clip((raw_feats - self._feat_min) / feat_span, 0, 1)

        # --- Asymmetric smoothing ---
        self._smoothed_centroid = self._asymmetric_smooth(self._smoothed_centroid, float(norm_feats[0]), dt)
        self._smoothed_flux = self._asymmetric_smooth(self._smoothed_flux, float(norm_feats[1]), dt)
        self._smoothed_flatness = self._asymmetric_smooth(self._smoothed_flatness, float(norm_feats[2]), dt)
        self._smoothed_rolloff = self._asymmetric_smooth(self._smoothed_rolloff, float(norm_feats[3]), dt)
        self._smoothed_bandwidth = self._asymmetric_smooth(self._smoothed_bandwidth, float(norm_feats[4]), dt)
        self._smoothed_zcr = self._asymmetric_smooth(self._smoothed_zcr, float(norm_feats[5]), dt)

        # RMS: dB scale + adaptive normalization
        rms_db = self._linear_to_db_norm(rms_linear)
        rms_norm, self._rms_running_min, self._rms_running_max = \
            self._adaptive_normalize_scalar(rms_db, self._rms_running_min, self._rms_running_max)
        self._smoothed_rms = self._asymmetric_smooth(self._smoothed_rms, rms_norm, dt)

        return AudioFrame(
            timestamp=now,
            rms=float(np.clip(self._smoothed_rms, 0, 1)),
            peak=min(peak, 1.0),
            energy_bands=energy_bands,
            raw_bass=raw_bass,
            spectral_centroid=float(np.clip(self._smoothed_centroid, 0, 1)),
            spectral_flux=float(np.clip(self._smoothed_flux, 0, 1)),
            spectral_flatness=float(np.clip(self._smoothed_flatness, 0, 1)),
            spectral_rolloff=float(np.clip(self._smoothed_rolloff, 0, 1)),
            spectral_bandwidth=float(np.clip(self._smoothed_bandwidth, 0, 1)),
            zero_crossing_rate=float(np.clip(self._smoothed_zcr, 0, 1)),
        )
