"""ServerState: mutable state shared between WebSocket handlers and render loop."""

from __future__ import annotations

import math
import os
import random
import time
from dataclasses import dataclass, replace

import torch

from easey_glyph.audio.capture import AudioCapture
from easey_glyph.audio.frame import AudioFrame
from easey_glyph.midi.input import MIDIInput
from easey_glyph.midi.mapping import MIDIMapping, load_midi_config, save_midi_config
from easey_glyph.output import HAS_NDI, HAS_SYPHON, HAS_SPOUT, PNGRecorder, VideoOutputSender
from easey_glyph.render.pipeline import EnhancementParams, GridPool

# Irrational-ratio oscillators for idle mode — effectively never repeats.
# (base_freq, secondary_freq, center, amplitude_range, drift_freq)
_TAU = 2.0 * math.pi
_IDLE_OSCILLATORS = [
    # bass:      slow, wide range
    (0.0732, 0.1291, 0.40, 0.30, 0.017),
    # mid:       moderate
    (0.1173, 0.1931, 0.35, 0.25, 0.023),
    # treble:    faster, narrower
    (0.1618, 0.2897, 0.25, 0.20, 0.031),
    # rms:       slow drift
    (0.0891, 0.1517, 0.35, 0.30, 0.019),
    # onset:     fast pulses (sparse, boosted on beat)
    (0.2314, 0.3793, 0.12, 0.10, 0.041),
    # centroid:  very slow
    (0.0523, 0.1137, 0.50, 0.15, 0.013),
    # flux:      moderate-fast
    (0.1937, 0.3109, 0.20, 0.18, 0.037),
    # flatness:  moderate
    (0.1357, 0.2273, 0.135, 0.115, 0.029),
    # rolloff:   slow
    (0.0673, 0.1709, 0.525, 0.175, 0.021),
    # bandwidth: moderate
    (0.1091, 0.1873, 0.35, 0.20, 0.025),
    # zcr:       moderate-fast
    (0.1491, 0.2713, 0.20, 0.15, 0.033),
]


def _idle_feature(t: float, idx: int) -> float:
    """Dual-frequency oscillator with amplitude drift for organic idle features."""
    base_f, sec_f, center, amp_range, drift_f = _IDLE_OSCILLATORS[idx]
    val = 0.65 * math.sin(t * base_f * _TAU) + 0.35 * math.sin(t * sec_f * _TAU + 1.37)
    amp_drift = 0.7 + 0.3 * math.sin(t * drift_f * _TAU)
    result = center + val * amp_range * amp_drift
    return max(0.0, min(1.0, result))


@dataclass
class AudioMapping:
    """Maps an audio feature to an effect parameter."""
    source: str = "none"
    min_val: float = 0.0
    max_val: float = 1.0
    invert: bool = False
    curve: str = "linear"
    lo_thresh: float = 0.0   # input floor (0-1)
    hi_thresh: float = 1.0   # input ceiling (0-1)


@dataclass
class PresetSnapshot:
    """Resolved preset state for A/B comparison."""
    enh: EnhancementParams
    mappings: dict  # {key: AudioMapping}
    superres_enabled: bool = False
    edge_mode_selected: str = "off"


class ServerState:
    """Single mutable state object for the live server."""

    def __init__(self, model, model_cfg, device, steps=8, preview_size=512, pool_size=64):
        self.model = model
        self.model_cfg = model_cfg
        self.device = device
        self.preview_base = preview_size  # base dimension (longest side)
        self.preview_size: tuple[int, int] = (preview_size, preview_size)

        # Enhancement params (current slider values)
        self.enh = EnhancementParams()

        # Audio-to-effect mappings {effect_key: AudioMapping}
        self.mappings: dict[str, AudioMapping] = {}

        # Grid pool
        self.grid_pool = GridPool(model, model_cfg, device, steps=steps,
                                   pool_size=pool_size, low_water=max(8, pool_size // 4))

        # Current display grids for SLERP
        self.current_grid: torch.Tensor | None = None
        self.prev_grid: torch.Tensor | None = None

        # Generation mode: "pool" | "realtime"
        self.gen_mode: str = "pool"

        # Feedback loop: 1.0 = off (pure noise), <1.0 = evolve current grid via img2img
        self.feedback_strength: float = 1.0

        # Grid persistence: 0 = off (new grids fully replace), >0 = blend old into new via slerp
        self.persistence: float = 0.0

        # Audio mode: "idle" | "file" | "live"
        self.mode: str = "idle"

        # Grid morph mode: "frozen" | "ambient" | "beat"
        self.morph_mode: str = "ambient"

        # Live audio capture
        self.capture: AudioCapture | None = None

        # File mode: pre-analyzed frames
        self.file_frames: list[AudioFrame] = []
        self.file_analysis_fps: float = 43.0  # sr/block_size
        self.audio_filename: str | None = None

        # Current playback position (from browser)
        self.audio_time: float = 0.0
        self._last_file_idx: int = -1

        # Beat edge detection
        self.last_beat_seen: bool = False

        # Onset threshold for live capture
        self.onset_threshold: float = 1.4

        # Idle mode timing
        self._idle_start: float = 0.0
        self._idle_cycle_duration: float = 5.0
        self._idle_cycle_start: float = 0.0
        self._idle_cycle_count: int = 0
        self._idle_last_beat_cycle: int = -1
        self._idle_cached: tuple[float, AudioFrame] | None = None

        # Ambient grid history (pool depletion resilience)
        self._idle_grid_history: list = []
        self._idle_history_max: int = 8

        # Alpha preview (WebP with transparency vs JPEG)
        self.alpha_preview: bool = False

        # Super-resolution CNN
        self.superres_model = None      # GlyphSuperRes or None
        self.superres_enabled: bool = False
        self.edge_mode_selected: str = "off"

        # CoreML backends (Mac, optional)
        self.coreml_model = None        # CoreMLFlowModel or None
        self.coreml_superres = None     # CoreMLSuperRes or None

        # Img2img state
        self.img2img_source: torch.Tensor | None = None  # [16, 32, 32] encoded grid
        self.img2img_strength: float = 0.5
        self.img2img_enabled: bool = False

        # Source input mode: "none" | "image" | "camera"
        self.source_mode: str = "none"
        self._camera_encode_future = None  # track in-flight glyph encoding

        # Glyph resources (loaded once for encoding)
        self._glyph_masks = None
        self._glyph_embeddings = None

        # MIDI controller
        self.midi_input: MIDIInput | None = None
        self.midi_mappings: dict[str, MIDIMapping] = {}     # param_key -> mapping
        self.midi_triggers: dict[str, MIDIMapping] = {}     # action_name -> mapping
        self.midi_cc_values: dict[tuple[str, int, int], float] = {}  # (type, num, chan) -> 0.0-1.0
        self.midi_learn_target: str | None = None           # key currently learning
        self.midi_learn_is_trigger: bool = False             # learning a trigger vs param
        self.midi_last_cc: tuple[str, int, int, int] | None = None  # (type, num, chan, val)
        self.midi_config_path = os.path.expanduser("~/.easey-glyph-midi.json")

        # A/B preset comparison
        self.ab_enabled: bool = False
        self.ab_preset_a: PresetSnapshot | None = None
        self.ab_preset_b: PresetSnapshot | None = None
        self.ab_preset_a_name: str = ""
        self.ab_preset_b_name: str = ""

        # Render FPS (smoothed, updated by send loop)
        self.fps_smooth: float = 60.0

        # Preview throttle: 1.0 = every frame, 0.5 = half, 0.0 = off
        self.preview_fraction: float = 1.0

        # High-resolution output
        self.target_fps: float = 60.0
        self.output_enabled: bool = False
        self.output_base: int = 1920
        self.output_size: tuple[int, int] = (1920, 1080)
        self.output_senders: list[VideoOutputSender] = []
        self.png_recorder: PNGRecorder | None = None

    def ensure_glyph_resources(self):
        """Lazy-load glyph atlas and embeddings (only needed for img2img)."""
        if self._glyph_masks is None:
            from easey_glyph.glyph.rasterizer import build_glyph_atlas
            from easey_glyph.glyph.embedding import compute_embeddings
            self._glyph_masks = build_glyph_atlas()
            self._glyph_embeddings = compute_embeddings()
        return self._glyph_masks, self._glyph_embeddings

    def set_source_mode(self, mode: str):
        """Switch source input mode."""
        self.source_mode = mode
        if mode == "none":
            self.img2img_source = None
            self.img2img_enabled = False
            if self.gen_mode == "pool":
                self.grid_pool.resume()
        elif mode in ("camera", "image"):
            self.img2img_enabled = True
            self.grid_pool.pause()

    def start(self):
        """Start grid pool background thread (non-blocking)."""
        now = time.monotonic()
        self._idle_start = now
        self._idle_cycle_start = now
        self._idle_cycle_count = 0
        self._idle_new_cycle()
        self.grid_pool.start()

    def try_seed_grids(self):
        """Seed initial prev/current grids from pool once available."""
        if self.current_grid is not None:
            return True  # already seeded
        g = self.grid_pool.next_grid()
        if g is not None:
            self.prev_grid = g
            self.current_grid = g
        g2 = self.grid_pool.next_grid()
        if g2 is not None:
            self.current_grid = g2
        return self.current_grid is not None

    def get_audio_frame(self) -> AudioFrame:
        """Get the current audio frame based on mode."""
        if self.mode == "live" and self.capture is not None:
            return self.capture.latest_frame
        elif self.mode == "file" and self.file_frames:
            idx = int(self.audio_time * self.file_analysis_fps)
            idx = max(0, min(idx, len(self.file_frames) - 1))
            frame = self.file_frames[idx]

            # Scan for beats that fell between render frames (analysis fps > render fps)
            if self._last_file_idx >= 0 and idx > self._last_file_idx + 1:
                scan_start = max(self._last_file_idx + 1, idx - 10)
                for i in range(scan_start, idx):
                    if self.file_frames[i].is_beat:
                        frame = replace(frame, is_beat=True, beat_phase=0.0,
                                        onset_strength=max(frame.onset_strength,
                                                           self.file_frames[i].onset_strength))
                        break
            self._last_file_idx = idx
            return frame
        return self._idle_frame()

    def _idle_new_cycle(self):
        """Pick a random duration for the next idle cycle (3.5-9.0s, biased 5-7)."""
        r = random.random() ** 0.7
        self._idle_cycle_duration = 3.5 + r * 5.5
        self._idle_cycle_start = time.monotonic()
        self._idle_cycle_count += 1

    def _idle_frame(self) -> AudioFrame:
        """Generate a synthetic ambient frame for idle/intermission mode.

        Variable-length cycles with smoothstep crossfades, irrational-ratio
        oscillators for rich feature variation, and per-frame caching to fix
        the double-call bug (get_audio_frame + morph both call this).
        """
        now = time.monotonic()

        # Cache: return same frame if called again within 1ms (fixes double-call bug)
        if self._idle_cached is not None:
            cached_t, cached_frame = self._idle_cached
            if now - cached_t < 0.001:
                return cached_frame

        # Variable cycle timing
        elapsed = now - self._idle_cycle_start
        if elapsed >= self._idle_cycle_duration:
            self._idle_new_cycle()
            elapsed = now - self._idle_cycle_start

        # Smoothstep phase (organic crossfade — dwells on endpoints)
        raw = max(0.0, min(1.0, elapsed / self._idle_cycle_duration))
        phase = raw * raw * (3.0 - 2.0 * raw)

        # Beat edge detection
        current_cycle = self._idle_cycle_count
        is_beat = current_cycle != self._idle_last_beat_cycle
        if is_beat:
            self._idle_last_beat_cycle = current_cycle

        # Total time for oscillators (not per-cycle)
        t = now - self._idle_start

        # Rich features from irrational-ratio oscillators
        bass_val = _idle_feature(t, 0)
        mid_val = _idle_feature(t, 1)
        treble_val = _idle_feature(t, 2)
        rms_val = _idle_feature(t, 3)
        onset_val = _idle_feature(t, 4)
        centroid_val = _idle_feature(t, 5)
        flux_val = _idle_feature(t, 6)
        flatness_val = _idle_feature(t, 7)
        rolloff_val = _idle_feature(t, 8)
        bandwidth_val = _idle_feature(t, 9)
        zcr_val = _idle_feature(t, 10)

        # Onset: sparse pulses (threshold > 0.2), boosted on beat
        if is_beat:
            onset_val = 0.6
        elif onset_val < 0.2:
            onset_val = 0.0

        frame = AudioFrame(
            timestamp=t,
            rms=rms_val,
            energy_bands=[
                bass_val * 1.2,   # sub_bass
                bass_val,         # bass
                mid_val * 0.8,    # low_mid
                mid_val,          # mid
                mid_val * 0.6,    # upper_mid
                treble_val,       # presence
                treble_val * 0.5, # brilliance
            ],
            raw_bass=bass_val * 1.1,
            beat_phase=phase,
            is_beat=is_beat,
            bpm=round(60.0 / self._idle_cycle_duration, 1),
            onset_strength=onset_val,
            spectral_centroid=centroid_val,
            spectral_flux=flux_val,
            spectral_flatness=flatness_val,
            spectral_rolloff=rolloff_val,
            spectral_bandwidth=bandwidth_val,
            zero_crossing_rate=zcr_val,
        )

        self._idle_cached = (now, frame)
        return frame

    def start_live(self):
        """Switch to live audio capture mode."""
        self.stop_capture()
        self._last_file_idx = -1
        self.capture = AudioCapture()
        if self.capture.start():
            self.capture.beat_detector.set_onset_threshold(self.onset_threshold)
            self.mode = "live"
            print("Live audio capture started")
        else:
            self.capture = None
            print("Failed to start live audio capture")

    def stop_capture(self):
        """Stop any active audio capture."""
        if self.capture is not None:
            self.capture.stop()
            self.capture = None

    def start_output_sender(self, sender_type: str, name: str = "EASEy-GLYPH") -> bool:
        """Start a video output sender by type name."""
        sender = None
        if sender_type == "ndi" and HAS_NDI:
            from easey_glyph.output import NDISender
            sender = NDISender()
        elif sender_type == "syphon" and HAS_SYPHON:
            from easey_glyph.output import SyphonSender
            sender = SyphonSender()
        elif sender_type == "spout" and HAS_SPOUT:
            from easey_glyph.output import SpoutSender
            sender = SpoutSender()

        if sender is None:
            print(f"Output: sender '{sender_type}' not available")
            return False

        w, h = self.output_size
        if sender.start(w, h, self.target_fps, name):
            self.output_senders.append(sender)
            self.output_enabled = True
            return True
        return False

    def stop_output_sender(self, sender_type: str):
        """Stop a specific output sender by type name."""
        remaining = []
        for sender in self.output_senders:
            if sender_type.lower() in sender.name.lower():
                sender.stop()
            else:
                remaining.append(sender)
        self.output_senders = remaining
        if not self.output_senders and (self.png_recorder is None or not self.png_recorder.active):
            self.output_enabled = False

    def restart_output_senders(self):
        """Restart all active output senders at the current output_size."""
        if not self.output_senders:
            return
        types = []
        for sender in self.output_senders:
            n = sender.name.lower()
            if "ndi" in n:
                types.append("ndi")
            elif "syphon" in n:
                types.append("syphon")
            elif "spout" in n:
                types.append("spout")
        for sender in self.output_senders:
            sender.stop()
        self.output_senders.clear()
        for t in types:
            self.start_output_sender(t)

    def stop_output_senders(self):
        """Stop all video output senders."""
        for sender in self.output_senders:
            sender.stop()
        self.output_senders.clear()
        if self.png_recorder is None or not self.png_recorder.active:
            self.output_enabled = False

    def start_recording(self, output_dir: str):
        """Start PNG sequence recording."""
        self.png_recorder = PNGRecorder()
        self.png_recorder.set_output_dir(output_dir)
        w, h = self.output_size
        if self.png_recorder.start(w, h, self.target_fps):
            self.output_enabled = True
        else:
            self.png_recorder = None

    def stop_recording(self):
        """Stop PNG sequence recording."""
        if self.png_recorder is not None:
            self.png_recorder.stop()
            self.png_recorder = None
        if not self.output_senders:
            self.output_enabled = False

    def start_midi(self, port_name: str | None = None):
        """Initialize MIDI input and load saved mappings."""
        self.midi_input = MIDIInput(port_name)
        if self.midi_input.start():
            # Load saved mappings
            params, triggers = load_midi_config(self.midi_config_path)
            self.midi_mappings = params
            self.midi_triggers = triggers
            if params or triggers:
                print(f"MIDI: loaded {len(params)} param + {len(triggers)} trigger mappings")
        else:
            self.midi_input = None

    def stop_midi(self):
        """Stop MIDI input."""
        if self.midi_input is not None:
            self.midi_input.stop()
            self.midi_input = None
        self.midi_cc_values.clear()

    def save_midi(self):
        """Persist current MIDI mappings to disk."""
        save_midi_config(self.midi_config_path, self.midi_mappings, self.midi_triggers)

    def stop(self):
        """Clean shutdown."""
        self.stop_capture()
        self.stop_midi()
        self.stop_output_senders()
        self.stop_recording()
        self.grid_pool.stop()
