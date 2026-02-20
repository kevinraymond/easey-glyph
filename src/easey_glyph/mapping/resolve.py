"""Centralized audio-to-effect mapping resolution.

Single source of truth for:
- Parameter definitions (type, range, defaults, side-effect flag)
- Audio feature extraction from AudioFrame
- Curve functions
- Mapping resolution (audio value → parameter value)
- Application to EnhancementParams (type coercion, side-effect separation)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from easey_glyph.audio.frame import AudioFrame
    from easey_glyph.render.pipeline import EnhancementParams
    from easey_glyph.server.state import AudioMapping


# ---------------------------------------------------------------------------
# Parameter registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParamDef:
    """Definition of a mappable parameter."""
    name: str
    param_type: str = "float"     # "float" | "int" | "bool"
    default: float = 0.0
    min_val: float = 0.0
    max_val: float = 1.0
    side_effect: bool = False     # True = writes to state, not EnhancementParams


PARAM_REGISTRY: dict[str, ParamDef] = {
    "fg_brightness":     ParamDef("fg_brightness", "float", 1.0, 0.2, 2.0),
    "opacity":           ParamDef("opacity", "float", 1.0, 0.0, 1.0),
    "saturation":        ParamDef("saturation", "float", 1.0, 0.0, 3.0),
    "contrast":          ParamDef("contrast", "float", 1.0, 0.0, 3.0),
    "gamma":             ParamDef("gamma", "float", 1.0, 0.2, 3.0),
    "alpha_curve":       ParamDef("alpha_curve", "float", 1.0, 0.1, 5.0),
    "sharpen":           ParamDef("sharpen", "float", 0.0, 0.0, 200.0),
    "grain":             ParamDef("grain", "float", 0.0, 0.0, 20.0),
    "render_scale":      ParamDef("render_scale", "int", 4, 1, 16),
    "posterize":         ParamDef("posterize", "int", 0, 0, 8),
    "scanlines":         ParamDef("scanlines", "int", 0, 0, 8),
    "edge_enhance":      ParamDef("edge_enhance", "bool", 0.0, 0.0, 1.0),
    "edge_intensity":    ParamDef("edge_intensity", "float", 1.0, 0.0, 1.0),
    "superres":          ParamDef("superres", "bool", 0.0, 0.0, 1.0, side_effect=True),
    "cfg_scale":         ParamDef("cfg_scale", "float", 0.0, 0.0, 5.0, side_effect=True),
    "feedback_strength": ParamDef("feedback_strength", "float", 1.0, 0.05, 1.0, side_effect=True),
    "persistence":       ParamDef("persistence", "float", 0.0, 0.0, 0.95, side_effect=True),
}


def get_param_ranges() -> dict[str, tuple[float, float]]:
    """Derive MIDI-compatible param ranges from registry."""
    return {k: (p.min_val, p.max_val) for k, p in PARAM_REGISTRY.items()}


# ---------------------------------------------------------------------------
# Audio feature extraction
# ---------------------------------------------------------------------------

def get_audio_feature(frame: AudioFrame, source: str) -> float:
    """Extract a normalized 0-1 audio feature from an AudioFrame."""
    if source == "bass": return frame.bass
    if source == "mid": return frame.mid
    if source == "treble": return frame.treble
    if source == "rms": return frame.rms
    if source == "beat_phase": return frame.beat_phase
    if source == "onset_strength": return frame.onset_strength
    if source == "spectral_centroid": return frame.spectral_centroid
    if source == "spectral_flux": return frame.spectral_flux
    if source == "spectral_flatness": return frame.spectral_flatness
    if source == "spectral_rolloff": return frame.spectral_rolloff
    if source == "spectral_bandwidth": return frame.spectral_bandwidth
    if source == "zero_crossing_rate": return frame.zero_crossing_rate
    return 0.0


# ---------------------------------------------------------------------------
# Curve functions
# ---------------------------------------------------------------------------

def apply_curve(t: float, curve: str) -> float:
    """Apply a curve function to a normalized 0-1 value."""
    if curve == "ease_in":
        return t * t
    elif curve == "ease_out":
        return 1.0 - (1.0 - t) * (1.0 - t)
    elif curve == "ease_in_out":
        return 3 * t * t - 2 * t * t * t  # smoothstep
    elif curve == "exponential":
        return t * t * t
    elif curve == "logarithmic":
        return math.log(1 + t * 9) / math.log(10)
    elif curve == "threshold":
        return 1.0 if t >= 0.5 else 0.0
    return t  # linear


# ---------------------------------------------------------------------------
# Mapping resolution (pure — no side effects)
# ---------------------------------------------------------------------------

def _threshold(audio_val: float, lo: float, hi: float) -> float:
    """Remap [lo, hi] → [0, 1]."""
    if hi > lo:
        return max(0.0, min(1.0, (audio_val - lo) / (hi - lo)))
    return audio_val


def resolve_mappings(
    mappings: dict[str, AudioMapping],
    audio_frame: AudioFrame,
    midi_vals: dict[str, float] | None = None,
) -> dict[str, float]:
    """Resolve all mappings to final float values. Pure function, no side effects.

    Returns dict[param_key, resolved_value].
    """
    resolved: dict[str, float] = {}
    for key, mapping in mappings.items():
        if mapping.source == "none":
            continue

        audio_val = get_audio_feature(audio_frame, mapping.source)
        audio_val = _threshold(audio_val, mapping.lo_thresh, mapping.hi_thresh)

        if mapping.invert:
            audio_val = 1.0 - audio_val
        audio_val = apply_curve(audio_val, mapping.curve)

        if midi_vals and key in midi_vals:
            # Bipolar: audio modulates around MIDI base value
            base = float(midi_vals[key])
            depth = mapping.max_val - mapping.min_val
            offset = (audio_val - 0.5) * depth
            pdef = PARAM_REGISTRY.get(key)
            if pdef:
                lo, hi = pdef.min_val, pdef.max_val
            else:
                lo, hi = mapping.min_val, mapping.max_val
            resolved[key] = max(lo, min(hi, base + offset))
        else:
            # Absolute: map audio [0,1] → [min_val, max_val]
            resolved[key] = mapping.min_val + (mapping.max_val - mapping.min_val) * audio_val

    return resolved


# ---------------------------------------------------------------------------
# Apply resolved values to EnhancementParams
# ---------------------------------------------------------------------------

def apply_resolved_to_enh(
    enh: EnhancementParams,
    resolved: dict[str, float],
    edge_mode_selected: str = "off",
) -> tuple[EnhancementParams, dict[str, Any]]:
    """Apply resolved values to EnhancementParams. Returns (new_enh, mapped_vals_for_ui).

    Skips side_effect params (those are applied separately via apply_side_effects).
    Handles type coercion per ParamDef.
    """
    result = replace(enh)
    mapped_vals: dict[str, Any] = {}

    for key, val in resolved.items():
        pdef = PARAM_REGISTRY.get(key)
        if pdef is None:
            continue
        if pdef.side_effect:
            # Side effects reported for UI but not applied to enh
            if key == "superres":
                mapped_vals[key] = bool(val > 0.5)
            else:
                mapped_vals[key] = round(val, 2)
            continue

        # Type coercion and special cases
        if key == "edge_enhance":
            if val > 0.5 and edge_mode_selected != "off":
                result.edge_enhance = edge_mode_selected
            else:
                result.edge_enhance = "off"
            mapped_vals[key] = result.edge_enhance != "off"
        elif pdef.param_type == "int":
            v = round(val)
            if key in ("posterize", "scanlines"):
                setattr(result, key, v if v > 0 else None)
            elif key == "render_scale":
                setattr(result, key, max(1, v))
            else:
                setattr(result, key, v)
            mapped_vals[key] = v
        else:
            # float
            setattr(result, key, val)
            mapped_vals[key] = round(val, 2)

    return result, mapped_vals


def apply_side_effects(resolved: dict[str, float], state: Any) -> None:
    """Apply side-effect mappings (cfg_scale, superres, etc.) to state."""
    for key, val in resolved.items():
        pdef = PARAM_REGISTRY.get(key)
        if pdef is None or not pdef.side_effect:
            continue

        if key == "cfg_scale":
            state.grid_pool.update_cfg(cfg_scale=max(0.0, val))
        elif key == "superres":
            state.superres_enabled = bool(val > 0.5)
        elif key == "feedback_strength":
            state.feedback_strength = val
        elif key == "persistence":
            state.persistence = val
