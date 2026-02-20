"""Audio-to-effect mapping resolution."""

from .resolve import (
    PARAM_REGISTRY,
    ParamDef,
    apply_curve,
    apply_resolved_to_enh,
    apply_side_effects,
    get_audio_feature,
    get_param_ranges,
    resolve_mappings,
)

__all__ = [
    "PARAM_REGISTRY",
    "ParamDef",
    "apply_curve",
    "apply_resolved_to_enh",
    "apply_side_effects",
    "get_audio_feature",
    "get_param_ranges",
    "resolve_mappings",
]
