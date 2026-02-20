"""MIDI mapping data model and JSON persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class MIDIMapping:
    """Maps a MIDI CC or Note to a parameter or trigger action."""
    cc: int               # CC number or note number (0-127)
    channel: int = 0      # 0 = omni (respond to all channels)
    msg_type: str = "cc"  # "cc" or "note"
    min_val: float = 0.0
    max_val: float = 1.0
    invert: bool = False


def load_midi_config(path: str | Path) -> tuple[dict[str, MIDIMapping], dict[str, MIDIMapping]]:
    """Load MIDI mappings from JSON file.

    Returns (param_mappings, trigger_mappings). Empty dicts if file missing or corrupt.
    """
    path = Path(path)
    if not path.exists():
        return {}, {}

    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}, {}

    params = {}
    for key, m in data.get("params", {}).items():
        params[key] = MIDIMapping(
            cc=int(m["cc"]),
            channel=int(m.get("channel", 0)),
            msg_type=str(m.get("type", "cc")),
            min_val=float(m.get("min", 0.0)),
            max_val=float(m.get("max", 1.0)),
            invert=bool(m.get("invert", False)),
        )

    triggers = {}
    for key, m in data.get("triggers", {}).items():
        triggers[key] = MIDIMapping(
            cc=int(m["cc"]),
            channel=int(m.get("channel", 0)),
            msg_type=str(m.get("type", "cc")),
        )

    return params, triggers


def save_midi_config(
    path: str | Path,
    param_mappings: dict[str, MIDIMapping],
    trigger_mappings: dict[str, MIDIMapping],
) -> None:
    """Save MIDI mappings to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "version": 1,
        "params": {
            key: {
                "cc": m.cc,
                "channel": m.channel,
                "type": m.msg_type,
                "min": m.min_val,
                "max": m.max_val,
                "invert": m.invert,
            }
            for key, m in param_mappings.items()
        },
        "triggers": {
            key: {
                "cc": m.cc,
                "channel": m.channel,
                "type": m.msg_type,
            }
            for key, m in trigger_mappings.items()
        },
    }

    path.write_text(json.dumps(data, indent=2) + "\n")
