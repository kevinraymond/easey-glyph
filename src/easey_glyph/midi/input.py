"""MIDI input handler — background thread with queue-based CC polling."""

from __future__ import annotations

import threading
from queue import SimpleQueue

try:
    import mido
    HAS_MIDI = True
except ImportError:
    HAS_MIDI = False


class MIDIInput:
    """Background MIDI listener that queues CC and Note messages for the render loop."""

    def __init__(self, port_name: str | None = None):
        self.port_name = port_name
        self._port = None
        # (msg_type, number, channel, value_0_127)
        # msg_type: "cc" or "note"
        self._queue: SimpleQueue[tuple[str, int, int, int]] = SimpleQueue()
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> bool:
        """Open MIDI port and start listener thread. Returns True on success."""
        if not HAS_MIDI:
            print("MIDI: mido/python-rtmidi not installed (pip install mido python-rtmidi)")
            return False

        try:
            if self.port_name:
                self._port = mido.open_input(self.port_name, callback=self._on_message)
            else:
                # Auto-select first available port
                ports = mido.get_input_names()
                if not ports:
                    print("MIDI: no input ports found")
                    return False
                self.port_name = ports[0]
                self._port = mido.open_input(self.port_name, callback=self._on_message)

            self._running = True
            print(f"MIDI: opened '{self.port_name}'")
            return True
        except Exception as e:
            print(f"MIDI: failed to open port: {e}")
            self._port = None
            return False

    def stop(self):
        """Stop listener and close port."""
        self._running = False
        if self._port is not None:
            try:
                self._port.close()
            except Exception:
                pass
            self._port = None

    def poll_messages(self) -> list[tuple[str, int, int, int]]:
        """Non-blocking drain of queued messages.

        Returns list of (msg_type, number, channel, value_0_127).
        msg_type is "cc" or "note".
        """
        msgs = []
        while not self._queue.empty():
            try:
                msgs.append(self._queue.get_nowait())
            except Exception:
                break
        return msgs

    @property
    def connected(self) -> bool:
        return self._port is not None and self._running

    def _on_message(self, msg):
        """Callback from mido — runs in mido's listener thread."""
        if not self._running:
            return
        if msg.type == "control_change":
            self._queue.put(("cc", msg.control, msg.channel, msg.value))
        elif msg.type == "note_on":
            self._queue.put(("note", msg.note, msg.channel, msg.velocity))
        elif msg.type == "note_off":
            self._queue.put(("note", msg.note, msg.channel, 0))

    @staticmethod
    def list_ports() -> tuple[list[str], str | None]:
        """List available MIDI input port names.

        Returns (ports, error) — error is None on success, a user-facing
        message string when ports can't be listed.
        """
        if not HAS_MIDI:
            return [], "mido/python-rtmidi not installed (uv sync --extra midi)"
        try:
            return mido.get_input_names(), None
        except Exception as e:
            return [], f"Failed to list MIDI ports: {e}"
