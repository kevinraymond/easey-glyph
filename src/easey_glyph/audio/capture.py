"""Audio capture via sounddevice (PortAudio wrapper).

Polling-based capture: PortAudio buffers audio in C while Python GIL is held.
Decoupled from AppConfig — uses simple args.
"""

import logging
import os
import platform
import subprocess
import threading
import time
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

from .analyzer import AudioAnalyzer
from .beat import BeatDetector
from .frame import AudioFrame

logger = logging.getLogger(__name__)


class AudioCapture:
    """Captures audio input and delivers AudioFrames via callback."""

    def __init__(
        self,
        sample_rate: int = 44100,
        block_size: int = 1024,
        device: Optional[int] = None,
        on_frame: Optional[Callable[[AudioFrame], None]] = None,
        smooth_tau: float = 0.05,
        beat_bpm_smooth: float = 0.7,
    ):
        """
        Args:
            sample_rate: Desired sample rate (auto-adjusted to device native).
            block_size: FFT block size.
            device: Sounddevice device index (None = auto-detect).
            on_frame: Callback receiving each analyzed AudioFrame.
            smooth_tau: Audio analyzer smoothing time constant.
            beat_bpm_smooth: BPM tracking EMA factor.
        """
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.device = device
        self.on_frame = on_frame

        self.analyzer = AudioAnalyzer(sample_rate, block_size, smooth_tau=smooth_tau)
        self.beat_detector = BeatDetector(sample_rate, bpm_smooth=beat_bpm_smooth)

        self._stream: Optional[sd.InputStream] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Latest frame (thread-safe via GIL for single-word writes)
        self.latest_frame: AudioFrame = AudioFrame()

    def _auto_detect_device(self) -> Optional[int]:
        """Find the best input device.

        On macOS: prefer BlackHole (loopback) > default input.
        On Linux: prefer 'pulse' or 'default' (shows in pavucontrol for routing)
                  over raw ALSA hw: devices (which bypass PulseAudio).
        """
        try:
            devices = sd.query_devices()
            num_devices = len(devices)

            if platform.system() == "Darwin":
                # macOS: prefer BlackHole for loopback
                for i in range(num_devices):
                    d = sd.query_devices(i)
                    if d["max_input_channels"] > 0 and "blackhole" in d["name"].lower():
                        logger.info(f"Auto-selected BlackHole device {i}: {d['name']}")
                        return i

            if platform.system() == "Linux":
                # Linux: prefer 'pulse' device (PulseAudio/PipeWire) so it appears
                # in pavucontrol and can be routed to a monitor source for loopback.
                for i in range(num_devices):
                    d = sd.query_devices(i)
                    if d["max_input_channels"] > 0 and d["name"] == "pulse":
                        logger.info(f"Auto-selected PulseAudio device {i}: {d['name']}")
                        print(f"Audio: using PulseAudio device [{i}]")
                        print("  TIP: Open pavucontrol -> Recording tab -> set this app's")
                        print("  input to 'Monitor of <your output>' for desktop audio loopback")
                        return i

            # Fallback: first input device
            for i in range(num_devices):
                d = sd.query_devices(i)
                if d["max_input_channels"] > 0:
                    logger.info(f"Auto-selected input device {i}: {d['name']}")
                    return i
        except Exception:
            pass
        return None

    def _poll_loop(self):
        """Read audio blocks in a dedicated thread."""
        while self._running:
            try:
                data, overflowed = self._stream.read(self.block_size)
                if overflowed:
                    logger.debug("Audio input overflow (buffered)")

                mono = data[:, 0] if data.ndim > 1 else data.flatten()
                frame = self.analyzer.analyze(mono)
                # Pass multi-resolution spectra to beat pipeline
                if self.analyzer.last_bass_spectrum is not None:
                    self.beat_detector.set_spectra(
                        self.analyzer.last_bass_spectrum,
                        self.analyzer.last_mid_spectrum,
                        self.analyzer.last_high_spectrum,
                    )
                frame = self.beat_detector.process(frame)

                self.latest_frame = frame

                if self.on_frame is not None:
                    self.on_frame(frame)

            except sd.PortAudioError as e:
                if not self._running:
                    break
                logger.debug(f"PortAudio error (will retry): {e}")
                time.sleep(0.1)

    def start(self) -> bool:
        """Start the audio input stream. Returns True if successful."""
        device = self.device
        if device is None:
            device = self._auto_detect_device()

        if device is None:
            logger.warning("No audio input device found")
            return False

        # Use device's native sample rate
        device_info = sd.query_devices(device)
        native_sr = int(device_info["default_samplerate"])
        if native_sr != self.sample_rate:
            logger.info(f"Using device native sample rate: {native_sr} Hz")
            self.sample_rate = native_sr
            self.analyzer = AudioAnalyzer(native_sr, self.block_size)
            self.beat_detector = BeatDetector(native_sr)

        try:
            self._stream = sd.InputStream(
                device=device,
                channels=1,
                samplerate=native_sr,
                blocksize=self.block_size,
                dtype="float32",
                latency="high",
            )
            self._stream.start()
            self._running = True
            self._thread = threading.Thread(target=self._poll_loop, daemon=True, name="audio-capture")
            self._thread.start()
            logger.info(f"Audio capture active: device={device_info['name']}, {native_sr} Hz")

            # On Linux, auto-redirect to monitor source for desktop audio loopback
            if platform.system() == "Linux":
                self._redirect_to_monitor()

            return True
        except Exception as e:
            logger.warning(f"Failed to open audio: {e}")
            return False

    def _redirect_to_monitor(self):
        """Redirect our PulseAudio/PipeWire input to the monitor of the active output sink.

        This gives us desktop audio loopback without requiring manual pavucontrol setup.
        """
        try:
            time.sleep(0.2)  # Let PulseAudio register our stream
            pid = str(os.getpid())

            # Find the monitor source of the running/default sink
            result = subprocess.run(
                ["pactl", "list", "short", "sources"],
                capture_output=True, text=True, timeout=5,
            )
            monitor_source = None
            for line in result.stdout.strip().split("\n"):
                parts = line.split("\t")
                if len(parts) >= 2 and ".monitor" in parts[1] and "RUNNING" in line:
                    monitor_source = parts[1]
                    break
            # Fallback: any monitor source
            if not monitor_source:
                for line in result.stdout.strip().split("\n"):
                    parts = line.split("\t")
                    if len(parts) >= 2 and ".monitor" in parts[1]:
                        monitor_source = parts[1]
                        break

            if not monitor_source:
                print("Audio: no monitor source found -- using default input (mic)")
                print("  TIP: Open pavucontrol -> Recording -> set to 'Monitor of <output>'")
                return

            # Find our source-output (recording stream) by PID
            result = subprocess.run(
                ["pactl", "list", "source-outputs"],
                capture_output=True, text=True, timeout=5,
            )
            our_index = None
            current_index = None
            for line in result.stdout.split("\n"):
                line = line.strip()
                if line.startswith("Source Output #"):
                    current_index = line.split("#")[1]
                elif "application.process.id" in line and pid in line:
                    our_index = current_index

            if not our_index:
                print("Audio: couldn't find our PulseAudio stream -- using default input")
                return

            # Move our stream to the monitor source
            subprocess.run(
                ["pactl", "move-source-output", our_index, monitor_source],
                capture_output=True, timeout=5,
            )
            short_name = monitor_source.split(".")[-2] if "." in monitor_source else monitor_source
            print(f"Audio: redirected to monitor of {short_name} (desktop audio loopback)")

        except FileNotFoundError:
            print("Audio: pactl not found -- using default input")
            print("  TIP: Open pavucontrol -> Recording -> set to 'Monitor of <output>'")
        except Exception as e:
            logger.debug(f"Monitor redirect failed: {e}")
            print(f"Audio: monitor redirect failed ({e}) -- using default input")

    def stop(self):
        """Stop the audio stream."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            logger.info("Audio capture stopped")
