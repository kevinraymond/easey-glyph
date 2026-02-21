"""FastAPI app: HTTP routes + WebSocket for live rendering."""

import asyncio
import io
import json
import random
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from PIL import Image

from easey_glyph.audio.features import analyze_file
from easey_glyph.audio.reactive import frame_to_render_params, render_pixel_frame
from easey_glyph.mapping import resolve_mappings, apply_resolved_to_enh, apply_side_effects, get_param_ranges
from easey_glyph.render.pipeline import (
    EDGE_KERNELS,
    INTERP_METHODS,
    apply_grain,
    apply_intermediate_enhancements,
    apply_output_enhancements,
    generate_single_grid,
    generate_single_grid_img2img,
    scale2x,
    slerp,
)
from easey_glyph.output import HAS_NDI, HAS_SYPHON, HAS_SPOUT
from easey_glyph.midi.input import MIDIInput
from easey_glyph.midi.mapping import MIDIMapping
from easey_glyph.server.state import AudioMapping, PresetSnapshot, ServerState

import numpy as np
import torch

STATIC_DIR = Path(__file__).parent.parent / "static"
UPLOAD_DIR = Path("uploads")

app = FastAPI(title="EASEy-GLYPH")

# Global state — set by __main__.py before starting uvicorn
state: ServerState | None = None

# Connection registry for multi-client sync
_connections: dict[WebSocket, asyncio.Queue] = {}
_connections_lock = asyncio.Lock()

# Shared render loop state — one render pass, all clients forward the same frame
_frame_cond: asyncio.Condition | None = None
_latest_frame: bytes | None = None
_latest_meta_json: str = ""
_frame_counter: int = 0

# Active preset name (for syncing between clients)
_active_preset: str = "Default"

# Non-blocking realtime generation — runs Euler solve off the frame loop
_rt_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rt-gen")
_rt_future = None
_pending_grid = None  # Async result held until next beat edge for smooth swap
_pending_is_evolution: bool = False  # Whether pending grid came from feedback evolution
_rt_pending_is_evolution: bool = False  # Whether in-flight generation is evolution

# ---------------------------------------------------------------------------
# Camera pipeline state
# ---------------------------------------------------------------------------
_camera_frames_recv = 0
_camera_frames_encoded = 0
_camera_frames_generated = 0
_camera_gen_future = None
_camera_result_grid = None
_camera_prev_grid = None
_camera_swap_time = 0.0
_CAMERA_CROSSFADE = 0.3  # seconds


def _clear_camera_state():
    """Reset all camera pipeline globals and resume pool if needed."""
    global _camera_frames_recv, _camera_frames_encoded, _camera_frames_generated
    global _camera_gen_future, _camera_result_grid, _camera_prev_grid, _camera_swap_time
    _camera_frames_recv = 0
    _camera_frames_encoded = 0
    _camera_frames_generated = 0
    if _camera_gen_future is not None and not _camera_gen_future.done():
        _camera_gen_future.cancel()
    _camera_gen_future = None
    _camera_result_grid = None
    _camera_prev_grid = None
    _camera_swap_time = 0.0

# ---------------------------------------------------------------------------
# Output thread — decouples output rendering from preview
# ---------------------------------------------------------------------------

class _OutputThread:
    """Dedicated thread for output rendering with single-slot latest-frame buffer."""

    def __init__(self):
        self._slot_lock = threading.Lock()
        self._slot = None  # (intermediate_arr, effective_enh, output_size)
        self._event = threading.Event()
        self._running = False
        self._thread: threading.Thread | None = None
        self.fps = 0.0

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="output-send")
        self._thread.start()

    def stop(self):
        self._running = False
        self._event.set()  # Wake thread so it can exit
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.fps = 0.0

    def submit(self, intermediate_arr, effective_enh, output_size):
        """Drop latest frame data into slot (non-blocking, <1us).

        Args:
            intermediate_arr: [H, W, 4] uint8 RGBA numpy array after intermediate enhancements.
            effective_enh: EnhancementParams for output-phase effects.
            output_size: (width, height) final output resolution.
        """
        with self._slot_lock:
            self._slot = (intermediate_arr, effective_enh, output_size)
        self._event.set()

    def _run(self):
        fps_smooth = 0.0
        last_send = time.monotonic()
        while self._running:
            self._event.wait()
            if not self._running:
                break
            self._event.clear()

            # Grab latest slot
            with self._slot_lock:
                data = self._slot
                self._slot = None
            if data is None:
                continue

            intermediate_arr, effective_enh, output_size = data
            try:
                output_arr = apply_output_enhancements(intermediate_arr, effective_enh, output_size)
                _send_output_frame(output_arr)
            except Exception as e:
                print(f"Output thread error: {e}")

            # Measure actual throughput (interval between sends), not processing time
            now = time.monotonic()
            dt = now - last_send
            last_send = now
            fps_smooth = fps_smooth * 0.9 + (1.0 / max(dt, 0.001)) * 0.1
            self.fps = fps_smooth


_output_thread: _OutputThread | None = None


def _ensure_output_thread():
    global _output_thread
    if _output_thread is None:
        _output_thread = _OutputThread()
    if not _output_thread._running:
        _output_thread.start()


def _stop_output_thread():
    global _output_thread
    if _output_thread is not None:
        _output_thread.stop()
        _output_thread = None


def set_state(s: ServerState):
    global state
    state = s


@app.on_event("startup")
async def _on_startup():
    global _frame_cond
    _frame_cond = asyncio.Condition()
    asyncio.create_task(_shared_render_loop())


# ---------------------------------------------------------------------------
# Multi-client state sync
# ---------------------------------------------------------------------------

def _get_param_value(key: str):
    """Read the canonical server-side value for a param key."""
    enh = state.enh
    if key == "fg_brightness": return enh.fg_brightness
    if key == "sharpen": return enh.sharpen
    if key == "edge_enhance": return enh.edge_enhance
    if key == "edge_intensity": return enh.edge_intensity
    if key == "saturation": return enh.saturation
    if key == "contrast": return enh.contrast
    if key == "auto_contrast": return enh.auto_contrast
    if key == "posterize": return enh.posterize if enh.posterize is not None else 0
    if key == "gamma": return enh.gamma
    if key == "grain": return enh.grain
    if key == "scanlines": return enh.scanlines if enh.scanlines is not None else 0
    if key == "pixel_upscale": return enh.pixel_upscale
    if key == "render_scale": return enh.render_scale
    if key == "interp": return enh.interp
    if key == "opacity": return enh.opacity
    if key == "alpha_curve": return enh.alpha_curve
    if key == "superres": return state.superres_enabled
    if key == "alpha_preview": return state.alpha_preview
    if key == "steps": return state.grid_pool.steps
    if key == "solver": return state.grid_pool.solver
    if key == "time_schedule": return state.grid_pool.schedule
    if key == "cfg_scale": return state.grid_pool.cfg_scale
    if key == "cfg_audio": return state.grid_pool.cfg_audio
    if key == "cfg_features": return "spectral4" if state.grid_pool.cfg_feature_mask is not None else "all"
    if key == "onset_threshold": return state.onset_threshold
    if key == "img2img_strength": return state.img2img_strength
    if key == "img2img_enabled": return state.img2img_enabled
    if key == "feedback_strength": return state.feedback_strength
    if key == "persistence": return state.persistence
    return None


def _build_state_snapshot() -> dict:
    """Build full state snapshot for init_state message."""
    enh = state.enh
    params = {
        "fg_brightness": enh.fg_brightness,
        "sharpen": enh.sharpen,
        "edge_enhance": enh.edge_enhance,
        "edge_intensity": enh.edge_intensity,
        "saturation": enh.saturation,
        "contrast": enh.contrast,
        "auto_contrast": enh.auto_contrast,
        "posterize": enh.posterize if enh.posterize is not None else 0,
        "gamma": enh.gamma,
        "grain": enh.grain,
        "scanlines": enh.scanlines if enh.scanlines is not None else 0,
        "pixel_upscale": enh.pixel_upscale,
        "render_scale": enh.render_scale,
        "interp": enh.interp,
        "opacity": enh.opacity,
        "alpha_curve": enh.alpha_curve,
        "superres": state.superres_enabled,
        "alpha_preview": state.alpha_preview,
        "steps": state.grid_pool.steps,
        "solver": state.grid_pool.solver,
        "time_schedule": state.grid_pool.schedule,
        "cfg_scale": state.grid_pool.cfg_scale,
        "cfg_audio": state.grid_pool.cfg_audio,
        "cfg_features": "spectral4" if state.grid_pool.cfg_feature_mask is not None else "all",
        "onset_threshold": state.onset_threshold,
        "img2img_strength": state.img2img_strength,
        "img2img_enabled": state.img2img_enabled,
        "feedback_strength": state.feedback_strength,
        "persistence": state.persistence,
    }
    mappings = {}
    for key, m in state.mappings.items():
        mappings[key] = {
            "source": m.source,
            "min": m.min_val,
            "max": m.max_val,
            "invert": m.invert,
            "curve": m.curve,
            "lo_thresh": m.lo_thresh,
            "hi_thresh": m.hi_thresh,
        }
    return {
        "type": "init_state",
        "params": params,
        "morph_mode": state.morph_mode,
        "gen_mode": state.gen_mode,
        "audio_mode": state.mode,
        "mappings": mappings,
        "target_fps": state.target_fps,
        "preview_fraction": state.preview_fraction,
        "preview_size": list(state.preview_size),
        "active_preset": _active_preset,
        "pool_ready": state.current_grid is not None,
        "audio_filename": state.audio_filename,
        "source_mode": state.source_mode,
        "ab_enabled": state.ab_enabled,
        "ab_preset_a": state.ab_preset_a_name,
        "ab_preset_b": state.ab_preset_b_name,
    }


async def _broadcast_sync(update: dict, exclude: WebSocket | None = None):
    """Send a state_update JSON to all connected clients except the sender."""
    msg = json.dumps({"type": "state_update", **update})
    async with _connections_lock:
        targets = [(ws, q) for ws, q in _connections.items() if ws is not exclude]
    for _, q in targets:
        try:
            q.put_nowait(msg)
        except asyncio.QueueFull:
            pass  # Drop for slow consumers — they'll get next update


# ---------------------------------------------------------------------------
# HTTP routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(
        (STATIC_DIR / "index.html").read_text(),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.get("/static/{path:path}")
async def serve_static(path: str):
    file = STATIC_DIR / path
    if file.exists() and file.is_file():
        suffix = file.suffix.lower()
        media = {
            ".css": "text/css",
            ".js": "application/javascript",
            ".html": "text/html",
            ".png": "image/png",
            ".svg": "image/svg+xml",
        }.get(suffix, "application/octet-stream")
        return FileResponse(
            file, media_type=media,
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )
    return HTMLResponse("Not found", status_code=404)


@app.post("/api/upload")
async def upload_audio(file: UploadFile):
    UPLOAD_DIR.mkdir(exist_ok=True)
    dest = UPLOAD_DIR / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Analyze in background thread to not block the event loop
    loop = asyncio.get_event_loop()
    frames, sr = await loop.run_in_executor(
        None, lambda: analyze_file(
            str(dest),
            onset_threshold_mult=state.onset_threshold,
        )
    )

    state.file_frames = frames
    state.file_analysis_fps = sr / 1024
    state.audio_filename = file.filename
    state.audio_time = 0.0
    state._last_file_idx = -1

    duration = len(frames) / state.file_analysis_fps
    beats = sum(1 for f in frames if f.is_beat)

    return {
        "filename": file.filename,
        "url": f"/audio/{file.filename}",
        "duration": round(duration, 2),
        "beats": beats,
        "analysis_fps": round(state.file_analysis_fps, 1),
    }


@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    file = UPLOAD_DIR / filename
    if file.exists():
        suffix = file.suffix.lower()
        media = {
            ".wav": "audio/wav",
            ".flac": "audio/flac",
            ".ogg": "audio/ogg",
            ".mp3": "audio/mpeg",
        }.get(suffix, "audio/wav")
        return FileResponse(file, media_type=media)
    return HTMLResponse("Not found", status_code=404)


@app.post("/api/upload_img2img")
async def upload_img2img(file: UploadFile):
    """Receive a source image, encode to glyph grid, store for img2img generation."""
    import numpy as np
    from easey_glyph.glyph.converter import image_to_glyph_grid

    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert("RGB").resize((256, 256), Image.LANCZOS)
    img_arr = np.array(img)

    masks, embeddings = state.ensure_glyph_resources()

    loop = asyncio.get_event_loop()
    grid, _ = await loop.run_in_executor(
        None, lambda: image_to_glyph_grid(img_arr, masks, embeddings)
    )

    state.img2img_source = grid
    state.set_source_mode("image")

    return {"status": "ok", "width": 256, "height": 256}


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    client = ws.client
    addr = f"{client.host}:{client.port}" if client else "unknown"

    # Register this connection with a sync queue
    sync_q: asyncio.Queue = asyncio.Queue(maxsize=64)
    async with _connections_lock:
        _connections[ws] = sync_q
        n = len(_connections)
    print(f"WS: client {addr} connected ({n} total, frame #{_frame_counter})")

    # Send full state snapshot (best-effort — don't block send/recv loops)
    try:
        snapshot = _build_state_snapshot()
        await ws.send_text(json.dumps(snapshot))
    except Exception as e:
        if isinstance(e, (WebSocketDisconnect, RuntimeError)):
            async with _connections_lock:
                _connections.pop(ws, None)
            return
        print(f"Warning: init_state send failed ({addr}): {e}")

    send_task = asyncio.create_task(_send_loop(ws, sync_q))
    recv_task = asyncio.create_task(_recv_loop(ws))

    try:
        done, pending = await asyncio.wait(
            [send_task, recv_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for t in done:
            name = "send" if t is send_task else "recv"
            if t.cancelled():
                print(f"WS: {addr} {name} loop cancelled")
            elif t.exception():
                print(f"WS: {addr} {name} loop error: {t.exception()}")
            else:
                print(f"WS: {addr} {name} loop ended cleanly")
        for t in pending:
            t.cancel()
    except Exception as e:
        print(f"WS: {addr} wait error: {e}")
        send_task.cancel()
        recv_task.cancel()
    finally:
        async with _connections_lock:
            _connections.pop(ws, None)
            n = len(_connections)
        print(f"WS: client {addr} disconnected ({n} remaining)")


async def _shared_render_loop():
    """Single render loop — produces frames consumed by all connected clients."""
    global _latest_frame, _latest_meta_json, _frame_counter

    fps_smooth = 60.0
    loop = asyncio.get_event_loop()
    frame_idx = 0

    while True:
        target_dt = 1.0 / state.target_fps
        t0 = time.monotonic()

        # Decide whether this frame includes preview
        frac = state.preview_fraction
        if frac <= 0:
            include_preview = False
        elif frac >= 1.0:
            include_preview = True
        else:
            interval = max(1, round(1.0 / frac))
            include_preview = (frame_idx % interval) == 0
        frame_idx += 1

        try:
            frame_bytes, meta_json = await loop.run_in_executor(
                None, _render_one_frame, include_preview, frac,
            )
        except Exception as e:
            print(f"Render error: {e}")
            await asyncio.sleep(target_dt)
            continue

        # Store latest frame and notify all send loops
        _latest_frame = frame_bytes
        _latest_meta_json = meta_json
        _frame_counter += 1
        async with _frame_cond:
            _frame_cond.notify_all()

        elapsed = time.monotonic() - t0
        fps_smooth = fps_smooth * 0.9 + (1.0 / max(elapsed, 0.001)) * 0.1
        state.fps_smooth = fps_smooth

        # Tight frame pacing: sleep most of the time, then yield-loop for precision
        remaining = target_dt - elapsed
        if remaining > 0.005:
            await asyncio.sleep(remaining - 0.005)
        while time.monotonic() - t0 < target_dt:
            await asyncio.sleep(0)


async def _send_loop(ws: WebSocket, sync_q: asyncio.Queue):
    """Forward shared render frames + sync messages to a single client."""
    last_counter = 0

    while True:
        # Wait for the shared render loop to produce a new frame
        async with _frame_cond:
            while _frame_counter == last_counter:
                await _frame_cond.wait()
            last_counter = _frame_counter
            frame_bytes = _latest_frame
            meta_json = _latest_meta_json

        try:
            if frame_bytes is not None:
                await ws.send_bytes(frame_bytes)
            await ws.send_text(meta_json)
            # Drain sync queue (peer state updates)
            while not sync_q.empty():
                try:
                    await ws.send_text(sync_q.get_nowait())
                except asyncio.QueueEmpty:
                    break
        except (WebSocketDisconnect, RuntimeError, ConnectionResetError, OSError):
            return


def _apply_mappings_main(enh, audio_frame, midi_vals=None):
    """Apply audio mappings to create effective enhancement params.

    Delegates to centralized mapping module. Returns (effective_enh, mapped_vals_dict).
    """
    if not state.mappings:
        return enh, {}

    resolved = resolve_mappings(state.mappings, audio_frame, midi_vals)
    effective_enh, mapped_vals = apply_resolved_to_enh(
        enh, resolved, state.edge_mode_selected)
    apply_side_effects(resolved, state)
    return effective_enh, mapped_vals


def _apply_mappings_pure(enh, mappings, audio_frame, edge_mode_selected):
    """Apply audio mappings — pure/side-effect-free for A/B comparison.

    Returns (effective_enh, superres_on).
    """
    if not mappings:
        return enh, False

    resolved = resolve_mappings(mappings, audio_frame)
    effective_enh, mapped_vals = apply_resolved_to_enh(
        enh, resolved, edge_mode_selected)
    superres_on = resolved.get("superres", 0.0) > 0.5
    return effective_enh, superres_on


def _resolve_preset_snapshot(data: dict) -> PresetSnapshot:
    """Build a PresetSnapshot from client-sent resolved preset data."""
    from easey_glyph.render.pipeline import EnhancementParams

    params = data.get("params", {})
    enh = EnhancementParams(
        fg_brightness=float(params.get("fg_brightness", 1.0)),
        render_scale=max(1, int(params.get("render_scale", 4))),
        sharpen=float(params.get("sharpen", 0)),
        saturation=float(params.get("saturation", 1.0)),
        contrast=float(params.get("contrast", 1.0)),
        gamma=float(params.get("gamma", 1.0)),
        posterize=int(params["posterize"]) if params.get("posterize") else None,
        grain=float(params.get("grain", 0)),
        scanlines=int(params["scanlines"]) if params.get("scanlines") else None,
        pixel_upscale=bool(params.get("pixel_upscale", False)),
        interp=str(params.get("interp", "bilinear")),
        opacity=float(params.get("opacity", 1.0)),
        alpha_curve=float(params.get("alpha_curve", 1.0)),
        edge_enhance=str(params.get("edge_enhance", "off")),
        edge_intensity=float(params.get("edge_intensity", 1.0)),
        auto_contrast=float(params["auto_contrast"]) if params.get("auto_contrast") else None,
    )

    mappings_data = data.get("mappings", {})
    mappings = {}
    for key, m in mappings_data.items():
        if not m or m.get("source", "none") == "none":
            continue
        mappings[key] = AudioMapping(
            source=str(m.get("source", "none")),
            min_val=float(m.get("min", 0)),
            max_val=float(m.get("max", 1)),
            invert=bool(m.get("invert", False)),
            curve=str(m.get("curve", "linear")),
            lo_thresh=float(m.get("lo_thresh", 0)),
            hi_thresh=float(m.get("hi_thresh", 1)),
        )

    toggles = data.get("toggles", {})
    selects = data.get("selects", {})

    return PresetSnapshot(
        enh=enh,
        mappings=mappings,
        superres_enabled=bool(toggles.get("superres", False)),
        edge_mode_selected=str(selects.get("edge_enhance", "off")),
    )


# ---------------------------------------------------------------------------
# MIDI processing
# ---------------------------------------------------------------------------

# Param ranges for MIDI learn — derived from mapping registry + MIDI-only extras
_PARAM_RANGES = get_param_ranges()
_PARAM_RANGES.update({
    "steps": (1, 50),
    "onset_threshold": (0.5, 5.0),
    "img2img_strength": (0, 1.0),
})


def _process_midi_messages():
    """Poll MIDI queue, update cc_values, handle learn mode, fire triggers.

    Trigger edge detection is per-message, not per-frame. This is critical
    because note_on + note_off often arrive in the same poll batch at 60fps.
    If we checked after all messages, the note_off would overwrite the value
    to 0 before the trigger ever saw the hit.
    """
    _triggers_fired.clear()

    if state.midi_input is None or not state.midi_input.connected:
        return

    messages = state.midi_input.poll_messages()
    if not messages:
        return

    for msg_type, number, channel, value_127 in messages:
        val_01 = value_127 / 127.0
        state.midi_cc_values[(msg_type, number, channel)] = val_01
        state.midi_last_cc = (msg_type, number, channel, value_127)

        # Learn mode — first message received becomes the mapping
        if state.midi_learn_target is not None:
            # For notes, only learn on note_on (velocity > 0)
            if msg_type == "note" and value_127 == 0:
                continue

            target = state.midi_learn_target
            state.midi_learn_target = None

            if state.midi_learn_is_trigger:
                state.midi_triggers[target] = MIDIMapping(
                    cc=number, channel=0, msg_type=msg_type,
                )
            else:
                lo, hi = _PARAM_RANGES.get(target, (0.0, 1.0))
                state.midi_mappings[target] = MIDIMapping(
                    cc=number, channel=0, msg_type=msg_type, min_val=lo, max_val=hi,
                )

            state.save_midi()
            label = f"CC{number}" if msg_type == "cc" else f"Note{number}"
            print(f"MIDI learn: {label} -> {target}")
            continue

        # Check triggers per-message (rising edge detection)
        is_on = val_01 > 0.5
        for action, mapping in state.midi_triggers.items():
            if mapping.msg_type != msg_type or mapping.cc != number:
                continue
            if mapping.channel != 0 and mapping.channel != channel:
                continue
            was_on = _trigger_prev_state.get(action, False)
            _trigger_prev_state[action] = is_on
            if is_on and not was_on:
                _triggers_fired.append(action)
                _execute_trigger(action)


def _get_midi_val(mapping: MIDIMapping) -> float | None:
    """Get current value for a mapping, handling omni channel."""
    mt = mapping.msg_type
    if mapping.channel == 0:
        # Omni — check all stored values for this type+number
        for (t, num, ch), val in state.midi_cc_values.items():
            if t == mt and num == mapping.cc:
                return val
        return None
    return state.midi_cc_values.get((mt, mapping.cc, mapping.channel))


def _apply_midi_mappings(enh):
    """Apply MIDI CC values to override slider base values.

    Returns (modified_enh, midi_vals_dict). MIDI replaces the slider value;
    audio maps modulate on top.
    """
    from dataclasses import replace

    if not state.midi_mappings or state.midi_input is None:
        return enh, {}

    result = replace(enh)
    midi_vals = {}
    for key, mapping in state.midi_mappings.items():
        cc_val = _get_midi_val(mapping)
        if cc_val is None:
            continue

        if mapping.invert:
            cc_val = 1.0 - cc_val
        mapped_val = mapping.min_val + (mapping.max_val - mapping.min_val) * cc_val

        if key == "superres":
            on = bool(mapped_val > 0.5)
            state.superres_enabled = on
            midi_vals[key] = round(mapped_val, 2)
        elif key == "edge_enhance":
            if mapped_val > 0.5 and state.edge_mode_selected != "off":
                result.edge_enhance = state.edge_mode_selected
            else:
                result.edge_enhance = "off"
            midi_vals[key] = round(mapped_val, 2)
        elif key in ("posterize", "scanlines"):
            v = round(mapped_val)
            setattr(result, key, v if v > 0 else None)
            midi_vals[key] = v
        elif key == "render_scale":
            v = max(1, round(mapped_val))
            setattr(result, key, v)
            midi_vals[key] = v
        elif key == "steps":
            v = max(1, round(mapped_val))
            state.grid_pool.update_steps(v)
            midi_vals[key] = v
        elif key == "cfg_scale":
            state.grid_pool.update_cfg(cfg_scale=max(0.0, mapped_val))
            midi_vals[key] = round(mapped_val, 2)
        elif key == "onset_threshold":
            state.onset_threshold = mapped_val
            if state.capture is not None:
                state.capture.beat_detector.set_onset_threshold(mapped_val)
            midi_vals[key] = round(mapped_val, 2)
        elif key == "feedback_strength":
            state.feedback_strength = mapped_val
            midi_vals[key] = round(mapped_val, 2)
        elif key == "persistence":
            state.persistence = mapped_val
            midi_vals[key] = round(mapped_val, 2)
        elif key == "img2img_strength":
            v = max(0.0, min(1.0, mapped_val))
            state.img2img_strength = v
            midi_vals[key] = round(v, 2)
        else:
            setattr(result, key, mapped_val)
            midi_vals[key] = round(mapped_val, 2)

    return result, midi_vals


# Track trigger edge detection to avoid repeated firing
_trigger_prev_state: dict[str, bool] = {}
# Triggers that fired this frame (for UI flash feedback)
_triggers_fired: list[str] = []


def _execute_trigger(action: str):
    """Execute a trigger action. Called on rising edge (already filtered)."""
    if action == "regenerate_pool":
        if state.gen_mode == "pool":
            state.grid_pool.flush_and_refill()
    elif action == "next_grid":
        g = state.grid_pool.next_grid()
        if g is not None:
            state.prev_grid = state.current_grid
            state.current_grid = g
    elif action == "morph_frozen":
        state.morph_mode = "frozen"
    elif action == "morph_ambient":
        state.morph_mode = "ambient"
    elif action == "morph_beat":
        state.morph_mode = "beat"
    elif action == "gen_pool":
        state.gen_mode = "pool"
        state.grid_pool.resume()
    elif action == "gen_realtime":
        state.gen_mode = "realtime"
        state.grid_pool.pause()
    elif action == "toggle_superres":
        state.superres_enabled = not state.superres_enabled


@torch.inference_mode()
def _apply_superres(frame_arr: np.ndarray, model, device) -> np.ndarray:
    """Run super-resolution CNN on a 32x32 RGBA frame -> 256x256 RGBA.

    Input:  [32, 32, 4] uint8 RGBA
    Output: [256, 256, 4] uint8 RGBA (alpha bilinear-upscaled separately)
    """
    # CoreML path — numpy in/out, no torch needed
    if state.coreml_superres is not None:
        rgb = frame_arr[:, :, :3].astype(np.float32) / 255.0
        rgb_nchw = np.transpose(rgb, (2, 0, 1))[np.newaxis]  # [1,3,32,32]
        sr_nchw = state.coreml_superres.upscale(rgb_nchw)     # [1,3,256,256]
        sr = (np.transpose(sr_nchw[0], (1, 2, 0)).clip(0, 1) * 255).astype(np.uint8)
        # Nearest-neighbor alpha upscale via np.kron (avoids PIL round-trip)
        scale = 256 // frame_arr.shape[0]
        alpha_up = np.kron(frame_arr[:, :, 3], np.ones((scale, scale), dtype=np.uint8))
        out = np.empty((256, 256, 4), dtype=np.uint8)
        out[:, :, :3] = sr
        out[:, :, 3] = alpha_up
        return out

    # PyTorch path — single GPU round-trip for full RGBA
    rgba_t = torch.from_numpy(frame_arr).to(device).float() / 255.0  # [32,32,4]
    rgb = rgba_t[:, :, :3].permute(2, 0, 1).unsqueeze(0)   # [1,3,32,32]
    alpha = rgba_t[:, :, 3].unsqueeze(0).unsqueeze(0)       # [1,1,32,32]
    # Auto-cast to model dtype (fp16 on CUDA, fp32 elsewhere)
    model_dtype = next(model.parameters()).dtype
    sr = model(rgb.to(model_dtype)).float()                  # [1,3,256,256]
    alpha_up = torch.nn.functional.interpolate(
        alpha, size=(256, 256), mode='bilinear', align_corners=False)
    out_t = torch.cat([sr[0], alpha_up[0]], dim=0)           # [4,256,256]
    return (out_t.permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy()


def _send_output_frame(frame: np.ndarray):
    """Send a frame to all active output senders and PNG recorder."""
    for sender in state.output_senders:
        sender.send_frame(frame)
    if state.png_recorder is not None and state.png_recorder.active:
        state.png_recorder.send_frame(frame)


def _audio_frame_to_tensor(audio_frame, device) -> torch.Tensor:
    """Extract 12 audio features from AudioFrame as [1, 12] tensor for CFG conditioning."""
    features = [
        audio_frame.bass, audio_frame.mid, audio_frame.treble,
        audio_frame.rms, audio_frame.beat_phase, audio_frame.onset_strength,
        audio_frame.spectral_centroid, audio_frame.spectral_flux,
        audio_frame.spectral_flatness, audio_frame.spectral_rolloff,
        audio_frame.spectral_bandwidth, audio_frame.zero_crossing_rate,
    ]
    return torch.tensor([features], dtype=torch.float32, device=device)


def _make_cfg_audio(n: int = 1) -> torch.Tensor | None:
    """Build audio tensor for realtime/img2img CFG calls."""
    pool = state.grid_pool
    if pool.cfg_scale <= 0:
        return None
    if pool.cfg_audio == "live":
        af = state.get_audio_frame()
        audio = _audio_frame_to_tensor(af, state.device).expand(n, -1)
        return pool._apply_feature_mask(audio)
    return torch.rand(n, 12, device=state.device)


def _submit_rt_generation():
    """Submit a realtime generation if GPU is free. Enables continuous generation."""
    global _rt_future, _rt_pending_is_evolution
    if _rt_future is not None or _camera_gen_future is not None:
        return
    if state.img2img_enabled and state.img2img_source is not None:
        _rt_future = _rt_executor.submit(
            generate_single_grid_img2img,
            state.model, state.model_cfg, state.device,
            state.img2img_source,
            strength=state.img2img_strength,
            steps=state.grid_pool.steps,
            solver=state.grid_pool.solver,
            schedule=state.grid_pool.schedule,
            audio=_make_cfg_audio(),
            cfg_scale=state.grid_pool.cfg_scale,
        )
        _rt_pending_is_evolution = False
    elif state.feedback_strength < 1.0 and state.current_grid is not None:
        evolution_source = _pending_grid if _pending_grid is not None else state.current_grid
        _rt_future = _rt_executor.submit(
            generate_single_grid_img2img,
            state.model, state.model_cfg, state.device,
            evolution_source,
            strength=state.feedback_strength,
            steps=state.grid_pool.steps,
            solver=state.grid_pool.solver,
            schedule=state.grid_pool.schedule,
            audio=_make_cfg_audio(),
            cfg_scale=state.grid_pool.cfg_scale,
        )
        _rt_pending_is_evolution = True
    elif state.gen_mode == "realtime":
        _rt_future = _rt_executor.submit(
            generate_single_grid,
            state.model, state.model_cfg, state.device,
            steps=state.grid_pool.steps,
            solver=state.grid_pool.solver,
            schedule=state.grid_pool.schedule,
            audio=_make_cfg_audio(),
            cfg_scale=state.grid_pool.cfg_scale,
        )
        _rt_pending_is_evolution = False


def _render_ab_side(snapshot, display_grid, audio_frame, params_base, side_size):
    """Render one side of A/B preview. Returns [h, w, 4] uint8 RGBA numpy array."""
    from dataclasses import replace as dc_replace

    # Apply audio mappings from this snapshot's config
    effective_enh, superres_on = _apply_mappings_pure(
        snapshot.enh, snapshot.mappings, audio_frame, snapshot.edge_mode_selected)

    # Render 32x32 pixel frame
    params = dc_replace(params_base, fg_brightness=effective_enh.fg_brightness)
    frame_arr = render_pixel_frame(display_grid, params)

    # Super-resolution CNN (32x32 -> 256x256)
    use_superres = (superres_on or snapshot.superres_enabled) and (
        state.superres_model is not None or state.coreml_superres is not None)
    if use_superres:
        frame_arr = _apply_superres(frame_arr, state.superres_model, state.device)

    # Optional scale2x (skipped when superres active)
    if not use_superres and effective_enh.pixel_upscale:
        frame_arr = scale2x(frame_arr)

    if use_superres:
        base_size = 256
    elif effective_enh.pixel_upscale:
        base_size = 64
    else:
        base_size = 32
    inter_size = base_size * effective_enh.render_scale

    # Resize to intermediate + apply intermediate enhancements
    frame_img = Image.fromarray(frame_arr, "RGBA")
    interp_method = INTERP_METHODS.get(effective_enh.interp, Image.BILINEAR)
    frame_img = frame_img.resize((inter_size, inter_size), interp_method)
    intermediate_arr = apply_intermediate_enhancements(frame_img, params, effective_enh)

    # Apply grain — use intermediate size as grain target to avoid a non-integer
    # NEAREST pre-upscale (128→170) that compounds with the output resize (170→512),
    # causing 1-2px horizontal jitter visible against the fixed A/B divider line.
    if effective_enh.grain > 0:
        intermediate_arr = apply_grain(intermediate_arr, effective_enh.grain, (inter_size, inter_size))

    # Final output enhancements — single NEAREST resize from inter_size to side_size
    return apply_output_enhancements(intermediate_arr, effective_enh, side_size)


def _render_ab_preview(display_grid, audio_frame, params_base, preview_size, alpha_preview):
    """Render side-by-side A/B preview. Returns encoded bytes."""
    w, h = preview_size

    frame_a = _render_ab_side(state.ab_preset_a, display_grid, audio_frame, params_base, (w, h))
    frame_b = _render_ab_side(state.ab_preset_b, display_grid, audio_frame, params_base, (w, h))

    composite = np.hstack([frame_a, frame_b])

    buf = io.BytesIO()
    if alpha_preview:
        img = Image.fromarray(composite, "RGBA")
        img.save(buf, format="WEBP", quality=80, method=0)
    else:
        img = Image.fromarray(composite[:, :, :3], "RGB")
        img.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


def _render_one_frame(include_preview: bool = True, preview_fraction: float = 1.0) -> tuple[bytes | None, str]:
    """Render a single frame. Called from the thread pool executor."""
    # Seed initial grids from pool once available
    state.try_seed_grids()

    audio_frame = state.get_audio_frame()

    # Snapshot live audio features for pool CFG (if using live audio mode)
    if state.grid_pool.cfg_scale > 0 and state.grid_pool.cfg_audio == "live":
        state.grid_pool.set_live_audio_features(
            _audio_frame_to_tensor(audio_frame, state.device))

    # Grid morph mode — fully decoupled from audio mappings
    if state.morph_mode == "frozen":
        morph_frame = None
    elif state.morph_mode == "ambient":
        morph_frame = state._idle_frame()
    else:  # "beat"
        morph_frame = audio_frame

    # Collect completed async result (defer swap to next beat edge)
    global _rt_future, _pending_grid, _pending_is_evolution, _rt_pending_is_evolution
    global _camera_gen_future, _camera_result_grid, _camera_prev_grid
    global _camera_swap_time, _camera_frames_generated

    # Pre-compute camera display state for guard checks below
    camera_display_active = (
        state.source_mode == "camera" and state.img2img_enabled
        and _camera_result_grid is not None
    )
    if _rt_future is not None and _rt_future.done():
        try:
            g = _rt_future.result()
            if g is not None:
                _pending_grid = g
                _pending_is_evolution = _rt_pending_is_evolution
        except Exception:
            pass
        _rt_future = None

        # Continuous realtime: immediately start next generation
        if state.gen_mode == "realtime" and not camera_display_active:
            _submit_rt_generation()

    # Ensure continuous realtime generation is running (initial startup / mode switch)
    if (state.gen_mode == "realtime" and _rt_future is None
            and not camera_display_active
            and _camera_gen_future is None):
        _submit_rt_generation()

    # ---------------------------------------------------------------------------
    # Continuous camera generation — runs independently of beat edges
    # ---------------------------------------------------------------------------
    if state.source_mode == "camera" and state.img2img_enabled:
        # Collect completed camera generation
        if _camera_gen_future is not None and _camera_gen_future.done():
            try:
                g = _camera_gen_future.result()
                if g is not None:
                    _camera_prev_grid = _camera_result_grid
                    _camera_result_grid = g
                    _camera_swap_time = time.monotonic()
                    _camera_frames_generated += 1
                    camera_display_active = True
            except Exception as e:
                print(f"Camera generation error: {e}")
            _camera_gen_future = None

        # Submit next generation if GPU is free and we have a source
        if (_camera_gen_future is None and _rt_future is None
                and state.img2img_source is not None):
            _camera_gen_future = _rt_executor.submit(
                generate_single_grid_img2img,
                state.model, state.model_cfg, state.device,
                state.img2img_source,
                strength=state.img2img_strength,
                steps=state.grid_pool.steps,
                solver=state.grid_pool.solver,
                schedule=state.grid_pool.schedule,
                audio=_make_cfg_audio(),
                cfg_scale=state.grid_pool.cfg_scale,
            )

    # Beat edge detection (skip when frozen or camera is actively displaying)
    if morph_frame is not None and not camera_display_active:
        if morph_frame.is_beat and not state.last_beat_seen:
            applied_pending = False

            # Phase 1: Apply pending async result
            if _pending_grid is not None:
                state.prev_grid = state.current_grid
                if not _pending_is_evolution and state.persistence > 0 and state.current_grid is not None:
                    # Non-evolution pending (realtime/img2img): apply persistence blend
                    state.current_grid = slerp(state.persistence, _pending_grid, state.current_grid)
                else:
                    # Evolution pending (feedback) or persistence off: direct replace
                    state.current_grid = _pending_grid
                _pending_grid = None
                applied_pending = True

            # Phase 2: Grab from pool (realtime gen is continuous, handled above)
            if state.gen_mode == "pool" and not applied_pending:
                g = state.grid_pool.next_grid()
                # Ambient resilience: recycle history if pool is empty
                if g is None and state.morph_mode == "ambient" and state._idle_grid_history:
                    g = random.choice(state._idle_grid_history).clone()
                if g is not None:
                    # Save to ambient history ring buffer
                    if state.morph_mode == "ambient":
                        h = state._idle_grid_history
                        if len(h) < state._idle_history_max:
                            h.append(g.clone())
                        else:
                            h[random.randint(0, len(h) - 1)] = g.clone()
                    state.prev_grid = state.current_grid
                    if state.persistence > 0 and state.current_grid is not None:
                        state.current_grid = slerp(state.persistence, g, state.current_grid)
                    else:
                        state.current_grid = g

            # Phase 3: When feedback evolution was applied AND persistence > 0,
            # also blend in a fresh pool grid
            if state.persistence > 0 and applied_pending and _pending_is_evolution:
                if state.gen_mode == "pool" and not (state.img2img_enabled and state.img2img_source is not None):
                    g = state.grid_pool.next_grid()
                    if g is not None and state.current_grid is not None:
                        state.current_grid = slerp(state.persistence, g, state.current_grid)

            state.last_beat_seen = True
        elif not morph_frame.is_beat:
            state.last_beat_seen = False

    # SLERP between prev and current grid
    if camera_display_active:
        # Camera mode: time-based crossfade between successive camera results
        if _camera_prev_grid is not None and _camera_result_grid is not None:
            elapsed = time.monotonic() - _camera_swap_time
            t = min(1.0, elapsed / _CAMERA_CROSSFADE)
            # smoothstep for organic transition
            t = t * t * (3.0 - 2.0 * t)
            display_grid = slerp(t, _camera_prev_grid, _camera_result_grid)
        else:
            display_grid = _camera_result_grid
    elif morph_frame is not None and state.prev_grid is not None and state.current_grid is not None:
        display_grid = slerp(morph_frame.beat_phase, state.prev_grid, state.current_grid)
    elif state.current_grid is not None:
        display_grid = state.current_grid
    else:
        display_grid = torch.zeros(16, 32, 32)

    # Process MIDI CC messages (update cc_values, handle learn, fire triggers)
    _process_midi_messages()

    # Apply MIDI mappings → override slider base values
    enh, midi_vals = _apply_midi_mappings(state.enh)

    # Apply audio mappings → effective enh with overridden values
    effective_enh, mapped_vals = _apply_mappings_main(enh, audio_frame, midi_vals)

    # No built-in auto-reactivity — effects only react to audio
    # when the user explicitly maps them via the UI dropdowns.
    params = frame_to_render_params(audio_frame)
    params.fg_brightness = effective_enh.fg_brightness

    # A/B comparison mode — side-by-side preview from two preset snapshots
    ab_active = (state.ab_enabled and state.ab_preset_a is not None
                 and state.ab_preset_b is not None)
    preview_bytes = None
    preview_format = "webp" if state.alpha_preview else "jpeg"

    if ab_active and include_preview:
        preview_bytes = _render_ab_preview(
            display_grid, audio_frame, params,
            state.preview_size, state.alpha_preview)
        preview_format = "webp" if state.alpha_preview else "jpeg"

    # Normal single-preset render (always needed for output senders,
    # and for preview when A/B is off)
    # Render 32x32 pixel frame
    frame_arr = render_pixel_frame(display_grid, params)

    # Super-resolution CNN (32x32 -> 256x256)
    use_superres = state.superres_enabled and (state.superres_model is not None or state.coreml_superres is not None)
    if use_superres:
        frame_arr = _apply_superres(frame_arr, state.superres_model, state.device)

    # Optional scale2x (skipped when superres active — already 256x256)
    if not use_superres and effective_enh.pixel_upscale:
        frame_arr = scale2x(frame_arr)

    if use_superres:
        base_size = 256
    elif effective_enh.pixel_upscale:
        base_size = 64
    else:
        base_size = 32
    inter_size = base_size * effective_enh.render_scale

    # Step 1: interp to intermediate + apply intermediate enhancements once
    # Returns numpy [H, W, 4] uint8 RGBA — shared by preview and output paths
    needs_frame_img = (include_preview and not ab_active) or state.output_enabled
    intermediate_arr = None
    if needs_frame_img:
        frame_img = Image.fromarray(frame_arr, "RGBA")
        interp_method = INTERP_METHODS.get(effective_enh.interp, Image.BILINEAR)
        frame_img = frame_img.resize((inter_size, inter_size), interp_method)
        intermediate_arr = apply_intermediate_enhancements(frame_img, params, effective_enh)
        # Apply grain — use intermediate size as grain target to avoid a non-integer
        # NEAREST pre-upscale (128→170) that compounds with the output resize (170→512),
        # causing 1-2px pixel shifting jitter (same fix as A/B path).
        if effective_enh.grain > 0:
            intermediate_arr = apply_grain(intermediate_arr, effective_enh.grain, (inter_size, inter_size))

    # Path 1: Preview — output enhancements + JPEG/WebP (skipped when A/B active or throttled)
    if include_preview and not ab_active and intermediate_arr is not None:
        frame_arr = apply_output_enhancements(intermediate_arr, effective_enh, state.preview_size)
        buf = io.BytesIO()
        if state.alpha_preview:
            img = Image.fromarray(frame_arr, "RGBA")
            img.save(buf, format="WEBP", quality=80, method=0)
            preview_format = "webp"
        else:
            img = Image.fromarray(frame_arr[:, :, :3], "RGB")
            img.save(buf, format="JPEG", quality=80)
        preview_bytes = buf.getvalue()

    # Path 2: Output — output enhancements at output resolution (async thread)
    # Output senders always use the primary (A) preset, not the A/B composite
    if state.output_enabled and intermediate_arr is not None:
        _ensure_output_thread()
        _output_thread.submit(intermediate_arr, effective_enh, state.output_size)

    meta = {
        "bpm": round(audio_frame.bpm, 1),
        "is_beat": audio_frame.is_beat,
        "bass": round(audio_frame.bass, 3),
        "mid": round(audio_frame.mid, 3),
        "treble": round(audio_frame.treble, 3),
        "rms": round(audio_frame.rms, 3),
        "pool": state.grid_pool.size,
        "pool_target": state.grid_pool.pool_size,
        "pool_ready": state.current_grid is not None,
        "mode": state.mode,
        "morph": state.morph_mode,
        "gen_mode": state.gen_mode,
        "features": {
            "bass": round(audio_frame.bass, 3),
            "mid": round(audio_frame.mid, 3),
            "treble": round(audio_frame.treble, 3),
            "rms": round(audio_frame.rms, 3),
            "beat_phase": round(audio_frame.beat_phase, 3),
            "onset_strength": round(audio_frame.onset_strength, 3),
            "spectral_centroid": round(audio_frame.spectral_centroid, 3),
            "spectral_flux": round(audio_frame.spectral_flux, 3),
            "spectral_flatness": round(audio_frame.spectral_flatness, 3),
            "spectral_rolloff": round(audio_frame.spectral_rolloff, 3),
            "spectral_bandwidth": round(audio_frame.spectral_bandwidth, 3),
            "zero_crossing_rate": round(audio_frame.zero_crossing_rate, 3),
        },
        "mapped": mapped_vals,
        "preview_format": preview_format,
        "preview_fraction": preview_fraction,
        "ab_mode": state.ab_enabled,
        "superres": state.superres_enabled,
        "img2img": state.img2img_enabled,
        "img2img_strength": state.img2img_strength,
        "source_mode": state.source_mode,
        "solver": state.grid_pool.solver,
        "schedule": state.grid_pool.schedule,
        "cfg_scale": state.grid_pool.cfg_scale,
        "fps_smooth": round(state.fps_smooth, 1),
        "output_active": state.output_enabled,
        "output_fps": round(_output_thread.fps, 1) if _output_thread is not None else 0,
        "output_size": list(state.output_size),
        "output_senders": [s.name for s in state.output_senders],
        "output_recording": state.png_recorder is not None and state.png_recorder.active,
        "output_rec_frames": state.png_recorder.frame_count if state.png_recorder and state.png_recorder.active else 0,
        "has_superres": state.superres_model is not None or state.coreml_superres is not None,
        "has_ndi": HAS_NDI,
        "has_syphon": HAS_SYPHON,
        "has_spout": HAS_SPOUT,
        "midi_active": state.midi_input is not None and state.midi_input.connected,
        "midi_learn": state.midi_learn_target,
        "midi_mappings": {
            k: {"cc": m.cc, "channel": m.channel, "type": m.msg_type, "min": m.min_val, "max": m.max_val, "invert": m.invert}
            for k, m in state.midi_mappings.items()
        },
        "midi_triggers": {
            k: {"cc": m.cc, "channel": m.channel, "type": m.msg_type}
            for k, m in state.midi_triggers.items()
        },
        "midi_vals": midi_vals,
        "midi_triggers_fired": list(_triggers_fired),
        "midi_last_cc": list(state.midi_last_cc) if state.midi_last_cc else None,
    }

    if state.source_mode == "camera":
        meta["camera_stats"] = {
            "recv": _camera_frames_recv,
            "encoded": _camera_frames_encoded,
            "generated": _camera_frames_generated,
        }

    return preview_bytes, json.dumps(meta)


def _process_camera_frame(jpeg_bytes: bytes):
    """Decode JPEG, encode to glyph grid. Runs in thread pool."""
    global _camera_frames_encoded
    try:
        from easey_glyph.glyph.converter import image_to_glyph_grid
        img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB").resize((256, 256), Image.LANCZOS)
        masks, embeddings = state.ensure_glyph_resources()
        grid, _ = image_to_glyph_grid(np.array(img), masks, embeddings)
        state.img2img_source = grid  # atomic swap (GIL)
        _camera_frames_encoded += 1
    except Exception as e:
        print(f"Camera frame encode error: {e}")


async def _recv_loop(ws: WebSocket):
    """Handle incoming JSON messages from the browser."""
    while True:
        try:
            raw = await ws.receive()
        except (WebSocketDisconnect, RuntimeError, ConnectionResetError, OSError) as e:
            code = getattr(e, 'code', None)
            if code:
                print(f"WS: recv disconnect code={code}")
            return

        if raw.get("type") == "websocket.disconnect":
            code = raw.get("code", "?")
            print(f"WS: client close code={code}")
            return

        data = raw.get("text")
        if not data:
            # Binary frame (text absent or empty string) — check for camera JPEG
            frame_bytes = raw.get("bytes")
            if frame_bytes and state.source_mode == "camera":
                global _camera_frames_recv
                _camera_frames_recv += 1
                # Only encode if previous frame is done (latest-wins, drop intermediates)
                if state._camera_encode_future is None or state._camera_encode_future.done():
                    state._camera_encode_future = asyncio.get_event_loop().run_in_executor(
                        None, _process_camera_frame, frame_bytes,
                    )
            continue

        try:
            msg = json.loads(data)
        except json.JSONDecodeError:
            continue

        msg_type = msg.get("type")

        if msg_type == "param":
            _handle_param(msg)
            key = msg.get("key")
            if key is not None:
                asyncio.create_task(_broadcast_sync(
                    {"params": {key: _get_param_value(key)}}, exclude=ws,
                ))
        elif msg_type == "mapping":
            _handle_mapping(msg)
            key = msg.get("key")
            if key is not None:
                m = state.mappings.get(key)
                mapping_data = None
                if m is not None:
                    mapping_data = {
                        "source": m.source, "min": m.min_val, "max": m.max_val,
                        "invert": m.invert, "curve": m.curve,
                        "lo_thresh": m.lo_thresh, "hi_thresh": m.hi_thresh,
                    }
                asyncio.create_task(_broadcast_sync(
                    {"mappings": {key: mapping_data}}, exclude=ws,
                ))
        elif msg_type == "mode":
            _handle_mode(msg)
            asyncio.create_task(_broadcast_sync(
                {"audio_mode": state.mode}, exclude=ws,
            ))
        elif msg_type == "audio_time":
            state.audio_time = float(msg.get("t", 0))
        elif msg_type == "morph":
            state.morph_mode = msg.get("mode", "ambient")
            asyncio.create_task(_broadcast_sync(
                {"morph_mode": state.morph_mode}, exclude=ws,
            ))
        elif msg_type == "aspect":
            _handle_aspect(msg)
            asyncio.create_task(_broadcast_sync(
                {"preview_size": list(state.preview_size)}, exclude=ws,
            ))
        elif msg_type == "gen_mode":
            mode = msg.get("mode", "pool")
            if mode in ("pool", "realtime"):
                state.gen_mode = mode
                if mode == "realtime":
                    state.grid_pool.pause()
                else:
                    state.grid_pool.resume()
                asyncio.create_task(_broadcast_sync(
                    {"gen_mode": state.gen_mode}, exclude=ws,
                ))
        elif msg_type == "regenerate":
            if state.gen_mode == "pool":
                state.grid_pool.flush_and_refill()
        elif msg_type == "reseed":
            g = state.grid_pool.next_grid()
            if g is not None:
                state.prev_grid = state.current_grid
                state.current_grid = g
        elif msg_type == "img2img_clear":
            _clear_camera_state()
            state.set_source_mode("none")
        elif msg_type == "img2img_toggle":
            state.img2img_enabled = bool(msg.get("enabled", False))
            if state.img2img_enabled and state.img2img_source is not None:
                state.grid_pool.pause()
            elif not state.img2img_enabled and state.gen_mode == "pool":
                state.grid_pool.resume()
            asyncio.create_task(_broadcast_sync(
                {"params": {"img2img_enabled": state.img2img_enabled}}, exclude=ws,
            ))
        elif msg_type == "camera_start":
            _clear_camera_state()
            state.set_source_mode("camera")
            print("Camera: source mode set to camera")
        elif msg_type == "camera_stop":
            _clear_camera_state()
            state.set_source_mode("none")
            print("Camera: source mode cleared")
        elif msg_type == "output_size":
            _handle_output_size(msg)
        elif msg_type == "output_aspect":
            _handle_output_aspect(msg)
        elif msg_type == "start_sender":
            sender_type = msg.get("sender", "")
            sender_name = msg.get("name", "EASEy-GLYPH")
            ok = state.start_output_sender(sender_type, sender_name)
            try:
                await ws.send_text(json.dumps({
                    "type": "sender_result",
                    "sender": sender_type,
                    "ok": ok,
                }))
            except (WebSocketDisconnect, RuntimeError):
                return
        elif msg_type == "stop_sender":
            sender_type = msg.get("sender", "")
            state.stop_output_sender(sender_type)
            if not state.output_enabled:
                _stop_output_thread()
        elif msg_type == "stop_output":
            state.stop_output_senders()
            if not state.output_enabled:
                _stop_output_thread()
        elif msg_type == "start_recording":
            output_dir = msg.get("dir", "output/recording")
            state.start_recording(output_dir)
        elif msg_type == "stop_recording":
            state.stop_recording()
            if not state.output_enabled:
                _stop_output_thread()
        elif msg_type == "target_fps":
            state.target_fps = max(1, min(120, float(msg.get("fps", 60))))
            asyncio.create_task(_broadcast_sync(
                {"target_fps": state.target_fps}, exclude=ws,
            ))
        elif msg_type == "preview_fraction":
            state.preview_fraction = max(0.0, min(1.0, float(msg.get("value", 1.0))))
            asyncio.create_task(_broadcast_sync(
                {"preview_fraction": state.preview_fraction}, exclude=ws,
            ))
        elif msg_type == "preset":
            global _active_preset
            _active_preset = str(msg.get("name", "Default"))
            asyncio.create_task(_broadcast_sync(
                {"active_preset": _active_preset}, exclude=ws,
            ))
        elif msg_type == "ab_enable":
            state.ab_preset_a = _resolve_preset_snapshot(msg.get("preset_a", {}))
            state.ab_preset_b = _resolve_preset_snapshot(msg.get("preset_b", {}))
            state.ab_preset_a_name = str(msg.get("preset_a", {}).get("name", ""))
            state.ab_preset_b_name = str(msg.get("preset_b", {}).get("name", ""))
            state.ab_enabled = True
            asyncio.create_task(_broadcast_sync({
                "ab_enabled": True,
                "ab_preset_a": state.ab_preset_a_name,
                "ab_preset_b": state.ab_preset_b_name,
            }, exclude=ws))
        elif msg_type == "ab_set_preset":
            side = msg.get("side", "a")
            preset_data = msg.get("preset", {})
            snapshot = _resolve_preset_snapshot(preset_data)
            name = str(preset_data.get("name", ""))
            if side == "a":
                state.ab_preset_a = snapshot
                state.ab_preset_a_name = name
            else:
                state.ab_preset_b = snapshot
                state.ab_preset_b_name = name
            asyncio.create_task(_broadcast_sync({
                "ab_enabled": state.ab_enabled,
                "ab_preset_a": state.ab_preset_a_name,
                "ab_preset_b": state.ab_preset_b_name,
            }, exclude=ws))
        elif msg_type == "ab_disable":
            state.ab_enabled = False
            asyncio.create_task(_broadcast_sync({
                "ab_enabled": False,
            }, exclude=ws))
        elif msg_type == "midi_learn":
            state.midi_learn_target = msg.get("key")
            state.midi_learn_is_trigger = False
        elif msg_type == "midi_learn_trigger":
            state.midi_learn_target = msg.get("key")
            state.midi_learn_is_trigger = True
        elif msg_type == "midi_cancel_learn":
            state.midi_learn_target = None
        elif msg_type == "midi_clear":
            key = msg.get("key")
            if key:
                state.midi_mappings.pop(key, None)
                state.midi_triggers.pop(key, None)
                state.save_midi()
        elif msg_type == "midi_set":
            _handle_midi_set(msg)
        elif msg_type == "midi_list_ports":
            ports, midi_error = MIDIInput.list_ports()
            current = state.midi_input.port_name if state.midi_input else None
            resp = {
                "type": "midi_ports",
                "ports": ports,
                "current": current,
            }
            if midi_error:
                resp["error"] = midi_error
            try:
                await ws.send_text(json.dumps(resp))
            except (WebSocketDisconnect, RuntimeError):
                return
        elif msg_type == "midi_select_port":
            port_name = msg.get("port")
            state.stop_midi()
            if port_name:
                state.start_midi(port_name)


def _handle_param(msg: dict):
    """Update an enhancement parameter from a slider/toggle message."""
    key = msg.get("key")
    value = msg.get("value")
    if key is None or value is None:
        return

    enh = state.enh
    if key == "fg_brightness":
        enh.fg_brightness = float(value)
    elif key == "sharpen":
        enh.sharpen = float(value)
    elif key == "edge_enhance":
        if value is True:
            enh.edge_enhance = "all"
            state.edge_mode_selected = "all"
        elif value is False:
            enh.edge_enhance = "off"
            state.edge_mode_selected = "off"
        elif str(value) in EDGE_KERNELS or str(value) == "off":
            enh.edge_enhance = str(value)
            state.edge_mode_selected = str(value)
        else:
            enh.edge_enhance = "off"
            state.edge_mode_selected = "off"
    elif key == "edge_intensity":
        enh.edge_intensity = max(0.0, min(1.0, float(value)))
    elif key == "saturation":
        enh.saturation = float(value)
    elif key == "contrast":
        enh.contrast = float(value)
    elif key == "auto_contrast":
        enh.auto_contrast = float(value) if value is not None and value is not False else None
    elif key == "posterize":
        enh.posterize = int(value) if value is not None and value > 0 else None
    elif key == "gamma":
        enh.gamma = float(value)
    elif key == "grain":
        enh.grain = float(value)
    elif key == "scanlines":
        enh.scanlines = int(value) if value is not None and value > 0 else None
    elif key == "pixel_upscale":
        enh.pixel_upscale = bool(value)
    elif key == "render_scale":
        enh.render_scale = max(1, int(value))
    elif key == "interp":
        if value in INTERP_METHODS:
            enh.interp = value
    elif key == "steps":
        state.grid_pool.update_steps(max(1, int(value)))
    elif key == "solver":
        if str(value) in ("euler", "heun", "midpoint"):
            state.grid_pool.update_solver(str(value))
    elif key == "time_schedule":
        if str(value) in ("uniform", "cosine", "poly"):
            state.grid_pool.update_schedule(str(value))
    elif key == "cfg_scale":
        state.grid_pool.update_cfg(cfg_scale=max(0.0, float(value)))
    elif key == "cfg_audio":
        if str(value) in ("random", "live"):
            state.grid_pool.update_cfg(cfg_audio=str(value))
    elif key == "cfg_features":
        value_str = str(value)
        _CFG_FEATURE_MASKS = {
            "all": None,
            "spectral4": [0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0],  # phase, centroid, flux, flatness
        }
        if value_str in _CFG_FEATURE_MASKS:
            state.grid_pool.update_cfg(cfg_feature_mask=_CFG_FEATURE_MASKS[value_str])
        elif value_str.startswith("["):
            try:
                mask = json.loads(value_str)
                if isinstance(mask, list) and len(mask) == 12:
                    state.grid_pool.update_cfg(cfg_feature_mask=[float(m) for m in mask])
            except (json.JSONDecodeError, ValueError):
                pass
    elif key == "onset_threshold":
        state.onset_threshold = float(value)
        if state.capture is not None:
            state.capture.beat_detector.set_onset_threshold(float(value))
    elif key == "superres":
        state.superres_enabled = bool(value)
    elif key == "img2img_strength":
        state.img2img_strength = max(0.0, min(1.0, float(value)))
    elif key == "opacity":
        enh.opacity = max(0.0, min(1.0, float(value)))
    elif key == "alpha_curve":
        enh.alpha_curve = max(0.1, min(5.0, float(value)))
    elif key == "alpha_preview":
        state.alpha_preview = bool(value)
    elif key == "feedback_strength":
        state.feedback_strength = max(0.0, min(1.0, float(value)))
    elif key == "persistence":
        state.persistence = max(0.0, min(1.0, float(value)))


def _handle_mapping(msg: dict):
    """Update an audio-to-effect mapping."""
    key = msg.get("key")
    source = msg.get("source", "none")
    if key is None:
        return

    if source == "none":
        state.mappings.pop(key, None)
    else:
        curve = str(msg.get("curve", "linear"))
        if curve not in ("linear", "ease_in", "ease_out", "ease_in_out", "exponential", "logarithmic", "threshold"):
            curve = "linear"
        lo_thresh = max(0.0, min(1.0, float(msg.get("lo_thresh", 0.0))))
        hi_thresh = max(0.0, min(1.0, float(msg.get("hi_thresh", 1.0))))
        state.mappings[key] = AudioMapping(
            source=source,
            min_val=float(msg.get("min", 0)),
            max_val=float(msg.get("max", 1)),
            invert=bool(msg.get("invert", False)),
            curve=curve,
            lo_thresh=lo_thresh,
            hi_thresh=hi_thresh,
        )


def _handle_mode(msg: dict):
    """Switch audio mode."""
    mode = msg.get("mode")
    if mode == "live":
        state.start_live()
    elif mode == "file":
        state.stop_capture()
        state.mode = "file"
    elif mode == "idle":
        state.stop_capture()
        state.mode = "idle"


ASPECT_RATIOS = {
    "1:1": (1, 1),
    "4:3": (4, 3),
    "16:9": (16, 9),
    "9:16": (9, 16),
    "16:10": (16, 10),
    "21:9": (21, 9),
}


def _handle_aspect(msg: dict):
    """Change preview aspect ratio."""
    aspect = msg.get("aspect", "1:1")
    ratio = ASPECT_RATIOS.get(aspect, (1, 1))
    base = state.preview_base
    if ratio[0] >= ratio[1]:
        w = base
        h = int(base * ratio[1] / ratio[0])
    else:
        h = base
        w = int(base * ratio[0] / ratio[1])
    state.preview_size = (w, h)


def _handle_output_size(msg: dict):
    """Set output resolution directly."""
    w = int(msg.get("width", 1920))
    h = int(msg.get("height", 1080))
    w = max(32, min(7680, w))
    h = max(32, min(4320, h))
    state.output_size = (w, h)
    state.output_base = max(w, h)
    _stop_output_thread()  # drain in-flight frame before sender restart
    state.restart_output_senders()


def _handle_output_aspect(msg: dict):
    """Change output aspect ratio using the output base dimension."""
    aspect = msg.get("aspect", "16:9")
    ratio = ASPECT_RATIOS.get(aspect, (16, 9))
    base = state.output_base
    if ratio[0] >= ratio[1]:
        w = base
        h = int(base * ratio[1] / ratio[0])
    else:
        h = base
        w = int(base * ratio[0] / ratio[1])
    state.output_size = (w, h)
    _stop_output_thread()  # drain in-flight frame before sender restart
    state.restart_output_senders()


def _handle_midi_set(msg: dict):
    """Manually create/edit a MIDI mapping."""
    key = msg.get("key")
    if not key:
        return
    cc = int(msg.get("cc", 0))
    channel = int(msg.get("channel", 0))
    msg_type = str(msg.get("msg_type", "cc"))
    is_trigger = bool(msg.get("trigger", False))

    if is_trigger:
        state.midi_triggers[key] = MIDIMapping(cc=cc, channel=channel, msg_type=msg_type)
    else:
        lo = float(msg.get("min", 0.0))
        hi = float(msg.get("max", 1.0))
        invert = bool(msg.get("invert", False))
        state.midi_mappings[key] = MIDIMapping(
            cc=cc, channel=channel, msg_type=msg_type, min_val=lo, max_val=hi, invert=invert,
        )
    state.save_midi()
