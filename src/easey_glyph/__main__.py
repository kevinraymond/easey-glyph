"""EASEy-GLYPH live control panel.

Usage:
    python -m easey_glyph --checkpoint training-runs/abstract-v2/final.pt
    uv run python -m easey_glyph --checkpoint training-runs/abstract-v2/final.pt
"""

import argparse
import logging
import webbrowser

import torch


def main():
    parser = argparse.ArgumentParser(description="EASEy-GLYPH live control panel")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--port", type=int, default=8420, help="Server port (default: 8420)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cuda, cpu")
    parser.add_argument("--steps", type=int, default=8, help="Initial ODE steps for grid generation")
    parser.add_argument("--solver", type=str, default="euler", choices=["euler", "heun", "midpoint"],
                        help="ODE solver (default: euler). Heun/midpoint use 2 model evals per step.")
    parser.add_argument("--schedule", type=str, default="uniform", choices=["uniform", "cosine", "poly"],
                        help="Time step schedule (default: uniform). Cosine concentrates at boundaries.")
    parser.add_argument("--cfg-scale", type=float, default=0.0,
                        help="CFG guidance scale (default: 0 = off). Doubles compute per step.")
    parser.add_argument("--cfg-audio", type=str, default="random", choices=["random", "live"],
                        help="CFG audio source (default: random). Live uses current audio features.")
    parser.add_argument("--preview-size", type=int, default=512, help="Preview resolution (default: 512)")
    parser.add_argument("--superres", type=str, default=None,
                        help="Super-resolution CNN checkpoint (32->256 upscale)")
    parser.add_argument("--no-open", action="store_true", help="Don't auto-open browser")
    parser.add_argument("--output-size", type=int, nargs=2, default=[1920, 1080],
                        metavar=("W", "H"), help="Output resolution (default: 1920 1080)")
    parser.add_argument("--target-fps", type=int, default=60,
                        help="Render loop target FPS (default: 60)")
    parser.add_argument("--ndi", action="store_true", help="Auto-start NDI sender")
    parser.add_argument("--ndi-name", type=str, default="EASEy-GLYPH",
                        help="NDI sender name (default: EASEy-GLYPH)")
    parser.add_argument("--syphon", action="store_true", help="Auto-start Syphon server (macOS)")
    parser.add_argument("--spout", action="store_true", help="Auto-start Spout sender (Windows)")
    parser.add_argument("--record", type=str, default=None, metavar="DIR",
                        help="Auto-start PNG recording to directory")
    parser.add_argument("--midi", action="store_true", help="Enable MIDI controller input")
    parser.add_argument("--midi-port", type=str, default=None,
                        help="MIDI input port name (default: auto-select first)")
    parser.add_argument("--pool-size", type=int, default=64,
                        help="Grid pool size (default: 64)")
    parser.add_argument("--debug-beat", action="store_true",
                        help="Enable debug logging for beat detection")
    parser.add_argument("--coreml-unet", type=str, default=None,
                        help="CoreML FlowUNet .mlpackage path (macOS, replaces PyTorch)")
    parser.add_argument("--coreml-superres", type=str, default=None,
                        help="CoreML SuperRes .mlpackage path (macOS, replaces PyTorch)")
    args = parser.parse_args()

    # Beat debug logging
    if args.debug_beat:
        logging.basicConfig(level=logging.WARNING, format="%(message)s")
        logging.getLogger("easey_glyph.audio.beat").setLevel(logging.DEBUG)

    # Device selection
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # Enable TF32 + cuDNN autotuner for faster GPU math
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    print(f"EASEy-GLYPH live server")
    print(f"  Device: {device}")
    print(f"  Checkpoint: {args.checkpoint}")

    # Load model
    from easey_glyph.render.pipeline import load_model
    model, model_cfg = load_model(args.checkpoint, device)

    # torch.compile for CUDA (significant speedup, ~30-60s one-time warmup)
    if device.type == "cuda":
        import warnings
        # Inductor warns about online softmax at small attention sizes — harmless
        warnings.filterwarnings("ignore", message=".*Online softmax is disabled.*")
        print("Compiling FlowUNet with torch.compile (one-time warmup)...")
        model = torch.compile(model)

    # Optionally load super-resolution model
    superres_model = None
    if args.superres:
        from easey_glyph.model.superres import load_superres
        superres_model = load_superres(args.superres, device)
        if device.type == "cuda":
            superres_model = superres_model.half()
            print("SuperRes: fp16 enabled")

    # CoreML backends (macOS)
    coreml_model = None
    coreml_superres = None
    if args.coreml_unet or args.coreml_superres:
        from easey_glyph.model.coreml_backend import CoreMLFlowModel, CoreMLSuperRes
        if args.coreml_unet:
            coreml_model = CoreMLFlowModel(args.coreml_unet)
            print(f"CoreML FlowUNet: {args.coreml_unet}")
        if args.coreml_superres:
            coreml_superres = CoreMLSuperRes(args.coreml_superres)
            print(f"CoreML SuperRes: {args.coreml_superres}")

    # Create server state
    from easey_glyph.server.state import ServerState
    state = ServerState(model, model_cfg, device, steps=args.steps, preview_size=args.preview_size,
                         pool_size=args.pool_size)
    state.grid_pool.update_solver(args.solver)
    state.grid_pool.update_schedule(args.schedule)
    state.grid_pool.update_cfg(cfg_scale=args.cfg_scale, cfg_audio=args.cfg_audio)
    if superres_model is not None:
        state.superres_model = superres_model
    if coreml_model is not None:
        state.coreml_model = coreml_model
        state.grid_pool.coreml_model = coreml_model
    if coreml_superres is not None:
        state.coreml_superres = coreml_superres

    # Configure output
    state.target_fps = float(args.target_fps)
    state.output_size = tuple(args.output_size)
    state.output_base = max(args.output_size)

    state.start()

    # Initialize requested output senders
    from easey_glyph.output import HAS_NDI, HAS_SYPHON, HAS_SPOUT
    if args.ndi:
        if HAS_NDI:
            state.start_output_sender("ndi", args.ndi_name)
        else:
            print("  Warning: --ndi requested but cyndilib not installed (pip install cyndilib)")
    if args.syphon:
        if HAS_SYPHON:
            state.start_output_sender("syphon")
        else:
            print("  Warning: --syphon requested but syphon-python not installed")
    if args.spout:
        if HAS_SPOUT:
            state.start_output_sender("spout")
        else:
            print("  Warning: --spout requested but SpoutGL not installed")
    if args.record:
        state.start_recording(args.record)

    # Initialize MIDI
    if args.midi:
        state.start_midi(args.midi_port)

    solver_label = args.solver.capitalize()
    nfe = args.steps * (2 if args.solver in ("heun", "midpoint") else 1)
    cfg_str = f", CFG {args.cfg_scale} ({args.cfg_audio})" if args.cfg_scale > 0 else ""
    print(f"  Solver: {solver_label} {args.steps} steps ({nfe} NFE, {args.schedule}){cfg_str}")
    print(f"  Pool: {state.grid_pool.pool_size} (refill below {state.grid_pool.low_water})")
    print(f"  Target FPS: {int(state.target_fps)}")
    print(f"  Output: {state.output_size[0]}x{state.output_size[1]}")
    if state.output_enabled:
        senders = [s.name for s in state.output_senders]
        if state.png_recorder and state.png_recorder.active:
            senders.append("Recording")
        print(f"  Active outputs: {', '.join(senders)}")

    # Wire up FastAPI app
    from easey_glyph.server.app import app, set_state
    set_state(state)

    import socket
    url = f"http://localhost:{args.port}"
    hostname = socket.gethostname()
    print(f"\n  Open: {url}")
    print(f"  Mobile: http://{hostname}.local:{args.port}\n")

    if not args.no_open:
        webbrowser.open(url)

    # Run uvicorn
    import uvicorn
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning",
                    ws="wsproto")
    finally:
        state.stop()


if __name__ == "__main__":
    main()
