#!/usr/bin/env python3
"""Render audio-reactive Braille glyph art video.

Beat-triggered: uses the full AudioAnalyzer + BeatDetector pipeline (same code
that runs live) to detect beats, generates a fresh glyph grid per beat, hard-cuts
between them. Dot size/visibility/brightness modulate continuously.

Usage:
    uv run scripts/render_audio_reactive.py \
        --audio music.wav \
        --checkpoint training-runs/abstract-v2/final.pt \
        --output output/reactive.mp4

    # Lower onset sensitivity = more beats = more image changes
    uv run scripts/render_audio_reactive.py \
        --audio music.wav \
        --checkpoint training-runs/abstract-v2/final.pt \
        --output output/reactive.mp4 \
        --onset-threshold 1.5
"""

import argparse
import bisect
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from easey_glyph.audio.features import analyze_file
from easey_glyph.audio.reactive import frame_to_render_params, render_pixel_frame
from easey_glyph.render.pipeline import (
    EnhancementParams,
    INTERP_METHODS,
    apply_enhancements,
    generate_grids,
    load_model,
    scale2x,
    slerp,
)


def main():
    parser = argparse.ArgumentParser(description="Audio-reactive Braille glyph video")
    parser.add_argument("--audio", type=str, required=True, help="Input audio file (WAV/FLAC/OGG)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--output", type=str, default="output/reactive.mp4", help="Output video path")
    parser.add_argument("--fps", type=float, default=30.0, help="Video framerate")
    parser.add_argument("--steps", type=int, default=50, help="Euler ODE steps for sampling")
    parser.add_argument("--resolution", type=int, nargs=2, default=[1920, 1080], metavar=("W", "H"),
                        help="Output video resolution (default: 1920 1080)")
    parser.add_argument("--onset-threshold", type=float, default=1.8,
                        help="Onset detection sensitivity (lower = more beats)")
    parser.add_argument("--render-scale", type=int, default=4,
                        help="Bilinear pre-upscale factor (2=chunky, 4=soft, 8=painterly, 16+=smooth)")

    # Interpolation
    parser.add_argument("--interp", type=str, default="bilinear",
                        choices=["bilinear", "bicubic", "lanczos", "nearest"],
                        help="Step 1 interpolation method (default: bilinear)")

    # Sharpening
    parser.add_argument("--sharpen", type=float, default=0,
                        help="UnsharpMask strength 0-200 (default: 0 = off). Audio-reactive: bass")
    parser.add_argument("--edge-enhance", type=str, default="off",
                        choices=["off", "all", "horizontal", "vertical",
                                 "diagonal_bwd", "diagonal_fwd", "emboss", "outline"],
                        help="Edge enhancement mode (default: off)")

    # Color
    parser.add_argument("--saturation", type=float, default=1.0,
                        help="Base saturation multiplier (default: 1.0). Audio-reactive: mid-band")
    parser.add_argument("--contrast", type=float, default=1.0,
                        help="Base contrast multiplier (default: 1.0). Audio-reactive: bass")
    parser.add_argument("--auto-contrast", type=float, default=None, metavar="CUTOFF",
                        help="Autocontrast with cutoff %% (default: off)")
    parser.add_argument("--posterize", type=int, default=None, metavar="BITS",
                        help="Reduce to N bits per channel, 1-8 (default: off)")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="Gamma correction (default: 1.0)")
    parser.add_argument("--alpha-curve", type=float, default=1.0,
                        help="Alpha power curve. 1.0=unchanged, <1=more opaque, >1=more transparent (default: 1.0)")

    # Stylistic
    parser.add_argument("--grain", type=float, default=0,
                        help="Film grain noise strength 0-20 (default: 0). Audio-reactive: inverse energy")
    parser.add_argument("--scanlines", type=int, default=None, metavar="SPACING",
                        help="Darken every Nth row for CRT look (default: off)")
    parser.add_argument("--pixel-upscale", type=str, default=None, choices=["scale2x"],
                        help="Pixel-art upscaler before standard 2-step (default: off)")
    parser.add_argument("--superres", type=str, default=None,
                        help="Super-resolution CNN checkpoint (32->256 upscale)")

    # Solver settings
    parser.add_argument("--solver", type=str, default="euler", choices=["euler", "heun", "midpoint"],
                        help="ODE solver (default: euler)")
    parser.add_argument("--schedule", type=str, default="uniform", choices=["uniform", "cosine", "poly"],
                        help="Time step schedule (default: uniform)")
    parser.add_argument("--cfg-scale", type=float, default=0.0,
                        help="CFG guidance scale (default: 0 = off)")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build EnhancementParams from CLI args
    enh = EnhancementParams(
        sharpen=args.sharpen,
        edge_enhance=args.edge_enhance,
        saturation=args.saturation,
        contrast=args.contrast,
        auto_contrast=args.auto_contrast,
        posterize=args.posterize,
        gamma=args.gamma,
        grain=args.grain,
        scanlines=args.scanlines,
        pixel_upscale=(args.pixel_upscale == "scale2x"),
        render_scale=args.render_scale,
        interp=args.interp,
        alpha_curve=args.alpha_curve,
    )

    # Output resolution
    res_w, res_h = args.resolution
    base_size = 32
    if enh.pixel_upscale:
        base_size = 64
    inter_size = base_size * enh.render_scale
    interp_name = enh.interp

    # Compute audio-reactive ranges for enhancements
    sharpen_range = (0.3, 1.0) if enh.sharpen > 0 else (0.0, 0.0)
    saturation_range = (0.8, 1.2) if enh.saturation != 1.0 else (1.0, 1.0)
    contrast_range = (0.8, 1.2) if enh.contrast != 1.0 else (1.0, 1.0)
    grain_range = (0.0, 1.0) if enh.grain > 0 else (0.0, 0.0)

    # Build enhancement summary
    enhancements = []
    if args.superres:
        enhancements.append("superres")
    if enh.pixel_upscale:
        enhancements.append("scale2x")
    if enh.sharpen > 0:
        enhancements.append(f"sharpen={enh.sharpen:.0f}")
    if enh.edge_enhance != "off":
        enhancements.append(f"edge={enh.edge_enhance}")
    if enh.saturation != 1.0:
        enhancements.append(f"saturation={enh.saturation}")
    if enh.contrast != 1.0:
        enhancements.append(f"contrast={enh.contrast}")
    if enh.auto_contrast is not None:
        enhancements.append(f"autocontrast={enh.auto_contrast}")
    if enh.posterize is not None:
        enhancements.append(f"posterize={enh.posterize}bit")
    if enh.gamma != 1.0:
        enhancements.append(f"gamma={enh.gamma}")
    if enh.grain > 0:
        enhancements.append(f"grain={enh.grain:.0f}")
    if enh.scanlines is not None:
        enhancements.append(f"scanlines={enh.scanlines}")
    enh_str = f" [{', '.join(enhancements)}]" if enhancements else ""
    scale2x_str = "scale2x -> " if enh.pixel_upscale else ""
    print(f"Output: {res_w}x{res_h} (render 32x32 -> {scale2x_str}{interp_name} {inter_size}x{inter_size} -> nearest){enh_str}")

    # 1. Analyze audio with the real pipeline
    print(f"Analyzing audio: {args.audio}")
    audio_frames, sr = analyze_file(
        args.audio,
        onset_threshold_mult=args.onset_threshold,
    )
    # Analysis rate is sr/block_size (e.g. 44100/1024 ≈ 43 fps)
    analysis_fps = sr / 1024
    duration_sec = len(audio_frames) / analysis_fps
    print(f"Audio: {duration_sec:.1f}s, {len(audio_frames)} analysis frames at {analysis_fps:.1f} fps")

    # Count beats
    beat_indices = [i for i, f in enumerate(audio_frames) if f.is_beat]
    # Always need at least one grid for frame 0
    if not beat_indices or beat_indices[0] != 0:
        beat_indices.insert(0, 0)
    num_beats = len(beat_indices)

    # Estimate BPM from detected beats
    if num_beats > 1 and duration_sec > 0:
        avg_bpm = (num_beats - 1) / duration_sec * 60
    else:
        avg_bpm = audio_frames[-1].bpm if audio_frames else 120.0
    print(f"Detected {num_beats} beats (~{avg_bpm:.0f} BPM)")

    # 2. Generate one grid per beat
    model, model_cfg = load_model(args.checkpoint, device)
    # Build CFG audio tensor if needed
    cfg_audio = None
    if args.cfg_scale > 0:
        cfg_audio = torch.rand(min(num_beats, 8), 12, device=device)
    grids = generate_grids(model, model_cfg, num_beats, args.steps, device,
                           solver=args.solver, schedule=args.schedule,
                           audio=cfg_audio, cfg_scale=args.cfg_scale)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Optionally load super-resolution model
    superres_model = None
    if args.superres:
        from easey_glyph.model.superres import load_superres
        superres_model = load_superres(args.superres, device)

    # 3. Build beat times for SLERP interpolation
    beat_times = [beat_indices[k] / analysis_fps for k in range(num_beats)]

    # 4. Resample to video fps
    video_num_frames = int(duration_sec * args.fps)
    print(f"Video: {video_num_frames} frames at {args.fps} fps")

    # 5. Stream frames to ffmpeg
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "rgba",
        "-s", f"{res_w}x{res_h}",
        "-r", str(args.fps),
        "-i", "-",
        "-i", args.audio,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "medium",
        "-crf", "18",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        str(output_path),
    ]

    print(f"Rendering {video_num_frames} frames -> {output_path}")
    proc = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    try:
        for vi in range(video_num_frames):
            video_time = vi / args.fps
            ai = min(int(video_time * analysis_fps), len(audio_frames) - 1)
            audio_frame = audio_frames[ai]

            # Find which beat interval we're in, SLERP between grids
            k = max(0, bisect.bisect_right(beat_times, video_time) - 1)
            if k < num_beats - 1:
                t0 = beat_times[k]
                t1 = beat_times[k + 1]
                frac = (video_time - t0) / (t1 - t0) if t1 > t0 else 0.0
                frac = max(0.0, min(1.0, frac))
                grid = slerp(frac, grids[k], grids[k + 1])
            else:
                grid = grids[k]

            # Compute render params from audio frame
            params = frame_to_render_params(
                audio_frame,
                sharpen_range=sharpen_range,
                saturation_range=saturation_range,
                contrast_range=contrast_range,
                grain_range=grain_range,
            )

            # Render 32x32 pixel colors
            frame = render_pixel_frame(grid, params)

            # Super-resolution CNN (32x32 -> 256x256)
            if superres_model is not None:
                rgb = frame[:, :, :3]
                alpha = frame[:, :, 3]
                with torch.no_grad():
                    x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
                    sr = superres_model(x)
                    sr = (sr[0].permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy()
                    # Bilinear-upscale alpha separately (CNN is RGB-only)
                    alpha_t = torch.from_numpy(alpha).unsqueeze(0).unsqueeze(0).float()
                    alpha_up = torch.nn.functional.interpolate(alpha_t, size=(256, 256), mode='bilinear', align_corners=False)
                frame = np.empty((256, 256, 4), dtype=np.uint8)
                frame[:, :, :3] = sr
                frame[:, :, 3] = alpha_up[0, 0].clamp(0, 255).byte().cpu().numpy()
                cur_base = 256
            else:
                cur_base = base_size

            # Optional Scale2x (32x32 -> 64x64, skipped when superres active)
            if superres_model is None and enh.pixel_upscale:
                frame = scale2x(frame)

            # Step 1: interpolation to intermediate size
            cur_inter = cur_base * enh.render_scale
            frame = Image.fromarray(frame, "RGBA")
            interp_method = INTERP_METHODS[interp_name]
            frame = frame.resize((cur_inter, cur_inter), interp_method)

            # Apply enhancements + step 2 (nearest to output) + final effects
            frame = apply_enhancements(frame, params, enh, (res_w, res_h))
            proc.stdin.write(frame.tobytes())

            if (vi + 1) % 100 == 0 or vi == video_num_frames - 1:
                print(f"  Frame {vi + 1}/{video_num_frames}", flush=True)

        proc.stdin.close()
        proc.wait()
        if proc.returncode != 0:
            print(f"ffmpeg error:\n{proc.stderr.read().decode()}", file=sys.stderr)
            sys.exit(1)
    except BrokenPipeError:
        proc.wait()
        print(f"ffmpeg pipe broke:\n{proc.stderr.read().decode()}", file=sys.stderr)
        sys.exit(1)

    print(f"Done! Output: {output_path}")


if __name__ == "__main__":
    main()
