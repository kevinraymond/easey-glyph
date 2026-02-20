# EASEy-GLYPH

Audio-reactive generative visuals for live performance. Turns your art into real-time visuals made from Unicode glyphs, driven by music.

<table>
<tr>
<td align="center"><img src="https://huggingface.co/kjraym/easey-glyph/resolve/main/images/abstract.png" width="180"><br><b>Abstract</b></td>
<td align="center"><img src="https://huggingface.co/kjraym/easey-glyph/resolve/main/images/nature.png" width="180"><br><b>Nature</b></td>
<td align="center"><img src="https://huggingface.co/kjraym/easey-glyph/resolve/main/images/ukiyoe.png" width="180"><br><b>Ukiyo-e</b></td>
<td align="center"><img src="https://huggingface.co/kjraym/easey-glyph/resolve/main/images/albums.png" width="180"><br><b>Albums</b></td>
</tr>
<tr>
<td align="center"><img src="https://huggingface.co/kjraym/easey-glyph/resolve/main/images/pixel.png" width="180"><br><b>Pixel</b></td>
<td align="center"><img src="https://huggingface.co/kjraym/easey-glyph/resolve/main/images/botanical.png" width="180"><br><b>Botanical</b></td>
<td align="center"><img src="https://huggingface.co/kjraym/easey-glyph/resolve/main/images/darkpsy.png" width="180"><br><b>Darkpsy</b></td>
<td></td>
</tr>
</table>

## What is this?

Initially released as a Stable Diffusion experiment, [EASE](https://github.com/kevinraymond/ease) was my first deep dive into audio-reactive visuals. It worked, but it was way too heavy for anything without a beefy CUDA GPU.

EASEy-GLYPH starts from the opposite end: an attempt to make real-time generative visuals accessible on Apple M1 hardware. That constraint shaped everything. Instead of generating pixels directly, a small model generates 32x32 grids of colored Unicode glyphs - chunky, graphic, with real transparency and color depth. The low resolution isn't a limitation, it's the aesthetic. An optional super-resolution CNN can upscale to 256x256 when the hardware budget allows.

Training is as easy as I could make it. You feed it ~10,000 images in whatever style you want. Your own renders, photos, textures, AI-generated images, etc., and it learns to generate infinite variations in that style. Your aesthetic, not someone else's.

_NOTE: AI-assisted content below_

## Features

- Real-time ~60fps on Apple Silicon (M1+) and NVIDIA GPUs
- Browser UI at `localhost:8420`
- 12 audio features (bass, mids, treble, onset, spectral centroid, etc.) mapped to 11 visual effects with per-mapping Lo/Hi range
- Beat detection with grid morphing
- RGBA transparency for layering over other visuals (real alpha, not black backgrounds)
- NDI, Syphon, and Spout output for VJ software (Resolume, VDMX, OBS, etc.)
- MIDI controller support with learn mode
- Camera and video file input (img2img)
- Optional super-resolution (32x32 to 256x256)
- 17 built-in presets with A/B comparison mode
- Multiple aspect ratios (1:1, 4:3, 16:9, 9:16)

## Quick Start

```bash
git clone https://github.com/kevinraymond/easey-glyph.git
cd easey-glyph
uv sync

# Download pre-trained models from HuggingFace
uv run pip install huggingface-hub
huggingface-cli download kjraym/easey-glyph --local-dir ./models

# Run with the abstract realtime model
uv run python -m easey_glyph \
    --checkpoint models/easey-glyph-flow-abstract-v2-realtime.safetensors

# Or add super-resolution for sharper output (trades some fps)
uv run python -m easey_glyph \
    --checkpoint models/easey-glyph-flow-abstract-v2-realtime.safetensors \
    --superres models/easey-glyph-superres-abstract-v2.safetensors
```

Opens a browser UI at **http://localhost:8420**. Connect audio (file or system monitor), map effects to audio features, and perform.

See the [HuggingFace repo](https://huggingface.co/kjraym/easey-glyph) for all model variants and details.

## Installation

Requires Python 3.10-3.12.

```bash
uv sync
```

### Optional extras

Install video output, MIDI, and other extras as needed:

```bash
# NDI output (network video)
uv sync --extra ndi

# Syphon output (macOS inter-app video)
uv sync --extra syphon

# Spout output (Windows inter-app video)
uv sync --extra spout

# MIDI controller input
uv sync --extra midi

# CoreML backend (macOS Apple Silicon)
uv sync --extra coreml

# HuggingFace model loading (safetensors)
uv sync --extra hf

# Everything
uv sync --all-extras
```

## Train Your Own

The whole pipeline — preprocessing, training, and super-resolution — runs on a single GPU and trains on as few as 10,000 images. Training takes roughly 4-8 hours on an RTX 3090/4090.

See the [pipeline walkthrough](docs/index.md) for the full step-by-step guide.

## Pre-trained Models

7 model variants are available on [HuggingFace](https://huggingface.co/kjraym/easey-glyph): Abstract, Nature, Ukiyo-e, Albums, Pixel, Botanical, and Darkpsy. Each variant includes a base model (unconditional), a realtime model (audio-reactive via CFG), and an optional super-resolution upscaler.

## License

[MIT](LICENSE)
