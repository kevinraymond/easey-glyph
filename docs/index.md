---
layout: default
title: Pipeline Walkthrough
---

## What is EASEy-GLYPH?

EASEy-GLYPH turns your art into live, audio-reactive visuals made entirely from text characters. You give it a collection of images, it learns to generate new ones as grids of Unicode block and Braille glyphs with color, and then you perform with them live — mapped to music in real time.

The whole pipeline runs on a single GPU and trains on as few as 10,000 images.

---

## The Pipeline

```
Your Art → Glyph Encoding → Train Model → Perform Live
 (images)   (32x32 grid of     (learns to     (audio-reactive
             colored glyphs)    generate new)   visuals)
```

Each step below shows you what happens and gives you the command to do it.

---

## Step 1: Create Your Art

Start with a collection of images — photographs, paintings, digital art, AI-generated images, anything. Any resolution works; the pipeline resizes everything to 256x256 internally.

For best results, aim for **~10,000 images** with a consistent visual theme. More images = more variety in the output.

**Prepare a dataset:**

Put your images in a folder and ZIP them up:

```bash
zip -r datasets/my-art.zip path/to/my/images/
```

---

## Step 2: Encode to Glyphs

This is where the magic starts. Each 256x256 image gets broken into a **32x32 grid** of cells. Each cell becomes:

- A **Unicode glyph** (block element or Braille character) that matches the cell's light/dark pattern
- A **foreground color** (the bright parts)
- A **background color** (the dark parts)
- **Alpha transparency** derived from luminance (dark = transparent, bright = opaque)

The result is a compact 16-channel grid: 8 channels for the glyph identity (as a PCA embedding), 4 for foreground RGBA, and 4 for background RGBA.

<!-- TODO: add sample images (original → pixel render → glyph render) -->

The **pixel render** shows the flat color grid (what the model actually sees). The **glyph render** shows it with actual Unicode characters — that blocky, textural quality is the aesthetic.

**Preprocess your dataset:**

```bash
uv run scripts/preprocess_dataset.py --input datasets/my-art-256-10k.zip
# -> datasets/my-art/glyph-my-art-10k.pt (auto-derived)
```

---

## Step 3: Train the Model

Now the model learns to generate new glyph grids that look like yours. It uses **flow matching** — a generative technique where the model learns to transform random noise into your glyph grids.

The model is intentionally small (34M parameters) so it trains fast and runs in real time.

**Train:**

```bash
uv run scripts/train_glyph.py \
  --config configs/glyph_base.yaml \
  --data datasets/my-art/glyph-my-art-10k.pt
# -> training-runs/my-art/ (auto-derived)
```

Training takes roughly **4-8 hours on a single GPU** (RTX 3090/4090) for 15,000 kimg. You'll see loss drop to ~0.13 when it's converged.

---

## Step 4: Super-Resolution (Optional)

The glyph grid is only 32x32 pixels — great for the textural glyph look, but sometimes you want more detail. The **super-resolution CNN** (779K params) upscales the pixel render from 32x32 to 256x256, recovering fine detail while keeping the glyph character.

<!-- TODO: add sample images (pixel render vs super-resolution) -->

**Prepare superres training data:**

```bash
uv run scripts/preprocess_superres.py --input datasets/my-art-256-10k.zip
# -> datasets/my-art/superres-my-art-10k.pt (auto-derived)
```

**Train superres:**

```bash
uv run scripts/train_superres.py \
  --data datasets/my-art/superres-my-art-10k.pt
# -> training-runs/superres-my-art/ (auto-derived)
```

---

## Step 5: Perform Live

Launch the live server and connect it to your audio:

```bash
uv run python -m easey_glyph \
  --checkpoint training-runs/my-art/final.pt \
  --superres training-runs/superres-my-art/final.pt
```

Open **http://localhost:8420** in your browser. You get:

- **Real-time generation** at ~60fps
- **12 audio features** (bass, mid, treble, onset, spectral centroid, etc.)
- **11 visual effects** mappable to any audio feature
- **Beat detection** for grid morphing
- **MIDI control** for hands-on tweaking
- **NDI/Syphon/Spout output** for VJ software integration
- **Transparent output** (RGBA) for layering over other visuals

See the [main README](https://github.com/kevinraymond/easey-glyph) for full documentation on the live server.

---

## Quick Reference

All commands in one place:

```bash
# 1. Preprocess your art into glyph grids
uv run scripts/preprocess_dataset.py --input datasets/my-art-256-10k.zip
# -> datasets/my-art/glyph-my-art-10k.pt

# 2. Train the generative model
uv run scripts/train_glyph.py \
  --config configs/glyph_base.yaml \
  --data datasets/my-art/glyph-my-art-10k.pt
# -> training-runs/my-art/

# 3. (Optional) Train super-resolution
uv run scripts/preprocess_superres.py --input datasets/my-art-256-10k.zip
# -> datasets/my-art/superres-my-art-10k.pt

uv run scripts/train_superres.py \
  --data datasets/my-art/superres-my-art-10k.pt
# -> training-runs/superres-my-art/

# 4. Launch the live server
uv run python -m easey_glyph \
  --checkpoint training-runs/my-art/final.pt \
  --superres training-runs/superres-my-art/final.pt

# Generate sample images for this walkthrough
uv run scripts/generate_pipeline_samples.py \
  --input datasets/my-art-256-10k.zip \
  --output docs/assets/samples/my-art \
  --num-samples 6 \
  --superres training-runs/superres-my-art/final.pt
```
