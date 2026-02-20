---
layout: default
title: Presets
---

## Audio Presets

EASEy-GLYPH ships with 16 presets (plus Default) that map audio features to visual effects. Each is designed around a specific concept, using curves and thresholds to shape response musically rather than just toggling effects on/off.

### Design Principles

- **No brightness flashing** — presets use contrast, gamma, and opacity for visual punch instead of `fg_brightness` mapping, which tends to create tiring strobe effects across the board.
- **Curves shape response** — `ease_out` for responsiveness to moderate input, `ease_in` for peak-only reactions, `ease_in_out` for smooth breathing, `exponential` for extreme-only triggers.
- **Floor/ceiling thresholds** — `lo_thresh` ignores the noise floor so effects don't twitch on silence; `hi_thresh` prevents maxing out on loud passages.
- **Tighter output ranges** — subtle, musical response rather than jarring full-range swings.

### Feature-Effect Matching

| Feature | Character | Best Effects |
|---|---|---|
| bass | Kick/low-end punch | contrast, render_scale, posterize, scanlines |
| mid | Melodic/harmonic content | saturation, gamma |
| treble | High-freq shimmer | sharpen, grain |
| rms | Overall loudness | opacity, subtle gamma |
| beat_phase | Cyclic 0-1 ramp between beats | gamma (breathing), saturation, persistence |
| onset_strength | Transient/hit detection | contrast (punch), posterize, sharpen |
| spectral_centroid | Timbral brightness | saturation, gamma |
| spectral_flux | Rate of spectral change | sharpen, contrast, grain |
| spectral_flatness | Noise-like vs tonal | grain, posterize, saturation (inverted) |
| spectral_rolloff | High-freq energy ceiling | gamma, opacity |
| spectral_bandwidth | Spectral spread | saturation, contrast |
| zero_crossing_rate | Percussiveness | sharpen, grain, scanlines |

---

### Preset Reference

| Preset | Concept | Mappings | Design Notes |
|---|---|---|---|
| **Default** | Clean slate — no mappings, no effects | None | Starting point for building your own mappings |
| **Bass Pulse** | Simple kick-driven punch | bass &rarr; contrast (0.95-1.5, ease_out), bass &rarr; gamma (0.9-1.15, ease_out) | Contrast is the primary punch; gamma provides subtle shadow lift. `ease_out` makes it responsive to even moderate kicks. Intentionally minimal — two mappings, one source. |
| **Full Reactive** | All six standard FFT features mapped — the "everything" preset for typical music | onset &rarr; contrast, mid &rarr; saturation, treble &rarr; sharpen, phase &rarr; gamma, bass &rarr; grain, rms &rarr; opacity | Each of the 6 standard features drives a different effect. Phase &rarr; gamma gives breathing, onset &rarr; contrast gives punch, bass &rarr; grain adds texture only on strong kicks (ease_in + high floor), rms &rarr; opacity provides subtle presence/absence. No superres. |
| **Dense Spectrum** | All six spectral analysis features mapped — the "spectral deep dive" preset | flux &rarr; contrast, flatness &rarr; saturation (inv), ZCR &rarr; sharpen, centroid &rarr; gamma, bandwidth &rarr; grain, rolloff &rarr; opacity | Each spectral feature drives an intuitively matched effect: flux = change = punch, flatness inverted = tonal is vivid / noise is muted, ZCR = percussive = sharp, centroid = bright timbre = lifted shadows, bandwidth = wide spectrum = texture, rolloff = high-freq presence = visibility. No superres. |
| **Retro CRT** | CRT aesthetic — scanlines, posterize, grain with bass warmth | flux &rarr; grain (ease_out), rms &rarr; scanlines (inv), flatness &rarr; posterize (ease_in), bass &rarr; contrast (ease_out) | Scanlines invert on RMS — loud = clean, quiet = CRT. Posterize uses ease_in so only strong tonal content triggers heavy crush. Bass &rarr; contrast adds subtle warmth on kicks. Pixel upscale + nearest for chunky look. |
| **Dark Noir** | Desaturated, high contrast, cinematic | bass &rarr; contrast (1.3-1.9), treble &rarr; sharpen (20-90), onset &rarr; gamma (0.65-0.95) | Base gamma is 0.8 (dark); onset lifts toward 0.95 on hits — a cinematic "flash from darkness" that's subtler than brightness mapping. Near-zero saturation (0.15) for monochrome noir feel. Edge enhance on. |
| **Pixel Pop** | Chunky pixel art, vivid colors | bass &rarr; render_scale (inv), mid &rarr; saturation (1.0-2.2), onset &rarr; contrast (1.1-1.5) | Bass kicks make pixels chunkier (inverted scale). Mid drives vivid saturation. Onset &rarr; contrast gives pop on transients instead of brightness flash. Pixel upscale + nearest for hard pixel edges. |
| **Dream Haze** | Soft, dreamy, ambient | phase &rarr; gamma (0.95-1.35, ease_in_out), centroid &rarr; saturation (0.9-1.5), rms &rarr; opacity (0.7-0.95) | Dream fades in/out with RMS energy via opacity (far subtler than brightness). Phase &rarr; gamma gives gentle breathing. High render_scale (8) + lanczos for soft, blobby look. Ambient morph mode. |
| **Flux Storm** | Chaotic, aggressive | flux &rarr; sharpen (0-180), onset &rarr; contrast (1.0-1.7), flatness &rarr; posterize (exponential), bass &rarr; grain (ease_in, hi floor) | Bass rumble = gritty grain texture instead of brightness flash. Exponential posterize means only strongly noisy content triggers bit-crush. Grain uses ease_in + high floor so only strong kicks produce grain bursts. Edge enhance on. |
| **Texture Ride** | Textural response to spectral character | flatness &rarr; grain (ease_out), rolloff &rarr; gamma (ease_in_out), bandwidth &rarr; saturation, bass &rarr; contrast (ease_out) | Noisy content = grainy texture (flatness &rarr; grain). High-freq rolloff lifts/drops shadows via gamma. Wide spectrum = vivid color. Subtle bass &rarr; contrast for gentle punch. Base grain of 6 provides constant texture. |
| **Transient Edge** | ZCR-driven sharpening, edge detail on percussive content | ZCR &rarr; sharpen (0-160), bandwidth &rarr; saturation (0.7-1.8), onset &rarr; contrast (1.0-1.5), rolloff &rarr; gamma (inv, 0.85-1.15) | Percussive content sharpens the image. Gamma inverted on rolloff — high-freq = slightly darker mood, low-freq = slight lift (very tight range for subtlety). Edge enhance on. |
| **SuperRes Snap** | CNN superres toggled by kicks — chunky when quiet, detailed on bass | bass &rarr; superres (ease_in, floor 0.25), onset &rarr; contrast, rms &rarr; saturation | The one preset that IS about superres. ease_in + floor 0.25 means only strong kicks trigger it (threshold >0.5 requires audio ~0.79 after curve). Energy = vivid via saturation instead of brightness. |
| **Phase Breath** | Coordinated breathing with beat phase | phase &rarr; gamma (ease_in_out), phase &rarr; saturation (ease_in_out), onset &rarr; contrast, phase &rarr; persistence (0.1-0.6) | Three effects breathe in sync with beat phase. Persistence ties grid evolution to the cycle — fresh grid (0.1) at beat start, holding (0.6) near end. Onset &rarr; contrast provides clean transient punch. Smoothstep curves for natural feel. |
| **Lo-Fi Glitch** | Lo-fi CRT glitch aesthetic — degraded look | onset &rarr; posterize (ease_out), bass &rarr; scanlines, flux &rarr; grain (ease_out), rolloff &rarr; gamma (ease_in_out) | Hits crush the color depth, kicks add scanlines, spectral change adds grain, rolloff breathes the gamma. Tighter gamma range (0.7-1.15). Pixel upscale + nearest for hard edges. |
| **Feedback Drift** | Gentle evolution with feedback — dreamy continuity | centroid &rarr; saturation (ease_out), rms &rarr; gamma (0.95-1.2), phase &rarr; opacity (ease_in_out) | Energy = subtle shadow lift via gamma instead of brightness flash. Phase &rarr; opacity adds dreamy breathing opacity. Feedback_strength 0.35 + persistence 0.6 for gentle trails. All ranges deliberately gentle. |
| **Feedback Storm** | Aggressive feedback — chaotic evolution with fresh content punching through | bass &rarr; contrast (1.0-1.7), flux &rarr; sharpen (0-130), onset &rarr; grain (ease_in, floor 0.25) | Bass crunch instead of brightness flash. Transient grain bursts fit the "storm" theme — onset drives grain with ease_in so only hard hits produce bursts. Feedback_strength 0.7 for aggressive trails. |
| **Ghost Layer** | Alpha transparency for VJ layering — opacity on bass is the core effect | bass &rarr; opacity (0.15-0.9), centroid &rarr; saturation (0.7-1.6), phase &rarr; gamma (ease_in_out), onset &rarr; contrast (1.0-1.4) | Designed for compositing over other sources. Max opacity capped at 0.9 (never fully opaque). Onset &rarr; contrast provides punch when visible. Alpha preview enabled by default. |
| **Darkpsy Acid** | Psytrance/darkpsy — seven maximally uncorrelated audio voices driving independent visual dimensions | bass &rarr; contrast (1.1-1.8, ease_out), phase &rarr; gamma (0.7-1.0, ease_in_out), mid &rarr; saturation (0.5-1.6, ease_out), flux &rarr; sharpen (0-150, ease_out), flatness &rarr; grain (0-7, ease_out), onset &rarr; posterize (0-4, ease_in), rms &rarr; opacity (0.85-1.0) | Designed from Pearson correlation analysis of psytrance audio. The golden triad — bass, flux, flatness — are pairwise independent (all |r| ≤ 0.09), so kick punch, spectral edges, and noise texture move completely independently. Phase↔onset anti-correlation (-0.80) creates the signature sidechain pump: gamma drops dark on the beat precisely when posterize fires, then breathes back between beats. Mid (acid line) drives saturation independently of the kick (r=-0.10). Dark base: gamma 0.85, contrast 1.3, saturation 0.9, render_scale 3 with edge enhance for gritty darkpsy feel. |
