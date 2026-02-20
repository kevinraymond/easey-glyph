#!/usr/bin/env bash
# Generate dataset images using two ComfyUI instances (one per GPU).
#
# Usage:
#   bash scripts/generate_dataset_dual.sh <theme> [model] [invert_fraction]
#
#   bash scripts/generate_dataset_dual.sh nature           # 9b model (default)
#   bash scripts/generate_dataset_dual.sh pixel 4b         # 4b model (faster)
#   bash scripts/generate_dataset_dual.sh darkpsy 9b 0.3   # 30% random inversion
#
# Prerequisites:
#   - Two GPUs available (CUDA devices 0 and 1)
#   - ComfyUI installed at /home/kevin/git/comfyui
#
# The script:
#   1. Starts two ComfyUI servers (ports 8188, 8189)
#   2. Runs two generation scripts in parallel (5K images each, different seed offsets)
#   3. Merges both output dirs, resizes to 256x256, creates zip

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash scripts/generate_dataset_dual.sh <theme> [model] [invert_fraction]

Runs two ComfyUI instances (one per GPU) to generate 10K images in parallel,
then merges and resizes to 256x256 zip.

Themes:
  abstract        Neon particles, tubes, spirals, marble on dark backgrounds
  nature          Diatoms, mineral thin-sections, satellite imagery, coral, frost crystals
  ukiyoe          Waves, cherry blossoms, cranes, koi — flat color, bold outlines
  albums          Psychedelic posters, glitch art, bold graphic design, vinyl grooves
  architecture    Brutalist facades, spiral stairs, gothic vaults, Islamic tiles
  pixel           16-bit RPG overworlds, sprite sheets, side-scrollers, limited palettes
  botanical       Pressed flowers, scientific illustration, cross-sections, light backgrounds
  darkpsy         B&W sacred geometry, glitch, biomechanical — neon accents, mixed polarity

Models:
  9b   flux-2-klein-9b-fp8 (default, better quality)
  4b   flux-2-klein-4b     (faster)

Invert fraction (optional):
  0.0-1.0  Fraction of images to randomly invert during resize (default: 0.0)

Examples:
  bash scripts/generate_dataset_dual.sh nature
  bash scripts/generate_dataset_dual.sh pixel 4b
  bash scripts/generate_dataset_dual.sh darkpsy 9b 0.3

Output: datasets/<theme>-256-10k.zip
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

THEME="${1:?$(usage >&2; echo "Error: theme argument required")}"
MODEL="${2:-9b}"
INVERT_FRACTION="${3:-0.0}"

COMFYUI_DIR="/home/kevin/git/comfyui"
COMFYUI_PYTHON="${COMFYUI_DIR}/.venv/bin/python"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
GEN_SCRIPT="${SCRIPT_DIR}/generate_dataset_comfyui.py"

GPU0_DIR="${PROJECT_DIR}/datasets/${THEME}-512-gpu0"
GPU1_DIR="${PROJECT_DIR}/datasets/${THEME}-512-gpu1"
OUTPUT_ZIP="${PROJECT_DIR}/datasets/${THEME}-256-10k.zip"

PORT0=8188
PORT1=8189

NUM_PER_GPU=5000
COMFYUI_PIDS=()

cleanup() {
    echo ""
    echo "Cleaning up ComfyUI processes..."
    for pid in "${COMFYUI_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
            echo "  Stopped PID $pid"
        fi
    done
}
trap cleanup EXIT

# -----------------------------------------------------------------------
# Phase A: Start two ComfyUI servers
# -----------------------------------------------------------------------
echo "=== Phase A: Starting ComfyUI servers ==="
echo "Theme: ${THEME}, Model: ${MODEL}"
echo ""

start_comfyui() {
    local gpu=$1
    local port=$2
    echo "Starting ComfyUI on GPU ${gpu}, port ${port}..."
    cd "$COMFYUI_DIR"
    CUDA_VISIBLE_DEVICES=$gpu "$COMFYUI_PYTHON" main.py --listen --port "$port" > "/tmp/comfyui_gpu${gpu}.log" 2>&1 &
    local pid=$!
    COMFYUI_PIDS+=("$pid")
    echo "  PID: $pid (log: /tmp/comfyui_gpu${gpu}.log)"
}

wait_for_server() {
    local port=$1
    local max_wait=120
    local elapsed=0
    echo "Waiting for ComfyUI on port ${port}..."
    while [ $elapsed -lt $max_wait ]; do
        if curl -s "http://127.0.0.1:${port}/system_stats" > /dev/null 2>&1; then
            echo "  ComfyUI on port ${port} is ready (${elapsed}s)"
            return 0
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done
    echo "ERROR: ComfyUI on port ${port} did not start within ${max_wait}s"
    echo "Check log: /tmp/comfyui_gpu*.log"
    exit 1
}

start_comfyui 0 $PORT0
start_comfyui 1 $PORT1

wait_for_server $PORT0
wait_for_server $PORT1

echo ""
echo "Both ComfyUI servers are ready."
echo ""

# -----------------------------------------------------------------------
# Phase B: Generate images in parallel
# -----------------------------------------------------------------------
echo "=== Phase B: Generating ${THEME} images (${NUM_PER_GPU} per GPU, model=${MODEL}) ==="

cd "$PROJECT_DIR"

echo "GPU 0: ${NUM_PER_GPU} images -> ${GPU0_DIR} (seed-offset 0)"
uv run python "$GEN_SCRIPT" \
    --theme "$THEME" \
    --model "$MODEL" \
    --num-images "$NUM_PER_GPU" \
    --output "$GPU0_DIR" \
    --server "127.0.0.1:${PORT0}" \
    --seed-offset 0 &
GEN_PID0=$!

echo "GPU 1: ${NUM_PER_GPU} images -> ${GPU1_DIR} (seed-offset ${NUM_PER_GPU})"
uv run python "$GEN_SCRIPT" \
    --theme "$THEME" \
    --model "$MODEL" \
    --num-images "$NUM_PER_GPU" \
    --output "$GPU1_DIR" \
    --server "127.0.0.1:${PORT1}" \
    --seed-offset "$NUM_PER_GPU" &
GEN_PID1=$!

echo ""
echo "Generation running (PIDs: ${GEN_PID0}, ${GEN_PID1})"
echo "Monitor progress:"
echo "  ls ${GPU0_DIR}/*.png 2>/dev/null | wc -l"
echo "  ls ${GPU1_DIR}/*.png 2>/dev/null | wc -l"
echo ""

# Wait for both to finish
FAIL=0
wait $GEN_PID0 || FAIL=1
wait $GEN_PID1 || FAIL=1

if [ $FAIL -ne 0 ]; then
    echo "WARNING: One or both generation processes exited with errors."
    echo "Check output dirs for partial results."
fi

GPU0_COUNT=$(ls "${GPU0_DIR}"/*.png 2>/dev/null | wc -l || echo 0)
GPU1_COUNT=$(ls "${GPU1_DIR}"/*.png 2>/dev/null | wc -l || echo 0)

echo ""
echo "Generation complete:"
echo "  GPU 0: ${GPU0_COUNT} images"
echo "  GPU 1: ${GPU1_COUNT} images"
echo ""

# -----------------------------------------------------------------------
# Phase C: Merge, resize, zip
# -----------------------------------------------------------------------
echo "=== Phase C: Resize to 256x256 and create zip ==="

uv run python "${SCRIPT_DIR}/resize_and_zip.py" \
    --input-dirs "$GPU0_DIR" "$GPU1_DIR" \
    --output "$OUTPUT_ZIP" \
    --size 256 \
    --invert-fraction "$INVERT_FRACTION"

echo ""
echo "=== Done! ==="
echo "Output: ${OUTPUT_ZIP}"
echo ""
echo "Next steps:"
echo "  1. Preprocess:"
echo "     uv run scripts/preprocess_dataset.py --input ${OUTPUT_ZIP}"
echo "  2. Train:"
echo "     uv run scripts/train_glyph.py --config configs/glyph_base.yaml --data ${PROJECT_DIR}/datasets/${THEME}/glyph-${THEME}-10k.pt"
