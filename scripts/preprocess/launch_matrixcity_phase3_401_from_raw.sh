#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/vePFS-buaa/linming/workspace/worldrender/FastVideo-Exp"
cd "${ROOT_DIR}"

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

GPU_NUM="${GPU_NUM:-6}"
MODEL_PATH="${MODEL_PATH:-/vePFS-buaa/linming/workspace/worldrender/Wan-AI/Wan2.2-TI2V-5B-Diffusers}"
RAW_ROOT="${RAW_ROOT:-/vePFS-buaa/yinli/datasets/matrixcity}"
MASK_ROOT="${MASK_ROOT:-/vePFS-buaa/yinli/datasets/matrixcity_mask_merge_new}"
OUT_DIR="${OUT_DIR:-/vePFS-buaa/linming/workspace/worldrender/MATRIXCITY_PARQUET_RGBN_401_RAW}"
STREET_SPLIT="${STREET_SPLIT:-train_dense}"
CLIP_LENGTH="${CLIP_LENGTH:-401}"
WINDOW_LEN="${WINDOW_LEN:-401}"
SAMPLES_PER_FILE="${SAMPLES_PER_FILE:-8}"
FLUSH_FREQUENCY="${FLUSH_FREQUENCY:-8}"

# Notes:
# 1. RAW_ROOT is the new MatrixCity raw root and provides rgb/depth/normal.
# 2. MASK_ROOT still points to the legacy mask clips. Those clips are currently 81-frame only.
# 3. We enable --require_full_control_sequence so this launch fails safe instead of silently
#    padding 81-frame controls to 401 frames. Replace MASK_ROOT with a true 401-frame warp/mask
#    root, or remove that flag only if you intentionally accept repeated control tail frames.

torchrun --nnodes 1 --nproc_per_node "${GPU_NUM}" \
  tools/preprocess_matrixcity_ti2v_controlnet_parquet.py \
  --model_path "${MODEL_PATH}" \
  --mask_root "${MASK_ROOT}" \
  --rgb_root "${RAW_ROOT}" \
  --depth_root "${RAW_ROOT}" \
  --normal_root "${RAW_ROOT}" \
  --output_dir "${OUT_DIR}" \
  --street_split "${STREET_SPLIT}" \
  --clip_length "${CLIP_LENGTH}" \
  --window_len "${WINDOW_LEN}" \
  --use_clip_start_global_ids \
  --first_frame_source rgb \
  --require_masked_rgb \
  --require_normal \
  --fps 16 \
  --max_width 512 \
  --max_height 384 \
  --dtype bf16 \
  --samples_per_file "${SAMPLES_PER_FILE}" \
  --flush_frequency "${FLUSH_FREQUENCY}" \
  --write_vae_latent \
  --require_full_control_sequence
