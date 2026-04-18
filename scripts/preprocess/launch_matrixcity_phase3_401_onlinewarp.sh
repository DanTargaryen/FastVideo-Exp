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
OUT_DIR="${OUT_DIR:-/vePFS-buaa/linming/workspace/worldrender/MATRIXCITY_PARQUET_PHASE3_401_ONLINEWARP}"
STREET_SPLIT="${STREET_SPLIT:-train_dense}"
CLIP_LENGTH="${CLIP_LENGTH:-401}"
WINDOW_LEN="${WINDOW_LEN:-401}"
ONLINE_WARP_WINDOW_FRAMES="${ONLINE_WARP_WINDOW_FRAMES:-81}"
ONLINE_WARP_OVERLAP_FRAMES="${ONLINE_WARP_OVERLAP_FRAMES:-1}"
ONLINE_WARP_NUM_KEYFRAMES="${ONLINE_WARP_NUM_KEYFRAMES:-4}"
ONLINE_WARP_NUM_TARGET_SAMPLES="${ONLINE_WARP_NUM_TARGET_SAMPLES:-3}"
ONLINE_WARP_VOXEL_SIZE="${ONLINE_WARP_VOXEL_SIZE:-0.1}"
RAW_CLIP_STRIDE="${RAW_CLIP_STRIDE:-80}"
CAMERA_MODE="${CAMERA_MODE:-B_inv}"
DEFAULT_PROMPT="${DEFAULT_PROMPT:-A continuous driving view through a city street.}"
SAMPLES_PER_FILE="${SAMPLES_PER_FILE:-8}"
FLUSH_FREQUENCY="${FLUSH_FREQUENCY:-8}"

# Notes:
# 1. This launch enumerates clips directly from RAW_ROOT train_dense scenes.
# 2. mask/masked_rgb are generated online from raw rgb/depth/camera pose
#    using the long-warp style windowed policy.
# 3. RAW_CLIP_STRIDE=80 matches 401 = 81 + 4 * (81 - 1), i.e. 5 long-video
#    windows with overlap=1.
# 4. We intentionally do not pass --write_vae_latent here because current
#    phase-3 training uses simulate_generator_forward=true.

torchrun --nnodes 1 --nproc_per_node "${GPU_NUM}" \
  tools/preprocess_matrixcity_ti2v_controlnet_parquet.py \
  --model_path "${MODEL_PATH}" \
  --rgb_root "${RAW_ROOT}" \
  --depth_root "${RAW_ROOT}" \
  --normal_root "${RAW_ROOT}" \
  --output_dir "${OUT_DIR}" \
  --street_split "${STREET_SPLIT}" \
  --clip_length "${CLIP_LENGTH}" \
  --window_len "${WINDOW_LEN}" \
  --use_clip_start_global_ids \
  --first_frame_source rgb \
  --control_source online_warp \
  --online_warp_camera_mode "${CAMERA_MODE}" \
  --online_warp_window_frames "${ONLINE_WARP_WINDOW_FRAMES}" \
  --online_warp_overlap_frames "${ONLINE_WARP_OVERLAP_FRAMES}" \
  --online_warp_num_keyframes "${ONLINE_WARP_NUM_KEYFRAMES}" \
  --online_warp_selection_num_target_samples "${ONLINE_WARP_NUM_TARGET_SAMPLES}" \
  --online_warp_selection_voxel_size "${ONLINE_WARP_VOXEL_SIZE}" \
  --online_warp_raw_clip_stride "${RAW_CLIP_STRIDE}" \
  --online_warp_default_prompt "${DEFAULT_PROMPT}" \
  --require_normal \
  --fps 16 \
  --max_width 512 \
  --max_height 384 \
  --dtype bf16 \
  --samples_per_file "${SAMPLES_PER_FILE}" \
  --flush_frequency "${FLUSH_FREQUENCY}"
