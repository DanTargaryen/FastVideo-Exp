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
OUT_DIR="${OUT_DIR:-/vePFS-buaa/linming/workspace/worldrender/MATRIXCITY_PARQUET_PHASE3_401_BOOTSTRAP_W0CTRL_DN401_POSEFILTERED_SEGMENTED_R5_ROT10}"
STREET_SPLIT="${STREET_SPLIT:-train_dense}"
CLIP_LENGTH="${CLIP_LENGTH:-401}"
WINDOW_LEN="${WINDOW_LEN:-401}"
ONLINE_WARP_WINDOW_FRAMES="${ONLINE_WARP_WINDOW_FRAMES:-81}"
ONLINE_WARP_OVERLAP_FRAMES="${ONLINE_WARP_OVERLAP_FRAMES:-1}"
ONLINE_WARP_DEPTH_NORMALIZATION_MODE="${ONLINE_WARP_DEPTH_NORMALIZATION_MODE:-percentile}"
RAW_CLIP_STRIDE="${RAW_CLIP_STRIDE:-80}"
CAMERA_MODE="${CAMERA_MODE:-B_inv}"
POSE_FILTER_TRANSLATION_RATIO="${POSE_FILTER_TRANSLATION_RATIO:-5}"
POSE_FILTER_MAX_ROTATION_DEG="${POSE_FILTER_MAX_ROTATION_DEG:-10}"
DEFAULT_PROMPT="${DEFAULT_PROMPT:-A continuous driving view through a city street.}"
SAMPLES_PER_FILE="${SAMPLES_PER_FILE:-8}"
FLUSH_FREQUENCY="${FLUSH_FREQUENCY:-8}"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a _visible_gpu_array <<< "${CUDA_VISIBLE_DEVICES}"
  VISIBLE_GPU_COUNT="${#_visible_gpu_array[@]}"
else
  VISIBLE_GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
fi

if (( VISIBLE_GPU_COUNT < GPU_NUM )); then
  echo "Error: GPU_NUM=${GPU_NUM}, but only ${VISIBLE_GPU_COUNT} GPUs are visible to this shell." >&2
  echo "Current CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}" >&2
  echo "Please unset CUDA_VISIBLE_DEVICES or set it to at least ${GPU_NUM} devices before launching." >&2
  exit 1
fi

# This pose-filtered launch writes:
# - first-window 81-frame control_latent cache
# - full 401-frame depth_latent / normal_latent cache
# Later training windows still regenerate mask/maskrgb online from RAW_ROOT.

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
  --online_warp_depth_normalization_mode "${ONLINE_WARP_DEPTH_NORMALIZATION_MODE}" \
  --online_warp_raw_clip_stride "${RAW_CLIP_STRIDE}" \
  --online_warp_pose_filter_translation_ratio "${POSE_FILTER_TRANSLATION_RATIO}" \
  --online_warp_pose_filter_max_rotation_deg "${POSE_FILTER_MAX_ROTATION_DEG}" \
  --online_warp_default_prompt "${DEFAULT_PROMPT}" \
  --require_normal \
  --bootstrap_only \
  --bootstrap_cache_first_window_control_latent \
  --fps 16 \
  --max_width 512 \
  --max_height 384 \
  --dtype bf16 \
  --samples_per_file "${SAMPLES_PER_FILE}" \
  --flush_frequency "${FLUSH_FREQUENCY}"
