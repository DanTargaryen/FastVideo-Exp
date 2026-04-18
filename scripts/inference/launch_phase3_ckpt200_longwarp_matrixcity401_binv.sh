#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/vePFS-buaa/linming/workspace/worldrender/FastVideo-Exp}"
PYTHON_BIN="${PYTHON_BIN:-/vePFS-buaa/linming/miniconda3/envs/fastvideo/bin/python3.12}"
MODEL_DIR="${MODEL_DIR:-/vePFS-buaa/linming/workspace/worldrender/Wan-AI/Wan2.2-TI2V-5B-Diffusers}"
CKPT_DIR="${CKPT_DIR:-/vePFS-buaa/linming/workspace/worldrender/phase3_dmd_out_causal_lr_ga12/checkpoint-200_weight_only}"
SAMPLE_ROOT="${SAMPLE_ROOT:-/vePFS-buaa/linming/Dataset/Matrixcity_sample_401}"

GPU_ID="${GPU_ID:-6}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29636}"
OUT_DIR="${OUT_DIR:-/vePFS-buaa/linming/workspace/worldrender/FastVideo-Exp/outputs/phase3dmd200_matrixcity_sample401_binv_longwarp_w81_o1_401f}"
RAW_PROMPT="${RAW_PROMPT:-A driving scene in city street.}"
NUM_FRAMES="${NUM_FRAMES:-401}"
CAUSAL_WINDOW_FRAMES="${CAUSAL_WINDOW_FRAMES:-81}"
CAUSAL_OVERLAP_FRAMES="${CAUSAL_OVERLAP_FRAMES:-1}"
SCHEDULER="${SCHEDULER:-flowmatch_euler}"
SCHEDULE_NUM_INFERENCE_STEPS="${SCHEDULE_NUM_INFERENCE_STEPS:-50}"
DMD_STEPS="${DMD_STEPS:-1000,750,500,250}"
UPDATE_RULE="${UPDATE_RULE:-renoise_x0}"
FULL_SCHEDULE="${FULL_SCHEDULE:-0}"
CACHE_RESET_INTERVAL_WINDOWS="${CACHE_RESET_INTERVAL_WINDOWS:-0}"
BOUNDARY_BLEND_FRAMES="${BOUNDARY_BLEND_FRAMES:-0}"
BOUNDARY_BLEND_STRENGTH="${BOUNDARY_BLEND_STRENGTH:-1.0}"
RAW_DEPTH_NORMALIZATION_MODE="${RAW_DEPTH_NORMALIZATION_MODE:-percentile}"

cd "${ROOT_DIR}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export FASTVIDEO_ATTENTION_BACKEND="${FASTVIDEO_ATTENTION_BACKEND:-TORCH_SDPA}"
export OPENCV_IO_ENABLE_OPENEXR="${OPENCV_IO_ENABLE_OPENEXR:-1}"
export MASTER_ADDR
export MASTER_PORT
export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

port_in_use() {
  local port="$1"
  if ! command -v ss >/dev/null 2>&1; then
    return 1
  fi
  ss -H -ltn "( sport = :${port} )" | grep -q .
}

if port_in_use "${MASTER_PORT}"; then
  original_port="${MASTER_PORT}"
  for candidate in $(seq $((MASTER_PORT + 1)) $((MASTER_PORT + 100))); do
    if ! port_in_use "${candidate}"; then
      MASTER_PORT="${candidate}"
      export MASTER_PORT
      break
    fi
  done
  if [[ "${MASTER_PORT}" == "${original_port}" ]]; then
    echo "Failed to find a free MASTER_PORT in [$((original_port + 1)), $((original_port + 100))]." >&2
    exit 1
  fi
fi

mkdir -p "${OUT_DIR}"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
LOG_FILE="${OUT_DIR}/infer_${TIMESTAMP}.log"
LATEST_LOG="${OUT_DIR}/latest.log"
CMD_FILE="${OUT_DIR}/launch_${TIMESTAMP}.sh"

cmd=(
  "${PYTHON_BIN}"
  tools/infer_wan_controlnet_ti2v_long_firstframe_warp.py
  --base_model "${MODEL_DIR}"
  --transformer_dir "${CKPT_DIR}/generator_inference_transformer"
  --controlnet_dir "${CKPT_DIR}/generator_inference_controlnet"
  --raw_sample_root "${SAMPLE_ROOT}"
  --raw_rgb_dir "${SAMPLE_ROOT}/rgb"
  --raw_depth_dir "${SAMPLE_ROOT}/depth"
  --raw_normal_dir "${SAMPLE_ROOT}/normal"
  --raw_require_normal
  --raw_prompt "${RAW_PROMPT}"
  --cam_k "${SAMPLE_ROOT}/camera_B_inv/camera_K.txt"
  --cam_rt_dir "${SAMPLE_ROOT}/camera_B_inv"
  --scheduler "${SCHEDULER}"
  --schedule_num_inference_steps "${SCHEDULE_NUM_INFERENCE_STEPS}"
  --dmd_steps "${DMD_STEPS}"
  --update_rule "${UPDATE_RULE}"
  --warp_denoising_step
  --guidance_scale 1
  --height 384
  --width 512
  --num_frames "${NUM_FRAMES}"
  --causal_window_frames "${CAUSAL_WINDOW_FRAMES}"
  --causal_overlap_frames "${CAUSAL_OVERLAP_FRAMES}"
  --local_attn_size 21
  --sink_size 1
  --warp_num_keyframes 4
  --raw_depth_normalization_mode "${RAW_DEPTH_NORMALIZATION_MODE}"
  --no_raw_depth_invert
  --first_frame_timestep_zero
  --first_frame_condition_mode hard_replace
  --dtype bf16
  --seed 42
  --cache_reset_interval_windows "${CACHE_RESET_INTERVAL_WINDOWS}"
  --boundary_blend_frames "${BOUNDARY_BLEND_FRAMES}"
  --boundary_blend_strength "${BOUNDARY_BLEND_STRENGTH}"
  --out_dir "${OUT_DIR}"
)

if [[ "${FULL_SCHEDULE}" == "1" ]]; then
  cmd+=(--full_schedule)
fi

{
  echo "[launch] utc_time=${TIMESTAMP}"
  echo "[launch] cwd=${ROOT_DIR}"
  echo "[launch] log_file=${LOG_FILE}"
  echo "[launch] cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
  echo "[launch] master_addr=${MASTER_ADDR}"
  echo "[launch] master_port=${MASTER_PORT}"
  echo "[launch] out_dir=${OUT_DIR}"
  echo "[launch] causal_window_frames=${CAUSAL_WINDOW_FRAMES}"
  echo "[launch] causal_overlap_frames=${CAUSAL_OVERLAP_FRAMES}"
  echo "[launch] scheduler=${SCHEDULER}"
  echo "[launch] schedule_num_inference_steps=${SCHEDULE_NUM_INFERENCE_STEPS}"
  echo "[launch] dmd_steps=${DMD_STEPS}"
  echo "[launch] update_rule=${UPDATE_RULE}"
  echo "[launch] full_schedule=${FULL_SCHEDULE}"
  echo "[launch] cache_reset_interval_windows=${CACHE_RESET_INTERVAL_WINDOWS}"
  echo "[launch] boundary_blend_frames=${BOUNDARY_BLEND_FRAMES}"
  echo "[launch] boundary_blend_strength=${BOUNDARY_BLEND_STRENGTH}"
  echo "[launch] opencv_io_enable_openexr=${OPENCV_IO_ENABLE_OPENEXR}"
  printf '[launch] command='
  printf '%q ' "${cmd[@]}"
  echo
} | tee "${LOG_FILE}"

{
  echo "#!/usr/bin/env bash"
  printf '%q ' env \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="${PYTHONPATH}" \
    FASTVIDEO_ATTENTION_BACKEND="${FASTVIDEO_ATTENTION_BACKEND}" \
    OPENCV_IO_ENABLE_OPENEXR="${OPENCV_IO_ENABLE_OPENEXR}" \
    MASTER_ADDR="${MASTER_ADDR}" \
    MASTER_PORT="${MASTER_PORT}" \
    RANK=0 \
    WORLD_SIZE=1 \
    LOCAL_RANK=0 \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
  printf ' '
  printf '%q ' "${cmd[@]}"
  echo
} > "${CMD_FILE}"

ln -sfn "${LOG_FILE}" "${LATEST_LOG}"

"${cmd[@]}" 2>&1 | tee -a "${LOG_FILE}"
