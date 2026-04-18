#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/vePFS-buaa/linming/workspace/worldrender/FastVideo-Exp}"
PYTHON_BIN="${PYTHON_BIN:-/vePFS-buaa/linming/miniconda3/envs/fastvideo/bin/python3.12}"
MODEL_DIR="${MODEL_DIR:-/vePFS-buaa/linming/workspace/worldrender/Wan-AI/Wan2.2-TI2V-5B-Diffusers}"
CKPT_DIR="${CKPT_DIR:-/vePFS-buaa/linming/workspace/worldrender/phase3_dmd_out_causal_lr_ga6_mc401_posefiltered_r5rot10_w2only_bootstrap_1000step/checkpoint-175_weight_only}"
TEST_ROOT="${TEST_ROOT:-/vePFS-buaa/wangyuzhen/Dataset/test}"
OUT_ROOT="${OUT_ROOT:-/vePFS-buaa/linming/workspace/worldrender/FastVideo-Exp/outputs/test_batch_phase3_ckpt175_wangyuzhen_localattn9_hardreplace_401f_o1_w81}"
GPU_ID="${GPU_ID:-7}"
WORKER_IDX="${WORKER_IDX:-0}"
NUM_WORKERS="${NUM_WORKERS:-1}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29780}"
NUM_FRAMES="${NUM_FRAMES:-401}"
CAUSAL_WINDOW_FRAMES="${CAUSAL_WINDOW_FRAMES:-81}"
CAUSAL_OVERLAP_FRAMES="${CAUSAL_OVERLAP_FRAMES:-1}"
LOCAL_ATTN_SIZE="${LOCAL_ATTN_SIZE:-9}"
SINK_SIZE="${SINK_SIZE:-1}"
WARP_NUM_KEYFRAMES="${WARP_NUM_KEYFRAMES:-4}"
FIRST_FRAME_CONDITION_MODE="${FIRST_FRAME_CONDITION_MODE:-hard_replace}"
FIRST_FRAME_TIMESTEP_ZERO="${FIRST_FRAME_TIMESTEP_ZERO:-1}"

cd "${ROOT_DIR}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export FASTVIDEO_ATTENTION_BACKEND="${FASTVIDEO_ATTENTION_BACKEND:-TORCH_SDPA}"
export OPENCV_IO_ENABLE_OPENEXR="${OPENCV_IO_ENABLE_OPENEXR:-1}"
export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

mkdir -p "${OUT_ROOT}"

if (( NUM_WORKERS <= 0 )); then
  echo "NUM_WORKERS must be > 0, got ${NUM_WORKERS}" >&2
  exit 1
fi
if (( WORKER_IDX < 0 || WORKER_IDX >= NUM_WORKERS )); then
  echo "WORKER_IDX must satisfy 0 <= WORKER_IDX < NUM_WORKERS, got WORKER_IDX=${WORKER_IDX} NUM_WORKERS=${NUM_WORKERS}" >&2
  exit 1
fi
if (( NUM_FRAMES <= 0 )); then
  echo "NUM_FRAMES must be > 0, got ${NUM_FRAMES}" >&2
  exit 1
fi

sample_idx=0
processed=0
skipped=0
shopt -s nullglob
samples=( "${TEST_ROOT}"/* )

for sample_root in "${samples[@]}"; do
  [[ -d "${sample_root}" ]] || continue

  if (( sample_idx % NUM_WORKERS != WORKER_IDX )); then
    sample_idx="$((sample_idx + 1))"
    continue
  fi

  sample_name="$(basename "${sample_root}")"
  rgb_dir="${sample_root}/rgb"
  depth_dir="${sample_root}/depth"
  normal_dir="${sample_root}/normal"
  camera_dir="${sample_root}/camera"
  cam_k="${camera_dir}/camera_K.txt"
  sample_out="${OUT_ROOT}/${sample_name}"

  if [[ ! -d "${rgb_dir}" || ! -d "${depth_dir}" || ! -d "${normal_dir}" || ! -d "${camera_dir}" || ! -f "${cam_k}" ]]; then
    echo "[skip] ${sample_name}: missing required rgb/depth/normal/camera inputs" >&2
    skipped="$((skipped + 1))"
    sample_idx="$((sample_idx + 1))"
    continue
  fi

  rgb_count="$(find "${rgb_dir}" -maxdepth 1 -type f | wc -l)"
  depth_count="$(find "${depth_dir}" -maxdepth 1 -type f | wc -l)"
  normal_count="$(find "${normal_dir}" -maxdepth 1 -type f | wc -l)"
  camera_count="$(find "${camera_dir}" -maxdepth 1 -type f -name 'camera_RT_*.txt' | wc -l)"
  usable_frames="${rgb_count}"
  if (( depth_count < usable_frames )); then usable_frames="${depth_count}"; fi
  if (( normal_count < usable_frames )); then usable_frames="${normal_count}"; fi
  if (( camera_count < usable_frames )); then usable_frames="${camera_count}"; fi

  if (( usable_frames < NUM_FRAMES )); then
    echo "[skip] ${sample_name}: usable_frames=${usable_frames} < required=${NUM_FRAMES}" >&2
    skipped="$((skipped + 1))"
    sample_idx="$((sample_idx + 1))"
    continue
  fi

  mkdir -p "${sample_out}"
  if [[ -f "${sample_out}/${sample_name}.mp4" ]]; then
    echo "[skip] worker=${WORKER_IDX} sample=${sample_name}: output already exists at ${sample_out}/${sample_name}.mp4"
    skipped="$((skipped + 1))"
    sample_idx="$((sample_idx + 1))"
    continue
  fi

  export MASTER_ADDR
  export MASTER_PORT="$((MASTER_PORT_BASE + sample_idx))"

  echo "[run] worker=${WORKER_IDX}/${NUM_WORKERS} gpu=${GPU_ID} sample=${sample_name} num_frames=${NUM_FRAMES} out=${sample_out}"

  cmd=(
    "${PYTHON_BIN}" tools/infer_wan_controlnet_ti2v_long_firstframe_warp.py
    --base_model "${MODEL_DIR}"
    --transformer_dir "${CKPT_DIR}/generator_inference_transformer"
    --controlnet_dir "${CKPT_DIR}/generator_inference_controlnet"
    --raw_sample_root "${sample_root}"
    --raw_rgb_dir "${rgb_dir}"
    --raw_depth_dir "${depth_dir}"
    --raw_normal_dir "${normal_dir}"
    --raw_require_normal
    --cam_k "${cam_k}"
    --cam_rt_dir "${camera_dir}"
    --scheduler flowmatch_euler
    --schedule_num_inference_steps 50
    --dmd_steps 1000,750,500,250
    --update_rule renoise_x0
    --warp_denoising_step
    --guidance_scale 1
    --height 384
    --width 512
    --num_frames "${NUM_FRAMES}"
    --causal_window_frames "${CAUSAL_WINDOW_FRAMES}"
    --causal_overlap_frames "${CAUSAL_OVERLAP_FRAMES}"
    --local_attn_size "${LOCAL_ATTN_SIZE}"
    --sink_size "${SINK_SIZE}"
    --warp_num_keyframes "${WARP_NUM_KEYFRAMES}"
    --raw_depth_normalization_mode percentile
    --no_raw_depth_invert
    --first_frame_condition_mode "${FIRST_FRAME_CONDITION_MODE}"
    --dtype bf16
    --seed 42
    --cache_reset_interval_windows 0
    --boundary_blend_frames 0
    --boundary_blend_strength 1.0
    --out_dir "${sample_out}"
  )

  if [[ "${FIRST_FRAME_TIMESTEP_ZERO}" != "0" ]]; then
    cmd+=( --first_frame_timestep_zero )
  fi

  "${cmd[@]}"

  processed="$((processed + 1))"
  sample_idx="$((sample_idx + 1))"
done

echo "[done] worker=${WORKER_IDX}/${NUM_WORKERS} gpu=${GPU_ID} processed=${processed} skipped=${skipped} out_root=${OUT_ROOT}"
