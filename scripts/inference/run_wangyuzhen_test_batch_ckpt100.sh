#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/vePFS-buaa/linming/workspace/worldrender/FastVideo-Exp}"
PYTHON_BIN="${PYTHON_BIN:-/vePFS-buaa/linming/miniconda3/envs/fastvideo/bin/python3.12}"
MODEL_DIR="${MODEL_DIR:-/vePFS-buaa/linming/workspace/worldrender/Wan-AI/Wan2.2-TI2V-5B-Diffusers}"
CKPT_DIR="${CKPT_DIR:-/vePFS-buaa/linming/workspace/worldrender/phase3_dmd_out_causal_lr_ga6_mc401_posefiltered_r5rot10_pctl_w2only_bootstrap_bugfix_100step/checkpoint-100_weight_only}"
TEST_ROOT="${TEST_ROOT:-/vePFS-buaa/wangyuzhen/Dataset/test}"
OUT_ROOT="${OUT_ROOT:-/vePFS-buaa/linming/workspace/worldrender/FastVideo-Exp/outputs/test_batch_phase3_ckpt100_wangyuzhen}"
GPU_ID="${GPU_ID:-7}"
WORKER_IDX="${WORKER_IDX:-0}"
NUM_WORKERS="${NUM_WORKERS:-1}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29720}"

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

sample_idx=0
processed=0
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
    continue
  fi

  rgb_count="$(find "${rgb_dir}" -maxdepth 1 -type f | wc -l)"
  depth_count="$(find "${depth_dir}" -maxdepth 1 -type f | wc -l)"
  normal_count="$(find "${normal_dir}" -maxdepth 1 -type f | wc -l)"
  camera_count="$(find "${camera_dir}" -maxdepth 1 -type f -name 'camera_RT_*.txt' | wc -l)"
  num_frames="${rgb_count}"
  if (( depth_count < num_frames )); then num_frames="${depth_count}"; fi
  if (( normal_count < num_frames )); then num_frames="${normal_count}"; fi
  if (( camera_count < num_frames )); then num_frames="${camera_count}"; fi

  if (( num_frames <= 0 )); then
    echo "[skip] ${sample_name}: no usable frames found" >&2
    continue
  fi

  export MASTER_ADDR
  export MASTER_PORT="$((MASTER_PORT_BASE + sample_idx))"

  mkdir -p "${sample_out}"
  if [[ -f "${sample_out}/${sample_name}.mp4" ]]; then
    echo "[skip] worker=${WORKER_IDX} sample=${sample_name}: output already exists at ${sample_out}/${sample_name}.mp4"
    sample_idx="$((sample_idx + 1))"
    continue
  fi
  echo "[run] sample=${sample_name} num_frames=${num_frames} out=${sample_out}"

  "${PYTHON_BIN}" tools/infer_wan_controlnet_ti2v_long_firstframe_warp.py \
    --base_model "${MODEL_DIR}" \
    --transformer_dir "${CKPT_DIR}/generator_inference_transformer" \
    --controlnet_dir "${CKPT_DIR}/generator_inference_controlnet" \
    --raw_sample_root "${sample_root}" \
    --raw_rgb_dir "${rgb_dir}" \
    --raw_depth_dir "${depth_dir}" \
    --raw_normal_dir "${normal_dir}" \
    --raw_require_normal \
    --cam_k "${cam_k}" \
    --cam_rt_dir "${camera_dir}" \
    --scheduler flowmatch_euler \
    --schedule_num_inference_steps 50 \
    --dmd_steps 1000,750,500,250 \
    --update_rule renoise_x0 \
    --warp_denoising_step \
    --guidance_scale 1 \
    --height 384 \
    --width 512 \
    --num_frames "${num_frames}" \
    --causal_window_frames 81 \
    --causal_overlap_frames 1 \
    --local_attn_size 21 \
    --sink_size 1 \
    --warp_num_keyframes 4 \
    --raw_depth_normalization_mode percentile \
    --no_raw_depth_invert \
    --first_frame_timestep_zero \
    --first_frame_condition_mode hard_replace \
    --dtype bf16 \
    --seed 42 \
    --cache_reset_interval_windows 0 \
    --boundary_blend_frames 0 \
    --boundary_blend_strength 1.0 \
    --out_dir "${sample_out}"

  processed="$((processed + 1))"
  sample_idx="$((sample_idx + 1))"
done

echo "[done] worker=${WORKER_IDX}/${NUM_WORKERS} processed ${processed} assigned samples into ${OUT_ROOT}"
