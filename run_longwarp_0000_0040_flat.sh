#!/usr/bin/env bash
set -euo pipefail

cd /vePFS-buaa/linming/workspace/worldrender/FastVideo-Exp
export PYTHONPATH=$PWD

MODEL=/vePFS-buaa/linming/workspace/worldrender/Wan-AI/Wan2.2-TI2V-5B-Diffusers
AR_CKPT=/vePFS-buaa/linming/workspace/worldrender/phase3_dmd_out/checkpoint-3200_weight_only

BASE=/vePFS-buaa/wangyuzhen/Dataset/test
GPU_ID=0
MASTER_PORT=29635

VIDEO_ROOT=/vePFS-buaa/linming/workspace/worldrender/results/video
FRAME_ROOT=/vePFS-buaa/linming/workspace/worldrender/results/frames

mkdir -p "$VIDEO_ROOT" "$FRAME_ROOT"

for i in $(seq 0 40); do
  sid=$(printf "%04d" "$i")

  sample_root="${BASE}/${sid}"
  rgb_dir="${sample_root}/rgb"
  depth_dir="${sample_root}/depth"
  normal_dir="${sample_root}/normal"

  if [[ ! -d "$sample_root" ]]; then
    echo "[skip] ${sid}: sample_root missing"
    continue
  fi
  if [[ ! -d "$rgb_dir" ]]; then
    echo "[skip] ${sid}: rgb missing"
    continue
  fi
  if [[ ! -d "$depth_dir" ]]; then
    echo "[skip] ${sid}: depth missing"
    continue
  fi
  if [[ ! -d "$normal_dir" ]]; then
    echo "[skip] ${sid}: normal missing"
    continue
  fi

  tmp_out="${VIDEO_ROOT}/tmp_${sid}"
  tmp_frame_dir="${tmp_out}/frames/${sid}"
  final_video="${VIDEO_ROOT}/${sid}.mp4"
  final_frame_dir="${FRAME_ROOT}/${sid}"

  rm -rf "$tmp_out" "$final_frame_dir"

  echo "[run] ${sid}"
  echo "      video=${final_video}"
  echo "      frames=${final_frame_dir}"

  MASTER_ADDR=127.0.0.1 \
  MASTER_PORT=${MASTER_PORT} \
  RANK=0 \
  WORLD_SIZE=1 \
  LOCAL_RANK=0 \
  CUDA_VISIBLE_DEVICES=${GPU_ID} \
  python tools/infer_wan_controlnet_ti2v_long_firstframe_warp.py \
    --base_model "$MODEL" \
    --transformer_dir "$AR_CKPT/generator_inference_transformer" \
    --controlnet_dir "$AR_CKPT/generator_inference_controlnet" \
    --raw_sample_root "$sample_root" \
    --raw_rgb_dir "$rgb_dir" \
    --raw_depth_dir "$depth_dir" \
    --raw_normal_dir "$normal_dir" \
    --raw_require_normal \
    --scheduler flowmatch_euler \
    --dmd_steps 1000,750,500,250 \
    --update_rule renoise_x0 \
    --warp_denoising_step \
    --guidance_scale 1 \
    --height 384 \
    --width 512 \
    --num_frames 397 \
    --causal_window_frames 45 \
    --causal_overlap_frames 1 \
    --local_attn_size 21 \
    --sink_size 1 \
    --no_raw_depth_invert \
    --first_frame_timestep_zero \
    --first_frame_condition_mode hard_replace \
    --seed 42 \
    --out_dir "$tmp_out" \
    --save_frames

  if [[ -f "${tmp_out}/${sid}.mp4" ]]; then
    mv "${tmp_out}/${sid}.mp4" "$final_video"
  else
    echo "[warn] ${sid}: missing video ${tmp_out}/${sid}.mp4"
  fi

  if [[ -d "$tmp_frame_dir" ]]; then
    mv "$tmp_frame_dir" "$final_frame_dir"
  else
    echo "[warn] ${sid}: missing frames ${tmp_frame_dir}"
  fi

  rm -rf "$tmp_out"
done
