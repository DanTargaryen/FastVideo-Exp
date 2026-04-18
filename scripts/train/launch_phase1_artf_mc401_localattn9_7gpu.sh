#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/vePFS-buaa/linming/workspace/worldrender/FastVideo-Exp"
cd "${ROOT_DIR}"

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

GPU_NUM="${GPU_NUM:-7}"
CONFIG_PATH="${CONFIG_PATH:-configs/phase1_ar_tf_mc401_localattn9_7gpu.yaml}"

torchrun --nnodes 1 --nproc_per_node "${GPU_NUM}" \
  fastvideo/training/wan_controlnet_ar_tf_pipeline.py \
  --inference_mode=False \
  --config "${CONFIG_PATH}"
