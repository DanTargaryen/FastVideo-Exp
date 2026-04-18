#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/vePFS-buaa/linming/workspace/worldrender/FastVideo-Exp}"
CONFIG_PATH="${CONFIG_PATH:-configs/train_wan_controlnet_self_forcing_phase3_rgbn_ablation.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-/vePFS-buaa/linming/workspace/worldrender/phase3_dmd_out_causal_lr_ga6_rgbn81_w1only_ablation50}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-phase3_dmd_from_phase2ode_ckpt3000_lr2e6_f4e7_ga6_rgbn81_w1only_ablation50}"
NPROC_PER_NODE="${NPROC_PER_NODE:-7}"

cd "${ROOT_DIR}"

export TOKENIZERS_PARALLELISM=false
export FASTVIDEO_ATTENTION_BACKEND="${FASTVIDEO_ATTENTION_BACKEND:-FLASH_ATTN}"
export FASTVIDEO_EMPTY_CACHE_BEFORE_GEN_STEP="${FASTVIDEO_EMPTY_CACHE_BEFORE_GEN_STEP:-1}"
export OPENCV_IO_ENABLE_OPENEXR="${OPENCV_IO_ENABLE_OPENEXR:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"
export WANDB_MODE="${WANDB_MODE:-online}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

if command -v python >/dev/null 2>&1; then
  TORCH_LIB_DIR="$(
    python - <<'PY'
import os
try:
    import torch
    print(os.path.join(os.path.dirname(torch.__file__), "lib"))
except Exception:
    pass
PY
  )"
  if [[ -n "${TORCH_LIB_DIR}" ]]; then
    export LD_LIBRARY_PATH="${TORCH_LIB_DIR}:${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"
  elif [[ -n "${CONDA_PREFIX:-}" ]]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
  fi
fi

if [[ "${WANDB_MODE}" == "online" ]]; then
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    if [[ "${#WANDB_API_KEY}" -lt 40 ]]; then
      echo "WANDB_API_KEY looks invalid: expected 40+ characters, got ${#WANDB_API_KEY}." >&2
      exit 1
    fi
  elif [[ ! -f "${HOME}/.netrc" ]] || ! grep -q "machine api.wandb.ai" "${HOME}/.netrc"; then
    echo "W&B online logging requested, but neither WANDB_API_KEY nor ~/.netrc wandb login is available." >&2
    exit 1
  fi
fi

mkdir -p "${OUTPUT_DIR}/logs"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
LOG_FILE="${OUTPUT_DIR}/logs/train_${TIMESTAMP}.log"
LATEST_LOG="${OUTPUT_DIR}/logs/latest.log"
CMD_FILE="${OUTPUT_DIR}/logs/launch_${TIMESTAMP}.sh"

cmd=(
  torchrun
  --nnodes 1
  --nproc_per_node "${NPROC_PER_NODE}"
)

if [[ -n "${MASTER_ADDR:-}" ]]; then
  cmd+=(--master_addr "${MASTER_ADDR}")
fi

if [[ -n "${MASTER_PORT:-}" ]]; then
  cmd+=(--master_port "${MASTER_PORT}")
fi

cmd+=(
  fastvideo/training/wan_controlnet_self_forcing_distillation_pipeline.py
  --inference_mode=False
  --config "${CONFIG_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --wandb_run_name "${WANDB_RUN_NAME}"
  --num_gpus "${NPROC_PER_NODE}"
  --hsdp_shard_dim "${NPROC_PER_NODE}"
)

if [[ "$#" -gt 0 ]]; then
  cmd+=("$@")
fi

{
  echo "[launch] utc_time=${TIMESTAMP}"
  echo "[launch] cwd=${ROOT_DIR}"
  echo "[launch] log_file=${LOG_FILE}"
  echo "[launch] wandb_mode=${WANDB_MODE}"
  echo "[launch] output_dir=${OUTPUT_DIR}"
  echo "[launch] wandb_run_name=${WANDB_RUN_NAME}"
  echo "[launch] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}"
  echo "[launch] nproc_per_node=${NPROC_PER_NODE}"
  echo "[launch] master_addr=${MASTER_ADDR:-<unset>}"
  echo "[launch] master_port=${MASTER_PORT:-<unset>}"
  echo "[launch] pytorch_alloc_conf=${PYTORCH_ALLOC_CONF}"
  echo "[launch] empty_cache_before_gen_step=${FASTVIDEO_EMPTY_CACHE_BEFORE_GEN_STEP}"
  printf '[launch] command='
  printf '%q ' "${cmd[@]}"
  echo
} | tee "${LOG_FILE}"

{
  echo "#!/usr/bin/env bash"
  printf '%q ' "${cmd[@]}"
  echo
} > "${CMD_FILE}"

ln -sfn "${LOG_FILE}" "${LATEST_LOG}"

"${cmd[@]}" 2>&1 | tee -a "${LOG_FILE}"
