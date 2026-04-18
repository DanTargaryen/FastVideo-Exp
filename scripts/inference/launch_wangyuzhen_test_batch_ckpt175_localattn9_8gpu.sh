#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/vePFS-buaa/linming/workspace/worldrender/FastVideo-Exp}"
WORKER_SCRIPT="${WORKER_SCRIPT:-${ROOT_DIR}/scripts/inference/run_wangyuzhen_test_batch_ckpt175_localattn9.sh}"
OUT_ROOT="${OUT_ROOT:-/vePFS-buaa/linming/workspace/worldrender/FastVideo-Exp/outputs/test_batch_phase3_ckpt175_wangyuzhen_localattn9_hardreplace_401f_o1_w81}"
LOG_ROOT="${LOG_ROOT:-${OUT_ROOT}/worker_logs}"
NUM_WORKERS="${NUM_WORKERS:-8}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29780}"

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"

IFS=',' read -r -a GPU_ARR <<< "${GPU_IDS}"
if (( ${#GPU_ARR[@]} != NUM_WORKERS )); then
  echo "GPU_IDS count (${#GPU_ARR[@]}) must match NUM_WORKERS (${NUM_WORKERS})" >&2
  exit 1
fi

echo "[launch] worker_script=${WORKER_SCRIPT}"
echo "[launch] out_root=${OUT_ROOT}"
echo "[launch] log_root=${LOG_ROOT}"
echo "[launch] num_workers=${NUM_WORKERS} gpu_ids=${GPU_IDS}"

for idx in $(seq 0 $((NUM_WORKERS - 1))); do
  gpu="${GPU_ARR[$idx]}"
  log_path="${LOG_ROOT}/worker_${idx}_gpu${gpu}.log"
  pid_path="${LOG_ROOT}/worker_${idx}_gpu${gpu}.pid"

  nohup bash -lc \
    "GPU_ID='${gpu}' WORKER_IDX='${idx}' NUM_WORKERS='${NUM_WORKERS}' MASTER_PORT_BASE='${MASTER_PORT_BASE}' OUT_ROOT='${OUT_ROOT}' '${WORKER_SCRIPT}'" \
    > "${log_path}" 2>&1 &

  echo $! > "${pid_path}"
  echo "[launched] worker=${idx} gpu=${gpu} pid=$(cat "${pid_path}") log=${log_path}"
done

echo "[done] launched ${NUM_WORKERS} workers"
