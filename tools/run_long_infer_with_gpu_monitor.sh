#!/usr/bin/env bash
set -eo pipefail

if [[ "$#" -lt 6 ]]; then
  echo "Usage: $0 <python_bin> <root_dir> <gpu_id> <master_port> <out_dir> <infer_script> [infer args...]" >&2
  exit 1
fi

PY_BIN="$1"
ROOT_DIR="$2"
GPU_ID="$3"
MASTER_PORT="$4"
OUT_DIR="$5"
INFER_SCRIPT="$6"
shift 6

mkdir -p "${OUT_DIR}"

CMD=(
  "${PY_BIN}" "${INFER_SCRIPT}" "$@"
)

{
  printf 'CUDA_VISIBLE_DEVICES=%q ' "${GPU_ID}"
  printf 'PYTHONUNBUFFERED=%q ' "1"
  printf 'PYTHONPATH=%q ' "${ROOT_DIR}:${PYTHONPATH:-}"
  printf 'FASTVIDEO_ATTENTION_BACKEND=%q ' "TORCH_SDPA"
  printf 'OPENCV_IO_ENABLE_OPENEXR=%q ' "1"
  printf 'MASTER_ADDR=%q ' "127.0.0.1"
  printf 'MASTER_PORT=%q ' "${MASTER_PORT}"
  printf 'RANK=%q ' "0"
  printf 'WORLD_SIZE=%q ' "1"
  printf 'LOCAL_RANK=%q ' "0"
  printf 'cd %q && ' "${ROOT_DIR}"
  printf '%q ' "${CMD[@]}"
  echo
} > "${OUT_DIR}/launch_command.sh"

(
  cd "${ROOT_DIR}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  PYTHONUNBUFFERED=1 \
  PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}" \
  FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA \
  OPENCV_IO_ENABLE_OPENEXR=1 \
  MASTER_ADDR=127.0.0.1 \
  MASTER_PORT="${MASTER_PORT}" \
  RANK=0 \
  WORLD_SIZE=1 \
  LOCAL_RANK=0 \
  "${CMD[@]}" 2>&1 | tee "${OUT_DIR}/infer.log"
) &
PID=$!

echo "timestamp,memory_used_mib,memory_total_mib,util_gpu" > "${OUT_DIR}/gpu_monitor.csv"
while kill -0 "${PID}" 2>/dev/null; do
  LINE="$(nvidia-smi --id="${GPU_ID}" --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits)"
  USED="$(echo "${LINE}" | cut -d',' -f1 | tr -d ' ')"
  TOTAL="$(echo "${LINE}" | cut -d',' -f2 | tr -d ' ')"
  UTIL="$(echo "${LINE}" | cut -d',' -f3 | tr -d ' ')"
  printf '%s,%s,%s,%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${USED}" "${TOTAL}" "${UTIL}" >> "${OUT_DIR}/gpu_monitor.csv"
  sleep 5
done

wait "${PID}"

python - <<PY
import csv
from pathlib import Path

csv_path = Path(r"${OUT_DIR}") / "gpu_monitor.csv"
rows = list(csv.DictReader(csv_path.open()))
peak_ratio = max((float(r["memory_used_mib"]) / float(r["memory_total_mib"]) * 100.0 for r in rows), default=0.0)
peak_mib = max((int(r["memory_used_mib"]) for r in rows), default=0)

summary_path = Path(r"${OUT_DIR}") / "gpu_monitor_summary.txt"
summary_path.write_text(
    f"peak_memory_used_mib={peak_mib}\n"
    f"peak_memory_ratio_pct={peak_ratio:.4f}\n",
    encoding="utf-8",
)
print(f"PEAK_RATIO={peak_ratio:.4f}")
PY
