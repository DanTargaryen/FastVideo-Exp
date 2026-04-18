#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/vePFS-buaa/linming/workspace/worldrender/FastVideo-Exp}"
CKPT_DIR="${CKPT_DIR:-/vePFS-buaa/linming/workspace/worldrender/phase2_ode_out/checkpoint-3000}"
GPU_ID="${GPU_ID:-7}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/vePFS-buaa/linming/workspace/worldrender/FastVideo-Exp/outputs/phase2_ckpt3000_mc401_batch_longwarp}"
RAW_PROMPT="${RAW_PROMPT:-A continuous driving view through a city street.}"
SAMPLE_PARENT="${SAMPLE_PARENT:-/vePFS-buaa/linming/workspace/worldrender/tmp/phase2_ckpt3000_mc401_batch}"

cd "${ROOT_DIR}"
mkdir -p "${OUTPUT_ROOT}"

samples=(
  "${SAMPLE_PARENT}/small_city_road_vertical_dense__1540_1940__clip_start_1540"
  "${SAMPLE_PARENT}/small_city_road_down_dense__42176_42576__clip_start_42176"
  "${SAMPLE_PARENT}/small_city_road_horizon_dense__34445_34845__clip_start_34445"
)

for sample_root in "${samples[@]}"; do
  sample_name="$(basename "${sample_root}")"
  out_dir="${OUTPUT_ROOT}/${sample_name}"
  out_mp4="${out_dir}/${sample_name}.mp4"

  if [[ -f "${out_mp4}" ]]; then
    echo "[skip] ${sample_name} already exists at ${out_mp4}"
    continue
  fi

  echo "[run] sample=${sample_name}"
  CKPT_DIR="${CKPT_DIR}" \
  SAMPLE_ROOT="${sample_root}" \
  GPU_ID="${GPU_ID}" \
  OUT_DIR="${out_dir}" \
  RAW_PROMPT="${RAW_PROMPT}" \
  bash scripts/inference/launch_phase3_ckpt200_longwarp_matrixcity401_binv.sh
done

echo "[done] all requested samples processed"
