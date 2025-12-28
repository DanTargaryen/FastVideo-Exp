# Example: preprocess OmniWorld-Game pickle + warp_out into TI2V+ControlNet parquet.
#
# Required:
#   - MODEL_PATH: Wan diffusers root (contains text_encoder/vae/etc)
#   - IN_PICKLE: phase2 manifest pickle (list[dict])
#   - WARP_OUT: output root from Diff-Factory/tools/depth_warp.py
#   - OUT_DIR: output parquet dir
#
# Notes:
#   - This script runs on 1 GPU. For speed, run multiple jobs with different --start/--end ranges.

GPU_NUM=1
MODEL_PATH="/path/to/Wan2.2-TI2V-5B-Diffusers"
IN_PICKLE="/path/to/OmniWorldGame_mask10_size3641.pickle"
WARP_OUT="/path/to/warp_out"
OUT_DIR="/path/to/omnigame_ti2v_controlnet_parquet"

torchrun --nproc_per_node=$GPU_NUM \
  fastvideo/pipelines/preprocess/v1_preprocess_omnigame_ti2v_controlnet.py \
  --model_path "$MODEL_PATH" \
  --in_pickle "$IN_PICKLE" \
  --warp_out_root "$WARP_OUT" \
  --output_dir "$OUT_DIR" \
  --max_height 384 \
  --max_width 512 \
  --clip_length 81 \
  --fps 30 \
  --samples_per_file 8 \
  --flush_frequency 8 \
  --start 0 \
  --end 100

