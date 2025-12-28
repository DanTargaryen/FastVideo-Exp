export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN

NUM_GPUS=8

# Preprocessed parquet dir from:
#   fastvideo/pipelines/preprocess/v1_preprocess_omnigame_ti2v_controlnet.py
DATA_DIR="/path/to/omnigame_ti2v_controlnet_parquet"

# Wan2.2 TI2V diffusers root (teacher backbone)
MODEL="/path/to/Wan2.2-TI2V-5B-Diffusers"

# Diffusers-format ControlNet component dir (teacher controlnet)
CONTROLNET="/path/to/world-renderer-controlnet-warp-mask"

torchrun --nnodes 1 --nproc_per_node $NUM_GPUS \
  fastvideo/training/wan_controlnet_self_forcing_distillation_pipeline.py \
  --inference_mode False \
  --pretrained_model_name_or_path "$MODEL" \
  --model_path "$MODEL" \
  --real_score_model_path "$MODEL" \
  --fake_score_model_path "$MODEL" \
  --controlnet-model-path "$CONTROLNET" \
  --real-score-controlnet-model-path "$CONTROLNET" \
  --fake-score-controlnet-model-path "$CONTROLNET" \
  --data_path "$DATA_DIR" \
  --train_batch_size 1 \
  --train_sp_batch_size 1 \
  --num_gpus $NUM_GPUS \
  --tp_size 1 \
  --sp_size 1 \
  --hsdp_replicate_dim $NUM_GPUS \
  --hsdp-shard-dim 1 \
  --dataloader_num_workers 0 \
  --gradient_accumulation_steps 4 \
  --max_train_steps 2000 \
  --learning_rate 2e-6 \
  --mixed_precision "bf16" \
  --output_dir "outputs/wan_controlnet_self_forcing_phase2" \
  --checkpoints_total_limit 2 \
  --training_state_checkpointing_steps 200 \
  --training_cfg_rate 0.0 \
  --simulate_generator_forward True \
  --num_height 384 \
  --num_width 512 \
  --num_frames 81 \
  --num_latent_t 21 \
  --flow_shift 8 \
  --num_frame_per_block 3 \
  --context_noise 0 \
  --dfake_gen_update_ratio 5 \
  --real_score_guidance_scale 3.5 \
  --seed 1024

