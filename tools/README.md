# Tools Directory

This directory collects project-specific utility scripts. Most of them are
meant for manual research workflows, so they are grouped by job instead of by
strict library boundaries.

## Common patterns

- Prefer environment variables over hard-coded secrets.
- Treat these scripts as user-facing CLIs; pass paths as arguments instead of
  editing source files.
- Large generated artifacts should go under ignored directories such as
  `outputs/`, `outputs_dmd/`, `tmp/`, or external dataset/model folders.

## Recommended entry points

- **Model / dataset download**
  - `download_weights.py`: thin Hugging Face snapshot downloader for local model
    pulls. Reads `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` when needed.
- **Dataset preparation**
  - `preprocess_matrixcity_ti2v_controlnet_parquet.py`
  - `preprocess_interiornet_ti2v_controlnet_parquet.py`
  - `preprocess_ode_trajectory_ti2v_controlnet.py`
  - `fill_matrixcity_vae_latent_from_raw.py`
- **Inference / visualization**
  - `infer_wan_controlnet_ti2v.py`
  - `infer_wan_controlnet_ti2v_long_firstframe_warp.py`
  - `infer_wan_controlnet_ti2v_long_bidirectional_warp.py`
  - `run_long_infer_with_gpu_monitor.sh`
  - `plot_phase3_current_losses.py`
- **Conversion / inspection**
  - `convert_pt_state_dict_to_safetensors.py`
  - `merge_distributed_checkpoint_to_inference.py`
  - `compare_safetensors_checkpoints.py`
  - `check_wan_union_parquet.py`
  - `review_matrixcity_alignment.py`
- **One-off research helpers**
  - `make_verify_0001_0_80.py`
  - `experiment_teacher_last_window_bidirectional.py`
  - `generate_matrixcity_sample_raw_warp.py`
  - `gen_caption_json.py`

## Notes on cleanup

- I intentionally keep the existing filenames stable to avoid breaking shell
  scripts or personal workflows.
- Some scripts still contain lab-specific defaults for convenience, but the
  main entry points above now support command-line arguments so they can be used
  without editing source.
