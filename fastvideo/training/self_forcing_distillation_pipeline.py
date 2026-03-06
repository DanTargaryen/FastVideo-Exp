# SPDX-License-Identifier: Apache-2.0
import copy
import os
import time
from collections import deque
from typing import Any

import imageio
import numpy as np
import torch
import torch.distributed as dist
from einops import rearrange
from tqdm.auto import tqdm

import fastvideo.envs as envs
from fastvideo.distributed import (cleanup_dist_env_and_memory,
                                   get_local_torch_device, get_sp_group,
                                   get_world_group)
from fastvideo.fastvideo_args import TrainingArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger

from fastvideo.models.schedulers.scheduling_self_forcing_flow_match import SelfForcingFlowMatchScheduler
from fastvideo.models.loader.component_loader import PipelineComponentLoader
from fastvideo.models.utils import pred_noise_to_pred_video
from fastvideo.pipelines import TrainingBatch
from fastvideo.training.distillation_pipeline import DistillationPipeline
from fastvideo.training.training_utils import (EMA_FSDP,
                                               normalize_dit_input,
                                               save_distillation_checkpoint)
from fastvideo.utils import is_vsa_available, set_random_seed
from fastvideo.profiler import profile_region

logger = init_logger(__name__)

vsa_available = is_vsa_available()
sync_list_verbose_log = bool(int(os.getenv("FASTVIDEO_SYNC_LIST_LOG", "0")))


def _to_log_scalar(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.item())
        return float(value.float().mean().item())
    return float(value)


def _wan_denormalize_dit_latents(latents: torch.Tensor, vae) -> torch.Tensor:
    latents_mean = torch.tensor(vae.latents_mean,
                                device=latents.device,
                                dtype=latents.dtype).view(1, -1, 1, 1, 1)
    latents_std = torch.tensor(vae.latents_std,
                               device=latents.device,
                               dtype=latents.dtype).view(1, -1, 1, 1, 1)
    return latents * latents_std + latents_mean


def _compute_negative_prompt_embeddings(
    *,
    tokenizer,
    text_encoder,
    negative_prompt: str,
    max_sequence_length: int,
    dtype: torch.dtype,
    target_device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert negative_prompt, "Negative prompt must be provided for CFG."
    encoder_device = next(text_encoder.parameters()).device
    text_encoder.eval()
    with torch.no_grad():
        tokens = tokenizer(
            [negative_prompt],
            padding="max_length",
            truncation=True,
            max_length=max_sequence_length,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        prompt_embeds = text_encoder(
            tokens.input_ids.to(encoder_device),
            tokens.attention_mask.to(encoder_device),
        ).last_hidden_state

    attn_mask = tokens.attention_mask
    seq_len = int(attn_mask.sum(dim=1)[0].item())
    seq_len = min(seq_len, prompt_embeds.shape[1])
    prompt_embeds = prompt_embeds[:, :seq_len, :]
    attn_mask = attn_mask[:, :seq_len]

    if prompt_embeds.shape[1] < max_sequence_length:
        pad_len = max_sequence_length - prompt_embeds.shape[1]
        pad_embed = prompt_embeds.new_zeros((1, pad_len, prompt_embeds.shape[-1]))
        prompt_embeds = torch.cat([prompt_embeds, pad_embed], dim=1)
        pad_mask = attn_mask.new_zeros((1, pad_len))
        attn_mask = torch.cat([attn_mask, pad_mask], dim=1)

    prompt_embeds = prompt_embeds.to(dtype=dtype, device=target_device)
    attn_mask = attn_mask.to(dtype=dtype, device=target_device)
    return prompt_embeds, attn_mask


class SelfForcingDistillationPipeline(DistillationPipeline):
    """
    A self-forcing distillation pipeline that alternates between training
    the generator and critic based on the self-forcing methodology.
    
    This implementation follows the self-forcing approach where:
    1. Generator and critic are trained in alternating steps
    2. Generator loss uses DMD-style loss with the critic as fake score
    3. Critic loss trains the fake score model to distinguish real vs fake
    """

    def initialize_training_pipeline(self, training_args: TrainingArgs):
        """Initialize the self-forcing training pipeline."""
        # Check if FSDP2 auto wrap is enabled - not supported for self-forcing distillation
        if os.environ.get("FASTVIDEO_FSDP2_AUTOWRAP", "0") == "1":
            raise NotImplementedError(
                "FASTVIDEO_FSDP2_AUTOWRAP is not implemented for self-forcing distillation. "
                "Please set FASTVIDEO_FSDP2_AUTOWRAP=0 or unset the environment variable."
            )

        logger.info("Initializing self-forcing distillation pipeline...")

        self.generator_ema: EMA_FSDP | None = None
        self.generator_ema_2: EMA_FSDP | None = None

        super().initialize_training_pipeline(training_args)
        try:
            logger.info("RANK: %s, entered initialize_training_pipeline",
                        self.global_rank,
                        local_main_process_only=False)
        except Exception:
            logger.info("Entered initialize_training_pipeline (rank unknown)")

        flow_shift = getattr(training_args.pipeline_config, "flow_shift", None)
        if flow_shift is None:
            flow_shift = 5.0
        self.noise_scheduler = SelfForcingFlowMatchScheduler(
            num_inference_steps=1000,
            shift=float(flow_shift),
            sigma_min=0.0,
            extra_one_step=True,
            training=True)
        self.dfake_gen_update_ratio = getattr(training_args,
                                              'dfake_gen_update_ratio', 5)

        self.num_frame_per_block = getattr(training_args, 'num_frame_per_block',
                                           3)
        self.independent_first_frame = getattr(training_args,
                                               'independent_first_frame', False)
        self.same_step_across_blocks = getattr(training_args,
                                               'same_step_across_blocks', False)
        self.last_step_only = getattr(training_args, 'last_step_only', False)
        self.context_noise = getattr(training_args, 'context_noise', 0)

        self.kv_cache1: list[dict[str, Any]] | None = None
        self.crossattn_cache: list[dict[str, Any]] | None = None

        if not self.same_step_across_blocks:
            logger.info(
                "Forcing same_step_across_blocks=True to align self-forcing DMD with Causal-Forcing."
            )
            self.same_step_across_blocks = True

        if getattr(self, "boundary_timestep", None) is not None:
            logger.info(
                "Disabling boundary_timestep for self-forcing DMD to avoid mixed-expert updates within one train step."
            )
            self.boundary_timestep = None
            self.train_transformer_2 = False
            self.train_fake_score_transformer_2 = False

        neg_prompt = str(getattr(training_args, "negative_prompt", "") or "").strip()
        if neg_prompt:
            model_root = training_args.pretrained_model_name_or_path or training_args.model_path
            tokenizer = PipelineComponentLoader.load_module(
                "tokenizer",
                os.path.join(model_root, "tokenizer"),
                "transformers",
                training_args,
            )
            text_encoder = PipelineComponentLoader.load_module(
                "text_encoder",
                os.path.join(model_root, "text_encoder"),
                "transformers",
                training_args,
            )
            max_text_len = int(getattr(self.transformer, "text_len", 226))
            self.negative_prompt_embeds, self.negative_prompt_attention_mask = (
                _compute_negative_prompt_embeddings(
                    tokenizer=tokenizer,
                    text_encoder=text_encoder,
                    negative_prompt=neg_prompt,
                    max_sequence_length=max_text_len,
                    dtype=torch.bfloat16,
                    target_device=get_local_torch_device(),
                )
            )
            logger.info("Using training negative_prompt: %s", neg_prompt)
            del tokenizer, text_encoder

        logger.info("Self-forcing generator update ratio: %s",
                    self.dfake_gen_update_ratio)
        logger.info("RANK: %s, exiting initialize_training_pipeline",
                    self.global_rank,
                    local_main_process_only=False)

    def _normalize_dit_input(self,
                             training_batch: TrainingBatch) -> TrainingBatch:
        # Self-forcing generator rollout synthesizes its own noise latents.
        # The placeholder batch.latents created by _get_next_batch are used for
        # shape bookkeeping only, so they should not be remapped from VAE latent
        # space through DiT input normalization.
        if getattr(self.training_args, "simulate_generator_forward", False):
            return training_batch
        return super()._normalize_dit_input(training_batch)

    def generate_and_sync_list(self, num_blocks: int, num_denoising_steps: int,
                               device: torch.device) -> list[int]:
        """Generate and synchronize random exit flags across distributed processes."""
        if sync_list_verbose_log:
            logger.info(
                "RANK: %s, enter generate_and_sync_list blocks=%s steps=%s device=%s",
                self.global_rank,
                num_blocks,
                num_denoising_steps,
                str(device),
                local_main_process_only=False)
        rank = dist.get_rank() if dist.is_initialized() else 0

        forced_exit_index = getattr(self, "_forced_exit_index", None)
        if rank == 0:
            if forced_exit_index is not None:
                indices = torch.full((num_blocks, ),
                                     int(forced_exit_index),
                                     device=device,
                                     dtype=torch.long)
            else:
                # Generate random indices
                indices = torch.randint(low=0,
                                        high=num_denoising_steps,
                                        size=(num_blocks, ),
                                        device=device)
                if self.last_step_only:
                    indices = torch.ones_like(indices) * (
                        num_denoising_steps - 1)
        else:
            indices = torch.empty(num_blocks, dtype=torch.long, device=device)

        if dist.is_initialized():
            dist.broadcast(indices,
                           src=0)  # Broadcast the random indices to all ranks
        flags = indices.tolist()
        if sync_list_verbose_log:
            logger.info(
                "RANK: %s, exit generate_and_sync_list flags_len=%s first=%s",
                self.global_rank,
                len(flags),
                flags[0] if len(flags) > 0 else None,
                local_main_process_only=False)
        return flags

    def _sample_shared_exit_index(self, *, device: torch.device) -> int:
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            if self.last_step_only:
                index = torch.tensor([len(self.denoising_step_list) - 1],
                                     dtype=torch.long,
                                     device=device)
            else:
                index = torch.randint(low=0,
                                      high=len(self.denoising_step_list),
                                      size=(1, ),
                                      device=device)
        else:
            index = torch.empty(1, dtype=torch.long, device=device)
        if dist.is_initialized():
            dist.broadcast(index, src=0)
        return int(index.item())

    def generator_loss(
            self, training_batch: TrainingBatch
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Compute generator loss using DMD-style approach.
        The generator tries to fool the critic (fake_score_transformer).
        """
        with set_forward_context(
                current_timestep=training_batch.timesteps,
                attn_metadata=training_batch.attn_metadata_vsa):
            rollout = self._generator_multi_step_simulation_forward(
                training_batch, return_sim_steps=True)
        if isinstance(rollout, tuple):
            (generator_pred_video, denoised_timestep_from,
             denoised_timestep_to, _) = rollout
        else:
            generator_pred_video = rollout
            denoised_timestep_from = None
            denoised_timestep_to = None
        training_batch.denoised_timestep_from = denoised_timestep_from
        training_batch.denoised_timestep_to = denoised_timestep_to

        with set_forward_context(current_timestep=training_batch.timesteps,
                                 attn_metadata=training_batch.attn_metadata):
            dmd_loss = self._dmd_forward(
                generator_pred_video=generator_pred_video,
                training_batch=training_batch)

        dmd_grad_norm = training_batch.dmd_latent_vis_dict.get(
            "dmdtrain_gradient_norm", torch.tensor(0.0, device=self.device))
        if not isinstance(dmd_grad_norm, torch.Tensor):
            dmd_grad_norm = torch.tensor(float(dmd_grad_norm),
                                         device=self.device)
        log_dict = {
            "dmdtrain_gradient_norm": dmd_grad_norm
        }

        return dmd_loss, log_dict

    def critic_loss(
            self, training_batch: TrainingBatch
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Compute critic loss using flow matching between noise and generator output.
        The critic learns to predict the flow from noise to the generator's output.
        """
        updated_batch, flow_matching_loss = self.faker_score_forward(
            training_batch)
        training_batch.fake_score_latent_vis_dict = updated_batch.fake_score_latent_vis_dict
        log_dict: dict[str, Any] = {}

        return flow_matching_loss, log_dict

    def _select_generator_model_for_timestep(self, timestep_value: float):
        if self.boundary_timestep is not None and self.transformer_2 is not None and timestep_value < self.boundary_timestep:
            return self.transformer_2
        return self.transformer

    def _decode_dit_latents_to_pixels(self, latents_bfchw: torch.Tensor,
                                      dtype: torch.dtype) -> torch.Tensor:
        latents = latents_bfchw.permute(0, 2, 1, 3, 4).contiguous()
        if hasattr(self.vae, "latents_mean") and hasattr(self.vae, "latents_std"):
            latents = _wan_denormalize_dit_latents(latents, self.vae)
        if isinstance(self.vae.scaling_factor, torch.Tensor):
            latents = latents / self.vae.scaling_factor.to(
                latents.device, latents.dtype)
        else:
            latents = latents / self.vae.scaling_factor
        if hasattr(self.vae, "shift_factor") and self.vae.shift_factor is not None:
            if isinstance(self.vae.shift_factor, torch.Tensor):
                latents = latents + self.vae.shift_factor.to(
                    latents.device, latents.dtype)
            else:
                latents = latents + self.vae.shift_factor
        if latents.device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pixels = self.vae.decode(latents)
        else:
            pixels = self.vae.decode(latents)
        return pixels.to(dtype)

    def _encode_pixels_to_dit_latents(self, pixels_bcfhw: torch.Tensor,
                                      dtype: torch.dtype) -> torch.Tensor:
        posterior = self.vae.encode(pixels_bcfhw)
        if hasattr(posterior, "mean"):
            latents = posterior.mean
        elif hasattr(posterior, "mode"):
            latents = posterior.mode()
        else:
            latents = posterior
        if hasattr(self.vae, "shift_factor") and self.vae.shift_factor is not None:
            if isinstance(self.vae.shift_factor, torch.Tensor):
                latents = latents - self.vae.shift_factor.to(
                    latents.device, latents.dtype)
            else:
                latents = latents - self.vae.shift_factor
        if getattr(self.vae, "scaling_factor", None) is not None:
            if isinstance(self.vae.scaling_factor, torch.Tensor):
                latents = latents * self.vae.scaling_factor.to(
                    latents.device, latents.dtype)
            else:
                latents = latents * self.vae.scaling_factor
        latents = normalize_dit_input("wan", latents, self.vae).to(dtype)
        return latents.permute(0, 2, 1, 3, 4).contiguous()

    def _generator_multi_step_simulation_forward(
            self,
            training_batch: TrainingBatch,
            return_sim_steps: bool = False):
        """Forward pass through student transformer matching inference procedure with KV cache management.
        
        This function is adapted from the reference self-forcing implementation's inference_with_trajectory
        and includes gradient masking logic for dynamic frame generation.
        """
        latents = training_batch.latents
        dtype = latents.dtype
        batch_size = latents.shape[0]
        initial_latent = getattr(training_batch, 'image_latent', None)

        # Dynamic frame generation logic (adapted from _run_generator)
        num_training_frames = getattr(self.training_args, 'num_latent_t', 21)

        # During training, the number of generated frames should be uniformly sampled from
        # [21, self.num_training_frames], but still being a multiple of self.num_frame_per_block
        min_num_frames = 20 if self.independent_first_frame else 21
        max_num_frames = num_training_frames - 1 if self.independent_first_frame else num_training_frames
        assert max_num_frames % self.num_frame_per_block == 0
        assert min_num_frames % self.num_frame_per_block == 0
        max_num_blocks = max_num_frames // self.num_frame_per_block
        min_num_blocks = min_num_frames // self.num_frame_per_block

        # Sample number of blocks and sync across processes
        num_generated_blocks = torch.randint(min_num_blocks,
                                             max_num_blocks + 1, (1, ),
                                             device=self.device)
        if dist.is_initialized():
            dist.broadcast(num_generated_blocks, src=0)
        num_generated_blocks = num_generated_blocks.item()
        num_generated_frames = num_generated_blocks * self.num_frame_per_block
        if self.independent_first_frame and initial_latent is None:
            num_generated_frames += 1
            min_num_frames += 1

        # Create noise with dynamic shape
        if initial_latent is not None:
            noise_shape = [
                batch_size, num_generated_frames - 1,
                *self.video_latent_shape[2:]
            ]
        else:
            noise_shape = [
                batch_size, num_generated_frames, *self.video_latent_shape[2:]
            ]

        noise = torch.randn(noise_shape, device=self.device, dtype=dtype)
        if self.sp_world_size > 1:
            noise = rearrange(noise,
                              "b (n t) c h w -> b n t c h w",
                              n=self.sp_world_size).contiguous()
            noise = noise[:, self.rank_in_sp_group, :, :, :, :]

        batch_size, num_frames, num_channels, height, width = noise.shape

        # Block size calculation
        if not self.independent_first_frame or (self.independent_first_frame
                                                and initial_latent is not None):
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block

        num_input_frames = initial_latent.shape[
            1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype)

        def get_model_device(model):
            if model is None:
                return "None"
            try:
                return next(model.parameters()).device
            except (StopIteration, AttributeError):
                return "Unknown"

        # Step 1: Initialize KV cache to all zeros
        cache_frames = num_generated_frames + num_input_frames
        self.kv_cache1, self.crossattn_cache = self._initialize_simulation_caches(
            batch_size, dtype, self.device, max_num_frames=cache_frames)
        self._init_additional_simulation_caches(batch_size, dtype, self.device,
                                                cache_frames)

        # Step 2: Cache context feature
        current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones(
                [batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            output[:, :1] = initial_latent
            with torch.no_grad():
                # Build input kwargs for initial latent
                training_batch_temp = self._build_distill_input_kwargs(
                    initial_latent, timestep * 0,
                    training_batch.conditional_dict, training_batch)
                current_model = self._select_generator_model_for_timestep(0.0)
                _ = self._simulation_model_forward_raw(
                    model=current_model,
                    training_batch_temp=training_batch_temp,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start_frame=current_start_frame,
                    start_frame=current_start_frame,
                    current_num_frames=1,
                )
            current_start_frame += 1

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        num_denoising_steps = len(self.denoising_step_list)
        exit_flags = self.generate_and_sync_list(len(all_num_frames),
                                                 num_denoising_steps,
                                                 device=noise.device)
        grad_last_n_frames = int(
            getattr(self.training_args, "gradient_mask_last_n_frames", 21) or 21
        )
        start_gradient_frame_index = max(0, num_output_frames - grad_last_n_frames)

        for block_index, current_num_frames in enumerate(all_num_frames):
            noisy_input = noise[:, current_start_frame -
                                num_input_frames:current_start_frame +
                                current_num_frames - num_input_frames]

            # Step 3.1: Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                if self.same_step_across_blocks:
                    exit_flag = (index == exit_flags[0])
                else:
                    exit_flag = (index == exit_flags[block_index])

                timestep = torch.ones([batch_size, current_num_frames],
                                      device=noise.device,
                                      dtype=torch.int64) * current_timestep

                current_model = self._select_generator_model_for_timestep(
                    float(current_timestep))

                if not exit_flag:
                    with torch.no_grad():
                        # Build input kwargs
                        training_batch_temp = self._build_distill_input_kwargs(
                            noisy_input, timestep,
                            training_batch.conditional_dict, training_batch)

                        pred_flow = self._simulation_predict_flow(
                            model=current_model,
                            training_batch_temp=training_batch_temp,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start_frame=current_start_frame,
                            start_frame=current_start_frame,
                            current_num_frames=current_num_frames,
                        )

                        denoised_pred = pred_noise_to_pred_video(
                            pred_noise=pred_flow.flatten(0, 1),
                            noise_input_latent=noisy_input.flatten(0, 1),
                            timestep=timestep,
                            scheduler=self.noise_scheduler).unflatten(
                                0, pred_flow.shape[:2])

                        next_timestep = self.denoising_step_list[index + 1]
                        noisy_input = self.noise_scheduler.add_noise(
                            denoised_pred.flatten(0, 1),
                            torch.randn_like(denoised_pred.flatten(0, 1)),
                            next_timestep *
                            torch.ones([batch_size * current_num_frames],
                                       device=noise.device,
                                       dtype=torch.long)).unflatten(
                                           0, denoised_pred.shape[:2])
                else:
                    # Final prediction with gradient control
                    if current_start_frame < start_gradient_frame_index:
                        with torch.no_grad():
                            training_batch_temp = self._build_distill_input_kwargs(
                                noisy_input, timestep,
                                training_batch.conditional_dict, training_batch)

                            pred_flow = self._simulation_predict_flow(
                                model=current_model,
                                training_batch_temp=training_batch_temp,
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                current_start_frame=current_start_frame,
                                start_frame=current_start_frame,
                                current_num_frames=current_num_frames,
                            )
                    else:
                        training_batch_temp = self._build_distill_input_kwargs(
                            noisy_input, timestep,
                            training_batch.conditional_dict, training_batch)

                        pred_flow = self._simulation_predict_flow(
                            model=current_model,
                            training_batch_temp=training_batch_temp,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start_frame=current_start_frame,
                            start_frame=current_start_frame,
                            current_num_frames=current_num_frames,
                        )

                    denoised_pred = pred_noise_to_pred_video(
                        pred_noise=pred_flow.flatten(0, 1),
                        noise_input_latent=noisy_input.flatten(0, 1),
                        timestep=timestep,
                        scheduler=self.noise_scheduler).unflatten(
                            0, pred_flow.shape[:2])
                    break

            # Step 3.2: record the model's output
            denoised_pred = self._simulation_postprocess_chunk_output(
                denoised_pred,
                training_batch=training_batch,
                current_start_frame=current_start_frame,
                current_num_frames=current_num_frames,
            )
            output[:, current_start_frame:current_start_frame +
                   current_num_frames] = denoised_pred

            # Step 3.3: rerun with timestep zero to update the cache
            context_timestep = torch.ones_like(timestep) * self.context_noise
            denoised_pred = self.noise_scheduler.add_noise(
                denoised_pred.flatten(0, 1),
                torch.randn_like(denoised_pred.flatten(0, 1)),
                context_timestep).unflatten(0, denoised_pred.shape[:2])

            with torch.no_grad():
                training_batch_temp = self._build_distill_input_kwargs(
                    denoised_pred, context_timestep,
                    training_batch.conditional_dict, training_batch)

                current_model = self._select_generator_model_for_timestep(
                    float(self.context_noise))
                _ = self._simulation_model_forward_raw(
                    model=current_model,
                    training_batch_temp=training_batch_temp,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start_frame=current_start_frame,
                    start_frame=current_start_frame,
                    current_num_frames=current_num_frames,
                )

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        # Handle "last N frames" logic (N = gradient_mask_last_n_frames).
        pred_image_or_video = output
        if num_input_frames > 0:
            pred_image_or_video = output[:, num_input_frames:]

        # Slice last N frames if we generated more
        gradient_mask = None
        if pred_image_or_video.shape[1] > grad_last_n_frames:
            with torch.no_grad():
                # Re-encode to get image latent
                # Keep the last (N-1) latent frames, and convert the preceding
                # part into a single "image latent" by decoding its last frame
                # then re-encoding.
                keep_last = max(0, grad_last_n_frames - 1)
                latent_to_decode = pred_image_or_video[:, :-keep_last, ...] if keep_last > 0 else pred_image_or_video
                pixels = self._decode_dit_latents_to_pixels(latent_to_decode,
                                                            dtype)
                frame = pixels[:, :, -1:, :, :].to(
                    dtype)  # Last frame [B, C, 1, H, W]
                image_latent = self._encode_pixels_to_dit_latents(frame, dtype)

            suffix = pred_image_or_video[:, -keep_last:, ...] if keep_last > 0 else pred_image_or_video[:, :0, ...]
            pred_image_or_video_last_n = torch.cat([image_latent, suffix], dim=1)
        else:
            pred_image_or_video_last_n = pred_image_or_video

        # Set up gradient mask if we generated more than minimum frames
        if num_generated_frames != min_num_frames:
            # Currently, we do not use gradient for the first chunk, since it contains image latents
            gradient_mask = torch.ones_like(pred_image_or_video_last_n,
                                            dtype=torch.bool)
            if self.independent_first_frame:
                gradient_mask[:, :1] = False
            else:
                gradient_mask[:, :self.num_frame_per_block] = False

        # Apply gradient masking if needed
        final_output = pred_image_or_video_last_n.to(dtype)
        if gradient_mask is not None:
            # Apply gradient masking: detach frames that shouldn't contribute gradients
            final_output = torch.where(
                gradient_mask,
                pred_image_or_video_last_n,  # Keep original values where gradient_mask is True
                pred_image_or_video_last_n.detach(
                )  # Detach where gradient_mask is False
            )

        # Store visualization data
        training_batch.dmd_latent_vis_dict["generator_timestep"] = (
            self.denoising_step_list[exit_flags[0]].detach().clone().to(
                device=self.device, dtype=torch.float32))

        # Store gradient mask information for debugging
        if gradient_mask is not None:
            training_batch.dmd_latent_vis_dict[
                "gradient_mask"] = gradient_mask.float()
            training_batch.dmd_latent_vis_dict[
                "num_generated_frames"] = torch.tensor(num_generated_frames,
                                                       dtype=torch.float32,
                                                       device=self.device)
            training_batch.dmd_latent_vis_dict["min_num_frames"] = torch.tensor(
                min_num_frames, dtype=torch.float32, device=self.device)

        # Clean up caches.
        # IMPORTANT: when gradients are enabled, the attention backward can save
        # K/V tensors (including from caches) to compute dQ. Resetting caches
        # in-place before backward would trigger "modified by an inplace operation"
        # autograd errors. In the grad-enabled path we simply keep the cache
        # tensors alive until backward completes; the next rollout will
        # re-initialize fresh caches anyway.
        if not torch.is_grad_enabled():
            assert self.kv_cache1 is not None
            assert self.crossattn_cache is not None
            self._reset_simulation_caches(self.kv_cache1, self.crossattn_cache)
            self._reset_additional_simulation_caches()

        output_tensor = final_output if gradient_mask is not None else pred_image_or_video
        denoised_timestep_from: int | None = None
        denoised_timestep_to: int | None = None
        if self.same_step_across_blocks and len(exit_flags) > 0:
            exit_idx = int(exit_flags[0])
            scheduler_timesteps = self.noise_scheduler.timesteps.to(self.device)
            from_t = self.denoising_step_list[exit_idx]
            denoised_timestep_from = int(self.num_train_timestep - torch.argmin(
                (scheduler_timesteps - from_t).abs()).item())
            if exit_idx == len(self.denoising_step_list) - 1:
                denoised_timestep_to = 0
            else:
                to_t = self.denoising_step_list[exit_idx + 1]
                denoised_timestep_to = int(self.num_train_timestep - torch.argmin(
                    (scheduler_timesteps - to_t).abs()).item())

        if return_sim_steps:
            return output_tensor, denoised_timestep_from, denoised_timestep_to, (
                int(exit_flags[0]) + 1 if len(exit_flags) > 0 else 0)
        return output_tensor

    def _simulation_postprocess_chunk_output(self,
                                             denoised_pred: torch.Tensor,
                                             *,
                                             training_batch: TrainingBatch,
                                             current_start_frame: int,
                                             current_num_frames: int
                                             ) -> torch.Tensor:
        """Hook for subclasses to postprocess each generated chunk output before caching/updating."""
        return denoised_pred

    def _init_additional_simulation_caches(self, batch_size: int,
                                           dtype: torch.dtype,
                                           device: torch.device,
                                           max_num_frames: int) -> None:
        """Hook for subclasses to initialize extra KV caches (e.g. ControlNet)."""
        return

    def _reset_additional_simulation_caches(self) -> None:
        """Hook for subclasses to reset extra KV caches (e.g. ControlNet)."""
        return

    def _simulation_model_forward_raw(
        self,
        *,
        model,
        training_batch_temp: TrainingBatch,
        kv_cache: list[dict[str, Any]],
        crossattn_cache: list[dict[str, Any]],
        current_start_frame: int,
        start_frame: int,
        current_num_frames: int,
    ) -> torch.Tensor:
        """Hookable model forward used by the self-forcing generator simulation."""
        return model(
            hidden_states=training_batch_temp.input_kwargs["hidden_states"],
            encoder_hidden_states=training_batch_temp.input_kwargs[
                "encoder_hidden_states"],
            timestep=training_batch_temp.input_kwargs["timestep"],
            encoder_hidden_states_image=training_batch_temp.input_kwargs.get(
                "encoder_hidden_states_image"),
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start=current_start_frame * self.frame_seq_length,
            start_frame=start_frame,
        )

    def _simulation_predict_flow(
        self,
        *,
        model,
        training_batch_temp: TrainingBatch,
        kv_cache: list[dict[str, Any]],
        crossattn_cache: list[dict[str, Any]],
        current_start_frame: int,
        start_frame: int,
        current_num_frames: int,
    ) -> torch.Tensor:
        """Return flow prediction in BFCHW for denoising loop."""
        pred = self._simulation_model_forward_raw(
            model=model,
            training_batch_temp=training_batch_temp,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start_frame=current_start_frame,
            start_frame=start_frame,
            current_num_frames=current_num_frames,
        )
        return pred.permute(0, 2, 1, 3, 4)

    def _initialize_simulation_caches(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
        *,
        max_num_frames: int | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Initialize KV cache and cross-attention cache for multi-step simulation."""
        num_transformer_blocks = len(self.transformer.blocks)
        latent_shape = self.video_latent_shape_sp
        _, num_frames, _, height, width = latent_shape

        _, p_h, p_w = self.transformer.patch_size
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        frame_seq_length = post_patch_height * post_patch_width
        self.frame_seq_length = frame_seq_length

        # Get model configuration parameters - handle FSDP wrapping
        num_attention_heads = getattr(self.transformer, 'num_attention_heads',
                                      None)
        attention_head_dim = getattr(self.transformer, 'attention_head_dim',
                                     None)
        text_len = getattr(self.transformer, 'text_len', None)

        if max_num_frames is None:
            max_num_frames = num_frames
        num_max_frames = max(max_num_frames, num_frames)
        kv_cache_size = num_max_frames * frame_seq_length

        kv_cache = []
        for _ in range(num_transformer_blocks):
            kv_cache.append({
                "k":
                torch.zeros([
                    batch_size, kv_cache_size, num_attention_heads,
                    attention_head_dim
                ],
                            dtype=dtype,
                            device=device),
                "v":
                torch.zeros([
                    batch_size, kv_cache_size, num_attention_heads,
                    attention_head_dim
                ],
                            dtype=dtype,
                            device=device),
                "global_end_index":
                torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index":
                torch.tensor([0], dtype=torch.long, device=device)
            })

        # Initialize cross-attention cache
        crossattn_cache = []
        for _ in range(num_transformer_blocks):
            crossattn_cache.append({
                "k":
                torch.zeros([
                    batch_size, text_len, num_attention_heads,
                    attention_head_dim
                ],
                            dtype=dtype,
                            device=device),
                "v":
                torch.zeros([
                    batch_size, text_len, num_attention_heads,
                    attention_head_dim
                ],
                            dtype=dtype,
                            device=device),
                "is_init":
                False
            })

        return kv_cache, crossattn_cache

    def _reset_simulation_caches(self, kv_cache: list[dict[str, Any]],
                                 crossattn_cache: list[dict[str, Any]]) -> None:
        """Reset KV cache and cross-attention cache to clean state."""
        if kv_cache is not None:
            for cache_dict in kv_cache:
                cache_dict["global_end_index"].fill_(0)
                cache_dict["local_end_index"].fill_(0)
                cache_dict["k"].zero_()
                cache_dict["v"].zero_()

        if crossattn_cache is not None:
            for cache_dict in crossattn_cache:
                cache_dict["is_init"] = False
                cache_dict["k"].zero_()
                cache_dict["v"].zero_()

    def _get_next_batch(self, training_batch: TrainingBatch) -> TrainingBatch:
        batch = next(self.train_loader_iter, None)  # type: ignore
        if batch is None:
            self.current_epoch += 1
            # Reset iterator for next epoch
            self.train_loader_iter = iter(self.train_dataloader)
            # Get first batch of new epoch
            batch = next(self.train_loader_iter)

        # latents, encoder_hidden_states, encoder_attention_mask, infos = batch
        encoder_hidden_states = batch['text_embedding']
        encoder_attention_mask = batch['text_attention_mask']
        infos = batch['info_list']

        batch_size = encoder_hidden_states.shape[0]
        vae_config = self.training_args.pipeline_config.vae_config.arch_config
        num_channels = vae_config.z_dim
        spatial_compression_ratio = vae_config.spatial_compression_ratio

        latent_height = self.training_args.num_height // spatial_compression_ratio
        latent_width = self.training_args.num_width // spatial_compression_ratio

        latents = torch.randn(batch_size, num_channels,
                              self.training_args.num_latent_t, latent_height,
                              latent_width).to(get_local_torch_device(),
                                               dtype=torch.bfloat16)

        training_batch.latents = latents.to(get_local_torch_device(),
                                            dtype=torch.bfloat16)
        training_batch.encoder_hidden_states = encoder_hidden_states.to(
            get_local_torch_device(), dtype=torch.bfloat16)
        training_batch.encoder_attention_mask = encoder_attention_mask.to(
            get_local_torch_device(), dtype=torch.bfloat16)
        training_batch.infos = infos
        return training_batch

    def train_one_step(self, training_batch: TrainingBatch) -> TrainingBatch:
        """
        Self-forcing training step that alternates between generator and critic training.
        """
        gradient_accumulation_steps = getattr(self.training_args,
                                              'gradient_accumulation_steps', 1)
        train_generator = (self.current_trainstep %
                           self.dfake_gen_update_ratio == 0)

        batches = []
        for _ in range(gradient_accumulation_steps):
            batch = self._get_next_batch(training_batch)
            batch = self._normalize_dit_input(batch)
            batch = self._prepare_dit_inputs(batch)
            batch = self._build_attention_metadata(batch)
            batch.attn_metadata_vsa = copy.deepcopy(batch.attn_metadata)
            if batch.attn_metadata is not None:
                batch.attn_metadata.VSA_sparsity = 0.0
            batches.append(batch)

        training_batch.dmd_latent_vis_dict = {}
        training_batch.fake_score_latent_vis_dict = {}

        if train_generator:
            logger.debug("Training generator at step %s",
                         self.current_trainstep)
            self.optimizer.zero_grad()
            if self.transformer_2 is not None:
                self.optimizer_2.zero_grad()
            total_generator_loss = 0.0
            generator_log_dict = {}
            self._forced_exit_index = self._sample_shared_exit_index(
                device=self.device)

            for batch in batches:
                # Create a new batch with detached tensors
                batch_gen = TrainingBatch()
                for key, value in batch.__dict__.items():
                    if isinstance(value, torch.Tensor):
                        setattr(batch_gen, key, value.detach().clone())
                    elif isinstance(value, dict):
                        setattr(
                            batch_gen, key, {
                                k:
                                v.detach().clone() if isinstance(
                                    v, torch.Tensor) else copy.deepcopy(v)
                                for k, v in value.items()
                            })
                    else:
                        setattr(batch_gen, key, copy.deepcopy(value))

                generator_loss, gen_log_dict = self.generator_loss(batch_gen)
                with set_forward_context(current_timestep=batch_gen.timesteps,
                                         attn_metadata=batch_gen.attn_metadata):
                    (generator_loss / gradient_accumulation_steps).backward()
                total_generator_loss += generator_loss.detach().item()
                generator_log_dict.update(gen_log_dict)
                # Store visualization data from generator training
                if hasattr(batch_gen, 'dmd_latent_vis_dict'):
                    training_batch.dmd_latent_vis_dict.update(
                        batch_gen.dmd_latent_vis_dict)

            generator_timestep = training_batch.dmd_latent_vis_dict.get(
                "generator_timestep", None)
            train_generator_expert_2 = bool(
                self.transformer_2 is not None and self.boundary_timestep is not None
                and generator_timestep is not None
                and float(torch.as_tensor(generator_timestep).float().mean().item())
                < float(self.boundary_timestep))
            training_batch.dmd_latent_vis_dict[
                "train_generator_expert_2"] = torch.tensor(
                    1.0 if train_generator_expert_2 else 0.0,
                    device=self.device)
            del self._forced_exit_index

            # Only clip gradients and step optimizer for the model that is currently training
            if train_generator_expert_2 and self.transformer_2 is not None:
                self._clip_model_grad_norm_(batch_gen, self.transformer_2)
                self.optimizer_2.step()
                self.lr_scheduler_2.step()
            else:
                self._clip_model_grad_norm_(batch_gen, self.transformer)
                self.optimizer.step()
                self.lr_scheduler.step()

            if self.generator_ema is not None:
                if train_generator_expert_2 and self.transformer_2 is not None:
                    # Update EMA for transformer_2 when training it
                    if self.generator_ema_2 is not None:
                        self.generator_ema_2.update(self.transformer_2)
                else:
                    self.generator_ema.update(self.transformer)

            avg_generator_loss = torch.tensor(total_generator_loss /
                                              gradient_accumulation_steps,
                                              device=self.device)
            world_group = get_world_group()
            world_group.all_reduce(avg_generator_loss,
                                   op=torch.distributed.ReduceOp.AVG)
            training_batch.generator_loss = avg_generator_loss.item()
        else:
            training_batch.generator_loss = 0.0

        logger.debug("Training critic at step %s", self.current_trainstep)
        self.fake_score_optimizer.zero_grad()
        total_critic_loss = 0.0
        critic_log_dict = {}

        for batch in batches:
            # Create a new batch with detached tensors
            batch_critic = TrainingBatch()
            for key, value in batch.__dict__.items():
                if isinstance(value, torch.Tensor):
                    setattr(batch_critic, key, value.detach().clone())
                elif isinstance(value, dict):
                    setattr(
                        batch_critic, key, {
                            k:
                            v.detach().clone()
                            if isinstance(v, torch.Tensor) else copy.deepcopy(v)
                            for k, v in value.items()
                        })
                else:
                    setattr(batch_critic, key, copy.deepcopy(value))

            critic_loss, crit_log_dict = self.critic_loss(batch_critic)
            with set_forward_context(current_timestep=batch_critic.timesteps,
                                     attn_metadata=batch_critic.attn_metadata):
                (critic_loss / gradient_accumulation_steps).backward()
            total_critic_loss += critic_loss.detach().item()
            critic_log_dict.update(crit_log_dict)
            # Store visualization data from critic training
            if hasattr(batch_critic, 'fake_score_latent_vis_dict'):
                training_batch.fake_score_latent_vis_dict.update(
                    batch_critic.fake_score_latent_vis_dict)

        if self.train_fake_score_transformer_2 and self.fake_score_transformer_2 is not None:
            self._clip_model_grad_norm_(batch_critic,
                                        self.fake_score_transformer_2)
            self.fake_score_optimizer_2.step()
            self.fake_score_lr_scheduler_2.step()
        else:
            self._clip_model_grad_norm_(batch_critic,
                                        self.fake_score_transformer)
            self.fake_score_optimizer.step()
            self.fake_score_lr_scheduler.step()

        avg_critic_loss = torch.tensor(total_critic_loss /
                                       gradient_accumulation_steps,
                                       device=self.device)
        world_group = get_world_group()
        world_group.all_reduce(avg_critic_loss,
                               op=torch.distributed.ReduceOp.AVG)
        training_batch.fake_score_loss = avg_critic_loss.item()

        training_batch.total_loss = training_batch.generator_loss + training_batch.fake_score_loss
        return training_batch

    def _log_training_info(self) -> None:
        """Log self-forcing specific training information."""
        super()._log_training_info()
        logger.info("Self-forcing specific settings:")
        logger.info("  Generator update ratio: %s", self.dfake_gen_update_ratio)

    def visualize_intermediate_latents(self, training_batch: TrainingBatch,
                                       training_args: TrainingArgs, step: int):
        """Add visualization data to tracker logging and save frames to disk."""
        tracker_loss_dict: dict[str, Any] = {}

        # Debug logging
        if hasattr(training_batch, 'dmd_latent_vis_dict'):
            logger.info("DMD latent keys: %s",
                        list(training_batch.dmd_latent_vis_dict.keys()))
        if hasattr(training_batch, 'fake_score_latent_vis_dict'):
            logger.info("Fake score latent keys: %s",
                        list(training_batch.fake_score_latent_vis_dict.keys()))

        # Process generator predictions if available
        if hasattr(
                training_batch,
                'dmd_latent_vis_dict') and training_batch.dmd_latent_vis_dict:
            dmd_latents_vis_dict = training_batch.dmd_latent_vis_dict
            dmd_log_keys = [
                'generator_pred_video', 'real_score_pred_video',
                'faker_score_pred_video'
            ]

            for latent_key in dmd_log_keys:
                if latent_key in dmd_latents_vis_dict:
                    logger.info("Processing DMD latent: %s", latent_key)
                    latents = dmd_latents_vis_dict[latent_key]
                    if not isinstance(latents, torch.Tensor):
                        logger.warning("Expected tensor for %s, got %s",
                                       latent_key, type(latents))
                        continue

                    latents = latents.detach()
                    video = self._decode_dit_latents_to_pixels(
                        latents, torch.float32)
                    video = (video / 2 + 0.5).clamp(0, 1)
                    video = video.cpu().float()
                    video = video.permute(0, 2, 1, 3, 4)
                    video = (video * 255).numpy().astype(np.uint8)
                    video_artifact = self.tracker.video(video,
                                                        fps=24,
                                                        format="mp4")
                    if video_artifact is not None:
                        tracker_loss_dict[f"dmd_{latent_key}"] = video_artifact
                    del video, latents

        # Process critic predictions
        if hasattr(training_batch, 'fake_score_latent_vis_dict'
                   ) and training_batch.fake_score_latent_vis_dict:
            fake_score_latents_vis_dict = training_batch.fake_score_latent_vis_dict
            fake_score_log_keys = ['generator_pred_video']

            for latent_key in fake_score_log_keys:
                if latent_key in fake_score_latents_vis_dict:
                    logger.info("Processing critic latent: %s", latent_key)
                    latents = fake_score_latents_vis_dict[latent_key]
                    if not isinstance(latents, torch.Tensor):
                        logger.warning("Expected tensor for %s, got %s",
                                       latent_key, type(latents))
                        continue

                    latents = latents.detach()
                    video = self._decode_dit_latents_to_pixels(
                        latents, torch.float32)
                    video = (video / 2 + 0.5).clamp(0, 1)
                    video = video.cpu().float()
                    video = video.permute(0, 2, 1, 3, 4)
                    video = (video * 255).numpy().astype(np.uint8)
                    video_artifact = self.tracker.video(video,
                                                        fps=24,
                                                        format="mp4")
                    if video_artifact is not None:
                        tracker_loss_dict[
                            f"critic_{latent_key}"] = video_artifact
                    del video, latents

        # Log metadata
        if hasattr(
                training_batch,
                'dmd_latent_vis_dict') and training_batch.dmd_latent_vis_dict:
            if "generator_timestep" in training_batch.dmd_latent_vis_dict:
                tracker_loss_dict[
                    "generator_timestep"] = training_batch.dmd_latent_vis_dict[
                        "generator_timestep"].float().mean().item()
            if "dmd_timestep" in training_batch.dmd_latent_vis_dict:
                tracker_loss_dict[
                    "dmd_timestep"] = training_batch.dmd_latent_vis_dict[
                        "dmd_timestep"].float().mean().item()

        if hasattr(
                training_batch, 'fake_score_latent_vis_dict'
        ) and training_batch.fake_score_latent_vis_dict and "fake_score_timestep" in training_batch.fake_score_latent_vis_dict:
            tracker_loss_dict[
                "fake_score_timestep"] = training_batch.fake_score_latent_vis_dict[
                    "fake_score_timestep"].float().mean().item()

        # Log final dict contents
        logger.info("Final tracker_loss_dict keys: %s",
                    list(tracker_loss_dict.keys()))

        if self.global_rank == 0 and tracker_loss_dict:
            self.tracker.log_artifacts(tracker_loss_dict, step)

    @torch.no_grad()
    def _decode_latents_to_video_uint8(
        self,
        latents_bfchw: torch.Tensor,
        *,
        max_frames: int = 0,
    ) -> np.ndarray:
        video = self._decode_dit_latents_to_pixels(latents_bfchw.detach(),
                                                   torch.float32)
        video = (video / 2 + 0.5).clamp(0, 1).cpu().float()
        video = video.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
        if max_frames and max_frames > 0:
            video = video[:, :max_frames]
        video = (video * 255).numpy().astype(np.uint8)
        return video

    @torch.no_grad()
    def _maybe_log_checkpoint_preview(self, training_batch: TrainingBatch,
                                      step: int, *, tag: str) -> None:
        if self.global_rank != 0:
            return
        if not getattr(self.training_args, "checkpoint_preview", False):
            return

        latents = None
        source = None
        if getattr(training_batch, "dmd_latent_vis_dict", None):
            latents = training_batch.dmd_latent_vis_dict.get(
                "generator_pred_video")
            source = "dmd/generator_pred_video"
        if latents is None and getattr(training_batch, "fake_score_latent_vis_dict",
                                       None):
            latents = training_batch.fake_score_latent_vis_dict.get(
                "generator_pred_video")
            source = "critic/generator_pred_video"

        if not isinstance(latents, torch.Tensor):
            logger.info(
                "Skipping checkpoint preview at step %s (no generator latents in vis dicts)",
                step)
            return

        max_frames = int(getattr(self.training_args, "checkpoint_preview_max_frames",
                                 0) or 0)
        fps = int(getattr(self.training_args, "checkpoint_preview_fps", 24) or 24)

        video_uint8 = self._decode_latents_to_video_uint8(latents,
                                                          max_frames=max_frames)

        # 1) Log to tracker (wandb if enabled)
        caption = f"{tag} step={step} source={source}"
        video_artifact = self.tracker.video(video_uint8,
                                            fps=fps,
                                            format="mp4",
                                            caption=caption)
        if video_artifact is not None:
            self.tracker.log_artifacts({f"checkpoint_preview/{tag}": video_artifact},
                                       step)

        # 2) Save mp4 to disk under output_dir for quick local inspection
        try:
            out_dir = os.path.join(self.training_args.output_dir,
                                   "checkpoint_previews")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{tag}_step_{step:07d}.mp4")
            frames = video_uint8[0].transpose(0, 2, 3, 1)  # (T,H,W,C)
            imageio.mimwrite(out_path, frames, fps=fps)
            logger.info("Saved checkpoint preview video: %s", out_path)
        except Exception as e:
            logger.warning("Failed to save checkpoint preview mp4: %s", e)

    @profile_region("profiler_region_training_train")
    def train(self) -> None:
        """Main training loop with self-forcing specific logging."""
        assert self.training_args.seed is not None, "seed must be set"
        seed = self.training_args.seed

        # Set the same seed within each SP group to ensure reproducibility
        if self.sp_world_size > 1:
            # Use the same seed for all processes within the same SP group
            sp_group_seed = seed + (self.global_rank // self.sp_world_size)
            set_random_seed(sp_group_seed)
        else:
            set_random_seed(seed + self.global_rank)

        self.noise_random_generator = torch.Generator(device="cpu").manual_seed(
            self.seed)
        self.noise_gen_cuda = torch.Generator(device="cuda").manual_seed(
            self.seed)
        self.validation_random_generator = torch.Generator(
            device="cpu").manual_seed(self.seed)
        logger.info("Initialized random seeds with seed: %s", seed)

        self.current_trainstep = self.init_steps

        if self.training_args.resume_from_checkpoint:
            self._resume_from_checkpoint()
            logger.info("Resumed from checkpoint, random states restored")
        else:
            logger.info("Starting training from scratch")

        self.train_loader_iter = iter(self.train_dataloader)

        step_times: deque[float] = deque(maxlen=100)

        self._log_training_info()
        self._log_validation(self.transformer, self.training_args,
                             self.init_steps)

        progress_bar = tqdm(
            range(0, self.training_args.max_train_steps),
            initial=self.init_steps,
            desc="Steps",
            disable=self.local_rank > 0,
        )

        use_vsa = vsa_available and envs.FASTVIDEO_ATTENTION_BACKEND == "VIDEO_SPARSE_ATTN"
        for step in range(self.init_steps + 1,
                          self.training_args.max_train_steps + 1):
            start_time = time.perf_counter()
            if use_vsa:
                vsa_sparsity = self.training_args.VSA_sparsity
                vsa_decay_rate = self.training_args.VSA_decay_rate
                vsa_decay_interval_steps = self.training_args.VSA_decay_interval_steps
                if vsa_decay_interval_steps > 1:
                    current_decay_times = min(step // vsa_decay_interval_steps,
                                              vsa_sparsity // vsa_decay_rate)
                    current_vsa_sparsity = current_decay_times * vsa_decay_rate
                else:
                    current_vsa_sparsity = vsa_sparsity
            else:
                current_vsa_sparsity = 0.0

            training_batch = TrainingBatch()
            self.current_trainstep = step
            training_batch.current_vsa_sparsity = current_vsa_sparsity

            if (step >= self.training_args.ema_start_step) and \
                    (self.generator_ema is None) and (self.training_args.ema_decay > 0):
                self.generator_ema = EMA_FSDP(
                    self.transformer, decay=self.training_args.ema_decay)
                logger.info("Created generator EMA at step %s with decay=%s",
                            step, self.training_args.ema_decay)

                # Create EMA for transformer_2 if it exists
                if self.transformer_2 is not None and self.generator_ema_2 is None:
                    self.generator_ema_2 = EMA_FSDP(
                        self.transformer_2, decay=self.training_args.ema_decay)
                    logger.info(
                        "Created generator EMA_2 at step %s with decay=%s",
                        step, self.training_args.ema_decay)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                training_batch = self.train_one_step(training_batch)

            total_loss = training_batch.total_loss
            generator_loss = training_batch.generator_loss
            fake_score_loss = training_batch.fake_score_loss
            grad_norm = training_batch.grad_norm

            step_time = time.perf_counter() - start_time
            step_times.append(step_time)
            avg_step_time = sum(step_times) / len(step_times)

            progress_bar.set_postfix({
                "total_loss":
                f"{total_loss:.4f}",
                "generator_loss":
                f"{generator_loss:.4f}",
                "fake_score_loss":
                f"{fake_score_loss:.4f}",
                "step_time":
                f"{step_time:.2f}s",
                "grad_norm":
                grad_norm,
                "ema":
                "✓" if (self.generator_ema is not None and self.is_ema_ready())
                else "✗",
                "ema2":
                "✓" if (self.generator_ema_2 is not None
                        and self.is_ema_ready()) else "✗",
            })
            progress_bar.update(1)

            if self.global_rank == 0:
                log_data = {
                    "train_total_loss":
                    total_loss,
                    "train_fake_score_loss":
                    fake_score_loss,
                    "learning_rate":
                    self.lr_scheduler.get_last_lr()[0],
                    "fake_score_learning_rate":
                    self.fake_score_lr_scheduler.get_last_lr()[0],
                    "step_time":
                    step_time,
                    "avg_step_time":
                    avg_step_time,
                    "grad_norm":
                    grad_norm,
                }
                if (step % self.dfake_gen_update_ratio == 0):
                    log_data["train_generator_loss"] = generator_loss
                if use_vsa:
                    log_data["VSA_train_sparsity"] = current_vsa_sparsity

                if self.generator_ema is not None or self.generator_ema_2 is not None:
                    log_data["ema_enabled"] = self.generator_ema is not None
                    log_data["ema_2_enabled"] = self.generator_ema_2 is not None
                    log_data["ema_decay"] = self.training_args.ema_decay
                else:
                    log_data["ema_enabled"] = False
                    log_data["ema_2_enabled"] = False

                ema_stats = self.get_ema_stats()
                log_data.update(ema_stats)

                if training_batch.dmd_latent_vis_dict:
                    if "generator_timestep" in training_batch.dmd_latent_vis_dict:
                        log_data["generator_timestep"] = training_batch.dmd_latent_vis_dict[
                            "generator_timestep"].float().mean().item()
                    if "dmd_timestep" in training_batch.dmd_latent_vis_dict:
                        log_data["dmd_timestep"] = training_batch.dmd_latent_vis_dict[
                            "dmd_timestep"].float().mean().item()
                    if "dmd_grad_normalizer" in training_batch.dmd_latent_vis_dict:
                        log_data["dmd_grad_normalizer"] = training_batch.dmd_latent_vis_dict[
                            "dmd_grad_normalizer"].float().mean().item()
                    if "dmd_grad_normalizer_raw" in training_batch.dmd_latent_vis_dict:
                        log_data["dmd_grad_normalizer_raw"] = training_batch.dmd_latent_vis_dict[
                            "dmd_grad_normalizer_raw"].float().mean().item()
                    if "dmd_grad_normalizer_ema" in training_batch.dmd_latent_vis_dict:
                        log_data["dmd_grad_normalizer_ema"] = training_batch.dmd_latent_vis_dict[
                            "dmd_grad_normalizer_ema"].float().mean().item()

                faker_score_additional_logs = {
                    "fake_score_timestep":
                    _to_log_scalar(training_batch.fake_score_latent_vis_dict[
                        "fake_score_timestep"]),
                }
                log_data.update(faker_score_additional_logs)

                self.tracker.log(log_data, step)

                if self.training_args.log_validation and step % self.training_args.validation_steps == 0 and self.training_args.log_visualization:
                    self.visualize_intermediate_latents(training_batch,
                                                        self.training_args,
                                                        step)

            if (self.training_args.training_state_checkpointing_steps > 0
                    and step %
                    self.training_args.training_state_checkpointing_steps == 0):
                print("rank", self.global_rank,
                      "save training state checkpoint at step", step)
                save_distillation_checkpoint(
                    self.transformer,
                    self.fake_score_transformer,
                    self.global_rank,
                    self.training_args.output_dir,
                    step,
                    self.optimizer,
                    self.fake_score_optimizer,
                    self.train_dataloader,
                    self.lr_scheduler,
                    self.fake_score_lr_scheduler,
                    self.noise_random_generator,
                    self.generator_ema,
                    save_consolidated_inference_checkpoint=False,
                    generator_controlnet=getattr(self, "controlnet", None),
                    fake_score_controlnet=getattr(self, "fake_score_controlnet",
                                                  None),
                    # MoE support
                    generator_transformer_2=getattr(self, 'transformer_2',
                                                    None),
                    real_score_transformer_2=getattr(
                        self, 'real_score_transformer_2', None),
                    fake_score_transformer_2=getattr(
                        self, 'fake_score_transformer_2', None),
                    generator_optimizer_2=getattr(self, 'optimizer_2', None),
                    fake_score_optimizer_2=getattr(self,
                                                   'fake_score_optimizer_2',
                                                   None),
                    generator_scheduler_2=getattr(self, 'lr_scheduler_2', None),
                    fake_score_scheduler_2=getattr(self,
                                                   'fake_score_lr_scheduler_2',
                                                   None),
                    generator_ema_2=getattr(self, 'generator_ema_2', None))

                self._maybe_log_checkpoint_preview(training_batch,
                                                   step,
                                                   tag="train_state")

                if self.transformer:
                    self.transformer.train()
                self.sp_group.barrier()

            if (self.training_args.weight_only_checkpointing_steps > 0
                    and step %
                    self.training_args.weight_only_checkpointing_steps == 0):
                print("rank", self.global_rank,
                      "save weight-only checkpoint at step", step)
                save_distillation_checkpoint(
                    self.transformer,
                    self.fake_score_transformer,
                    self.global_rank,
                    self.training_args.output_dir,
                    f"{step}_weight_only",
                    only_save_generator_weight=True,
                    generator_ema=self.generator_ema,
                    generator_controlnet=getattr(self, "controlnet", None),
                    fake_score_controlnet=getattr(self, "fake_score_controlnet",
                                                  None),
                    # MoE support
                    generator_transformer_2=getattr(self, 'transformer_2',
                                                    None),
                    real_score_transformer_2=getattr(
                        self, 'real_score_transformer_2', None),
                    fake_score_transformer_2=getattr(
                        self, 'fake_score_transformer_2', None),
                    generator_optimizer_2=getattr(self, 'optimizer_2', None),
                    fake_score_optimizer_2=getattr(self,
                                                   'fake_score_optimizer_2',
                                                   None),
                    generator_scheduler_2=getattr(self, 'lr_scheduler_2', None),
                    fake_score_scheduler_2=getattr(self,
                                                   'fake_score_lr_scheduler_2',
                                                   None),
                    generator_ema_2=getattr(self, 'generator_ema_2', None))

                self._maybe_log_checkpoint_preview(training_batch,
                                                   step,
                                                   tag="weight_only")

                if self.training_args.use_ema and self.is_ema_ready():
                    self.save_ema_weights(self.training_args.output_dir, step)

            if self.training_args.log_validation and step % self.training_args.validation_steps == 0:
                self._log_validation(self.transformer, self.training_args, step)

        self.tracker.finish()

        print("rank", self.global_rank,
              "save final training state checkpoint at step",
              self.training_args.max_train_steps)
        save_distillation_checkpoint(
            self.transformer,
            self.fake_score_transformer,
            self.global_rank,
            self.training_args.output_dir,
            self.training_args.max_train_steps,
            self.optimizer,
            self.fake_score_optimizer,
            self.train_dataloader,
            self.lr_scheduler,
            self.fake_score_lr_scheduler,
            self.noise_random_generator,
            self.generator_ema,
            generator_controlnet=getattr(self, "controlnet", None),
            fake_score_controlnet=getattr(self, "fake_score_controlnet", None),
            # MoE support
            generator_transformer_2=getattr(self, 'transformer_2', None),
            real_score_transformer_2=getattr(self, 'real_score_transformer_2',
                                             None),
            fake_score_transformer_2=getattr(self, 'fake_score_transformer_2',
                                             None),
            generator_optimizer_2=getattr(self, 'optimizer_2', None),
            fake_score_optimizer_2=getattr(self, 'fake_score_optimizer_2',
                                           None),
            generator_scheduler_2=getattr(self, 'lr_scheduler_2', None),
            fake_score_scheduler_2=getattr(self, 'fake_score_lr_scheduler_2',
                                           None),
            generator_ema_2=getattr(self, 'generator_ema_2', None))

        if self.training_args.use_ema and self.is_ema_ready():
            self.save_ema_weights(self.training_args.output_dir,
                                  self.training_args.max_train_steps)

        if envs.FASTVIDEO_TORCH_PROFILER_DIR:
            logger.info("Stopping profiler...")
            self.profiler_controller.stop()
            logger.info("Profiler stopped.")

        if get_sp_group():
            cleanup_dist_env_and_memory()
