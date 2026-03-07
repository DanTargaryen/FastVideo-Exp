# SPDX-License-Identifier: Apache-2.0
import math
import sys
from copy import deepcopy
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F

from fastvideo.configs.sample.base import SamplingParam
from fastvideo.dataset.dataloader.schema import pyarrow_schema_ti2v_controlnet
from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.logger import init_logger
from fastvideo.models.utils import pred_noise_to_pred_video
from fastvideo.models.loader.component_loader import PipelineComponentLoader
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.forward_context import set_forward_context
from fastvideo.training.self_forcing_distillation_pipeline import (
    SelfForcingDistillationPipeline,
)
from fastvideo.training.activation_checkpoint import apply_activation_checkpointing
from fastvideo.training.training_utils import (
    clip_grad_norm_while_handling_failing_dtensor_cases,
    get_scheduler,
    normalize_dit_input,
)
from fastvideo.utils import is_vsa_available, shallow_asdict
from fastvideo.models.dits.controlnet_union_components import WanControlNetUnionInput

vsa_available = is_vsa_available()
logger = init_logger(__name__)


def _is_union_controlnet(model) -> bool:
    return "union" in model.__class__.__name__.lower()


def _split_union_control_latent(control_latent: torch.Tensor,
                                num_channels_latents: int
                                ) -> tuple[torch.Tensor, torch.Tensor | None,
                                           torch.Tensor, torch.Tensor]:
    c = int(num_channels_latents)
    if control_latent.shape[1] == 3 * c:
        depth = control_latent[:, :c]
        masked = control_latent[:, c:2 * c]
        mask = control_latent[:, 2 * c:3 * c]
        normal = None
        return depth, normal, masked, mask
    if control_latent.shape[1] == 4 * c:
        depth = control_latent[:, :c]
        normal = control_latent[:, c:2 * c]
        masked = control_latent[:, 2 * c:3 * c]
        mask = control_latent[:, 3 * c:4 * c]
        return depth, normal, masked, mask
    raise ValueError(
        f"Union control_latent channel mismatch: got {control_latent.shape[1]}, "
        f"expected 3*C or 4*C (C={c}).")


def _build_controlnet_kwargs(controlnet, control_latent: torch.Tensor,
                             num_channels_latents: int) -> dict:
    if not _is_union_controlnet(controlnet):
        return {"controlnet_states": control_latent}
    depth, normal, masked, mask = _split_union_control_latent(
        control_latent, num_channels_latents)
    return {
        "controlnet_cond": WanControlNetUnionInput(depth=depth, normal=normal),
        "mask": mask,
        "masked_latent": masked,
    }


def _normalize_first_frame_latent(first_frame_latent: torch.Tensor,
                                  vae) -> torch.Tensor:
    if first_frame_latent.ndim != 5:
        return first_frame_latent
    latent_bcfhw = first_frame_latent.permute(0, 2, 1, 3, 4).contiguous()
    latent_bcfhw = normalize_dit_input("wan", latent_bcfhw, vae)
    return latent_bcfhw.permute(0, 2, 1, 3, 4).contiguous()


def _normalize_control_latent(control_latent: torch.Tensor, vae,
                              num_channels_latents: int) -> torch.Tensor:
    if control_latent.ndim != 5:
        return control_latent
    chunks = []
    total_channels = control_latent.shape[1]
    if total_channels % num_channels_latents != 0:
        raise ValueError(
            f"control_latent channels {total_channels} not divisible by latent channels {num_channels_latents}"
        )
    for start in range(0, total_channels, num_channels_latents):
        chunk = control_latent[:, start:start + num_channels_latents]
        chunks.append(normalize_dit_input("wan", chunk, vae))
    return torch.cat(chunks, dim=1)


def _apply_first_frame_latent(
    hidden_states: torch.Tensor,
    first_frame_latent: torch.Tensor | None,
) -> torch.Tensor:
    if first_frame_latent is None:
        return hidden_states
    if hidden_states.ndim != 5 or first_frame_latent.ndim != 5:
        return hidden_states
    image_latent = first_frame_latent.permute(0, 2, 1, 3, 4).contiguous()
    if hidden_states.shape[1] != image_latent.shape[1] or hidden_states.shape[
            2] < 1:
        return hidden_states
    conditioned = hidden_states.clone()
    conditioned[:, :, :1] = image_latent[:, :, :1]
    return conditioned


def _sample_uniform_score_timestep(*,
                                   batch_size: int,
                                   device: torch.device,
                                   num_train_timestep: int,
                                   timestep_shift: float,
                                   min_timestep: int,
                                   max_timestep: int,
                                   denoised_timestep_from: int | None,
                                   denoised_timestep_to: int | None
                                   ) -> torch.Tensor:
    raw_min_timestep = int(min_timestep)
    raw_max_timestep = int(max_timestep)
    if denoised_timestep_to is not None:
        raw_min_timestep = max(raw_min_timestep, int(denoised_timestep_to))
    if denoised_timestep_from is not None:
        raw_max_timestep = min(raw_max_timestep, int(denoised_timestep_from))
    if raw_max_timestep < raw_min_timestep:
        raw_min_timestep, raw_max_timestep = min_timestep, max_timestep

    timestep = torch.randint(raw_min_timestep,
                             raw_max_timestep + 1, [batch_size],
                             device=device,
                             dtype=torch.long)
    from fastvideo.training.training_utils import shift_timestep
    timestep = shift_timestep(timestep, timestep_shift, num_train_timestep)
    return timestep.clamp(min_timestep, max_timestep)


def _np_dtype(dtype_str: str | None) -> np.dtype:
    if dtype_str is None or dtype_str == "":
        return np.float32
    s = dtype_str.lower()
    if s in ("float", "float32", "fp32"):
        return np.float32
    if s in ("float16", "fp16"):
        return np.float16
    if s in ("int64", "long"):
        return np.int64
    if s in ("int32",):
        return np.int32
    raise ValueError(f"Unsupported dtype in validation row: {dtype_str}")


def _decode_validation_tensor(row: dict[str, Any], prefix: str) -> torch.Tensor:
    shape = row.get(f"{prefix}_shape")
    blob = row.get(f"{prefix}_bytes")
    dtype_str = row.get(f"{prefix}_dtype")
    if blob is None or shape is None:
        raise KeyError(
            f"Missing {prefix}_bytes/{prefix}_shape in validation sample")
    array = np.frombuffer(blob, dtype=_np_dtype(dtype_str)).reshape(shape).copy()
    return torch.from_numpy(array)


def _samplewise_std(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.float().flatten(1).std(dim=1, unbiased=False).view(
        tensor.shape[0], 1, 1, 1, 1)


def _ensure_first_frame_bcfhw(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        x = x.unsqueeze(0).unsqueeze(2)
    elif x.dim() == 4:
        x = x.unsqueeze(2)
    elif x.dim() == 5 and x.shape[1] in (1, 3) and x.shape[2] >= 8:
        x = x.permute(0, 2, 1, 3, 4).contiguous()
    if x.dim() != 5 or x.shape[2] != 1:
        raise ValueError(
            f"first_frame_latent must be [B,C,1,H,W], got {tuple(x.shape)}")
    return x


def _ensure_control_latent_bcfhw(x: torch.Tensor, *,
                                 latent_channels: int) -> torch.Tensor:
    valid_channels = (3 * int(latent_channels), 4 * int(latent_channels))
    if x.dim() == 4:
        if int(x.shape[0]) not in valid_channels:
            raise ValueError(
                f"control_latent 4D shape must be [C_total,F,H,W], got {tuple(x.shape)}"
            )
        return x.unsqueeze(0)
    if x.dim() == 5:
        if int(x.shape[1]) in valid_channels:
            return x
        if int(x.shape[2]) in valid_channels:
            return x.permute(0, 2, 1, 3, 4).contiguous()
    raise ValueError(
        f"control_latent must be [B,C_total,F,H,W] or [B,F,C_total,H,W], got {tuple(x.shape)}"
    )


class WanControlnetSelfForcingDistillationPipeline(SelfForcingDistillationPipeline):
    """
    Self-forcing DMD distillation with an external ControlNet module.

    - Student/generator trains: `transformer` + `controlnet`
    - Critic trains: `fake_score_transformer` + `fake_score_controlnet`
    - Teacher is frozen: `real_score_transformer` + `real_score_controlnet`

    ControlNet residuals are injected into both causal and bidirectional Wan transformers
    via `block_controlnet_hidden_states`.

    Control latent convention:
      If Wan VAE latent channel size is `C_lat` (z_dim), then:
        - video latents: (B, C_lat, T_lat, H_lat, W_lat)
        - control latents: (B, 3*C_lat, T_lat, H_lat, W_lat) = cat(depth, masked_rgb, mask)
    """

    _required_config_modules = ["scheduler", "transformer", "vae"]
    trainable_transformer_names = [
        "transformer",
        "controlnet",
        "fake_score_transformer",
        "fake_score_controlnet",
    ]

    def set_schemas(self):
        self.train_dataset_schema = pyarrow_schema_ti2v_controlnet

    def load_modules(self,
                     fastvideo_args: FastVideoArgs,
                     loaded_modules: dict[str, torch.nn.Module] | None = None):
        training_args = cast(TrainingArgs, fastvideo_args)

        # Student/generator must run with chunk-wise causal transformer to match
        # inference rollout (KV cache). Teacher/critic transformers are loaded
        # later and explicitly forced to bidirectional in `DistillationPipeline`.
        prev_override = getattr(training_args, "override_transformer_cls_name",
                                None)
        training_args.override_transformer_cls_name = "CausalWanTransformer3DModel"
        try:
            modules = super().load_modules(fastvideo_args, loaded_modules)
        finally:
            training_args.override_transformer_cls_name = prev_override

        # Load student ControlNet (component dir, diffusers format)
        if training_args.controlnet_model_path:
            logger.info("Loading student controlnet from: %s",
                        training_args.controlnet_model_path)
            prev_cn_override = getattr(training_args,
                                       "override_controlnet_cls_name", None)
            training_args.override_controlnet_cls_name = "CausalWanControlnetUnion3DModel"
            logger.info("Student controlnet override class: %s",
                        training_args.override_controlnet_cls_name)
            try:
                self.controlnet = PipelineComponentLoader.load_module(
                    module_name="controlnet",
                    component_model_path=training_args.controlnet_model_path,
                    transformers_or_diffusers="diffusers",
                    fastvideo_args=training_args,
                )
            finally:
                training_args.override_controlnet_cls_name = prev_cn_override
            if not _is_union_controlnet(self.controlnet):
                raise ValueError(
                    "Phase-2 student controlnet must be Union. "
                    "Please use a Union ControlNet checkpoint/config."
                )
            modules["controlnet"] = self.controlnet
        else:
            self.controlnet = None

        # Teacher ControlNet
        if training_args.real_score_controlnet_model_path:
            logger.info("Loading teacher controlnet from: %s",
                        training_args.real_score_controlnet_model_path)
            # Prevent student custom init weights from being applied to teacher.
            setattr(training_args, "_loading_teacher_critic_model", True)
            prev_cn_override = getattr(training_args,
                                       "override_controlnet_cls_name", None)
            training_args.override_controlnet_cls_name = "WanControlnetUnion3DModel"
            try:
                self.real_score_controlnet = PipelineComponentLoader.load_module(
                    module_name="controlnet",
                    component_model_path=training_args.real_score_controlnet_model_path,
                    transformers_or_diffusers="diffusers",
                    fastvideo_args=training_args,
                )
            finally:
                training_args.override_controlnet_cls_name = prev_cn_override
                if hasattr(training_args, "_loading_teacher_critic_model"):
                    delattr(training_args, "_loading_teacher_critic_model")
            if not _is_union_controlnet(self.real_score_controlnet):
                raise ValueError(
                    "Teacher controlnet must be Union for phase-2 Union training."
                )
            modules["real_score_controlnet"] = self.real_score_controlnet
        else:
            self.real_score_controlnet = None

        # Critic ControlNet
        if training_args.fake_score_controlnet_model_path:
            logger.info("Loading critic controlnet from: %s",
                        training_args.fake_score_controlnet_model_path)
            # Prevent student custom init weights from being applied to critic.
            setattr(training_args, "_loading_teacher_critic_model", True)
            prev_cn_override = getattr(training_args,
                                       "override_controlnet_cls_name", None)
            training_args.override_controlnet_cls_name = "WanControlnetUnion3DModel"
            try:
                self.fake_score_controlnet = PipelineComponentLoader.load_module(
                    module_name="controlnet",
                    component_model_path=training_args.fake_score_controlnet_model_path,
                    transformers_or_diffusers="diffusers",
                    fastvideo_args=training_args,
                )
            finally:
                training_args.override_controlnet_cls_name = prev_cn_override
                if hasattr(training_args, "_loading_teacher_critic_model"):
                    delattr(training_args, "_loading_teacher_critic_model")
            if not _is_union_controlnet(self.fake_score_controlnet):
                raise ValueError(
                    "Critic controlnet must be Union for phase-2 Union training."
                )
            modules["fake_score_controlnet"] = self.fake_score_controlnet
        else:
            self.fake_score_controlnet = None

        return modules

    def _init_additional_simulation_caches(self, batch_size: int,
                                           dtype: torch.dtype,
                                           device: torch.device,
                                           max_num_frames: int) -> None:
        if getattr(self, "controlnet", None) is None:
            self.controlnet_kv_cache1 = None
            self.controlnet_crossattn_cache = None
            return

        num_transformer_blocks = len(self.controlnet.blocks)
        kv_cache_size = max_num_frames * self.frame_seq_length
        num_attention_heads = getattr(self.controlnet, "num_attention_heads",
                                      None)
        attention_head_dim = getattr(self.controlnet, "attention_head_dim",
                                     None)
        text_len = getattr(self.controlnet, "text_len", None)

        kv_cache = []
        for _ in range(num_transformer_blocks):
            kv_cache.append({
                "k":
                torch.zeros([batch_size, kv_cache_size, num_attention_heads, attention_head_dim],
                            dtype=dtype,
                            device=device),
                "v":
                torch.zeros([batch_size, kv_cache_size, num_attention_heads, attention_head_dim],
                            dtype=dtype,
                            device=device),
                "global_end_index":
                torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index":
                torch.tensor([0], dtype=torch.long, device=device)
            })

        crossattn_cache = []
        for _ in range(num_transformer_blocks):
            crossattn_cache.append({
                "k":
                torch.zeros([batch_size, text_len, num_attention_heads, attention_head_dim],
                            dtype=dtype,
                            device=device),
                "v":
                torch.zeros([batch_size, text_len, num_attention_heads, attention_head_dim],
                            dtype=dtype,
                            device=device),
                "is_init":
                False
            })

        self.controlnet_kv_cache1 = kv_cache
        self.controlnet_crossattn_cache = crossattn_cache

    def _reset_additional_simulation_caches(self) -> None:
        if getattr(self, "controlnet_kv_cache1", None) is not None and getattr(
                self, "controlnet_crossattn_cache", None) is not None:
            self._reset_simulation_caches(self.controlnet_kv_cache1,
                                          self.controlnet_crossattn_cache)

    def _simulation_model_forward_raw(
        self,
        *,
        model,
        training_batch_temp,
        kv_cache,
        crossattn_cache,
        current_start_frame: int,
        start_frame: int,
        current_num_frames: int,
    ) -> torch.Tensor:
        control_latent = getattr(training_batch_temp, "control_latent", None)
        first_frame_latent = getattr(training_batch_temp, "first_frame_latent",
                                     None)
        if getattr(self, "controlnet", None) is None or control_latent is None:
            return super()._simulation_model_forward_raw(
                model=model,
                training_batch_temp=training_batch_temp,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start_frame=current_start_frame,
                start_frame=start_frame,
                current_num_frames=current_num_frames,
            )

        assert self.controlnet_kv_cache1 is not None
        assert self.controlnet_crossattn_cache is not None

        control_chunk = control_latent[:, :, start_frame:start_frame +
                                       current_num_frames]

        hidden_states = training_batch_temp.input_kwargs["hidden_states"]
        # TI2V: enforce the first latent frame of the whole sequence (start_frame==0) to be the given image latent.
        if start_frame == 0 and first_frame_latent is not None:
            # first_frame_latent: BFCHW -> BCFHW
            img = first_frame_latent.permute(0, 2, 1, 3, 4).contiguous()
            if hidden_states.shape[2] >= 1:
                hidden_states = torch.cat([img, hidden_states[:, :, 1:]], dim=2)

        timestep = training_batch_temp.input_kwargs.get("timestep", 0)
        if isinstance(timestep, torch.Tensor):
            if timestep.numel() == 0:
                context_timestep = 0
            else:
                context_timestep = int(timestep.reshape(-1)[0].item())
        else:
            context_timestep = int(timestep)

        num_channels_latents = getattr(model, "num_channels_latents",
                                       control_chunk.shape[1] // 3)
        with set_forward_context(current_timestep=context_timestep,
                                 attn_metadata=None):
            control_res = self.controlnet(
                hidden_states=hidden_states,
                encoder_hidden_states=training_batch_temp.input_kwargs[
                    "encoder_hidden_states"],
                timestep=training_batch_temp.input_kwargs["timestep"],
                encoder_hidden_states_image=training_batch_temp.input_kwargs.get(
                    "encoder_hidden_states_image"),
                **_build_controlnet_kwargs(self.controlnet, control_chunk,
                                           num_channels_latents),
                kv_cache=self.controlnet_kv_cache1,
                crossattn_cache=self.controlnet_crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
                start_frame=start_frame,
            )

            return model(
                hidden_states=hidden_states,
                encoder_hidden_states=training_batch_temp.input_kwargs[
                    "encoder_hidden_states"],
                timestep=training_batch_temp.input_kwargs["timestep"],
                encoder_hidden_states_image=training_batch_temp.input_kwargs.get(
                    "encoder_hidden_states_image"),
                block_controlnet_hidden_states=control_res,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
                start_frame=start_frame,
            )

    def _simulation_postprocess_chunk_output(self,
                                             denoised_pred: torch.Tensor,
                                             *,
                                             training_batch,
                                             current_start_frame: int,
                                             current_num_frames: int
                                             ) -> torch.Tensor:
        first_frame_latent = getattr(training_batch, "first_frame_latent", None)
        if current_start_frame == 0 and first_frame_latent is not None:
            # denoised_pred: BFCHW, enforce frame 0
            denoised_pred = denoised_pred.clone()
            denoised_pred[:, :1] = first_frame_latent
        return denoised_pred

    def _predict_noise_with_controlnet(self, *, transformer, controlnet,
                                       input_kwargs: dict[str, Any],
                                       control_latent: torch.Tensor | None,
                                       first_frame_latent: torch.Tensor
                                       | None = None):
        conditioned_input_kwargs = dict(input_kwargs)
        conditioned_hidden_states = _apply_first_frame_latent(
            conditioned_input_kwargs["hidden_states"], first_frame_latent)
        conditioned_input_kwargs["hidden_states"] = conditioned_hidden_states
        if controlnet is None or control_latent is None:
            return transformer(**conditioned_input_kwargs)
        num_channels_latents = getattr(transformer, "num_channels_latents",
                                       control_latent.shape[1] // 3)
        control_res = controlnet(
            hidden_states=conditioned_hidden_states,
            encoder_hidden_states=conditioned_input_kwargs[
                "encoder_hidden_states"],
            timestep=conditioned_input_kwargs["timestep"],
            encoder_hidden_states_image=conditioned_input_kwargs.get(
                "encoder_hidden_states_image"),
            **_build_controlnet_kwargs(controlnet, control_latent,
                                       num_channels_latents),
        )
        return transformer(
            **conditioned_input_kwargs, block_controlnet_hidden_states=control_res)

    def _score_context_timestep(self, timestep: torch.Tensor) -> int:
        return int(timestep.reshape(-1)[0].item())

    def _build_score_attn_metadata(self, training_batch, timestep: torch.Tensor):
        if getattr(training_batch, "raw_latent_shape", None) is None:
            return getattr(training_batch, "attn_metadata", None)
        original_timesteps = getattr(training_batch, "timesteps", None)
        original_attn_metadata = getattr(training_batch, "attn_metadata", None)
        try:
            training_batch.timesteps = timestep
            self._build_attention_metadata(training_batch)
            return training_batch.attn_metadata
        finally:
            training_batch.timesteps = original_timesteps
            training_batch.attn_metadata = original_attn_metadata

    def _dmd_forward(self, generator_pred_video: torch.Tensor,
                     training_batch) -> torch.Tensor:
        original_latent = generator_pred_video
        control_latent = getattr(training_batch, "control_latent", None)
        first_frame_latent = getattr(training_batch, "first_frame_latent", None)
        denoised_timestep_from = getattr(training_batch, "denoised_timestep_from",
                                         None)
        denoised_timestep_to = getattr(training_batch, "denoised_timestep_to",
                                       None)
        if not hasattr(training_batch, "dmd_latent_vis_dict") or training_batch.dmd_latent_vis_dict is None:
            training_batch.dmd_latent_vis_dict = {}
        with torch.no_grad():
            batch_size, num_frames = original_latent.shape[:2]
            timestep = _sample_uniform_score_timestep(
                batch_size=batch_size,
                device=self.device,
                num_train_timestep=self.num_train_timestep,
                timestep_shift=self.timestep_shift,
                min_timestep=self.min_timestep,
                max_timestep=self.max_timestep,
                denoised_timestep_from=denoised_timestep_from,
                denoised_timestep_to=denoised_timestep_to,
            )
            timestep_for_noise = timestep.repeat_interleave(num_frames)
            score_attn_metadata = self._build_score_attn_metadata(
                training_batch, timestep)
            score_context_timestep = self._score_context_timestep(timestep)

            noise = torch.randn_like(generator_pred_video)

            noisy_latent = self.noise_scheduler.add_noise(
                generator_pred_video.flatten(0, 1), noise.flatten(0, 1),
                timestep_for_noise).detach().unflatten(
                    0, (batch_size, generator_pred_video.shape[1]))

            # fake_score forward (critic)
            training_batch = self._build_distill_input_kwargs(
                noisy_latent, timestep, training_batch.conditional_dict,
                training_batch)
            current_fake_score_transformer = self._get_fake_score_transformer(
                timestep)
            with set_forward_context(current_timestep=score_context_timestep,
                                     attn_metadata=score_attn_metadata):
                fake_score_pred_noise = self._predict_noise_with_controlnet(
                    transformer=current_fake_score_transformer,
                    controlnet=getattr(self, "fake_score_controlnet", None),
                    input_kwargs=training_batch.input_kwargs,
                    control_latent=control_latent,
                    first_frame_latent=first_frame_latent,
                ).permute(0, 2, 1, 3, 4)

            faker_score_pred_video = pred_noise_to_pred_video(
                pred_noise=fake_score_pred_noise.flatten(0, 1),
                noise_input_latent=noisy_latent.flatten(0, 1),
                timestep=timestep_for_noise,
                scheduler=self.noise_scheduler).unflatten(
                    0, fake_score_pred_noise.shape[:2])

            # real_score forward (teacher) conditional
            training_batch = self._build_distill_input_kwargs(
                noisy_latent, timestep, training_batch.conditional_dict,
                training_batch)
            current_real_score_transformer = self._get_real_score_transformer(
                timestep)
            with set_forward_context(current_timestep=score_context_timestep,
                                     attn_metadata=score_attn_metadata):
                real_score_pred_noise_cond = self._predict_noise_with_controlnet(
                    transformer=current_real_score_transformer,
                    controlnet=getattr(self, "real_score_controlnet", None),
                    input_kwargs=training_batch.input_kwargs,
                    control_latent=control_latent,
                    first_frame_latent=first_frame_latent,
                ).permute(0, 2, 1, 3, 4)

            pred_real_video_cond = pred_noise_to_pred_video(
                pred_noise=real_score_pred_noise_cond.flatten(0, 1),
                noise_input_latent=noisy_latent.flatten(0, 1),
                timestep=timestep_for_noise,
                scheduler=self.noise_scheduler).unflatten(
                    0, real_score_pred_noise_cond.shape[:2])

            # real_score forward (teacher) unconditional
            training_batch = self._build_distill_input_kwargs(
                noisy_latent, timestep, training_batch.unconditional_dict,
                training_batch)
            with set_forward_context(current_timestep=score_context_timestep,
                                     attn_metadata=score_attn_metadata):
                real_score_pred_noise_uncond = self._predict_noise_with_controlnet(
                    transformer=current_real_score_transformer,
                    controlnet=getattr(self, "real_score_controlnet", None),
                    input_kwargs=training_batch.input_kwargs,
                    control_latent=control_latent,
                    first_frame_latent=first_frame_latent,
                ).permute(0, 2, 1, 3, 4)

            pred_real_video_uncond = pred_noise_to_pred_video(
                pred_noise=real_score_pred_noise_uncond.flatten(0, 1),
                noise_input_latent=noisy_latent.flatten(0, 1),
                timestep=timestep_for_noise,
                scheduler=self.noise_scheduler).unflatten(
                    0, real_score_pred_noise_uncond.shape[:2])

            teacher_cfg_delta = (
                pred_real_video_cond - pred_real_video_uncond
            ) * self.real_score_guidance_scale
            teacher_cfg_delta_std = _samplewise_std(teacher_cfg_delta)
            teacher_cfg_clip_scale = torch.ones_like(teacher_cfg_delta_std)
            teacher_cfg_clip_ratio = float(
                getattr(self.training_args,
                        "dmd_teacher_cfg_delta_max_ratio", 1.5))
            if teacher_cfg_clip_ratio > 0:
                cond_std = _samplewise_std(pred_real_video_cond).clamp_min(1e-6)
                max_cfg_std = cond_std * teacher_cfg_clip_ratio
                teacher_cfg_clip_scale = torch.clamp(
                    max_cfg_std / teacher_cfg_delta_std.clamp_min(1e-6),
                    max=1.0,
                )
                teacher_cfg_delta = teacher_cfg_delta * teacher_cfg_clip_scale
            real_score_pred_video = pred_real_video_cond + teacher_cfg_delta

            teacher_residual = real_score_pred_video - original_latent
            teacher_residual_std = _samplewise_std(teacher_residual)
            teacher_residual_clip_scale = torch.ones_like(
                teacher_residual_std)
            teacher_residual_clip_ratio = float(
                getattr(self.training_args,
                        "dmd_teacher_residual_max_ratio", 1.5))
            if teacher_residual_clip_ratio > 0:
                generator_std = _samplewise_std(original_latent).clamp_min(1e-6)
                max_teacher_residual_std = (
                    generator_std * teacher_residual_clip_ratio)
                teacher_residual_clip_scale = torch.clamp(
                    max_teacher_residual_std /
                    teacher_residual_std.clamp_min(1e-6),
                    max=1.0,
                )
                real_score_pred_video = original_latent + (
                    teacher_residual * teacher_residual_clip_scale)

            # Stabilize DMD normalization when student is close to teacher.
            # Use raw normalizer with EMA-based floor to prevent tiny early-step
            # denominators from amplifying updates.
            grad_normalizer_raw = torch.abs(
                original_latent - real_score_pred_video).mean(
                    dim=(1, 2, 3, 4), keepdim=True)
            ema_decay = float(
                getattr(self.training_args, "dmd_grad_normalizer_ema_decay",
                        0.99))
            ema_floor_ratio = float(
                getattr(self.training_args,
                        "dmd_grad_normalizer_ema_floor_ratio", 0.1))
            normalizer_min = float(
                getattr(self.training_args, "dmd_grad_normalizer_min", 1e-3))

            prev_ema = getattr(self, "_dmd_grad_norm_ema", None)
            if prev_ema is None:
                grad_normalizer_ema = grad_normalizer_raw.detach().mean()
            else:
                grad_normalizer_ema = prev_ema * ema_decay + grad_normalizer_raw.detach(
                ).mean() * (1.0 - ema_decay)
            self._dmd_grad_norm_ema = grad_normalizer_ema

            grad_normalizer = torch.maximum(
                grad_normalizer_raw,
                grad_normalizer_ema * ema_floor_ratio,
            ).clamp_min(normalizer_min)
            grad = (faker_score_pred_video -
                    real_score_pred_video) / grad_normalizer
            grad_clip = float(
                getattr(self.training_args, "dmd_grad_value_clip", 10.0))
            if grad_clip > 0:
                grad = grad.clamp(min=-grad_clip, max=grad_clip)
            first_frame_grad_zeroed = 0.0
            if first_frame_latent is not None and grad.ndim == 5 and grad.shape[
                    1] > 0:
                grad = grad.clone()
                grad[:, :1] = 0
                first_frame_grad_zeroed = 1.0
            grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
            grad_abs_mean = grad.abs().mean().detach()

        dmd_loss = 0.5 * F.mse_loss(original_latent.float(),
                                    (original_latent.float() -
                                     grad.float()).detach())

        training_batch.dmd_latent_vis_dict.update({
            "training_batch_dmd_fwd_clean_latent": training_batch.latents,
            "generator_pred_video": original_latent.detach(),
            "real_score_pred_video": real_score_pred_video.detach(),
            "faker_score_pred_video": faker_score_pred_video.detach(),
            "dmd_timestep": timestep.detach(),
            "denoised_timestep_from": torch.tensor(
                -1 if denoised_timestep_from is None else denoised_timestep_from,
                device=self.device,
                dtype=torch.float32),
            "denoised_timestep_to": torch.tensor(
                -1 if denoised_timestep_to is None else denoised_timestep_to,
                device=self.device,
                dtype=torch.float32),
            "dmd_grad_normalizer": grad_normalizer.detach(),
            "dmd_grad_normalizer_raw": grad_normalizer_raw.detach(),
            "dmd_grad_normalizer_ema": grad_normalizer_ema.detach(),
            "dmdtrain_gradient_norm": grad_abs_mean,
            "teacher_cfg_delta_std": teacher_cfg_delta_std.detach(),
            "teacher_cfg_clip_scale": teacher_cfg_clip_scale.detach(),
            "teacher_residual_std": teacher_residual_std.detach(),
            "teacher_residual_clip_scale":
            teacher_residual_clip_scale.detach(),
            "dmd_first_frame_grad_zeroed": torch.tensor(
                first_frame_grad_zeroed,
                device=self.device,
                dtype=torch.float32),
        })
        return dmd_loss

    def faker_score_forward(self, training_batch):
        control_latent = getattr(training_batch, "control_latent", None)
        first_frame_latent = getattr(training_batch, "first_frame_latent", None)
        denoised_timestep_from = getattr(training_batch, "denoised_timestep_from",
                                         None)
        denoised_timestep_to = getattr(training_batch, "denoised_timestep_to",
                                       None)
        # The Wan attention stack requires ForwardContext to be set for every
        # forward pass (including ControlNet during simulation).
        with torch.no_grad(), set_forward_context(
                current_timestep=training_batch.timesteps,
                attn_metadata=training_batch.attn_metadata):
            if self.training_args.simulate_generator_forward:
                rollout = self._generator_multi_step_simulation_forward(
                    training_batch, return_sim_steps=True)
                if isinstance(rollout, tuple):
                    (generator_pred_video, denoised_timestep_from,
                     denoised_timestep_to, _) = rollout
                else:
                    generator_pred_video = rollout
            else:
                generator_pred_video = self._generator_forward(training_batch)

        batch_size, num_frames = generator_pred_video.shape[:2]
        training_batch.denoised_timestep_from = denoised_timestep_from
        training_batch.denoised_timestep_to = denoised_timestep_to
        fake_score_timestep = _sample_uniform_score_timestep(
            batch_size=batch_size,
            device=self.device,
            num_train_timestep=self.num_train_timestep,
            timestep_shift=self.timestep_shift,
            min_timestep=self.min_timestep,
            max_timestep=self.max_timestep,
            denoised_timestep_from=denoised_timestep_from,
            denoised_timestep_to=denoised_timestep_to,
        )
        fake_score_timestep_for_noise = fake_score_timestep.repeat_interleave(
            num_frames)
        score_attn_metadata = self._build_score_attn_metadata(
            training_batch, fake_score_timestep)
        score_context_timestep = self._score_context_timestep(
            fake_score_timestep)

        fake_score_noise = torch.randn_like(generator_pred_video)
        noisy_generator_pred_video = self.noise_scheduler.add_noise(
            generator_pred_video.flatten(0, 1), fake_score_noise.flatten(0, 1),
            fake_score_timestep_for_noise).unflatten(
                0, (batch_size, generator_pred_video.shape[1]))

        training_batch = self._build_distill_input_kwargs(
            noisy_generator_pred_video, fake_score_timestep,
            training_batch.conditional_dict, training_batch)

        with set_forward_context(current_timestep=score_context_timestep,
                                 attn_metadata=score_attn_metadata):
            current_fake_score_transformer = self._get_fake_score_transformer(
                fake_score_timestep)
            fake_score_pred_noise = self._predict_noise_with_controlnet(
                transformer=current_fake_score_transformer,
                controlnet=getattr(self, "fake_score_controlnet", None),
                input_kwargs=training_batch.input_kwargs,
                control_latent=control_latent,
                first_frame_latent=first_frame_latent,
            ).permute(0, 2, 1, 3, 4)

        target = fake_score_noise - generator_pred_video
        flow_matching_loss = torch.mean((fake_score_pred_noise - target)**2)

        training_batch.fake_score_latent_vis_dict = {
            "generator_pred_video": generator_pred_video,
            "fake_score_timestep": fake_score_timestep,
            "denoised_timestep_from": torch.tensor(
                -1 if denoised_timestep_from is None else denoised_timestep_from,
                device=self.device,
                dtype=torch.float32),
            "denoised_timestep_to": torch.tensor(
                -1 if denoised_timestep_to is None else denoised_timestep_to,
                device=self.device,
                dtype=torch.float32),
        }
        return training_batch, flow_matching_loss

    def _clip_model_grad_norm_(self, training_batch, transformer):
        max_grad_norm = self.training_args.max_grad_norm
        if max_grad_norm is None:
            training_batch.grad_norm = 0.0
            return training_batch

        model_parts = [transformer]
        if transformer is self.transformer or transformer is getattr(
                self, "transformer_2", None):
            controlnet = getattr(self, "controlnet", None)
            if controlnet is not None:
                model_parts.append(controlnet)
        elif transformer is self.fake_score_transformer or transformer is getattr(
                self, "fake_score_transformer_2", None):
            fake_score_controlnet = getattr(self, "fake_score_controlnet", None)
            if fake_score_controlnet is not None:
                model_parts.append(fake_score_controlnet)

        grad_norm = clip_grad_norm_while_handling_failing_dtensor_cases(
            [p for m in model_parts for p in m.parameters() if p.requires_grad],
            max_grad_norm,
            foreach=None,
        )
        grad_norm = grad_norm.item() if grad_norm is not None else 0.0
        if not math.isfinite(grad_norm):
            raise ValueError(f"Detected non-finite gradient norm: {grad_norm}")
        training_batch.grad_norm = grad_norm
        return training_batch

    def initialize_training_pipeline(self, training_args: TrainingArgs):
        super().initialize_training_pipeline(training_args)

        if getattr(self, "rollout_add_context_noise", True):
            logger.info(
                "Forcing rollout_add_context_noise=False in Wan ControlNet self-forcing to align with causal inference cache update."
            )
        self.rollout_add_context_noise = False

        # Activation checkpointing for ControlNet modules (important for memory in
        # self-forcing rollout, where the KV-cache path would otherwise retain a
        # large autograd graph).
        if training_args.enable_gradient_checkpointing_type is not None:
            for name in ("controlnet", "fake_score_controlnet",
                         "real_score_controlnet"):
                m = getattr(self, name, None)
                if m is not None:
                    setattr(
                        self,
                        name,
                        apply_activation_checkpointing(
                            m,
                            checkpointing_type=training_args.
                            enable_gradient_checkpointing_type),
                    )

        # Freeze teacher controlnet
        if getattr(self, "real_score_controlnet", None) is not None:
            self.real_score_controlnet.requires_grad_(False)
            self.real_score_controlnet.eval()

        # Make teacher/critic controlnet effectively bidirectional on full sequence (when not using KV cache)
        for name in ("real_score_controlnet", "fake_score_controlnet"):
            m = getattr(self, name, None)
            if m is not None and hasattr(m, "num_frame_per_block"):
                try:
                    m.num_frame_per_block = int(self.training_args.num_latent_t)
                except Exception:
                    pass
                if hasattr(m, "_block_mask_cache"):
                    try:
                        m._block_mask_cache = {}
                    except Exception:
                        pass

        # Make sure student/critic controlnet are trainable (if provided)
        if getattr(self, "controlnet", None) is not None:
            self.controlnet.requires_grad_(True)
            self.controlnet.train()
        if getattr(self, "fake_score_controlnet", None) is not None:
            self.fake_score_controlnet.requires_grad_(True)
            self.fake_score_controlnet.train()

        # Rebuild generator optimizer to include controlnet params
        if getattr(self, "controlnet", None) is not None:
            gen_params = list(
                filter(lambda p: p.requires_grad,
                       list(self.transformer.parameters()) +
                       list(self.controlnet.parameters())))
            betas_str = training_args.betas
            betas = tuple(float(x.strip()) for x in betas_str.split(","))
            self.optimizer = torch.optim.AdamW(
                gen_params,
                lr=training_args.learning_rate,
                betas=betas,
                weight_decay=training_args.weight_decay,
                eps=1e-8,
            )
            self.lr_scheduler = get_scheduler(
                training_args.lr_scheduler,
                optimizer=self.optimizer,
                num_warmup_steps=training_args.lr_warmup_steps,
                num_training_steps=training_args.max_train_steps,
                num_cycles=training_args.lr_num_cycles,
                power=training_args.lr_power,
                min_lr_ratio=training_args.min_lr_ratio,
                last_epoch=self.init_steps - 1,
            )
            if self.transformer_2 is not None:
                gen_params_2 = list(
                    filter(lambda p: p.requires_grad,
                           list(self.transformer_2.parameters()) +
                           list(self.controlnet.parameters())))
                self.optimizer_2 = torch.optim.AdamW(
                    gen_params_2,
                    lr=training_args.learning_rate,
                    betas=betas,
                    weight_decay=training_args.weight_decay,
                    eps=1e-8,
                )
                self.lr_scheduler_2 = get_scheduler(
                    training_args.lr_scheduler,
                    optimizer=self.optimizer_2,
                    num_warmup_steps=training_args.lr_warmup_steps,
                    num_training_steps=training_args.max_train_steps,
                    num_cycles=training_args.lr_num_cycles,
                    power=training_args.lr_power,
                    min_lr_ratio=training_args.min_lr_ratio,
                    last_epoch=self.init_steps - 1,
                )

        # Rebuild critic optimizer to include critic controlnet params
        if getattr(self, "fake_score_controlnet", None) is not None:
            fake_score_lr = training_args.fake_score_learning_rate
            if fake_score_lr == 0.0:
                fake_score_lr = training_args.learning_rate
            betas_str = training_args.fake_score_betas
            betas = tuple(float(x.strip()) for x in betas_str.split(","))
            critic_params = list(
                filter(
                    lambda p: p.requires_grad,
                    list(self.fake_score_transformer.parameters()) +
                    list(self.fake_score_controlnet.parameters()),
                ))
            self.fake_score_optimizer = torch.optim.AdamW(
                critic_params,
                lr=fake_score_lr,
                betas=betas,
                weight_decay=training_args.weight_decay,
                eps=1e-8,
            )
            self.fake_score_lr_scheduler = get_scheduler(
                training_args.fake_score_lr_scheduler,
                optimizer=self.fake_score_optimizer,
                num_warmup_steps=training_args.lr_warmup_steps,
                num_training_steps=training_args.max_train_steps,
                num_cycles=training_args.lr_num_cycles,
                power=training_args.lr_power,
                min_lr_ratio=training_args.min_lr_ratio,
                last_epoch=self.init_steps - 1,
            )
            if self.fake_score_transformer_2 is not None:
                critic_params_2 = list(
                    filter(
                        lambda p: p.requires_grad,
                        list(self.fake_score_transformer_2.parameters()) +
                        list(self.fake_score_controlnet.parameters()),
                    ))
                self.fake_score_optimizer_2 = torch.optim.AdamW(
                    critic_params_2,
                    lr=fake_score_lr,
                    betas=betas,
                    weight_decay=training_args.weight_decay,
                    eps=1e-8,
                )
                self.fake_score_lr_scheduler_2 = get_scheduler(
                    training_args.fake_score_lr_scheduler,
                    optimizer=self.fake_score_optimizer_2,
                    num_warmup_steps=training_args.lr_warmup_steps,
                    num_training_steps=training_args.max_train_steps,
                    num_cycles=training_args.lr_num_cycles,
                    power=training_args.lr_power,
                    min_lr_ratio=training_args.min_lr_ratio,
                    last_epoch=self.init_steps - 1,
                )

    def _get_next_batch(self, training_batch):
        batch = next(self.train_loader_iter, None)  # type: ignore
        if batch is None:
            self.current_epoch += 1
            self.train_loader_iter = iter(self.train_dataloader)
            batch = next(self.train_loader_iter)

        device = get_local_torch_device()
        dtype = torch.bfloat16

        encoder_hidden_states = batch["text_embedding"]
        encoder_attention_mask = batch["text_attention_mask"]
        infos = batch["info_list"]

        batch_size = encoder_hidden_states.shape[0]
        vae_config = self.training_args.pipeline_config.vae_config.arch_config
        num_channels = vae_config.z_dim
        spatial_compression_ratio = vae_config.spatial_compression_ratio
        latent_height = self.training_args.num_height // spatial_compression_ratio
        latent_width = self.training_args.num_width // spatial_compression_ratio

        latents = torch.randn(
            batch_size,
            num_channels,
            self.training_args.num_latent_t,
            latent_height,
            latent_width,
            device=device,
            dtype=dtype,
        )

        training_batch.latents = latents
        training_batch.encoder_hidden_states = encoder_hidden_states.to(device,
                                                                        dtype=dtype)
        training_batch.encoder_attention_mask = encoder_attention_mask.to(
            device, dtype=dtype)
        training_batch.infos = infos

        # Required fields for TI2V + ControlNet
        control_latent = batch["control_latent"].to(device, dtype=dtype)
        first_frame_latent = batch["first_frame_latent"].to(device, dtype=dtype)
        # Store separately: we keep TI2V first-frame as an in-chunk constraint (chunk 0, frame 0),
        # instead of treating it as an extra "context frame" (which would require 1+3k frames).
        # Expected shape: BFCHW (B,1,16,H,W). Also allow BCFHW (B,16,1,H,W).
        if first_frame_latent.ndim == 5 and first_frame_latent.shape[1] == 16 and first_frame_latent.shape[2] == 1:
            first_frame_latent = first_frame_latent.permute(0, 2, 1, 3, 4).contiguous()
        first_frame_latent = _normalize_first_frame_latent(first_frame_latent,
                                                           self.vae)

        num_channels = int(self.training_args.pipeline_config.vae_config.arch_config.z_dim)
        control_latent = _normalize_control_latent(control_latent, self.vae,
                                                   num_channels)

        training_batch.control_latent = control_latent
        training_batch.first_frame_latent = first_frame_latent

        return training_batch

    def _prepare_validation_batch(self, sampling_param: SamplingParam,
                                  training_args: TrainingArgs,
                                  validation_batch: dict[str, Any],
                                  num_inference_steps: int) -> ForwardBatch:
        required_columns = (
            "first_frame_latent_bytes",
            "first_frame_latent_shape",
            "control_latent_bytes",
            "control_latent_shape",
        )
        missing_columns = [
            key for key in required_columns if validation_batch.get(key) is None
        ]
        if missing_columns:
            raise ValueError(
                "ControlNet validation requires parquet/arrow validation rows with "
                f"{missing_columns}. Current validation sample keys: {sorted(validation_batch.keys())}"
            )

        sampling_param.prompt = validation_batch["prompt"]
        sampling_param.height = training_args.num_height
        sampling_param.width = training_args.num_width
        sampling_param.num_inference_steps = num_inference_steps
        sampling_param.data_type = "ti2v_controlnet"
        if training_args.validation_guidance_scale:
            sampling_param.guidance_scale = float(
                training_args.validation_guidance_scale)
        assert self.seed is not None
        sampling_param.seed = self.seed

        temporal_compression_factor = training_args.pipeline_config.vae_config.arch_config.temporal_compression_ratio
        spatial_compression_factor = training_args.pipeline_config.vae_config.arch_config.spatial_compression_ratio
        sampling_param.num_frames = ((training_args.num_latent_t - 1) *
                                     temporal_compression_factor + 1)
        latents_size = [(sampling_param.num_frames - 1) // temporal_compression_factor + 1,
                        sampling_param.height // spatial_compression_factor,
                        sampling_param.width // spatial_compression_factor]
        n_tokens = latents_size[0] * latents_size[1] * latents_size[2]

        first_frame_latent = _ensure_first_frame_bcfhw(
            _decode_validation_tensor(validation_batch, "first_frame_latent"))
        control_latent = _ensure_control_latent_bcfhw(
            _decode_validation_tensor(validation_batch, "control_latent"),
            latent_channels=int(first_frame_latent.shape[1]))
        if int(control_latent.shape[2]) != int(training_args.num_latent_t):
            raise ValueError(
                "Validation control_latent temporal length does not match num_latent_t: "
                f"control_latent.shape[2]={int(control_latent.shape[2])}, "
                f"num_latent_t={int(training_args.num_latent_t)}")

        device = get_local_torch_device()
        dtype = torch.bfloat16
        first_frame_latent = _normalize_first_frame_latent(
            first_frame_latent.to(device=device, dtype=dtype), self.vae)
        control_latent = _normalize_control_latent(
            control_latent.to(device=device, dtype=dtype), self.vae,
            int(training_args.pipeline_config.vae_config.arch_config.z_dim))

        batch = ForwardBatch(
            **shallow_asdict(sampling_param),
            latents=None,
            generator=self.validation_random_generator,
            n_tokens=n_tokens,
            eta=0.0,
            VSA_sparsity=training_args.VSA_sparsity,
            first_frame_latent=first_frame_latent,
            control_latent=control_latent,
        )
        fps = validation_batch.get("fps")
        if fps is not None:
            batch.fps = int(fps)
        return batch

    def initialize_validation_pipeline(self, training_args: TrainingArgs):
        # Keep same validation pipeline as WanSelfForcingDistillationPipeline
        logger.info("Initializing validation pipeline...")
        args_copy = deepcopy(training_args)
        args_copy.inference_mode = True
        if (hasattr(args_copy, "pipeline_config")
                and hasattr(args_copy.pipeline_config, "dit_config")
                and hasattr(args_copy.pipeline_config.dit_config,
                            "boundary_ratio")):
            args_copy.pipeline_config.dit_config.boundary_ratio = None
        from fastvideo.pipelines.basic.wan.wan_controlnet_causal_dmd_pipeline import (
            WanControlnetCausalDMDPipeline)

        validation_pipeline = WanControlnetCausalDMDPipeline.from_pretrained(
            training_args.model_path,
            args=args_copy,  # type: ignore
            inference_mode=True,
            loaded_modules={
                "transformer": self.get_module("transformer"),
                "transformer_2": self.get_module("transformer_2", None),
                "controlnet": self.get_module("controlnet"),
            },
            tp_size=training_args.tp_size,
            sp_size=training_args.sp_size,
            num_gpus=training_args.num_gpus,
            pin_cpu_memory=training_args.pin_cpu_memory,
            dit_cpu_offload=True)

        self.validation_pipeline = validation_pipeline


def main(args) -> None:
    logger.info("Starting Wan ControlNet self-forcing distillation pipeline...")
    pipeline = WanControlnetSelfForcingDistillationPipeline.from_pretrained(
        args.pretrained_model_name_or_path, args=args)
    pipeline.train()
    logger.info("Wan ControlNet self-forcing distillation pipeline completed")


if __name__ == "__main__":
    argv = sys.argv
    from fastvideo.utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser()
    parser = TrainingArgs.add_cli_args(parser)
    parser = FastVideoArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)
