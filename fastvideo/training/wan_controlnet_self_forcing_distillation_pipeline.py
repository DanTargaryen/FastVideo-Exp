# SPDX-License-Identifier: Apache-2.0
import sys
from copy import deepcopy
from typing import Any, cast

import torch
import torch.nn.functional as F

from fastvideo.dataset.dataloader.schema import pyarrow_schema_ti2v_controlnet
from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.logger import init_logger
from fastvideo.models.utils import pred_noise_to_pred_video
from fastvideo.models.loader.component_loader import PipelineComponentLoader
from fastvideo.forward_context import set_forward_context
from fastvideo.training.self_forcing_distillation_pipeline import (
    SelfForcingDistillationPipeline,
)
from fastvideo.training.activation_checkpoint import apply_activation_checkpointing
from fastvideo.training.training_utils import get_scheduler
from fastvideo.utils import is_vsa_available
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
            # Align student with causal inference path.
            training_args.override_controlnet_cls_name = "CausalWanControlnet3DModel"
            try:
                self.controlnet = PipelineComponentLoader.load_module(
                    module_name="controlnet",
                    component_model_path=training_args.controlnet_model_path,
                    transformers_or_diffusers="diffusers",
                    fastvideo_args=training_args,
                )
            finally:
                training_args.override_controlnet_cls_name = prev_cn_override
            modules["controlnet"] = self.controlnet
        else:
            self.controlnet = None

        # Teacher ControlNet
        if training_args.real_score_controlnet_model_path:
            logger.info("Loading teacher controlnet from: %s",
                        training_args.real_score_controlnet_model_path)
            # Prevent student custom init weights from being applied to teacher.
            setattr(training_args, "_loading_teacher_critic_model", True)
            try:
                self.real_score_controlnet = PipelineComponentLoader.load_module(
                    module_name="controlnet",
                    component_model_path=training_args.real_score_controlnet_model_path,
                    transformers_or_diffusers="diffusers",
                    fastvideo_args=training_args,
                )
            finally:
                if hasattr(training_args, "_loading_teacher_critic_model"):
                    delattr(training_args, "_loading_teacher_critic_model")
            modules["real_score_controlnet"] = self.real_score_controlnet
        else:
            self.real_score_controlnet = None

        # Critic ControlNet
        if training_args.fake_score_controlnet_model_path:
            logger.info("Loading critic controlnet from: %s",
                        training_args.fake_score_controlnet_model_path)
            # Prevent student custom init weights from being applied to critic.
            setattr(training_args, "_loading_teacher_critic_model", True)
            try:
                self.fake_score_controlnet = PipelineComponentLoader.load_module(
                    module_name="controlnet",
                    component_model_path=training_args.fake_score_controlnet_model_path,
                    transformers_or_diffusers="diffusers",
                    fastvideo_args=training_args,
                )
            finally:
                if hasattr(training_args, "_loading_teacher_critic_model"):
                    delattr(training_args, "_loading_teacher_critic_model")
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

        num_channels_latents = getattr(model, "num_channels_latents",
                                       control_chunk.shape[1] // 3)
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
                                       control_latent: torch.Tensor | None):
        if controlnet is None or control_latent is None:
            return transformer(**input_kwargs)
        num_channels_latents = getattr(transformer, "num_channels_latents",
                                       control_latent.shape[1] // 3)
        control_res = controlnet(
            hidden_states=input_kwargs["hidden_states"],
            encoder_hidden_states=input_kwargs["encoder_hidden_states"],
            timestep=input_kwargs["timestep"],
            encoder_hidden_states_image=input_kwargs.get(
                "encoder_hidden_states_image"),
            **_build_controlnet_kwargs(controlnet, control_latent,
                                       num_channels_latents),
        )
        return transformer(
            **input_kwargs, block_controlnet_hidden_states=control_res)

    def _dmd_forward(self, generator_pred_video: torch.Tensor,
                     training_batch) -> torch.Tensor:
        original_latent = generator_pred_video
        control_latent = getattr(training_batch, "control_latent", None)
        if not hasattr(training_batch, "dmd_latent_vis_dict") or training_batch.dmd_latent_vis_dict is None:
            training_batch.dmd_latent_vis_dict = {}
        with torch.no_grad():
            timestep = torch.randint(0,
                                     self.num_train_timestep, [1],
                                     device=self.device,
                                     dtype=torch.long)
            from fastvideo.training.training_utils import shift_timestep

            timestep = shift_timestep(timestep, self.timestep_shift,
                                      self.num_train_timestep)
            timestep = timestep.clamp(self.min_timestep, self.max_timestep)

            noise = torch.randn(self.video_latent_shape,
                                device=self.device,
                                dtype=generator_pred_video.dtype)

            noisy_latent = self.noise_scheduler.add_noise(
                generator_pred_video.flatten(0, 1), noise.flatten(0, 1),
                timestep).detach().unflatten(0,
                                             (1, generator_pred_video.shape[1]))

            # fake_score forward (critic)
            training_batch = self._build_distill_input_kwargs(
                noisy_latent, timestep, training_batch.conditional_dict,
                training_batch)
            current_fake_score_transformer = self._get_fake_score_transformer(
                timestep)
            fake_score_pred_noise = self._predict_noise_with_controlnet(
                transformer=current_fake_score_transformer,
                controlnet=getattr(self, "fake_score_controlnet", None),
                input_kwargs=training_batch.input_kwargs,
                control_latent=control_latent,
            ).permute(0, 2, 1, 3, 4)

            faker_score_pred_video = pred_noise_to_pred_video(
                pred_noise=fake_score_pred_noise.flatten(0, 1),
                noise_input_latent=noisy_latent.flatten(0, 1),
                timestep=timestep,
                scheduler=self.noise_scheduler).unflatten(
                    0, fake_score_pred_noise.shape[:2])

            # real_score forward (teacher) conditional
            training_batch = self._build_distill_input_kwargs(
                noisy_latent, timestep, training_batch.conditional_dict,
                training_batch)
            current_real_score_transformer = self._get_real_score_transformer(
                timestep)
            real_score_pred_noise_cond = self._predict_noise_with_controlnet(
                transformer=current_real_score_transformer,
                controlnet=getattr(self, "real_score_controlnet", None),
                input_kwargs=training_batch.input_kwargs,
                control_latent=control_latent,
            ).permute(0, 2, 1, 3, 4)

            pred_real_video_cond = pred_noise_to_pred_video(
                pred_noise=real_score_pred_noise_cond.flatten(0, 1),
                noise_input_latent=noisy_latent.flatten(0, 1),
                timestep=timestep,
                scheduler=self.noise_scheduler).unflatten(
                    0, real_score_pred_noise_cond.shape[:2])

            # real_score forward (teacher) unconditional
            training_batch = self._build_distill_input_kwargs(
                noisy_latent, timestep, training_batch.unconditional_dict,
                training_batch)
            real_score_pred_noise_uncond = self._predict_noise_with_controlnet(
                transformer=current_real_score_transformer,
                controlnet=getattr(self, "real_score_controlnet", None),
                input_kwargs=training_batch.input_kwargs,
                control_latent=control_latent,
            ).permute(0, 2, 1, 3, 4)

            pred_real_video_uncond = pred_noise_to_pred_video(
                pred_noise=real_score_pred_noise_uncond.flatten(0, 1),
                noise_input_latent=noisy_latent.flatten(0, 1),
                timestep=timestep,
                scheduler=self.noise_scheduler).unflatten(
                    0, real_score_pred_noise_uncond.shape[:2])

            real_score_pred_video = pred_real_video_cond + (
                pred_real_video_cond -
                pred_real_video_uncond) * self.real_score_guidance_scale

            grad = (faker_score_pred_video -
                    real_score_pred_video) / torch.abs(
                        original_latent - real_score_pred_video).mean()
            grad = torch.nan_to_num(grad)

        dmd_loss = 0.5 * F.mse_loss(original_latent.float(),
                                    (original_latent.float() -
                                     grad.float()).detach())

        training_batch.dmd_latent_vis_dict.update({
            "training_batch_dmd_fwd_clean_latent": training_batch.latents,
            "generator_pred_video": original_latent.detach(),
            "real_score_pred_video": real_score_pred_video.detach(),
            "faker_score_pred_video": faker_score_pred_video.detach(),
            "dmd_timestep": timestep.detach(),
        })
        return dmd_loss

    def faker_score_forward(self, training_batch):
        control_latent = getattr(training_batch, "control_latent", None)
        # The Wan attention stack requires ForwardContext to be set for every
        # forward pass (including ControlNet during simulation).
        with torch.no_grad(), set_forward_context(
                current_timestep=training_batch.timesteps,
                attn_metadata=training_batch.attn_metadata):
            if self.training_args.simulate_generator_forward:
                generator_pred_video = self._generator_multi_step_simulation_forward(
                    training_batch)
            else:
                generator_pred_video = self._generator_forward(training_batch)

        fake_score_timestep = torch.randint(0,
                                            self.num_train_timestep, [1],
                                            device=self.device,
                                            dtype=torch.long)
        from fastvideo.training.training_utils import shift_timestep
        fake_score_timestep = shift_timestep(fake_score_timestep,
                                             self.timestep_shift,
                                             self.num_train_timestep)
        fake_score_timestep = fake_score_timestep.clamp(self.min_timestep,
                                                        self.max_timestep)

        fake_score_noise = torch.randn(self.video_latent_shape,
                                       device=self.device,
                                       dtype=generator_pred_video.dtype)
        noisy_generator_pred_video = self.noise_scheduler.add_noise(
            generator_pred_video.flatten(0, 1), fake_score_noise.flatten(0, 1),
            fake_score_timestep).unflatten(0,
                                           (1, generator_pred_video.shape[1]))

        training_batch = self._build_distill_input_kwargs(
            noisy_generator_pred_video, fake_score_timestep,
            training_batch.conditional_dict, training_batch)

        with set_forward_context(current_timestep=training_batch.timesteps,
                                 attn_metadata=training_batch.attn_metadata):
            current_fake_score_transformer = self._get_fake_score_transformer(
                fake_score_timestep)
            fake_score_pred_noise = self._predict_noise_with_controlnet(
                transformer=current_fake_score_transformer,
                controlnet=getattr(self, "fake_score_controlnet", None),
                input_kwargs=training_batch.input_kwargs,
                control_latent=control_latent,
            ).permute(0, 2, 1, 3, 4)

        target = fake_score_noise - generator_pred_video
        flow_matching_loss = torch.mean((fake_score_pred_noise - target)**2)

        training_batch.fake_score_latent_vis_dict = {
            "generator_pred_video": generator_pred_video,
            "fake_score_timestep": fake_score_timestep,
        }
        return training_batch, flow_matching_loss

    def initialize_training_pipeline(self, training_args: TrainingArgs):
        super().initialize_training_pipeline(training_args)

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
        training_batch.control_latent = batch["control_latent"].to(device,
                                                                   dtype=dtype)
        first_frame_latent = batch["first_frame_latent"].to(device, dtype=dtype)
        # Store separately: we keep TI2V first-frame as an in-chunk constraint (chunk 0, frame 0),
        # instead of treating it as an extra "context frame" (which would require 1+3k frames).
        # Expected shape: BFCHW (B,1,16,H,W). Also allow BCFHW (B,16,1,H,W).
        if first_frame_latent.ndim == 5 and first_frame_latent.shape[1] == 16 and first_frame_latent.shape[2] == 1:
            first_frame_latent = first_frame_latent.permute(0, 2, 1, 3, 4).contiguous()
        training_batch.first_frame_latent = first_frame_latent

        return training_batch

    def initialize_validation_pipeline(self, training_args: TrainingArgs):
        # Keep same validation pipeline as WanSelfForcingDistillationPipeline
        logger.info("Initializing validation pipeline...")
        args_copy = deepcopy(training_args)
        args_copy.inference_mode = True
        from fastvideo.pipelines.basic.wan.wan_causal_dmd_pipeline import (
            WanCausalDMDPipeline)

        validation_pipeline = WanCausalDMDPipeline.from_pretrained(
            training_args.model_path,
            args=args_copy,  # type: ignore
            inference_mode=True,
            loaded_modules={
                "transformer": self.get_module("transformer"),
                "transformer_2": self.get_module("transformer_2", None),
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
