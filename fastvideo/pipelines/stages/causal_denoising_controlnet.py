import torch  # type: ignore

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.models.dits.controlnet_union_components import (
    WanControlNetUnionInput)
from fastvideo.models.utils import pred_noise_to_pred_video, pred_noise_to_x_bound
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.causal_denoising import (
    CausalDMDDenosingStage, SlidingTileAttentionBackend,
    VideoSparseAttentionBackend, st_attn_available, vsa_available)
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult

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


class ControlnetCausalDMDDenosingStage(CausalDMDDenosingStage):

    def __init__(self,
                 transformer,
                 controlnet,
                 scheduler,
                 transformer_2=None,
                 vae=None) -> None:
        super().__init__(transformer=transformer,
                         scheduler=scheduler,
                         transformer_2=transformer_2,
                         vae=vae)
        self.controlnet = controlnet

    def verify_input(self, batch: ForwardBatch,
                     fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = super().verify_input(batch, fastvideo_args)
        result.add_check("first_frame_latent", batch.first_frame_latent,
                         V.none_or_tensor_with_dims(5))
        result.add_check("control_latent", batch.control_latent,
                         [V.is_tensor, V.with_dims(5)])
        return result

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        target_dtype = torch.bfloat16
        autocast_enabled = (target_dtype != torch.float32
                            ) and not fastvideo_args.disable_autocast

        latent_seq_length = batch.latents.shape[-1] * batch.latents.shape[-2]
        patch_ratio = self.transformer.config.arch_config.patch_size[
            -1] * self.transformer.config.arch_config.patch_size[-2]
        self.frame_seq_length = latent_seq_length // patch_ratio
        independent_first_frame = self.transformer.independent_first_frame if hasattr(
            self.transformer, 'independent_first_frame') else False
        use_full_schedule = bool(
            getattr(fastvideo_args.pipeline_config, "validation_full_schedule",
                    False))
        validation_timestep_indices = getattr(
            fastvideo_args.pipeline_config, "validation_timestep_indices", None)
        parsed_timestep_indices: list[int] = []
        if isinstance(validation_timestep_indices, str):
            parsed_timestep_indices = [
                int(x.strip()) for x in validation_timestep_indices.split(",")
                if x.strip()
            ]
        elif isinstance(validation_timestep_indices, (list, tuple)):
            parsed_timestep_indices = [
                int(x) for x in validation_timestep_indices
            ]

        if parsed_timestep_indices:
            self.scheduler.set_timesteps(int(batch.num_inference_steps),
                                         device=get_local_torch_device())
            scheduler_timesteps = self.scheduler.timesteps.to(
                get_local_torch_device())
            max_idx = int(scheduler_timesteps.shape[0] - 1)
            select_indices = [
                min(max(i, 0), max_idx) for i in parsed_timestep_indices
            ]
            timesteps = scheduler_timesteps[torch.tensor(
                select_indices,
                device=scheduler_timesteps.device,
                dtype=torch.long)]
        elif use_full_schedule:
            self.scheduler.set_timesteps(int(batch.num_inference_steps),
                                         device=get_local_torch_device())
            timesteps = self.scheduler.timesteps.to(get_local_torch_device())
        else:
            timesteps = torch.tensor(
                fastvideo_args.pipeline_config.dmd_denoising_steps,
                dtype=torch.long).cpu()
            if fastvideo_args.pipeline_config.warp_denoising_step:
                scheduler_timesteps = torch.cat(
                    (self.scheduler.timesteps.cpu(),
                     torch.tensor([0], dtype=torch.float32)))
                timesteps = scheduler_timesteps[1000 - timesteps]
            timesteps = timesteps.to(get_local_torch_device())
        update_rule = str(
            getattr(fastvideo_args.pipeline_config, "validation_update_rule",
                    "euler_dt")).strip().lower()
        if update_rule not in ("euler_dt", "renoise_x0"):
            logger.warning(
                "Unknown validation_update_rule=%s. Falling back to euler_dt.",
                update_rule)
            update_rule = "euler_dt"

        if fastvideo_args.pipeline_config.dit_config.boundary_ratio is not None:
            boundary_timestep = fastvideo_args.pipeline_config.dit_config.boundary_ratio * self.scheduler.num_train_timesteps
            high_noise_timesteps = timesteps[timesteps >= boundary_timestep]
        else:
            boundary_timestep = None
            high_noise_timesteps = None

        image_kwargs: dict = {}
        pos_cond_kwargs = self.prepare_extra_func_kwargs(
            self.transformer.forward,
            {
                "encoder_attention_mask": batch.prompt_attention_mask,
            },
        )

        if st_attn_available and self.attn_backend == SlidingTileAttentionBackend:
            self.prepare_sta_param(batch, fastvideo_args)

        assert batch.latents is not None, "latents must be provided"
        assert batch.control_latent is not None, "control_latent must be provided"
        latents = batch.latents
        b, c, t, h, w = latents.shape
        self._last_t_lat = int(t)
        prompt_embeds = batch.prompt_embeds[0]
        negative_prompt_embeds = None
        do_cfg = bool(batch.do_classifier_free_guidance and
                      batch.guidance_scale is not None
                      and float(batch.guidance_scale) != 1.0
                      and batch.negative_prompt_embeds
                      and len(batch.negative_prompt_embeds) > 0)
        if do_cfg:
            negative_prompt_embeds = batch.negative_prompt_embeds[0]
        control_latent = batch.control_latent.to(latents.device,
                                                 dtype=target_dtype)
        first_frame_latent = batch.first_frame_latent
        if first_frame_latent is not None:
            first_frame_latent = first_frame_latent.to(latents.device,
                                                       dtype=target_dtype)

        kv_cache1 = self._initialize_kv_cache(batch_size=latents.shape[0],
                                              dtype=target_dtype,
                                              device=latents.device)
        kv_cache2 = None
        if boundary_timestep is not None and self.transformer_2 is not None:
            kv_cache2 = self._initialize_kv_cache(batch_size=latents.shape[0],
                                                  dtype=target_dtype,
                                                  device=latents.device,
                                                  model=self.transformer_2)

        def _get_kv_cache(timestep: float) -> list[dict]:
            if boundary_timestep is not None:
                if timestep >= boundary_timestep:
                    return kv_cache1
                assert kv_cache2 is not None
                return kv_cache2
            return kv_cache1
        kv_cache1_uncond = None
        kv_cache2_uncond = None
        if do_cfg:
            kv_cache1_uncond = self._initialize_kv_cache(
                batch_size=latents.shape[0],
                dtype=target_dtype,
                device=latents.device)
            if boundary_timestep is not None and self.transformer_2 is not None:
                kv_cache2_uncond = self._initialize_kv_cache(
                    batch_size=latents.shape[0],
                    dtype=target_dtype,
                    device=latents.device,
                    model=self.transformer_2)

        def _get_kv_cache_uncond(timestep: float) -> list[dict] | None:
            if not do_cfg:
                return None
            assert kv_cache1_uncond is not None
            if boundary_timestep is not None:
                if timestep >= boundary_timestep:
                    return kv_cache1_uncond
                assert kv_cache2_uncond is not None
                return kv_cache2_uncond
            return kv_cache1_uncond

        crossattn_cache = self._initialize_crossattn_cache(
            batch_size=latents.shape[0],
            max_text_len=fastvideo_args.pipeline_config.text_encoder_configs[0].
            arch_config.text_len,
            dtype=target_dtype,
            device=latents.device)
        crossattn_cache_uncond = None
        if do_cfg:
            crossattn_cache_uncond = self._initialize_crossattn_cache(
                batch_size=latents.shape[0],
                max_text_len=fastvideo_args.pipeline_config.
                text_encoder_configs[0].arch_config.text_len,
                dtype=target_dtype,
                device=latents.device)
        control_kv_cache = self._initialize_kv_cache(batch_size=latents.shape[0],
                                                     dtype=target_dtype,
                                                     device=latents.device,
                                                     model=self.controlnet)
        control_kv_cache_uncond = None
        if do_cfg:
            control_kv_cache_uncond = self._initialize_kv_cache(
                batch_size=latents.shape[0],
                dtype=target_dtype,
                device=latents.device,
                model=self.controlnet)
        control_crossattn_cache = self._initialize_crossattn_cache(
            batch_size=latents.shape[0],
            max_text_len=getattr(self.controlnet, "text_len",
                                 fastvideo_args.pipeline_config.
                                 text_encoder_configs[0].arch_config.text_len),
            dtype=target_dtype,
            device=latents.device,
            model=self.controlnet)
        control_crossattn_cache_uncond = None
        if do_cfg:
            control_crossattn_cache_uncond = self._initialize_crossattn_cache(
                batch_size=latents.shape[0],
                max_text_len=getattr(
                    self.controlnet, "text_len",
                    fastvideo_args.pipeline_config.text_encoder_configs[0].
                    arch_config.text_len),
                dtype=target_dtype,
                device=latents.device,
                model=self.controlnet)

        num_blocks = t // self.num_frames_per_block
        block_sizes = [self.num_frames_per_block] * num_blocks
        start_index = 0
        if boundary_timestep is not None:
            block_sizes[0] = 1

        pos_start_base = 0
        if first_frame_latent is not None:
            t_zero = torch.zeros([latents.shape[0], 1],
                                 device=latents.device,
                                 dtype=torch.long)
            control_chunk = control_latent[:, :, :1]
            with torch.autocast(device_type="cuda",
                                dtype=target_dtype,
                                enabled=autocast_enabled), \
                set_forward_context(current_timestep=0,
                                    attn_metadata=None,
                                    forward_batch=batch):
                num_channels_latents = getattr(self.transformer,
                                               "num_channels_latents",
                                               control_chunk.shape[1] // 3)
                control_res = self.controlnet(
                    hidden_states=first_frame_latent,
                    encoder_hidden_states=prompt_embeds,
                    timestep=t_zero,
                    **_build_controlnet_kwargs(self.controlnet, control_chunk,
                                               num_channels_latents),
                    kv_cache=control_kv_cache,
                    crossattn_cache=control_crossattn_cache,
                    current_start=(pos_start_base + start_index) *
                    self.frame_seq_length,
                    start_frame=start_index,
                    **image_kwargs,
                    **pos_cond_kwargs,
                )
                self.transformer(
                    first_frame_latent,
                    prompt_embeds,
                    t_zero,
                    kv_cache=kv_cache1,
                    crossattn_cache=crossattn_cache,
                    current_start=(pos_start_base + start_index) *
                    self.frame_seq_length,
                    start_frame=start_index,
                    block_controlnet_hidden_states=control_res,
                    **image_kwargs,
                    **pos_cond_kwargs,
                )
                if boundary_timestep is not None and self.transformer_2 is not None:
                    self.transformer_2(
                        first_frame_latent,
                        prompt_embeds,
                        t_zero,
                        kv_cache=kv_cache2,
                        crossattn_cache=crossattn_cache,
                        current_start=(pos_start_base + start_index) *
                        self.frame_seq_length,
                        start_frame=start_index,
                        block_controlnet_hidden_states=control_res,
                        **image_kwargs,
                        **pos_cond_kwargs,
                    )
                if do_cfg and negative_prompt_embeds is not None:
                    assert kv_cache1_uncond is not None
                    assert control_kv_cache_uncond is not None
                    assert crossattn_cache_uncond is not None
                    assert control_crossattn_cache_uncond is not None
                    control_res_uncond = self.controlnet(
                        hidden_states=first_frame_latent,
                        encoder_hidden_states=negative_prompt_embeds,
                        timestep=t_zero,
                        **_build_controlnet_kwargs(self.controlnet,
                                                   control_chunk,
                                                   num_channels_latents),
                        kv_cache=control_kv_cache_uncond,
                        crossattn_cache=control_crossattn_cache_uncond,
                        current_start=(pos_start_base + start_index) *
                        self.frame_seq_length,
                        start_frame=start_index,
                        **image_kwargs,
                        **pos_cond_kwargs,
                    )
                    self.transformer(
                        first_frame_latent,
                        negative_prompt_embeds,
                        t_zero,
                        kv_cache=kv_cache1_uncond,
                        crossattn_cache=crossattn_cache_uncond,
                        current_start=(pos_start_base + start_index) *
                        self.frame_seq_length,
                        start_frame=start_index,
                        block_controlnet_hidden_states=control_res_uncond,
                        **image_kwargs,
                        **pos_cond_kwargs,
                    )
                    if boundary_timestep is not None and self.transformer_2 is not None:
                        assert kv_cache2_uncond is not None
                        self.transformer_2(
                            first_frame_latent,
                            negative_prompt_embeds,
                            t_zero,
                            kv_cache=kv_cache2_uncond,
                            crossattn_cache=crossattn_cache_uncond,
                            current_start=(pos_start_base + start_index) *
                            self.frame_seq_length,
                            start_frame=start_index,
                            block_controlnet_hidden_states=control_res_uncond,
                            **image_kwargs,
                            **pos_cond_kwargs,
                        )
            start_index += 1
            block_sizes.pop(0)
            latents[:, :, :1, :, :] = first_frame_latent

        steps_per_block = max(1, len(timesteps) -
                              1) if update_rule == "euler_dt" else len(
                                  timesteps)
        with self.progress_bar(total=len(block_sizes) *
                               steps_per_block) as progress_bar:
            for current_num_frames in block_sizes:
                current_latents = latents[:, :, start_index:start_index +
                                          current_num_frames, :, :]
                noise_latents_btchw = current_latents.permute(0, 2, 1, 3, 4)
                video_raw_latent_shape = noise_latents_btchw.shape
                control_chunk = control_latent[:, :, start_index:start_index +
                                               current_num_frames]
                attn_metadata = None
                max_step_index = len(timesteps) if update_rule != "euler_dt" else max(
                    0, len(timesteps) - 1)
                for i in range(max_step_index):
                    t_cur = timesteps[i]
                    if boundary_timestep is not None and t_cur < boundary_timestep:
                        current_model = self.transformer_2
                    else:
                        current_model = self.transformer
                    if current_model is None:
                        current_model = self.transformer

                    noise_latents = noise_latents_btchw.clone()
                    latent_model_input = current_latents.to(target_dtype)
                    if first_frame_latent is not None and start_index == 0:
                        latent_model_input = latent_model_input.clone()
                        latent_model_input[:, :, :1] = first_frame_latent[:, :,
                                                                          :1]
                    if batch.image_latent is not None and independent_first_frame and start_index == 0:
                        latent_model_input = torch.cat([
                            latent_model_input,
                            batch.image_latent.to(target_dtype)
                        ],
                                                       dim=2)

                    t_expand = t_cur.repeat(latent_model_input.shape[0])
                    if (vsa_available and self.attn_backend
                            == VideoSparseAttentionBackend):
                        self.attn_metadata_builder_cls = self.attn_backend.get_builder_cls(
                        )
                        if self.attn_metadata_builder_cls is not None:
                            self.attn_metadata_builder = self.attn_metadata_builder_cls(
                            )
                            attn_metadata = self.attn_metadata_builder.build(  # type: ignore
                                current_timestep=i,  # type: ignore
                                raw_latent_shape=(current_num_frames, h, w),  # type: ignore
                                patch_size=fastvideo_args.pipeline_config.
                                dit_config.patch_size,  # type: ignore
                                STA_param=batch.STA_param,  # type: ignore
                                VSA_sparsity=fastvideo_args.VSA_sparsity,  # type: ignore
                                device=get_local_torch_device(),  # type: ignore
                            )  # type: ignore
                            assert attn_metadata is not None
                        else:
                            attn_metadata = None
                    else:
                        attn_metadata = None

                    with torch.autocast(device_type="cuda",
                                        dtype=target_dtype,
                                        enabled=autocast_enabled), \
                        set_forward_context(current_timestep=i,
                                            attn_metadata=attn_metadata,
                                            forward_batch=batch):
                        t_expanded_noise = t_cur * torch.ones(
                            (latent_model_input.shape[0], 1),
                            device=latent_model_input.device,
                            dtype=torch.long)
                        num_channels_latents = getattr(current_model,
                                                       "num_channels_latents",
                                                       control_chunk.shape[1] //
                                                       3)
                        control_res = self.controlnet(
                            hidden_states=latent_model_input,
                            encoder_hidden_states=prompt_embeds,
                            timestep=t_expanded_noise,
                            **_build_controlnet_kwargs(self.controlnet,
                                                       control_chunk,
                                                       num_channels_latents),
                            kv_cache=control_kv_cache,
                            crossattn_cache=control_crossattn_cache,
                            current_start=(pos_start_base + start_index) *
                            self.frame_seq_length,
                            start_frame=start_index,
                            **image_kwargs,
                            **pos_cond_kwargs,
                        )
                        pred_noise_btchw = current_model(
                            latent_model_input,
                            prompt_embeds,
                            t_expanded_noise,
                            kv_cache=_get_kv_cache(t_cur),
                            crossattn_cache=crossattn_cache,
                            current_start=(pos_start_base + start_index) *
                            self.frame_seq_length,
                            start_frame=start_index,
                            block_controlnet_hidden_states=control_res,
                            **image_kwargs,
                            **pos_cond_kwargs,
                        ).permute(0, 2, 1, 3, 4)
                        if do_cfg and negative_prompt_embeds is not None:
                            assert control_kv_cache_uncond is not None
                            assert control_crossattn_cache_uncond is not None
                            assert crossattn_cache_uncond is not None
                            kv_cache_uncond = _get_kv_cache_uncond(t_cur)
                            assert kv_cache_uncond is not None
                            control_res_uncond = self.controlnet(
                                hidden_states=latent_model_input,
                                encoder_hidden_states=negative_prompt_embeds,
                                timestep=t_expanded_noise,
                                **_build_controlnet_kwargs(
                                    self.controlnet, control_chunk,
                                    num_channels_latents),
                                kv_cache=control_kv_cache_uncond,
                                crossattn_cache=control_crossattn_cache_uncond,
                                current_start=(pos_start_base + start_index) *
                                self.frame_seq_length,
                                start_frame=start_index,
                                **image_kwargs,
                                **pos_cond_kwargs,
                            )
                            pred_noise_uncond_btchw = current_model(
                                latent_model_input,
                                negative_prompt_embeds,
                                t_expanded_noise,
                                kv_cache=kv_cache_uncond,
                                crossattn_cache=crossattn_cache_uncond,
                                current_start=(pos_start_base + start_index) *
                                self.frame_seq_length,
                                start_frame=start_index,
                                block_controlnet_hidden_states=
                                control_res_uncond,
                                **image_kwargs,
                                **pos_cond_kwargs,
                            ).permute(0, 2, 1, 3, 4)
                            pred_noise_btchw = pred_noise_uncond_btchw + float(
                                batch.guidance_scale) * (
                                    pred_noise_btchw - pred_noise_uncond_btchw)

                    if update_rule == "euler_dt":
                        timesteps_1d = self.scheduler.timesteps.to(
                            device=latents.device, dtype=torch.float32)
                        sigmas_1d = self.scheduler.sigmas.to(device=latents.device,
                                                             dtype=torch.float32)
                        t_next = timesteps[i + 1]
                        idx_cur = torch.argmin(
                            (timesteps_1d - t_cur.float()).abs())
                        idx_next = torch.argmin(
                            (timesteps_1d - t_next.float()).abs())
                        sigma_cur = sigmas_1d[idx_cur]
                        sigma_next = sigmas_1d[idx_next]
                        dt = (sigma_next - sigma_cur).to(
                            dtype=pred_noise_btchw.dtype)
                        current_latents = current_latents + dt * pred_noise_btchw.permute(
                            0, 2, 1, 3, 4).contiguous()
                        if progress_bar is not None:
                            progress_bar.update()
                        continue

                    if boundary_timestep is not None and t_cur >= boundary_timestep:
                        pred_video_btchw = pred_noise_to_x_bound(
                            pred_noise=pred_noise_btchw.flatten(0, 1),
                            noise_input_latent=noise_latents.flatten(0, 1),
                            timestep=t_expand,
                            boundary_timestep=torch.ones_like(t_expand) *
                            boundary_timestep,
                            scheduler=self.scheduler).unflatten(
                                0, pred_noise_btchw.shape[:2])
                    else:
                        pred_video_btchw = pred_noise_to_pred_video(
                            pred_noise=pred_noise_btchw.flatten(0, 1),
                            noise_input_latent=noise_latents.flatten(0, 1),
                            timestep=t_expand,
                            scheduler=self.scheduler).unflatten(
                                0, pred_noise_btchw.shape[:2])

                    if i < len(timesteps) - 1:
                        next_timestep = timesteps[i + 1] * torch.ones(
                            [1],
                            dtype=torch.long,
                            device=pred_video_btchw.device)
                        noise = torch.randn(
                            video_raw_latent_shape,
                            dtype=pred_video_btchw.dtype,
                            generator=(batch.generator[0] if isinstance(
                                batch.generator, list) else batch.generator)
                        ).to(latents.device)
                        noise_btchw = noise
                        if boundary_timestep is not None and i < len(
                                high_noise_timesteps) - 1:
                            noise_latents_btchw = self.scheduler.add_noise_high(
                                pred_video_btchw.flatten(0, 1),
                                noise_btchw.flatten(0, 1), next_timestep,
                                torch.ones_like(next_timestep) *
                                boundary_timestep).unflatten(
                                    0, pred_video_btchw.shape[:2])
                        elif boundary_timestep is not None and i == len(
                                high_noise_timesteps) - 1:
                            noise_latents_btchw = pred_video_btchw
                        else:
                            noise_latents_btchw = self.scheduler.add_noise(
                                pred_video_btchw.flatten(0, 1),
                                noise_btchw.flatten(0, 1),
                                next_timestep).unflatten(
                                    0, pred_video_btchw.shape[:2])
                        current_latents = noise_latents_btchw.permute(
                            0, 2, 1, 3, 4)
                    else:
                        current_latents = pred_video_btchw.permute(
                            0, 2, 1, 3, 4)

                    if progress_bar is not None:
                        progress_bar.update()

                latents[:, :, start_index:start_index +
                        current_num_frames, :, :] = current_latents

                context_noise = getattr(fastvideo_args.pipeline_config,
                                        "context_noise", 0)
                t_context = torch.ones([latents.shape[0]],
                                       device=latents.device,
                                       dtype=torch.float32) * float(
                                           context_noise)
                context_bcthw = current_latents.to(target_dtype)
                if float(context_noise) > 0.0:
                    if hasattr(self.scheduler, "timesteps"
                               ) and self.scheduler.timesteps is not None and self.scheduler.timesteps.numel(
                               ) > 0:
                        schedule_ts = self.scheduler.timesteps.to(
                            device=latents.device, dtype=t_context.dtype)
                        diff = (t_context[:, None] - schedule_ts[None, :]).abs()
                        nearest_idx = diff.argmin(dim=1)
                        t_context = schedule_ts.index_select(0, nearest_idx)
                    ctx_noise = torch.randn_like(context_bcthw.flatten(0, 1))
                    context_bcthw = self.scheduler.add_noise(
                        context_bcthw.flatten(0, 1),
                        ctx_noise,
                        t_context.repeat_interleave(current_num_frames),
                    ).unflatten(0, context_bcthw.shape[:2])
                if first_frame_latent is not None and start_index == 0:
                    context_bcthw = context_bcthw.clone()
                    context_bcthw[:, :, :1] = first_frame_latent[:, :, :1]
                with torch.autocast(device_type="cuda",
                                    dtype=target_dtype,
                                    enabled=autocast_enabled), \
                    set_forward_context(current_timestep=0,
                                        attn_metadata=attn_metadata,
                                        forward_batch=batch):
                    t_expanded_context = t_context.unsqueeze(1)
                    num_channels_latents = getattr(self.transformer,
                                                   "num_channels_latents",
                                                   control_chunk.shape[1] // 3)
                    control_res_ctx = self.controlnet(
                        hidden_states=context_bcthw,
                        encoder_hidden_states=prompt_embeds,
                        timestep=t_expanded_context,
                        **_build_controlnet_kwargs(self.controlnet,
                                                   control_chunk,
                                                   num_channels_latents),
                        kv_cache=control_kv_cache,
                        crossattn_cache=control_crossattn_cache,
                        current_start=(pos_start_base + start_index) *
                        self.frame_seq_length,
                        start_frame=start_index,
                        **image_kwargs,
                        **pos_cond_kwargs,
                    )
                    if boundary_timestep is not None and self.transformer_2 is not None:
                        self.transformer_2(
                            context_bcthw,
                            prompt_embeds,
                            t_expanded_context,
                            kv_cache=kv_cache2,
                            crossattn_cache=crossattn_cache,
                            current_start=(pos_start_base + start_index) *
                            self.frame_seq_length,
                            start_frame=start_index,
                            block_controlnet_hidden_states=control_res_ctx,
                            **image_kwargs,
                            **pos_cond_kwargs,
                        )

                    self.transformer(
                        context_bcthw,
                        prompt_embeds,
                        t_expanded_context,
                        kv_cache=kv_cache1,
                        crossattn_cache=crossattn_cache,
                        current_start=(pos_start_base + start_index) *
                        self.frame_seq_length,
                        start_frame=start_index,
                        block_controlnet_hidden_states=control_res_ctx,
                        **image_kwargs,
                        **pos_cond_kwargs,
                    )
                    if do_cfg and negative_prompt_embeds is not None:
                        assert control_kv_cache_uncond is not None
                        assert control_crossattn_cache_uncond is not None
                        assert kv_cache1_uncond is not None
                        assert crossattn_cache_uncond is not None
                        control_res_ctx_uncond = self.controlnet(
                            hidden_states=context_bcthw,
                            encoder_hidden_states=negative_prompt_embeds,
                            timestep=t_expanded_context,
                            **_build_controlnet_kwargs(
                                self.controlnet, control_chunk,
                                num_channels_latents),
                            kv_cache=control_kv_cache_uncond,
                            crossattn_cache=control_crossattn_cache_uncond,
                            current_start=(pos_start_base + start_index) *
                            self.frame_seq_length,
                            start_frame=start_index,
                            **image_kwargs,
                            **pos_cond_kwargs,
                        )
                        if boundary_timestep is not None and self.transformer_2 is not None:
                            assert kv_cache2_uncond is not None
                            self.transformer_2(
                                context_bcthw,
                                negative_prompt_embeds,
                                t_expanded_context,
                                kv_cache=kv_cache2_uncond,
                                crossattn_cache=crossattn_cache_uncond,
                                current_start=(pos_start_base + start_index) *
                                self.frame_seq_length,
                                start_frame=start_index,
                                block_controlnet_hidden_states=
                                control_res_ctx_uncond,
                                **image_kwargs,
                                **pos_cond_kwargs,
                            )

                        self.transformer(
                            context_bcthw,
                            negative_prompt_embeds,
                            t_expanded_context,
                            kv_cache=kv_cache1_uncond,
                            crossattn_cache=crossattn_cache_uncond,
                            current_start=(pos_start_base + start_index) *
                            self.frame_seq_length,
                            start_frame=start_index,
                            block_controlnet_hidden_states=
                            control_res_ctx_uncond,
                            **image_kwargs,
                            **pos_cond_kwargs,
                        )

                start_index += current_num_frames

        if boundary_timestep is not None:
            num_frames_to_remove = self.num_frames_per_block - 1
            latents = latents[:, :, :-num_frames_to_remove, :, :]

        batch.latents = latents
        return batch
