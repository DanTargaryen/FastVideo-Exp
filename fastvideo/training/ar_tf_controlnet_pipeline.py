# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import sys
from typing import Any, cast

import torch
import torch.nn.functional as F

from fastvideo.dataset.dataloader.schema import pyarrow_schema_ti2v_controlnet
from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.models.dits.controlnet_union_components import WanControlNetUnionInput
from fastvideo.models.loader.component_loader import PipelineComponentLoader
from fastvideo.models.schedulers.scheduling_self_forcing_flow_match import (
    SelfForcingFlowMatchScheduler,
)
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.training.activation_checkpoint import apply_activation_checkpointing
from fastvideo.training.training_pipeline import TrainingPipeline
from fastvideo.training.training_utils import (
    clip_grad_norm_while_handling_failing_dtensor_cases,
    get_scheduler,
)
from fastvideo.utils import FlexibleArgumentParser, get_compute_dtype

logger = init_logger(__name__)


def _is_union_controlnet(model) -> bool:
    return "union" in model.__class__.__name__.lower()


def _split_union_control_latent(
    control_latent: torch.Tensor, num_channels_latents: int
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
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
        f"expected 3*C or 4*C (C={c})."
    )


def _build_controlnet_kwargs(
    controlnet, control_latent: torch.Tensor, num_channels_latents: int
) -> dict:
    if not _is_union_controlnet(controlnet):
        return {"controlnet_states": control_latent}
    depth, normal, masked, mask = _split_union_control_latent(
        control_latent, num_channels_latents
    )
    return {
        "controlnet_cond": WanControlNetUnionInput(depth=depth, normal=normal),
        "mask": mask,
        "masked_latent": masked,
    }


def _ensure_first_frame(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        x = x.unsqueeze(0).unsqueeze(2)
    elif x.dim() == 4:
        x = x.unsqueeze(2)
    elif x.dim() == 5:
        # [B, F, C, H, W] -> [B, C, F, H, W]
        if x.shape[1] in (1, 3) and x.shape[2] >= 8:
            x = x.permute(0, 2, 1, 3, 4).contiguous()
    else:
        raise ValueError(f"Unsupported first_frame_latent shape: {tuple(x.shape)}")
    if x.shape[2] != 1:
        raise ValueError(
            f"first_frame_latent must have F==1, got shape={tuple(x.shape)}"
        )
    return x


def _ensure_control_latent(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        x = x.unsqueeze(0)
    return x


class ARTFControlnetTrainingPipeline(TrainingPipeline):
    """
    Stage-1 AR diffusion (teacher-forcing style) with TI2V + ControlNet inputs.

    Expected parquet schema:
      - `pyarrow_schema_ti2v_controlnet`
      - `vae_latent` must be populated (non-empty)
    """

    _required_config_modules = ["scheduler", "transformer", "vae"]
    trainable_transformer_names = ["transformer", "controlnet"]

    def set_schemas(self) -> None:
        self.train_dataset_schema = pyarrow_schema_ti2v_controlnet

    def load_modules(
        self,
        fastvideo_args: FastVideoArgs,
        loaded_modules: dict[str, torch.nn.Module] | None = None,
    ):
        training_args = cast(TrainingArgs, fastvideo_args)
        modules = super().load_modules(fastvideo_args, loaded_modules)

        if not training_args.controlnet_model_path:
            raise ValueError(
                "controlnet_model_path is required for AR-TF ControlNet training"
            )

        logger.info("Loading student controlnet from: %s", training_args.controlnet_model_path)
        prev_cn_override = getattr(training_args, "override_controlnet_cls_name", None)
        training_args.override_controlnet_cls_name = "CausalWanControlnetUnion3DModel"
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
                "AR-TF student controlnet must be Union. "
                "Please use a Union ControlNet checkpoint/config."
            )
        logger.info("AR-TF student controlnet class: %s", self.controlnet.__class__.__name__)
        modules["controlnet"] = self.controlnet
        return modules

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):
        self.modules["scheduler"] = SelfForcingFlowMatchScheduler(
            num_inference_steps=1000,
            num_train_timesteps=1000,
            shift=fastvideo_args.pipeline_config.flow_shift,
            training=True,
        )

    def initialize_training_pipeline(self, training_args: TrainingArgs):
        super().initialize_training_pipeline(training_args)

        self.modules["scheduler"] = SelfForcingFlowMatchScheduler(
            num_inference_steps=1000,
            num_train_timesteps=1000,
            shift=training_args.pipeline_config.flow_shift,
            training=True,
        )
        self.noise_scheduler = self.modules["scheduler"]

        self.transformer = self.get_module("transformer")
        self.controlnet = self.get_module("controlnet")
        self.vae = self.get_module("vae")
        self.vae.requires_grad_(False)

        if training_args.enable_gradient_checkpointing_type is not None:
            self.transformer = apply_activation_checkpointing(
                self.transformer,
                checkpointing_type=training_args.enable_gradient_checkpointing_type,
            )
            self.controlnet = apply_activation_checkpointing(
                self.controlnet,
                checkpointing_type=training_args.enable_gradient_checkpointing_type,
            )

        betas = tuple(float(x.strip()) for x in training_args.betas.split(","))
        params_to_optimize: list[torch.nn.Parameter] = []
        for module in self.trainable_transformer_modules.values():
            if not isinstance(module, torch.nn.Module):
                continue
            params_to_optimize.extend(
                [p for p in module.parameters() if p.requires_grad]
            )
        self.optimizer = torch.optim.AdamW(
            params_to_optimize,
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

        num_train_t = int(self.noise_scheduler.config.num_train_timesteps)
        min_ratio = float(max(0.0, min(1.0, training_args.min_timestep_ratio)))
        max_ratio = float(max(0.0, min(1.0, training_args.max_timestep_ratio)))
        if max_ratio <= min_ratio:
            max_ratio = min(1.0, min_ratio + 0.01)

        self.min_timestep_index = int(min_ratio * num_train_t)
        self.max_timestep_index = max(self.min_timestep_index + 1, int(max_ratio * num_train_t))
        logger.info(
            "AR-TF timestep index range: [%s, %s), num_train_t=%s, num_frame_per_block=%s",
            self.min_timestep_index,
            self.max_timestep_index,
            num_train_t,
            training_args.num_frame_per_block,
        )

    def initialize_validation_pipeline(self, training_args: TrainingArgs):
        logger.info("Validation pipeline is not used for AR-TF controlnet training.")

    def _sample_blockwise_timestep_indices(
        self,
        batch_size: int,
        num_frame: int,
        num_frame_per_block: int,
        device: torch.device,
        min_index: int | None = None,
        max_index: int | None = None,
    ) -> torch.Tensor:
        lo = self.min_timestep_index if min_index is None else int(min_index)
        hi = self.max_timestep_index if max_index is None else int(max_index)
        if hi <= lo:
            hi = lo + 1
        idx = torch.randint(
            lo,
            hi,
            (batch_size, num_frame),
            device=device,
            dtype=torch.long,
        )
        nfb = max(1, int(num_frame_per_block))
        idx = idx.reshape(batch_size, -1, nfb)
        idx[:, :, 1:] = idx[:, :, :1]
        return idx.reshape(batch_size, -1)

    def _get_next_batch(self, training_batch):
        batch = next(self.train_loader_iter, None)  # type: ignore
        if batch is None:
            self.current_epoch += 1
            logger.info("Starting epoch %s", self.current_epoch)
            self.train_loader_iter = iter(self.train_dataloader)
            batch = next(self.train_loader_iter)

        vae_latent = batch["vae_latent"]
        if vae_latent.numel() == 0 or vae_latent.ndim < 5:
            raise ValueError(
                "AR-TF requires parquet with non-empty `vae_latent`. "
                "Current sample has empty/invalid vae_latent. "
                "Please regenerate parquet with video latents populated."
            )
        vae_latent = vae_latent[:, :, :self.training_args.num_latent_t]

        encoder_hidden_states = batch["text_embedding"]
        encoder_attention_mask = batch["text_attention_mask"]
        infos = batch["info_list"]
        first_frame_latent = _ensure_first_frame(batch["first_frame_latent"])
        control_latent = _ensure_control_latent(batch["control_latent"])
        control_latent = control_latent[:, :, :self.training_args.num_latent_t]

        bsz, c, t, h, w = vae_latent.shape
        training_batch.raw_latent_shape = torch.Size([bsz, c, t, h, w])

        device = get_local_torch_device()
        dtype = torch.bfloat16
        training_batch.latents = vae_latent.to(device, dtype=dtype)
        training_batch.encoder_hidden_states = encoder_hidden_states.to(device, dtype=dtype)
        training_batch.encoder_attention_mask = encoder_attention_mask.to(device, dtype=dtype)
        training_batch.first_frame_latent = first_frame_latent.to(device, dtype=dtype)
        training_batch.control_latent = control_latent.to(device, dtype=dtype)
        training_batch.infos = infos
        return training_batch

    def train_one_step(self, training_batch):  # type: ignore[override]
        self.transformer.train()
        self.controlnet.train()
        self.optimizer.zero_grad()
        training_batch.total_loss = 0.0

        args = cast(TrainingArgs, self.training_args)
        device = get_local_torch_device()
        model_dtype = get_compute_dtype()
        # Base TrainingPipeline.train() may overwrite self.noise_scheduler with
        # FlowMatchEulerDiscreteScheduler, which does not implement add_noise.
        # AR-TF requires SelfForcingFlowMatchScheduler APIs.
        scheduler = self.noise_scheduler
        if not hasattr(scheduler, "add_noise"):
            scheduler = self.modules["scheduler"]
            self.noise_scheduler = scheduler

        for _ in range(args.gradient_accumulation_steps):
            training_batch = self._get_next_batch(training_batch)

            clean_latent = training_batch.latents
            assert clean_latent is not None
            first_frame_latent = training_batch.first_frame_latent
            control_latent = training_batch.control_latent
            encoder_hidden_states = training_batch.encoder_hidden_states
            encoder_attention_mask = training_batch.encoder_attention_mask
            assert first_frame_latent is not None
            assert control_latent is not None
            assert encoder_hidden_states is not None
            assert encoder_attention_mask is not None

            bsz, c, num_frames, height, width = clean_latent.shape
            clean_latent_btfhw = clean_latent.permute(0, 2, 1, 3, 4).contiguous()

            noise_btfhw = torch.randn(
                clean_latent_btfhw.shape,
                generator=self.noise_gen_cuda,
                device=device,
                dtype=clean_latent.dtype,
            )

            timestep_indices = self._sample_blockwise_timestep_indices(
                bsz,
                num_frames,
                int(args.num_frame_per_block),
                device,
            )
            timestep = scheduler.timesteps.to(device).index_select(
                0, timestep_indices.flatten()
            ).reshape(bsz, num_frames)

            noisy_input_btfhw = scheduler.add_noise(
                clean_latent_btfhw.flatten(0, 1),
                noise_btfhw.flatten(0, 1),
                timestep,
            ).unflatten(0, (bsz, num_frames))

            clean_context_btfhw = clean_latent_btfhw
            context_timestep = None
            if int(getattr(args, "context_noise", 0)) > 0:
                max_ctx = min(int(args.context_noise),
                              int(scheduler.timesteps.numel()))
                if max_ctx > 0:
                    context_indices = self._sample_blockwise_timestep_indices(
                        bsz,
                        num_frames,
                        int(args.num_frame_per_block),
                        device,
                        min_index=0,
                        max_index=max_ctx,
                    )
                    context_timestep = scheduler.timesteps.to(
                        device).index_select(0, context_indices.flatten()).reshape(
                            bsz, num_frames)
                    clean_context_btfhw = scheduler.add_noise(
                        clean_latent_btfhw.flatten(0, 1),
                        noise_btfhw.flatten(0, 1),
                        context_timestep,
                    ).unflatten(0, (bsz, num_frames))

            # TI2V teacher-forcing anchor: keep first latent frame fixed to first frame condition.
            noisy_input = noisy_input_btfhw.permute(0, 2, 1, 3, 4).contiguous()
            if noisy_input.shape[2] >= 1:
                noisy_input[:, :, :1] = first_frame_latent
                if args.independent_first_frame:
                    timestep = timestep.clone()
                    timestep[:, 0] = 0.0
            clean_context_bcfhw = clean_context_btfhw.permute(0, 2, 1, 3, 4).contiguous()
            if hasattr(scheduler, "training_target"):
                target_flow = scheduler.training_target(
                    clean_latent_btfhw.flatten(0, 1),
                    noise_btfhw.flatten(0, 1),
                    timestep,
                ).unflatten(0, (bsz, num_frames))
            else:
                target_flow = noise_btfhw - clean_latent_btfhw

            forward_batch = ForwardBatch(data_type="ti2v_controlnet")
            forward_batch.prompt_embeds = [encoder_hidden_states]
            forward_batch.height = int(height) * 8
            forward_batch.width = int(width) * 8
            forward_batch.num_frames = int(num_frames)

            with set_forward_context(
                current_timestep=int(timestep[0, 0].item()),
                attn_metadata=None,
                forward_batch=forward_batch,
            ):
                num_channels_latents = getattr(
                    self.transformer, "num_channels_latents", control_latent.shape[1] // 3
                )
                control_res = self.controlnet(
                    hidden_states=noisy_input.to(dtype=model_dtype),
                    encoder_hidden_states=[encoder_hidden_states.to(dtype=model_dtype)],
                    timestep=timestep.to(device, dtype=model_dtype),
                    clean_hidden_states=clean_context_bcfhw.to(dtype=model_dtype),
                    aug_t=None if context_timestep is None else context_timestep.to(
                        device, dtype=model_dtype),
                    **_build_controlnet_kwargs(
                        self.controlnet,
                        control_latent.to(dtype=model_dtype),
                        num_channels_latents,
                    ),
                )
                pred_flow = self.transformer(
                    noisy_input.to(dtype=model_dtype),
                    [encoder_hidden_states.to(dtype=model_dtype)],
                    timestep.to(device, dtype=model_dtype),
                    block_controlnet_hidden_states=control_res,
                    clean_x=clean_context_bcfhw.to(dtype=model_dtype),
                    aug_t=None if context_timestep is None else context_timestep.to(
                        device, dtype=model_dtype),
                ).permute(0, 2, 1, 3, 4)

            per_frame_loss = (pred_flow.float() - target_flow.float()).pow(2).mean(
                dim=(2, 3, 4)
            )
            if hasattr(scheduler, "training_weight"):
                weight = scheduler.training_weight(timestep).unflatten(
                    0, (bsz, num_frames))
                per_frame_loss = per_frame_loss * weight.to(
                    device=per_frame_loss.device, dtype=per_frame_loss.dtype)
            nonzero_mask = timestep != 0
            if torch.any(nonzero_mask):
                loss = per_frame_loss[nonzero_mask].mean()
            else:
                loss = per_frame_loss.mean()
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            training_batch.total_loss += float(loss.detach().item())

        grad_norm = clip_grad_norm_while_handling_failing_dtensor_cases(
            [p for p in self.transformer.parameters() if p.requires_grad]
            + [p for p in self.controlnet.parameters() if p.requires_grad],
            args.max_grad_norm if args.max_grad_norm is not None else 0.0,
        )
        self.optimizer.step()
        self.lr_scheduler.step()

        if grad_norm is None:
            training_batch.grad_norm = 0.0
        else:
            try:
                training_batch.grad_norm = float(
                    grad_norm.detach().float().item()
                    if isinstance(grad_norm, torch.Tensor)
                    else grad_norm
                )
            except Exception:
                training_batch.grad_norm = 0.0
        return training_batch


def main(args) -> None:
    logger.info("Starting Wan ControlNet AR-TF training pipeline...")
    pipeline = ARTFControlnetTrainingPipeline.from_pretrained(
        args.pretrained_model_name_or_path, args=args
    )
    pipeline.train()
    logger.info("Wan ControlNet AR-TF training pipeline completed")


if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    parser = TrainingArgs.add_cli_args(parser)
    parser = FastVideoArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)
