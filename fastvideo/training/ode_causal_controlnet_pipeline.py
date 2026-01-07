# SPDX-License-Identifier: Apache-2.0
import sys
from typing import Any, cast

import torch
import torch.nn.functional as F

from fastvideo.dataset.dataloader.schema import (
    pyarrow_schema_ode_trajectory_ti2v_controlnet)
from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.models.loader.component_loader import PipelineComponentLoader
from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from fastvideo.models.utils import pred_noise_to_pred_video
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.training.activation_checkpoint import (
    apply_activation_checkpointing)
from fastvideo.training.training_pipeline import TrainingPipeline
from fastvideo.training.training_utils import (
    clip_grad_norm_while_handling_failing_dtensor_cases, get_scheduler)

logger = init_logger(__name__)


class ODEInitControlnetTrainingPipeline(TrainingPipeline):
    """
    Phase-1 ODE-init training (TI2V + ControlNet).

    - Trajectories are recorded with a bidirectional teacher (see preprocess tool).
    - Student runs with causal masking (CausalWanTransformer3DModel + CausalWanControlnet3DModel).
    - Loss: predict x0 from x_t (flow) and MSE against teacher x0.
    """

    _required_config_modules = ["scheduler", "transformer", "vae"]
    trainable_transformer_names = ["transformer", "controlnet"]

    def set_schemas(self) -> None:
        self.train_dataset_schema = pyarrow_schema_ode_trajectory_ti2v_controlnet

    def load_modules(self,
                     fastvideo_args: FastVideoArgs,
                     loaded_modules: dict[str, torch.nn.Module] | None = None):
        training_args = cast(TrainingArgs, fastvideo_args)
        modules = super().load_modules(fastvideo_args, loaded_modules)

        if not training_args.controlnet_model_path:
            raise ValueError(
                "controlnet_model_path is required for ODE-init ControlNet training"
            )

        logger.info("Loading student controlnet from: %s",
                    training_args.controlnet_model_path)
        self.controlnet = PipelineComponentLoader.load_module(
            module_name="controlnet",
            component_model_path=training_args.controlnet_model_path,
            transformers_or_diffusers="diffusers",
            fastvideo_args=training_args,
        )
        modules["controlnet"] = self.controlnet
        return modules

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):
        # Replace base scheduler with FlowMatchEuler for ODE-init.
        self.modules["scheduler"] = FlowMatchEulerDiscreteScheduler(
            shift=fastvideo_args.pipeline_config.flow_shift)

    def initialize_training_pipeline(self, training_args: TrainingArgs):
        super().initialize_training_pipeline(training_args)

        # Re-bind scheduler to FlowMatchEuler (ensure training path uses Euler).
        self.modules["scheduler"] = FlowMatchEulerDiscreteScheduler(
            shift=training_args.pipeline_config.flow_shift)
        self.noise_scheduler = self.modules["scheduler"]

        self.transformer = self.get_module("transformer")
        self.controlnet = self.get_module("controlnet")
        self.vae = self.get_module("vae")
        self.vae.requires_grad_(False)

        if training_args.enable_gradient_checkpointing_type is not None:
            self.transformer = apply_activation_checkpointing(
                self.transformer,
                checkpointing_type=training_args.
                enable_gradient_checkpointing_type)
            self.controlnet = apply_activation_checkpointing(
                self.controlnet,
                checkpointing_type=training_args.
                enable_gradient_checkpointing_type)

        # Rebuild optimizer to include controlnet parameters.
        betas_str = training_args.betas
        betas = tuple(float(x.strip()) for x in betas_str.split(","))
        params_to_optimize = []
        for name, module in self.trainable_transformer_modules.items():
            if not isinstance(module, torch.nn.Module):
                continue
            params_to_optimize.extend(
                [p for p in module.parameters() if p.requires_grad])
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

        # Build DMD step list (values in 0..1000 grid)
        raw_steps = training_args.pipeline_config.dmd_denoising_steps or [
            1000, 750, 500, 250
        ]
        dmd_steps = torch.tensor(raw_steps,
                                 dtype=torch.long,
                                 device=get_local_torch_device())
        if training_args.warp_denoising_step:
            # Mirror distillation pipeline behavior.
            schedule_ts = torch.cat(
                (self.noise_scheduler.timesteps.to(device=get_local_torch_device()),
                 torch.tensor([0.0], device=get_local_torch_device())),
                dim=0,
            )
            idx = (self.noise_scheduler.num_train_timesteps -
                   dmd_steps).clamp(0, schedule_ts.numel() - 1)
            self.dmd_denoising_steps = schedule_ts.index_select(0, idx)
        else:
            self.dmd_denoising_steps = dmd_steps.to(dtype=torch.float32)

        self._cached_closest_idx_per_dmd = None
        logger.info("denoising_step_list: %s", self.dmd_denoising_steps)

    def initialize_validation_pipeline(self, training_args: TrainingArgs):
        logger.info("Validation pipeline not used for ODE-init controlnet.")

    def _get_next_batch(
            self,
            training_batch) -> tuple[Any, torch.Tensor, torch.Tensor,
                                     torch.Tensor, torch.Tensor]:
        batch = next(self.train_loader_iter, None)  # type: ignore
        if batch is None:
            self.current_epoch += 1
            logger.info("Starting epoch %s", self.current_epoch)
            self.train_loader_iter = iter(self.train_dataloader)
            batch = next(self.train_loader_iter)

        encoder_hidden_states = batch['text_embedding']
        encoder_attention_mask = batch['text_attention_mask']
        infos = batch['info_list']
        first_frame_latent = batch['first_frame_latent']
        control_latent = batch['control_latent']

        trajectory_latents = batch['trajectory_latents']
        if trajectory_latents.dim() == 7:
            trajectory_latents = trajectory_latents[:, 0]
        elif trajectory_latents.dim() == 6:
            pass
        else:
            raise ValueError(
                f"Unexpected trajectory_latents dim: {trajectory_latents.dim()}"
            )

        trajectory_timesteps = batch['trajectory_timesteps']
        if trajectory_timesteps.dim() == 3:
            trajectory_timesteps = trajectory_timesteps[:, 0]
        elif trajectory_timesteps.dim() == 2:
            pass
        else:
            raise ValueError(
                f"Unexpected trajectory_timesteps dim: {trajectory_timesteps.dim()}"
            )

        # [B, S, C, T, H, W] -> [B, S, T, C, H, W]
        trajectory_latents = trajectory_latents.permute(0, 1, 3, 2, 4, 5)

        device = get_local_torch_device()
        training_batch.encoder_hidden_states = encoder_hidden_states.to(
            device, dtype=torch.bfloat16)
        training_batch.encoder_attention_mask = encoder_attention_mask.to(
            device, dtype=torch.bfloat16)
        training_batch.infos = infos

        return (training_batch,
                trajectory_latents.to(device, dtype=torch.bfloat16),
                trajectory_timesteps.to(device),
                _ensure_first_frame(first_frame_latent).to(
                    device, dtype=torch.bfloat16),
                _ensure_control_latent(control_latent).to(
                    device, dtype=torch.bfloat16))

    def _get_timestep(self,
                      min_timestep: int,
                      max_timestep: int,
                      batch_size: int,
                      num_frame: int,
                      num_frame_per_block: int,
                      uniform_timestep: bool = False) -> torch.Tensor:
        if uniform_timestep:
            timestep = torch.randint(min_timestep,
                                     max_timestep, [batch_size, 1],
                                     device=self.device,
                                     dtype=torch.long).repeat(1, num_frame)
            return timestep
        timestep = torch.randint(min_timestep,
                                 max_timestep, [batch_size, num_frame],
                                 device=self.device,
                                 dtype=torch.long)
        timestep = timestep.reshape(timestep.shape[0], -1, num_frame_per_block)
        timestep[:, :, 1:] = timestep[:, :, 0:1]
        timestep = timestep.reshape(timestep.shape[0], -1)
        return timestep

    def _step_predict_next_latent(
            self, traj_latents: torch.Tensor, traj_timesteps: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            encoder_attention_mask: torch.Tensor,
            first_frame_latent: torch.Tensor,
            control_latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor,
                                                   torch.Tensor, dict[str,
                                                                      torch.Tensor]]:
        latent_vis_dict: dict[str, torch.Tensor] = {}
        device = get_local_torch_device()
        target_latent = traj_latents[:, -1]

        # Shapes: traj_latents [B, S, T, C, H, W], traj_timesteps [B, S]
        bsz, steps, num_frames, num_channels, height, width = traj_latents.shape

        if self._cached_closest_idx_per_dmd is None:
            # Compute closest indices to the desired DMD timesteps.
            ts = traj_timesteps[0].to(dtype=torch.float32)
            dmd = self.dmd_denoising_steps.to(device=device,
                                              dtype=torch.float32)
            idx = torch.argmin((ts.unsqueeze(0) - dmd.unsqueeze(1)).abs(),
                               dim=1)
            self._cached_closest_idx_per_dmd = idx.cpu()
            logger.info("closest dmd indices: %s",
                        self._cached_closest_idx_per_dmd)
            logger.info("traj timesteps at indices: %s",
                        ts.index_select(0, idx).detach().cpu())

        assert self._cached_closest_idx_per_dmd is not None
        relevant_traj_latents = torch.index_select(
            traj_latents,
            dim=1,
            index=self._cached_closest_idx_per_dmd.to(traj_latents.device))

        indexes = self._get_timestep(0, len(self.dmd_denoising_steps), bsz,
                                     num_frames, 3, uniform_timestep=False)
        noisy_input = torch.gather(
            relevant_traj_latents,
            dim=1,
            index=indexes.reshape(bsz, 1, num_frames, 1, 1,
                                  1).expand(-1, -1, -1, num_channels, height,
                                            width).to(self.device)).squeeze(1)

        timestep = self.dmd_denoising_steps[indexes]
        if self.training_args.independent_first_frame:
            timestep = timestep.clone()
            timestep[:, 0] = 0

        # Prepare inputs for transformer/controlnet
        latent_vis_dict["noisy_input"] = noisy_input.permute(
            0, 2, 1, 3, 4).detach().clone().cpu()
        latent_vis_dict["x0"] = target_latent.permute(0, 2, 1, 3,
                                                      4).detach().clone().cpu()

        latent_model_input = noisy_input.permute(0, 2, 1, 3, 4).contiguous()
        latent_model_input[:, :, :1] = first_frame_latent
        model_dtype = next(self.transformer.parameters()).dtype
        latent_model_input = latent_model_input.to(dtype=model_dtype)
        encoder_hidden_states = encoder_hidden_states.to(dtype=model_dtype)
        control_latent = control_latent.to(dtype=model_dtype)

        batch = ForwardBatch(data_type="ti2v_controlnet")
        batch.prompt_embeds = [encoder_hidden_states]
        batch.height = int(height) * 8
        batch.width = int(width) * 8
        batch.num_frames = int(num_frames)

        with set_forward_context(current_timestep=int(timestep[0, 0].item()),
                                 attn_metadata=None,
                                 forward_batch=batch):
            control_res = self.controlnet(
                hidden_states=latent_model_input,
                encoder_hidden_states=[encoder_hidden_states],
                timestep=timestep.to(device, dtype=model_dtype),
                controlnet_states=control_latent,
            )
            pred_flow = self.transformer(
                latent_model_input,
                [encoder_hidden_states],
                timestep.to(device, dtype=model_dtype),
                block_controlnet_hidden_states=control_res,
            ).permute(0, 2, 1, 3, 4)

        pred_video = pred_noise_to_pred_video(
            pred_noise=pred_flow.flatten(0, 1),
            noise_input_latent=noisy_input.flatten(0, 1),
            timestep=timestep.to(dtype=model_dtype).flatten(0, 1),
            scheduler=self.noise_scheduler).unflatten(0, pred_flow.shape[:2])

        latent_vis_dict["pred_video"] = pred_video.permute(
            0, 2, 1, 3, 4).detach().clone().cpu()

        return pred_video, target_latent, timestep, latent_vis_dict

    def train_one_step(self, training_batch):  # type: ignore[override]
        self.transformer.train()
        self.controlnet.train()
        self.optimizer.zero_grad()
        training_batch.total_loss = 0.0
        args = cast(TrainingArgs, self.training_args)

        for _ in range(args.gradient_accumulation_steps):
            (training_batch, traj_latents, traj_timesteps, first_frame_latent,
             control_latent) = self._get_next_batch(training_batch)
            text_embeds = training_batch.encoder_hidden_states
            text_attention_mask = training_batch.encoder_attention_mask
            assert traj_latents.shape[0] == 1

            pred_video, target_latent, t, latent_vis_dict = self._step_predict_next_latent(
                traj_latents, traj_timesteps, text_embeds, text_attention_mask,
                first_frame_latent, control_latent)

            training_batch.latent_vis_dict.update(latent_vis_dict)

            mask = t != 0
            loss = F.mse_loss(pred_video[mask],
                              target_latent[mask],
                              reduction="mean")
            loss = loss / args.gradient_accumulation_steps
            with set_forward_context(current_timestep=int(t[0, 0].item()),
                                     attn_metadata=None,
                                     forward_batch=None):
                loss.backward()
            training_batch.total_loss += float(loss.detach().clone().item())

        grad_norm = clip_grad_norm_while_handling_failing_dtensor_cases(
            [p for p in self.transformer.parameters() if p.requires_grad] +
            [p for p in self.controlnet.parameters() if p.requires_grad],
            args.max_grad_norm if args.max_grad_norm is not None else 0.0)

        self.optimizer.step()
        self.lr_scheduler.step()

        if grad_norm is None:
            training_batch.grad_norm = 0.0
        else:
            try:
                training_batch.grad_norm = float(
                    grad_norm.detach().float().item()
                    if isinstance(grad_norm, torch.Tensor) else grad_norm)
            except Exception:
                training_batch.grad_norm = 0.0
        return training_batch


def _ensure_first_frame(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        x = x.unsqueeze(0).unsqueeze(2)
    elif x.dim() == 4:
        x = x.unsqueeze(2)
    elif x.dim() == 5:
        # Accept [B, F, C, H, W] and convert to [B, C, F, H, W].
        if x.shape[1] in (1, 3) and x.shape[2] >= 8:
            x = x.permute(0, 2, 1, 3, 4).contiguous()
    else:
        raise ValueError(f"Unsupported first_frame_latent shape: {tuple(x.shape)}")
    if x.shape[2] != 1:
        raise ValueError(
            f"first_frame_latent must have F==1, got shape={tuple(x.shape)}")
    return x


def _ensure_control_latent(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        x = x.unsqueeze(0)
    return x


def main(args) -> None:
    logger.info("Starting ODE-init controlnet training pipeline...")
    pipeline = ODEInitControlnetTrainingPipeline.from_pretrained(
        args.pretrained_model_name_or_path, args=args)
    pipeline.train()
    logger.info("ODE-init controlnet training pipeline done")


if __name__ == "__main__":
    argv = sys.argv
    from fastvideo.fastvideo_args import TrainingArgs
    from fastvideo.utils import FlexibleArgumentParser
    parser = FlexibleArgumentParser()
    parser = TrainingArgs.add_cli_args(parser)
    parser = FastVideoArgs.add_cli_args(parser)
    args = parser.parse_args()
    # Force student to use causal transformer/controlnet
    args.override_transformer_cls_name = "CausalWanTransformer3DModel"
    args.override_controlnet_cls_name = "CausalWanControlnet3DModel"
    args.dit_cpu_offload = False
    main(args)
