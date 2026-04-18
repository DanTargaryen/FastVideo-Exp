#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import sys

from fastvideo.distributed import cleanup_dist_env_and_memory
from fastvideo.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.logger import init_logger
from fastvideo.training.training_utils import save_distillation_checkpoint
from fastvideo.training.wan_controlnet_self_forcing_distillation_pipeline import (
    WanControlnetSelfForcingDistillationPipeline,
)
from fastvideo.utils import FlexibleArgumentParser

logger = init_logger(__name__)


def main(args) -> None:
    logger.info("Starting phase3 init weight-only export...")
    pipeline = WanControlnetSelfForcingDistillationPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        args=args,
    )

    step_tag = str(getattr(args, "export_step_tag", "init_export"))
    logger.info("Exporting initialized generator/controlnet weights with step tag: %s",
                step_tag)
    save_distillation_checkpoint(
        pipeline.transformer,
        pipeline.fake_score_transformer,
        pipeline.global_rank,
        pipeline.training_args.output_dir,
        step_tag,
        only_save_generator_weight=True,
        generator_ema=pipeline.generator_ema,
        generator_controlnet=getattr(pipeline, "controlnet", None),
        fake_score_controlnet=getattr(pipeline, "fake_score_controlnet", None),
        generator_transformer_2=getattr(pipeline, "transformer_2", None),
        real_score_transformer_2=getattr(pipeline, "real_score_transformer_2",
                                         None),
        fake_score_transformer_2=getattr(pipeline, "fake_score_transformer_2",
                                         None),
        generator_optimizer_2=getattr(pipeline, "optimizer_2", None),
        fake_score_optimizer_2=getattr(pipeline, "fake_score_optimizer_2",
                                       None),
        generator_scheduler_2=getattr(pipeline, "lr_scheduler_2", None),
        fake_score_scheduler_2=getattr(pipeline, "fake_score_lr_scheduler_2",
                                       None),
        generator_ema_2=getattr(pipeline, "generator_ema_2", None),
    )

    try:
        pipeline.sp_group.barrier()
    except Exception:
        logger.warning("Barrier after init export failed; continuing cleanup")

    try:
        pipeline.tracker.finish()
    except Exception:
        logger.warning("Tracker shutdown after init export failed")

    logger.info("Phase3 init weight-only export completed")
    cleanup_dist_env_and_memory()


if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    parser.add_argument("--export_step_tag", type=str, default="init_export")
    parser = TrainingArgs.add_cli_args(parser)
    parser = FastVideoArgs.add_cli_args(parser)
    parsed_args = parser.parse_args()
    main(parsed_args)
