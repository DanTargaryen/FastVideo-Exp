# SPDX-License-Identifier: Apache-2.0
import sys

from fastvideo.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.logger import init_logger
from fastvideo.training.ode_causal_controlnet_pipeline import (
    ODEInitControlnetTrainingPipeline,
)
from fastvideo.utils import FlexibleArgumentParser

logger = init_logger(__name__)


def main(args) -> None:
    logger.info("Starting Wan ControlNet ODE-init training pipeline...")

    if not args.override_transformer_cls_name:
        args.override_transformer_cls_name = "CausalWanTransformer3DModel"
    if not args.override_controlnet_cls_name:
        args.override_controlnet_cls_name = "CausalWanControlnet3DModel"

    pipeline = ODEInitControlnetTrainingPipeline.from_pretrained(
        args.pretrained_model_name_or_path, args=args)
    pipeline.train()
    logger.info("Wan ControlNet ODE-init training pipeline completed")


if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    parser = TrainingArgs.add_cli_args(parser)
    parser = FastVideoArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)
