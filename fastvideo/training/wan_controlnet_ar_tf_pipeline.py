# SPDX-License-Identifier: Apache-2.0
from fastvideo.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.logger import init_logger
from fastvideo.training.ar_tf_controlnet_pipeline import (
    ARTFControlnetTrainingPipeline,
)
from fastvideo.utils import FlexibleArgumentParser

logger = init_logger(__name__)


def main(args) -> None:
    logger.info("Starting Wan ControlNet AR-TF pipeline...")

    args.override_transformer_cls_name = "CausalWanTransformer3DModel"
    args.override_controlnet_cls_name = "CausalWanControlnetUnion3DModel"
    logger.info(
        "Force overrides: transformer=%s controlnet=%s",
        args.override_transformer_cls_name,
        args.override_controlnet_cls_name,
    )

    pipeline = ARTFControlnetTrainingPipeline.from_pretrained(
        args.pretrained_model_name_or_path, args=args
    )
    pipeline.train()
    logger.info("Wan ControlNet AR-TF pipeline completed")


if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    parser = TrainingArgs.add_cli_args(parser)
    parser = FastVideoArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)
