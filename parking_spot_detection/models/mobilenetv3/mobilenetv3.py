"""MobileNet-v3 module."""

import argparse
import importlib
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


MOBILENETV3_SIZE = "small"
USE_TORCHVISION_MODEL = True


class MobileNetV3(nn.Module):
    """Implementation of MobileNet-v3.

    Args:
        data_config: a dictionary containing information about the data.
        args (optional): args from argparser.
    """

    def __init__(
        self,
        data_config: Dict[str, Any],
        args: Optional[argparse.Namespace] = None
    ):
        super().__init__()

        if args is None:
            self.args = {}
        else:
            self.args = vars(args)

        input_dims = data_config["input_dims"]
        if len(input_dims) != 4:
            raise ValueError(f"Expected input_dims to have 4 dimensions got {len(input_dims)} ({input_dims})")
        num_classes = len(data_config["mapping"])
        mobilenetv3_size = self.args.get("mobilenetv3_size", MOBILENETV3_SIZE)
        if mobilenetv3_size not in ("large", "small"):
            raise ValueError(f"Expected mobilenetv3_size to be 'small' or 'large' got {mobilenetv3_size}.")
        use_torchvision = self.args.get("use_torchvision_model", USE_TORCHVISION_MODEL)
        if use_torchvision:
            tv_models_module = importlib.import_module("torchvision.models")
            self.model = getattr(tv_models_module, f"mobilenet_v3_{mobilenetv3_size}")(pretrained=True)
            self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, num_classes)
        else:
            raise NotImplementedError()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns tensor of logits for each class."""
        return self.model(x)

    @staticmethod
    def add_to_argparse(
        parser: argparse.ArgumentParser,
        main_parser: argparse.ArgumentParser  # pylint: disable=unused-argument
    ) -> argparse.ArgumentParser:
        """Adds possible args to the given parser."""
        parser.add_argument(
            "--mobilenetv3_size", type=str, default=MOBILENETV3_SIZE,
            help="Size of mobilenetv3 to use ('large' or 'small')."
        )
        parser.add_argument(
            "--use_torchvision_model", default=False, action="store_true",
            help="If true, will use pretrained mobilenetv3 architecture from torchvision."
        )

        return parser
