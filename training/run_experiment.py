"""A module for running experiments in a unifed way."""

import argparse
import importlib
import json
from pathlib import Path
import random
import warnings

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torch

from parking_spot_detection import lit_models

DATA_CLASS_TEMPLATE = "parking_spot_detection.data.{}"
MODEL_CLASS_TEMPLATE = "parking_spot_detection.models.{}"
DEFAULT_SAVE_PATH = Path(__file__).resolve().parent / "saved_models"
SEED = 0


def _import_class(module_and_class_name: str) -> type:
    """Imports class from the module.

    Args:
        module_and_class_name: A string containing module and class name ("<module>.<class>").
    """
    module_name, class_name = module_and_class_name.rsplit('.', 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


def _setup_parser() -> argparse.ArgumentParser:
    """Setups ArgumentParser with all needed args (from data, model, trainer, etc.)."""
    parser = argparse.ArgumentParser(add_help=False)

    trainer_parser = pl.Trainer.add_argparse_args(parser)
    trainer_parser._action_groups[1].title = "Trainer Args"  # pylint: disable=protected-access
    parser = argparse.ArgumentParser(add_help=False, parents=[trainer_parser])

    parser.add_argument("--data_class", type=str, default="CIFAR10", help="Name of data module to use.")
    parser.add_argument("--model_class", type=str, default="MLP", help="Name of model module to use.")

    parser.add_argument("--use_wandb", default=False, action="store_true", help="If True, will use wandb.")
    parser.add_argument("--group", type=str, default="", help="Experiment group to log in wandb.")

    parser.add_argument("--es_patience", type=int, default=12, help="Patience for Early Stopping.")
    parser.add_argument(
        "--use_lr_monitor", default=False, action="store_true", help="If True. will use LRMonitor callback."
    )

    parser.add_argument(
        "--show_image_examples", default=False, action="store_true",
        help="If True, will show a batch of images (after transforms) instead of training."
    )
    parser.add_argument(
        "--count_classes", default=False, action="store_true",
        help="If True, will count items in each class in train, val and test and print the counts instead of training."
    )

    parser.add_argument(
        "--save_torchscript", default=False, action="store_true",
        help="If True, will save torchscript model at save_model_path."
    )
    parser.add_argument(
        "--save_path", type=str, default=DEFAULT_SAVE_PATH, help="File path at which to save the trained model."
    )

    parser.add_argument("--seed", type=int, default=SEED)

    temp_args, _ = parser.parse_known_args()
    data_class = _import_class(DATA_CLASS_TEMPLATE.format(temp_args.data_class))
    model_class = _import_class(MODEL_CLASS_TEMPLATE.format(temp_args.model_class))

    data_group = parser.add_argument_group(f"Data Args for {temp_args.data_class}")
    data_class.add_to_argparse(data_group, parser)

    model_group = parser.add_argument_group(f"Model Args for {temp_args.model_class}")
    model_class.add_to_argparse(model_group, parser)

    lit_model_group = parser.add_argument_group("LitModel Args")
    lit_models.BaseLitModel.add_to_argparse(lit_model_group, parser)

    parser.add_argument("--help", "-h", action="help")

    return parser


def _set_seeds(seed: int) -> None:
    """Sets seed for RNGs."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def _save_model(args: argparse.Namespace, lit_model: pl.LightningModule, data: pl.LightningDataModule):
    """Saves the model and the description file."""
    if args.save_torchscript:
        args.save_path.mkdir(parents=True, exist_ok=True)

        model_description = {
            "mapping": data.mapping,
            "input_shape": data.dims,
            "output_shape": data.output_dims,
        }
        preprocessing = {}
        normalize = {
            "step": 0,
            "mean": data.val_transform.transforms[1].mean,
            "std": data.val_transform.transforms[1].std
        }
        preprocessing["normalize"] = normalize
        model_description["preprocessing"] = preprocessing

        example_inputs = torch.rand(model_description["input_shape"])
        script = lit_model.cpu().eval().to_torchscript(method="trace", example_inputs=example_inputs)
        torch.jit.save(script.model, str(args.save_path / "model.pt"))
        with open(args.save_path / "model_desc.json", "w+") as f:
            json.dump(model_description, f)


def main() -> None:
    """Runs an experiment with specified args."""
    parser = _setup_parser()
    args, unknown_args = parser.parse_known_args()
    if unknown_args:
        warnings.warn(f"Got unknown args: {unknown_args}")
    print("Running an experiment with specified args:")
    print(args)

    _set_seeds(args.seed)

    data_class = _import_class(DATA_CLASS_TEMPLATE.format(args.data_class))
    model_class = _import_class(MODEL_CLASS_TEMPLATE.format(args.model_class))
    data = data_class(args=args)
    if args.show_image_examples:
        from run_experiment_utils import _show_image_examples  # pylint: disable=import-outside-toplevel
        _show_image_examples(data)
        return
    if args.count_classes:
        from run_experiment_utils import _count_classes  # pylint: disable=import-outside-toplevel
        _count_classes(data)
        return

    model = model_class(data_config=data.config(), args=args)

    lit_model = lit_models.BaseLitModel(model, args=args)

    loggers = []
    if args.use_wandb:
        wandb_logger = WandbLogger(config=args, group=args.group)
        loggers.append(wandb_logger)

    callbacks = []
    callbacks.append(pl.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=args.es_patience))
    if args.use_lr_monitor:
        callbacks.append(pl.callbacks.LearningRateMonitor())

    args.weights_summary = "full"

    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, logger=loggers, default_root_dir="training/logs")

    trainer.tune(lit_model, datamodule=data)  # pylint: disable=no-member
    trainer.fit(lit_model, datamodule=data)  # pylint: disable=no-member
    trainer.test(lit_model, datamodule=data)  # pylint: disable=no-member

    _save_model(args, lit_model, data)


if __name__ == "__main__":
    main()
