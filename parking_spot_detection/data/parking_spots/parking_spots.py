"""Parking Spots data module."""

import argparse
import json
from typing import Optional, Tuple

import h5py
from torch.utils.data import DataLoader
from torchvision import transforms

from ..base_data_module import BaseDataModule
from ..hdf5_dataset import HDF5Dataset
from .processing import ParkingSpotsProcessor


RAW_DATA_DIRNAME = BaseDataModule.data_dirname() / "raw" / "parking_dataset"
METADATA_FILENAME = RAW_DATA_DIRNAME / "metadata.toml"
DL_DATA_DIRNAME = BaseDataModule.data_dirname() / "downloaded" / "parking_dataset"
IMAGES_DIRNAME = DL_DATA_DIRNAME / "parking_spot_images"
PROCESSED_DATA_DIRNAME = BaseDataModule.data_dirname() / "processed" / "parking_dataset"
PROCESSED_DATA_FILENAME = PROCESSED_DATA_DIRNAME / "processed.h5"
ESSENTIALS_FILENAME = PROCESSED_DATA_DIRNAME / "parking_dataset_essentials.json"

REWRITE = False
SEED = 0
NUM_PROCESSING_WORKERS = -1
PROCESSING_BATCH_SIZE = 1000
TRAINVAL_SPLIT = 0.9
TRAIN_SPLIT = 0.95

# Calculated using ..utils.calculate_mean_and_std with 4320 train images, seed=0.
DATASET_MEAN = [0.4690, 0.4777, 0.4608]
DATASET_STD = [0.2022, 0.1973, 0.2252]

COLOR_JITTER = 0.0
HORIZONTAL_FLIP = False
DEGREES_AFFINE = 0
TRANSLATE_AFFINE = 0.0
SCALE_MARGIN_AFFINE = 0.0
SHEAR_AFFINE = 0
RANDOM_RESIZED_CROP = False
RANDOM_ERASING = False

USE_LOCAL = True
WORKER_BUFFER_SIZE = 5


class ParkingSpots(BaseDataModule):
    """Parking Spots data module."""

    def __init__(self, args: Optional[argparse.Namespace] = None):
        super().__init__(args)

        if args is None:
            self.args = {}
        else:
            self.args = vars(args)

        self.rewrite = self.args.get("rewrite", REWRITE)
        self.num_processing_workers = self.args.get("num_processing_workers", NUM_PROCESSING_WORKERS)
        self.seed = self.args.get("seed", SEED)
        self.use_local = self.args.get("use_local", USE_LOCAL)
        self.worker_buffer_size = self.args.get("worker_buffer_size", WORKER_BUFFER_SIZE)

        self.color_jitter = self.args.get("color_jitter", COLOR_JITTER)
        self.horizontal_flip = self.args.get("horizontal_flip", HORIZONTAL_FLIP)
        self.degrees_affine = self.args.get("degrees_affine", DEGREES_AFFINE)
        self.translate_affine = self.args.get("translate_affine", TRANSLATE_AFFINE)
        self.scale_margin_affine = self.args.get("scale_margin_affine", SCALE_MARGIN_AFFINE)
        self.shear_affine = self.args.get("shear_affine", SHEAR_AFFINE)
        self.random_resized_crop = self.args.get("random_resized_crop", RANDOM_RESIZED_CROP)
        self.random_erasing = self.args.get("random_erasing", RANDOM_ERASING)

        with open(ESSENTIALS_FILENAME) as f:
            essentials = json.load(f)

        self.mapping = essentials["mapping"]
        self.dims = (1, essentials["input_size"][2], *essentials["input_size"][:2])
        self.output_dims = (len(self.mapping))
        self.data_train = None
        self.data_val = None
        self.data_test = None

        self._set_transforms()

    def _set_transforms(self):
        """Inits and assigns train and val transform members."""
        train_transform_list = [
            transforms.ToTensor()
        ]
        if self.color_jitter:
            train_transform_list.append(transforms.ColorJitter(self.color_jitter, self.color_jitter, self.color_jitter))
        # transforms.RandomResizedCrop(size?, scale=(0.9, 1.0))
        #TODO: add or remove RandomResizedCrop
        train_transform_list.append(transforms.RandomAffine(
            degrees=self.degrees_affine,
            translate=(self.translate_affine, self.translate_affine),
            scale=(1 - self.scale_margin_affine, 1 + self.scale_margin_affine),
            shear=self.shear_affine
        ))
        if self.horizontal_flip:
            train_transform_list.append(transforms.RandomHorizontalFlip())
        train_transform_list.append(transforms.Normalize(DATASET_MEAN, DATASET_STD))
        if self.random_erasing:
            train_transform_list.append(transforms.RandomErasing())
        self.train_transform = transforms.Compose(train_transform_list)
        self.val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(DATASET_MEAN, DATASET_STD)
        ])

    @staticmethod
    def add_to_argparse(
        parser: argparse.ArgumentParser,
        main_parser: argparse.ArgumentParser
    ) -> argparse.ArgumentParser:
        """Adds argumets to parser required for Parking Spots Dataset."""
        parser = BaseDataModule.add_to_argparse(parser, main_parser)
        parser.add_argument(
            "--num_processing_workers", type=int, default=NUM_PROCESSING_WORKERS,
            help="Number of workers to use for processing raw data. If -1 will use <num_cpu_cores> - 1."
        )
        parser.add_argument(
            "--rewrite", default=False, action="store_true",
            help="If True, will processes raw data even if a processed file already exists."
        )
        parser.add_argument(
            "--use_local", default=False, action="store_true",
            help="If True, will use data from data folder, otherwise will download images from urls."
        )
        parser.add_argument(
            "--worker_buffer_size", type=int, default=WORKER_BUFFER_SIZE,
            help="A size of the buffer of each dataloader worker."
        )
        parser.add_argument(
            "--color_jitter", type=float, default=COLOR_JITTER,
            help="Specifies the maximum random change in the brigtness, contranst and saturation of an image."
        )
        parser.add_argument(
            "--horizontal_flip", default=False, action="store_true",
            help="If True, will flip an image horizontally with p=0.5."
        )
        parser.add_argument(
            "--degrees_affine", type=int, default=DEGREES_AFFINE,
            help="A range (will be (-degrees_affine, +degrees_affine)) of degrees a random rotation is chosen from."
        )
        parser.add_argument(
            "--translate_affine", type=float, default=TRANSLATE_AFFINE,
            help="A maximum absolute fraction for translations."
        )
        parser.add_argument(
            "--scale_margin_affine", type=float, default=SCALE_MARGIN_AFFINE,
            help="Scaling factor margin (1 - scale_margin_affine < scale < 1 + scale_margin_affine)."
        )
        parser.add_argument(
            "--shear_affine", type=int, default=SHEAR_AFFINE,
            help="A range (-shear_affine, +shear_affine) of degrees to select from."
        )
        parser.add_argument(
            "--random_resized_crop", default=False, action="store_true",
            help="If True, will perform random resize and rescaling before cropping to input size."
        )
        parser.add_argument(
            "--random_erasing", default=False, action="store_true",
            help="If True, will randomly erase pixels in a rectange region of an image."
        )
        return parser

    def prepare_data(self) -> None:
        """Prepares data for the node globally."""
        if self.rewrite or not PROCESSED_DATA_FILENAME.exists():
            _process_dataset(
                use_local=self.use_local,
                num_processing_workers=self.num_processing_workers,
                rewrite=self.rewrite,
                seed=self.seed
            )
        with ESSENTIALS_FILENAME.open("r") as f:
            self.essentials = json.load(f)

    def setup(self, stage: Optional[str]) -> None:
        """Prepares data for each process given a stage."""
        if stage == "fit" or stage is None:
            self.data_train = HDF5Dataset(
                filename=PROCESSED_DATA_FILENAME,
                data_dataset_name="x_train",
                targets_dataset_name="y_train",
                worker_buffer_size=self.worker_buffer_size,
                batch_size=self.batch_size,
                transform=self.train_transform,
                transform_target=None
            )
            self.data_val = HDF5Dataset(
                filename=PROCESSED_DATA_FILENAME,
                data_dataset_name="x_val",
                targets_dataset_name="y_val",
                worker_buffer_size=self.worker_buffer_size,
                batch_size=self.batch_size,
                transform=self.val_transform,
                transform_target=None
            )
        if stage == "test" or stage is None:
            self.data_test = HDF5Dataset(
                filename=PROCESSED_DATA_FILENAME,
                data_dataset_name="x_test",
                targets_dataset_name="y_test",
                worker_buffer_size=self.worker_buffer_size,
                batch_size=self.batch_size,
                transform=self.val_transform,
                transform_target=None
            )

    def train_dataloader(self) -> DataLoader:
        """Returns prepared train dataloader."""
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=False
        )


def _process_dataset(
    use_local: bool,
    num_processing_workers: int,
    rewrite: bool,
    seed: int,
    resize_size: Optional[Tuple[int, int]] = None
) -> None:
    """Processes dataset for training.

    Args:
        use_local: If True, will use data from data folder, otherwise will download images from urls.
        num_processing_workers: Number of workers to use for processing raw data. If -1 will use <num_cpu_cores> - 1.
        rewrite: If True, will processes raw data even if a processed file already exists.
        seed: An int used to seed train, val and test random split.
        resize_size (optional): A tuple of a size of images in the resulting file. If None, will not resize.
    """
    processor = ParkingSpotsProcessor(
        use_local=use_local,
        dl_data_dirname=DL_DATA_DIRNAME,
        images_dirname=IMAGES_DIRNAME,
        processed_data_filename=PROCESSED_DATA_FILENAME,
        essentials_filename=ESSENTIALS_FILENAME,
        trainval_split=TRAINVAL_SPLIT,
        train_split=TRAIN_SPLIT,
        num_processing_workers=num_processing_workers,
        processing_batch_size=PROCESSING_BATCH_SIZE,
        rewrite=rewrite,
        seed=seed,
        resize_size=resize_size
    )
    processor.process()


if __name__ == "__main__":
    _process_dataset(use_local=True, num_processing_workers=9, rewrite=True, seed=0, resize_size=None)

    import matplotlib.pyplot as plt
    def showid(idx, imgs, labels):
        print(labels[idx])
        plt.imshow(imgs[idx])
        plt.show()
    with h5py.File(PROCESSED_DATA_FILENAME, "r") as fi:
        x_train = fi['x_train']
        x_val = fi['x_val']
        x_test = fi['x_test']
        y_train = fi['y_train']
        showi = lambda idx: showid(idx, x_train, y_train)
        import ipdb
        ipdb.set_trace()
        print(1)
