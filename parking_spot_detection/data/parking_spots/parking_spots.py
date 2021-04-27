"""Parking Spots data module."""

import argparse
import json
import multiprocessing as mp
from pathlib import Path
from time import time
from typing import List, Optional, Tuple, Union

import cv2
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms

from ..base_data_module import BaseDataModule
from ..hdf5_dataset import HDF5Dataset
from .parse_labels import get_labels_and_paths


RAW_DATA_DIRNAME = BaseDataModule.data_dirname() / "raw" / "parking_dataset"
METADATA_FILENAME = RAW_DATA_DIRNAME / "metadata.toml"
DL_DATA_DIRNAME = BaseDataModule.data_dirname() / "downloaded" / "parking_dataset"
IMAGES_DIRNAME = DL_DATA_DIRNAME / "parking_spot_images"
PROCESSED_DATA_DIRNAME = BaseDataModule.data_dirname() / "processed" / "parking_dataset"
PROCESSED_DATA_FILENAME = PROCESSED_DATA_DIRNAME / "processed.h5"
ESSENTIALS_FILENAME = PROCESSED_DATA_DIRNAME / "parking_dataset_essentials.json"

SEED = 0
TRAINVAL_SPLIT = 0.9
TRAIN_SPLIT = 0.95

# Not yet calculated
DATASET_MEAN = 0.0
DATASET_STD = 1.0

USE_LOCAL = True
WORKER_BUFFER_SIZE = 5000

class ParkingSpots(BaseDataModule):
    """Parking Spots data module."""

    def __init__(self, args: Optional[argparse.Namespace] = None):
        super().__init__(args)

        if args is None:
            self.args = {}
        else:
            self.args = vars(args)

        self.seed = self.args.get("seed", SEED)
        self.use_local = self.args.get("use_local", USE_LOCAL)
        self.worker_buffer_size = self.args.get("worker_buffer_size", WORKER_BUFFER_SIZE)

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
        train_transform_list.append(transforms.Normalize(DATASET_MEAN, DATASET_STD))
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
            "--use_local", default=False, action="store_true",
            help="If True, will use data from data folder, otherwise will download images from urls."
        )
        parser.add_argument(
            "--worker_buffer_size", type=int, default=WORKER_BUFFER_SIZE,
            help="A size of the buffer of each dataloader worker."
        )
        return parser

    def prepare_data(self) -> None:
        """Prepares data for the node globally."""
        if not PROCESSED_DATA_FILENAME.exists():
            _process_dataset(use_local=self.use_local)
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
                transform=self.train_transform,
                transform_target=None
            )
            self.data_val = HDF5Dataset(
                filename=PROCESSED_DATA_FILENAME,
                data_dataset_name="x_val",
                targets_dataset_name="y_val",
                worker_buffer_size=self.worker_buffer_size,
                transform=self.val_transforms,
                transform_target=None
            )
        if stage == "test" or stage is None:
            self.data_test = HDF5Dataset(
                filename=PROCESSED_DATA_FILENAME,
                data_dataset_name="x_test",
                targets_dataset_name="y_test",
                worker_buffer_size=self.worker_buffer_size,
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
    rewrite: bool,
    seed: int,
    resize_size: Optional[Tuple[int, int]] = None
) -> None:
    """Processes dataset for training.

    Args:
        use_local: If True, will use data from data folder, otherwise will download images from urls.
        rewrite: If True, will processes raw data even if a processed file already exists.
        seed: An int used to seed train, val and test random split.
        resize_size (optional): A tuple of a size of images in the resulting file. If None, will not resize.
    """
    if not rewrite and PROCESSED_DATA_FILENAME.exists():
        print("Dataset already processed and rewrite=False.")
        return None

    start_t = time()

    print(f"Parsing export json file. {time() - start_t:.2f}")
    labels_and_paths = get_labels_and_paths(
        downloaded_dirname=DL_DATA_DIRNAME,
        return_url=(not use_local)
    )

    name_to_class = {}
    y = []
    paths = []
    for label, path in labels_and_paths:
        if label not in name_to_class:
            name_to_class[label] = len(name_to_class)
        y.append(name_to_class[label])
        paths.append(IMAGES_DIRNAME / path if use_local else path)
    mapping = {name_to_class[name]: name for name in name_to_class}

    y_trainval, y_test, paths_trainval, paths_test = train_test_split(
        y, paths, train_size=TRAINVAL_SPLIT, stratify=y, shuffle=True, random_state=seed
    )
    y_train, y_val, paths_train, paths_val = train_test_split(
        y_trainval, paths_trainval, train_size=TRAIN_SPLIT, stratify=y_trainval, shuffle=True, random_state=seed
    )

    print(f"Will save train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")
    print(f"Creating HDF5 file. {time() - start_t:.2f}")

    sample_image = _get_image(paths_train[0], resize_size)
    image_shape = sample_image.shape

    chunk_shape = (1, *image_shape)
    x_train_shape = (len(y_train), *image_shape)
    x_val_shape = (len(y_val), *image_shape)
    x_test_shape = (len(y_test), *image_shape)
    PROCESSED_DATA_FILENAME.parent.mkdir(exist_ok=True, parents=True)
    with h5py.File(PROCESSED_DATA_FILENAME, "w") as f:
        dataset_train = f.create_dataset("x_train", shape=x_train_shape, chunks=chunk_shape, dtype=np.uint8)
        f.create_dataset("y_train", data=y_train, dtype=np.int32)
        dataset_val = f.create_dataset("x_val", shape=x_val_shape, chunks=chunk_shape, dtype=np.uint8)
        f.create_dataset("y_val", data=y_train, dtype=np.int32)
        dataset_test = f.create_dataset("x_test", shape=x_test_shape, chunks=chunk_shape, dtype=np.uint8)
        f.create_dataset("y_test", data=y_train, dtype=np.int32)

        with mp.Pool(11) as pool:
            _process_split(pool, paths_train, dataset_train, resize_size, start_t, "train")
            _process_split(pool, paths_val, dataset_val, resize_size, start_t, "val")
            _process_split(pool, paths_test, dataset_test, resize_size, start_t, "test")

    print(f"Saving essential dataset parameters. {time() - start_t:.2f}")
    essentials = {
        "mapping": mapping,
        "input_size": image_shape
    }
    with ESSENTIALS_FILENAME.open("w") as f:
        json.dump(essentials, f)

    print(f"Done processing. {time() - start_t:.2f}")

def _process_split(
    pool: mp.Pool,
    image_paths: List[Union[Path, str]],
    dataset: h5py.Dataset,
    resize_size: Optional[Tuple[int, int]],
    start_t: float,
    name: str
) -> None:
    """Processes a split (e.g. train) of data.

    Args:
        image_paths: A list of paths or urls to images.
        dataset: A h5py dataset where to write data.
        resize_size: A tuple of a size of images in the resulting file. If None, will not resize.
        start_t: Time from which to generate log.
        name: Name of the split (e.g. "train").
    """
    processing_batch_size = 1000 #TODO: refactor
    n_batches = (len(image_paths) - 1) // processing_batch_size + 1
    for i in range(n_batches):
        start = i * processing_batch_size
        end = min((i + 1) * processing_batch_size, len(image_paths))
        print(f"Processing {name} batch {i + 1}/{n_batches} of size {end - start}. {time() - start_t:.2f}")
        images = pool.starmap(
            _get_image,
            zip(
                image_paths[start:end],
                (resize_size for _ in range(start, end))
            )
        )
        dataset[start:end] = images

def _get_image(
    filename: Union[Path, str],
    resize_size: Optional[Tuple[int, int]]
) -> np.ndarray:
    """Reads or downloads and resizes an image given its path.

    Args:
        filename: A path to file or a url string.
        resize_size: A size of output image. If None, will not resize.
    """
    if isinstance(filename, str):
        raise NotImplementedError("URL download is not supported.")

    image = cv2.imread(str(filename))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if resize_size is not None:
        image = cv2.resize(image, resize_size)
    return image


if __name__ == "__main__":
    _process_dataset(use_local=True, rewrite=True, seed=0)

    import matplotlib.pyplot as plt
    def showid(idx, imgs, labels):
        print(labels[idx])
        plt.imshow(imgs[idx])
        plt.show()
    with h5py.File(PROCESSED_DATA_FILENAME, "r") as f:
        x_train = f['x_train']
        x_val = f['x_val']
        x_test = f['x_test']
        y_train = f['y_train']
        showi = lambda idx: showid(idx, x_train, y_train)
        import ipdb
        ipdb.set_trace()
        print(1)
