"""Module for processing raw parking spot dataset."""


import multiprocessing as mp
from pathlib import Path
from time import time
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import h5py
import json
import numpy as np
from sklearn.model_selection import train_test_split

from .parse_labels import get_labels_and_paths


ImagePath = Union[Path, str]


class ParkingSpotsProcessor:
    def __init__(
        self,
        use_local: bool,
        dl_data_dirname: Path,
        images_dirname: Path,
        processed_data_filename: Path,
        essentials_filename: Path,
        trainval_split: float,
        train_split: float,
        num_processing_workers: int,
        processing_batch_size: int,
        rewrite: bool,
        seed: int,
        resize_size: Optional[Tuple[int, int]] = None
    ):
        self.use_local = use_local
        self.dl_data_dirname = dl_data_dirname
        self.images_dirname = images_dirname
        self.processed_data_filename = processed_data_filename
        self.essentials_filename = essentials_filename
        self.trainval_split = trainval_split
        self.train_split = train_split
        self.num_processing_workers = num_processing_workers
        self.processing_batch_size = processing_batch_size
        if num_processing_workers == -1:
            self.num_processing_workers = mp.cpu_count() - 1            
        self.rewrite = rewrite
        self.seed = seed
        self.resize_size = resize_size

    def process(self) -> None:
        """Processes raw dataset."""
        self._process_dataset()

    def _process_dataset(
        self,
    ) -> None:
        """Processes dataset for training.

        Args:
            use_local: If True, will use data from data folder, otherwise will download images from urls.
            rewrite: If True, will processes raw data even if a processed file already exists.
            seed: An int used to seed train, val and test random split.
            resize_size (optional): A tuple of a size of images in the resulting file. If None, will not resize.
        """
        if not self.rewrite and self.processed_data_filename.exists():
            print("Dataset already processed and rewrite=False.")
            return

        start_t = time()

        print(f"Parsing export json file. {time() - start_t:.2f}")
        y_train, y_val, y_test, paths_train, paths_val, paths_test, mapping = self._parse_export_file()

        print(f"Will save train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")
        print(f"Creating HDF5 file. {time() - start_t:.2f}")
        image_shape = self._create_hdf5(y_train, y_val, y_test, paths_train, paths_val, paths_test, start_t)

        print(f"Saving essential dataset parameters. {time() - start_t:.2f}")
        self._create_essentials(mapping, image_shape)

        print(f"Done processing. {time() - start_t:.2f}")

    def _parse_export_file(
        self
    ) -> Tuple[List[int], List[int], List[int], List[ImagePath], List[ImagePath], List[ImagePath], Dict[int, str]]:
        labels_and_paths = get_labels_and_paths(
            downloaded_dirname=self.dl_data_dirname,
            return_url=(not self.use_local)
        )

        name_to_class = {}
        y = []
        paths = []
        for label, path in labels_and_paths:
            if label not in name_to_class:
                name_to_class[label] = len(name_to_class)
            y.append(name_to_class[label])
            paths.append(self.images_dirname / path if self.use_local else path)
        mapping = {name_to_class[name]: name for name in name_to_class}

        y_trainval, y_test, paths_trainval, paths_test = train_test_split(
            y, paths,
            train_size=self.trainval_split, stratify=y, shuffle=True, random_state=self.seed
        )
        y_train, y_val, paths_train, paths_val = train_test_split(
            y_trainval, paths_trainval,
            train_size=self.train_split, stratify=y_trainval, shuffle=True, random_state=self.seed
        )
        return y_train, y_val, y_test, paths_train, paths_val, paths_test, mapping

    def _create_hdf5(
        self,
        y_train: List[int],
        y_val: List[int],
        y_test: List[int],
        paths_train: List[ImagePath],
        paths_val: List[ImagePath],
        paths_test: List[ImagePath],
        start_t: float
    ) -> Tuple[int, int, int]:
        """Create HDF5 file from data.

        Args:
            y_train: A list of labels for train split.
            y_val: A list of labels for val split.
            y_test: A list of labels for test split.
            paths_train: A list of paths/urls for train split.
            paths_val: A list of paths/urls for val split.
            paths_test: A list of paths/urls for test split.
            image_shape: A shape of processed images.
            start_t: A start time for logging.
        """
        image_shape = _get_image(paths_train[0], self.resize_size).shape
        chunk_shape = (1, *image_shape)
        x_train_shape = (len(y_train), *image_shape)
        x_val_shape = (len(y_val), *image_shape)
        x_test_shape = (len(y_test), *image_shape)
        self.processed_data_filename.parent.mkdir(exist_ok=True, parents=True)
        with h5py.File(self.processed_data_filename, "w") as f:
            dataset_train = f.create_dataset("x_train", shape=x_train_shape, chunks=chunk_shape, dtype=np.uint8)
            f.create_dataset("y_train", data=y_train, dtype=np.int64)
            dataset_val = f.create_dataset("x_val", shape=x_val_shape, chunks=chunk_shape, dtype=np.uint8)
            f.create_dataset("y_val", data=y_val, dtype=np.int64)
            dataset_test = f.create_dataset("x_test", shape=x_test_shape, chunks=chunk_shape, dtype=np.uint8)
            f.create_dataset("y_test", data=y_test, dtype=np.int64)

            with mp.Pool(self.num_processing_workers) as pool:
                self._process_split(pool, paths_train, dataset_train, self.resize_size, start_t, "train")
                self._process_split(pool, paths_val, dataset_val, self.resize_size, start_t, "val")
                self._process_split(pool, paths_test, dataset_test, self.resize_size, start_t, "test")
        return image_shape

    def _process_split(
        self,
        pool: mp.Pool,
        image_paths: List[ImagePath],
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
        n_batches = (len(image_paths) - 1) // self.processing_batch_size + 1
        for i in range(n_batches):
            start = i * self.processing_batch_size
            end = min((i + 1) * self.processing_batch_size, len(image_paths))
            print(f"Processing {name} batch {i + 1}/{n_batches} of size {end - start}. {time() - start_t:.2f}")
            images = pool.starmap(
                _get_image,
                zip(
                    image_paths[start:end],
                    (resize_size for _ in range(start, end))
                )
            )
            dataset[start:end] = images

    def _create_essentials(
        self,
        mapping: Dict[Any, Any],
        image_shape: Tuple[int, int, int]
    ) -> None:
        """Creates essentials file for the dataset.

        Args:
            mapping: A dict containing a mapping between classes and their names.
            image_shape: A shape of images in processed dataset.
        """
        essentials = {
            "mapping": mapping,
            "input_size": image_shape
        }
        with self.essentials_filename.open("w") as f:
            json.dump(essentials, f)


def _get_image(
    filename: ImagePath,
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
