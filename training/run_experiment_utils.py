"""Utils module for run_experiment."""

from collections import defaultdict
from typing import Dict, DefaultDict

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import tqdm


def _show_image_examples(data: pl.LightningDataModule) -> None:
    """Shows a batch of example images after transforms."""
    data.prepare_data()
    data.setup(None)
    for batch in data.train_dataloader():
        x, y = batch
        for i, img in enumerate(x):
            print(y[i].item(), data.mapping[str(y[i].item())])
            plt.imshow(img.permute(1, 2, 0))
            plt.show()
        return


def _count_classes_for_split(
    dataloader: torch.utils.data.DataLoader,
    mapping: Dict[str, str],
    counts: DefaultDict[str, int]
) -> None:
    """Counts occurences of each class in a particular split."""
    for batch in tqdm.tqdm(dataloader):
        for label in batch[1]:
            counts[mapping[str(label.item())]] += 1


def _count_classes(data: pl.LightningDataModule) -> None:
    """Counts occurences of each class for all splits."""
    data.prepare_data()
    data.setup(None)
    counts = {}
    counts["train"] = defaultdict(int)
    counts["val"] = defaultdict(int)
    counts["test"] = defaultdict(int)
    _count_classes_for_split(data.train_dataloader(), data.mapping, counts["train"])
    _count_classes_for_split(data.val_dataloader(), data.mapping, counts["val"])
    _count_classes_for_split(data.test_dataloader(), data.mapping, counts["test"])
    print(counts)
