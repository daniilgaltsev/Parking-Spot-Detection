"""Module for parsing Label Studio export file for Parking Spot Detection."""


import json
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union
import warnings

DATASET_LABELS_FILENAME = "parking_spot_labels.json"

def get_label_and_path_for_item(item: Any, return_url: bool) -> Tuple[Optional[str], Union[Path, str]]:
    """Extracts a label and path to image from the item.

    Args:
        item: A dict representing one item.
        return_url: If True, will return raw url from export json. Otherwise, will return path_to_downloaded + basename.
    """
    item_annotation = item['annotations'][0]
    item_result = item_annotation['result']
    if item_result:
        tmp = item_annotation['result'][0]
        item_label = tmp['value']['choices'][0]
    else:
        item_label = None

    item_data_url = item['data']['image']
    if return_url:
        return item_label, item_data_url

    item_data_name = item_data_url.rsplit('/', 1)[-1]
    return item_label, item_data_name


def get_labels_and_paths(
    downloaded_dirname: Path,
    return_url: bool,
    dataset_labels_filename: str = DATASET_LABELS_FILENAME
) -> List[Tuple[str, Union[Path, str]]]:
    """Parses an export json file from Label Studio.

    Args:
        downloaded_dirname: A path to the downloaded folder of parking_dataset.
        return_url: If True, will return raw url from export json. Otherwise, will return path_to_downloaded + basename.
        dataset_labels_filename (optional): A string containing the name of the export json file.

    Returns:
        A list of tuples for each item ([(<label>, <path>), ...]).
    """
    with open(downloaded_dirname / dataset_labels_filename, "r") as f:
        data = json.load(f)
    print(type(data))

    labels_and_paths = []

    for item in data:
        label_and_path = get_label_and_path_for_item(item, return_url)
        if label_and_path[0] is None:
            warnings.warn(f"No result for {label_and_path[1]}", UserWarning)
            continue
        labels_and_paths.append(label_and_path)

    return labels_and_paths


if __name__ == "__main__":
    PATH_TO_LABELS = Path(__file__).resolve().parents[3] / "data" / "downloaded" / "parking_dataset"
    get_labels_and_paths(PATH_TO_LABELS, False)
