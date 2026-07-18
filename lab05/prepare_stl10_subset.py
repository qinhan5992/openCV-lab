"""Create a reproducible five-class STL-10 subset from the binary files.

Split policy:
  - train: 300 images per class from the official training set
  - val:   100 images per class from the remaining official training set
  - test:  100 images per class from the official test set
"""

from __future__ import annotations

import csv
import tarfile
from pathlib import Path

import imageio.v3 as iio
import numpy as np


DATA_DIR = Path("data")
ARCHIVE_PATH = DATA_DIR / "stl10_binary.tar.gz"
BINARY_DIR = DATA_DIR / "stl10_binary"
OUTPUT_DIR = DATA_DIR / "selected_5_classes"

SEED = 42
TRAIN_PER_CLASS = 300
VAL_PER_CLASS = 100
TEST_PER_CLASS = 100

# STL-10 original labels are 1..10. Our new labels are remapped to 0..4.
SELECTED_CLASSES = {
    1: (0, "airplane"),
    3: (1, "car"),
    4: (2, "cat"),
    6: (3, "dog"),
    9: (4, "ship"),
}


def ensure_extracted() -> None:
    """Extract the official archive if the binary directory is absent."""
    if BINARY_DIR.is_dir():
        return
    if not ARCHIVE_PATH.is_file():
        raise FileNotFoundError(
            f"Cannot find {ARCHIVE_PATH}. Run download.py first."
        )

    print(f"Extracting {ARCHIVE_PATH} ...")
    with tarfile.open(ARCHIVE_PATH, "r:gz") as archive:
        # filter='data' prevents archive members from escaping DATA_DIR.
        archive.extractall(DATA_DIR, filter="data")


def read_images(path: Path) -> np.ndarray:
    """Read STL-10 binary images as (N, 96, 96, 3) RGB uint8."""
    raw = np.fromfile(path, dtype=np.uint8)
    expected_image_size = 3 * 96 * 96
    if raw.size % expected_image_size != 0:
        raise ValueError(f"Invalid STL-10 image file: {path}")
    images = raw.reshape(-1, 3, 96, 96)
    return images.transpose(0, 3, 2, 1)


def read_labels(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.uint8)


def save_split(
    split: str,
    images: np.ndarray,
    original_labels: np.ndarray,
    selected_indices: dict[int, np.ndarray],
) -> int:
    """Save one split and its CSV metadata; return its image count."""
    csv_path = OUTPUT_DIR / "labels" / f"{split}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for original_label, (new_label, class_name) in SELECTED_CLASSES.items():
        class_dir = OUTPUT_DIR / "images" / split / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        for number, source_index in enumerate(selected_indices[original_label]):
            filename = f"{class_name}_{number:04d}.png"
            image_path = class_dir / filename
            iio.imwrite(image_path, images[source_index])
            rows.append(
                {
                    "image_path": image_path.relative_to(OUTPUT_DIR).as_posix(),
                    "label": new_label,
                    "class_name": class_name,
                    "split": split,
                    "source_index": int(source_index),
                    "original_label": original_label,
                }
            )

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    ensure_extracted()

    train_images = read_images(BINARY_DIR / "train_X.bin")
    train_labels = read_labels(BINARY_DIR / "train_y.bin")
    test_images = read_images(BINARY_DIR / "test_X.bin")
    test_labels = read_labels(BINARY_DIR / "test_y.bin")

    if len(train_images) != len(train_labels) or len(test_images) != len(test_labels):
        raise ValueError("The number of images does not match the number of labels.")

    rng = np.random.default_rng(SEED)
    train_indices = {}
    val_indices = {}
    test_indices = {}

    for original_label in SELECTED_CLASSES:
        official_train = np.flatnonzero(train_labels == original_label)
        official_test = np.flatnonzero(test_labels == original_label)
        rng.shuffle(official_train)
        rng.shuffle(official_test)

        required_train = TRAIN_PER_CLASS + VAL_PER_CLASS
        if len(official_train) < required_train:
            raise ValueError(f"Label {original_label} has too few training images.")
        if len(official_test) < TEST_PER_CLASS:
            raise ValueError(f"Label {original_label} has too few test images.")

        train_indices[original_label] = official_train[:TRAIN_PER_CLASS]
        val_indices[original_label] = official_train[
            TRAIN_PER_CLASS:required_train
        ]
        test_indices[original_label] = official_test[:TEST_PER_CLASS]

    counts = {
        "train": save_split(
            "train", train_images, train_labels, train_indices
        ),
        "val": save_split("val", train_images, train_labels, val_indices),
        "test": save_split("test", test_images, test_labels, test_indices),
    }

    print("\nSubset created successfully:")
    for split, count in counts.items():
        print(f"  {split:<5}: {count:4d} images")
    print(f"  total: {sum(counts.values()):4d} images")
    print(f"Output: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
