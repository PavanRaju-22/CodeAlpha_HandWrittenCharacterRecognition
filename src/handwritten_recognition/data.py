
import gzip
import struct
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "gzip"

TRAIN_IMAGES = DATA_DIR / "emnist-balanced-train-images-idx3-ubyte.gz"
TRAIN_LABELS = DATA_DIR / "emnist-balanced-train-labels-idx1-ubyte.gz"

TEST_IMAGES = DATA_DIR / "emnist-balanced-test-images-idx3-ubyte.gz"
TEST_LABELS = DATA_DIR / "emnist-balanced-test-labels-idx1-ubyte.gz"

CLASS_MAPPING = DATA_DIR / "emnist-balanced-mapping.txt"


def load_idx_images(path: Path) -> np.ndarray:
    """Load images from a compressed IDX image file."""
    with gzip.open(path, "rb") as file:
        header = file.read(16)

        if len(header) != 16:
            raise ValueError(f"Invalid IDX image header: {path}")

        magic, count, rows, cols = struct.unpack(">IIII", header)

        if magic != 2051:
            raise ValueError(f"Invalid IDX image magic number: {magic}")

        image_bytes = file.read()
        expected_bytes = count * rows * cols

        if len(image_bytes) != expected_bytes:
            raise ValueError(
                f"Expected {expected_bytes} image bytes, "
                f"but found {len(image_bytes)} in {path}"
            )

    images = np.frombuffer(image_bytes, dtype=np.uint8)
    return images.reshape(count, rows, cols).copy()


def load_idx_labels(path: Path) -> np.ndarray:
    """Load labels from a compressed IDX label file."""
    with gzip.open(path, "rb") as file:
        header = file.read(8)

        if len(header) != 8:
            raise ValueError(f"Invalid IDX label header: {path}")

        magic, count = struct.unpack(">II", header)

        if magic != 2049:
            raise ValueError(f"Invalid IDX label magic number: {magic}")

        label_bytes = file.read()

        if len(label_bytes) != count:
            raise ValueError(
                f"Expected {count} labels, "
                f"but found {len(label_bytes)} in {path}"
            )

    return np.frombuffer(label_bytes, dtype=np.uint8).copy()
