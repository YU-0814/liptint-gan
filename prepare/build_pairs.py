"""Step 4: pack pairs into the pix2pix layout.

Every image is padded to a square with black borders (so lips are not distorted by resizing), resized to
256x256, and written as
    dataset/{train,val}/input/NNNNN.png   RGBA: bare lip + mask as alpha
    dataset/{train,val}/target/NNNNN.png  RGB:  tinted lip

    python prepare/build_pairs.py data/pairs data/dataset --val 0.2
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np


def pad_to_square(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    side = max(h, w)
    top, left = (side - h) // 2, (side - w) // 2
    return cv2.copyMakeBorder(img, top, side - h - top, left, side - w - left, cv2.BORDER_CONSTANT, value=0)


def square(img: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(pad_to_square(img), (size, size), interpolation=cv2.INTER_AREA)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("pairs_dir", type=Path)
    p.add_argument("dataset_dir", type=Path)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--val", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    pairs = sorted(a.pairs_dir.glob("*/*/tinted.jpg"))
    random.seed(a.seed)
    random.shuffle(pairs)
    n_val = int(len(pairs) * a.val)
    for split, subset in [("val", pairs[:n_val]), ("train", pairs[n_val:])]:
        for kind in ("input", "target"):
            (a.dataset_dir / split / kind).mkdir(parents=True, exist_ok=True)
        for i, tinted_path in enumerate(subset):
            d = tinted_path.parent
            tinted = square(cv2.imread(str(d / "tinted.jpg")), a.size)
            bare = square(cv2.imread(str(d / "bare.jpg")), a.size)
            mask = square(cv2.imread(str(d / "mask.png"), cv2.IMREAD_GRAYSCALE), a.size)
            cv2.imwrite(str(a.dataset_dir / split / "input" / f"{i:05d}.png"), np.dstack([bare, mask]))
            cv2.imwrite(str(a.dataset_dir / split / "target" / f"{i:05d}.png"), tinted)
        print(f"{split}: {len(subset)} pairs")


if __name__ == "__main__":
    main()
