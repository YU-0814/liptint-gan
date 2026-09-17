"""Step 2. Make a bare-lip image from each tinted photo: desaturate and darken inside the mask, fill highlights, blur.

    python prepare/bare_lip.py data/raw data/pairs
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def synthesize_bare_lip(tinted: np.ndarray, mask: np.ndarray) -> np.ndarray:
    lip = mask > 128
    h, s, v = cv2.split(cv2.cvtColor(tinted, cv2.COLOR_BGR2HSV))
    s_scale = 0.4 + 0.6 * (1 - s / 255.0)
    v_scale = 0.85 + 0.15 * (1 - v / 255.0)
    s = np.where(lip, (s * s_scale).astype(np.uint8), s)
    v_new = np.where(lip, (v * v_scale).astype(np.uint8), v)
    bare = cv2.cvtColor(cv2.merge([h, s, v_new]), cv2.COLOR_HSV2BGR)

    gloss = (v > 240) & (s < 50) & lip
    matte = bare[lip & ~gloss]
    if len(matte):
        bare[gloss] = matte.mean(axis=0).astype(np.uint8)
    return cv2.GaussianBlur(bare, (7, 7), 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("raw_dir", type=Path)
    p.add_argument("pairs_dir", type=Path)
    a = p.parse_args()

    n = 0
    for person in sorted(d for d in a.raw_dir.iterdir() if d.is_dir()):
        images = sorted(f for f in person.iterdir() if f.suffix.lower() in {".jpg", ".jpeg"} and f.with_suffix(".png").exists())
        for i, path in enumerate(images, 1):
            tinted = cv2.imread(str(path))
            mask = cv2.imread(str(path.with_suffix(".png")), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (tinted.shape[1], tinted.shape[0]))
            out = a.pairs_dir / person.name / str(i)
            out.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / "tinted.jpg"), tinted)
            cv2.imwrite(str(out / "mask.png"), mask)
            cv2.imwrite(str(out / "bare.jpg"), synthesize_bare_lip(tinted, mask))
            n += 1
    print(f"{n} pairs -> {a.pairs_dir}")


if __name__ == "__main__":
    main()
