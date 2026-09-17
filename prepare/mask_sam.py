"""Step 1. Click a point on the lips; keep one of SAM's three masks; save it as <image>.png.

    python prepare/mask_sam.py data/raw/<person> --checkpoint sam_vit_b_01ec64.pth
"""
import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry

IMAGE_SUFFIXES = {".jpg", ".jpeg"}


def clean_mask(mask: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(mask.astype(np.uint8) * 255, (5, 5), 0)
    return cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY)[1]


def choose_mask(predictor: SamPredictor, image: np.ndarray) -> np.ndarray | None:
    predictor.set_image(image)
    while True:
        answer = input(f"lip point as x,y (image {image.shape[1]}x{image.shape[0]}) or 's' to skip: ").strip()
        if answer.lower() == "s":
            return None
        try:
            x, y = map(int, answer.split(","))
        except ValueError:
            continue
        masks, _, _ = predictor.predict(point_coords=np.array([[x, y]]), point_labels=np.array([1]), multimask_output=True)
        _, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, mask in zip(axes, masks, strict=True):
            ax.imshow(image)
            ax.imshow(mask, alpha=0.5)
            ax.scatter([x], [y], color="red", s=40)
            ax.axis("off")
        plt.show()
        pick = input("save mask 1/2/3, or 0 to retry: ").strip()
        if pick in {"1", "2", "3"}:
            return masks[int(pick) - 1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("person_dir", type=Path)
    p.add_argument("--checkpoint", default="sam_vit_b_01ec64.pth")
    p.add_argument("--model", default="vit_b")
    a = p.parse_args()

    sam = sam_model_registry[a.model](checkpoint=a.checkpoint).to("cuda" if torch.cuda.is_available() else "cpu")
    predictor = SamPredictor(sam)
    for path in sorted(a.person_dir.iterdir()):
        if path.suffix.lower() not in IMAGE_SUFFIXES or path.with_suffix(".png").exists():
            continue
        print(path.name)
        mask = choose_mask(predictor, np.array(Image.open(path).convert("RGB")))
        if mask is not None:
            Image.fromarray(clean_mask(mask)).save(path.with_suffix(".png"))


if __name__ == "__main__":
    main()
