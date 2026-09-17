"""Step 3. Top every person up to N pairs with flips, zooms, shifts and brightness changes.

    python prepare/augment.py data/pairs --per-person 15
"""
import argparse
import random
from pathlib import Path

from PIL import Image, ImageEnhance

OPS = ["translate_up", "translate_down", "translate_left", "translate_right", "hflip", "zoom", "brighter", "darker"]


def translate(img, direction, amount=10):
    w, h = img.size
    box = {"up": (0, amount, w, h), "down": (0, 0, w, h - amount),
           "left": (amount, 0, w, h), "right": (0, 0, w - amount, h)}[direction]
    return img.crop(box)


def zoom(img, scale=0.95):
    w, h = img.size
    dw, dh = int(w * (1 - scale) / 2), int(h * (1 - scale) / 2)
    return img.crop((dw, dh, w - dw, h - dh))


def apply(op, tinted, mask, bare):
    if op.startswith("translate_"):
        d = op.split("_")[1]
        return translate(tinted, d), translate(mask, d), translate(bare, d)
    if op == "hflip":
        flip = Image.FLIP_LEFT_RIGHT
        return tinted.transpose(flip), mask.transpose(flip), bare.transpose(flip)
    if op == "zoom":
        return zoom(tinted), zoom(mask), zoom(bare)
    factor = random.uniform(1.03, 1.05) if op == "brighter" else random.uniform(0.95, 0.97)
    return ImageEnhance.Brightness(tinted).enhance(factor), mask, ImageEnhance.Brightness(bare).enhance(factor)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("pairs_dir", type=Path)
    p.add_argument("--per-person", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    random.seed(a.seed)

    for person in sorted(d for d in a.pairs_dir.iterdir() if d.is_dir()):
        originals = sorted(d for d in person.iterdir() if d.is_dir())
        ops = OPS.copy()
        next_index = len(originals) + 1
        while next_index <= a.per_person and ops:
            src = random.choice(originals)
            op = ops.pop(random.randrange(len(ops)))
            tinted, mask, bare = apply(op, Image.open(src / "tinted.jpg").convert("RGB"),
                                       Image.open(src / "mask.png").convert("L"), Image.open(src / "bare.jpg").convert("RGB"))
            out = person / str(next_index)
            out.mkdir()
            tinted.save(out / "tinted.jpg")
            mask.save(out / "mask.png")
            bare.save(out / "bare.jpg")
            next_index += 1
    total = sum(1 for _ in a.pairs_dir.glob("*/*/tinted.jpg"))
    print(f"{total} pairs after augmentation")


if __name__ == "__main__":
    main()
