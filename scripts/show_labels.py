"""Draw converted COCO boxes on sample frames, to check the conversion by eye.

Also prints box counts and sizes per class (the ball is the hard one).

Usage:
    uv run python scripts/show_labels.py --split valid --n 6
"""

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

COLORS = {
    "player": (66, 135, 245),
    "goalkeeper": (245, 197, 66),
    "referee": (235, 64, 52),
    "ball": (255, 255, 255),
}
PREVIEW_WIDTH = 1280


def print_box_stats(sizes: dict[str, list[tuple[float, float]]]) -> None:
    print(f"{'class':<11}{'boxes':>8}{'median w x h (px)':>20}{'smallest':>12}")
    for name, wh in sorted(sizes.items()):
        med_w = statistics.median(w for w, _ in wh)
        med_h = statistics.median(h for _, h in wh)
        small_w, small_h = min(wh, key=lambda s: s[0] * s[1])
        print(f"{name:<11}{len(wh):>8}{f'{med_w:.0f} x {med_h:.0f}':>20}{f'{small_w:.0f} x {small_h:.0f}':>12}")


def draw(image_path: Path, anns: list[dict], names: dict[int, str]) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    canvas = ImageDraw.Draw(img)
    for ann in anns:
        name = names[ann["category_id"]]
        x, y, w, h = ann["bbox"]
        color = COLORS.get(name, (0, 255, 0))
        canvas.rectangle([x, y, x + w, y + h], outline=color, width=3)
        if name == "ball":  # a ring so the tiny ball is visible in the preview
            cx, cy = x + w / 2, y + h / 2
            canvas.ellipse([cx - 25, cy - 25, cx + 25, cy + 25], outline=color, width=3)
        canvas.text((x, max(0, y - 12)), name, fill=color)
    scale = PREVIEW_WIDTH / img.width
    return img.resize((PREVIEW_WIDTH, round(img.height * scale)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview converted COCO labels.")
    parser.add_argument("--data", type=Path, default=Path("data/rfdetr_gsr"))
    parser.add_argument("--split", default="valid")
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("outputs/label_check"))
    args = parser.parse_args()

    split_dir = args.data / args.split
    coco = json.loads((split_dir / "_annotations.coco.json").read_text())
    names = {c["id"]: c["name"] for c in coco["categories"]}

    by_image: dict[int, list[dict]] = defaultdict(list)
    sizes: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for ann in coco["annotations"]:
        by_image[ann["image_id"]].append(ann)
        sizes[names[ann["category_id"]]].append((ann["bbox"][2], ann["bbox"][3]))

    print(f"{args.split}: {len(coco['images'])} frames, {len(coco['annotations'])} boxes")
    print_box_stats(sizes)

    args.out.mkdir(parents=True, exist_ok=True)
    sample = random.Random(args.seed).sample(coco["images"], k=min(args.n, len(coco["images"])))
    for im in sample:
        preview = draw(split_dir / im["file_name"], by_image[im["id"]], names)
        dest = args.out / f"{args.split}_{im['file_name']}"
        preview.save(dest, quality=90)
        print(f"wrote {dest} ({len(by_image[im['id']])} boxes)")


if __name__ == "__main__":
    main()
