import os
import shutil
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

def make_demo_dataset(base_dir="datasets/isic2018_classwise_demo", n_images=18, size=512, seed=1234):
    out_dir = Path(base_dir)
    train_in = out_dir / "ISIC2018_Task1-2_Training_Input"
    gt_out   = out_dir / "ISIC2018_Task1_Training_GroundTruth"

    # Check if already exists
    if train_in.exists() and gt_out.exists():
        existing_imgs = list(train_in.rglob("*.jpg"))
        if len(existing_imgs) >= n_images:
            print(f"Dataset already exists at {out_dir} with {len(existing_imgs)} images.")
            return str(out_dir)

    print(f"Creating demo ISIC dataset at {out_dir}...")
    for cat in ("1", "2", "3"):
        (train_in / cat).mkdir(parents=True, exist_ok=True)
    gt_out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    counts = {"1": 0, "2": 0, "3": 0}

    for i in range(n_images):
        img_id = f"ISIC_{i:07d}"
        cat = str(i % 3 + 1)
        img_path = train_in / cat / f"{img_id}.jpg"
        mask_path = gt_out / f"{img_id}_segmentation.png"

        H = W = size
        bg = np.zeros((H, W, 3), np.uint8)
        bg[..., 0] = rng.integers(170, 225)
        bg[..., 1] = rng.integers(120, 175)
        bg[..., 2] = rng.integers(95, 145)
        bg = np.clip(bg + rng.normal(0, 10, (H, W, 3)), 0, 255).astype(np.uint8)
        img = Image.fromarray(bg)

        cx, cy = (int(v) for v in rng.integers(size // 3, 2 * size // 3, size=2))
        rx, ry = (int(v) for v in rng.integers(size // 8, size // 3, size=2))
        color = tuple(int(v) for v in rng.integers(45, 165, size=3))
        ImageDraw.Draw(img).ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=color)
        img = img.filter(ImageFilter.GaussianBlur(1.2))
        img.save(img_path, quality=92)

        mask = Image.new("L", (W, H), 0)
        ImageDraw.Draw(mask).ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=255)
        mask.save(mask_path)
        counts[cat] += 1

    print(f"Created demo dataset successfully: {counts}")
    return str(out_dir)

if __name__ == "__main__":
    make_demo_dataset()

