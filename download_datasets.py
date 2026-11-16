import os
import subprocess
import sys
from pathlib import Path

DATASETS = [
    {
        "name": "isic",
        "handle": "heyoujue/isic2018-classwise",
        "path": "datasets/isic2018-classwise",
        "description": "ISIC 2018 Skin Lesion (~300MB)"
    },
    {
        "name": "lung",
        "handle": "heyoujue/lungsegmentation",
        "path": "datasets/lungsegmentation",
        "description": "Chest X-Ray / Lung Segmentation (~700MB)"
    },
    {
        "name": "fss",
        "handle": "meowmeowmeowmeowmeow/fss1000-a-1000-class-fewshot-segmentation",
        "path": "datasets/fss1000-a-1000-class-fewshot-segmentation",
        "description": "FSS-1000 Few-Shot Segmentation (~1.5GB)"
    },
    {
        "name": "deepglobe",
        "handle": "heyoujue/deepglobe",
        "path": "datasets/deepglobe",
        "description": "DeepGlobe Land Cover (~1.0GB)"
    },
    {
        "name": "suim",
        "handle": "heyoujue/suim-merged",
        "path": "datasets/suim-merged",
        "description": "SUIM Underwater Segmentation (~1.5GB)"
    }
]

def download_dataset(item):
    name = item["name"]
    handle = item["handle"]
    target_path = Path(item["path"])
    desc = item["description"]

    target_path.mkdir(parents=True, exist_ok=True)
    
    # Check if already populated
    files = list(target_path.glob("*"))
    if len(files) > 1:
        print(f"[SKIP] {name} ({desc}) already downloaded at {target_path}")
        return True

    print(f"\n[DOWNLOADING] {name} ({desc}) -> {target_path}...")
    cmd = [
        "kaggle", "datasets", "download",
        "-d", handle,
        "-p", str(target_path),
        "--unzip"
    ]
    try:
        proc = subprocess.run(cmd, check=True, text=True, capture_output=False)
        print(f"[SUCCESS] {name} downloaded and extracted.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to download {name}: {e}")
        return False

def main():
    print("=== Downloading all required datasets for ABCDFSS ===")
    results = {}
    for item in DATASETS:
        success = download_dataset(item)
        results[item["name"]] = success

    print("\n=== Download Summary ===")
    for k, v in results.items():
        status = "OK" if v else "FAILED"
        print(f" - {k:12s}: {status}")

if __name__ == "__main__":
    main()

