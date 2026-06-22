"""
Egypt Dataset Splitter
======================
Splits ~600 satellite images into four sets:

  train   60%  — model training
  val     15%  — validation during training
  test    15%  — standard test evaluation
  reserve 10%  — untouched until final final testing

Usage
-----
  python split_dataset.py --src "path/to/egypt_images"
  python split_dataset.py --src "path/to/egypt_images" --out "data/egypt_split"
  python split_dataset.py --src "path/to/egypt_images" --dry-run

The script copies (does NOT move) images into:
  <out>/train/
  <out>/val/
  <out>/test/
  <out>/reserve/

A manifest CSV is saved to <out>/split_manifest.csv so you can always
trace which image ended up in which set.

Supported image extensions: .tif, .tiff, .png, .jpg, .jpeg
"""

import argparse
import csv
import random
import shutil
from pathlib import Path
from typing import List, Dict

from config.settings import DATASET_SPLIT_CONFIG

SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def find_images(src_dir: Path) -> List[Path]:
    images = [
        p for p in sorted(src_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return images


def split_indices(n: int, cfg: Dict) -> Dict[str, List[int]]:
    """Return shuffled index lists for each split."""
    seed  = cfg["random_seed"]
    train = cfg["train_ratio"]
    val   = cfg["val_ratio"]
    test  = cfg["test_ratio"]
    # reserve gets the remainder

    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)

    n_train   = round(n * train)
    n_val     = round(n * val)
    n_test    = round(n * test)
    n_reserve = n - n_train - n_val - n_test   # absorbs rounding remainder

    return {
        "train":   indices[:n_train],
        "val":     indices[n_train : n_train + n_val],
        "test":    indices[n_train + n_val : n_train + n_val + n_test],
        "reserve": indices[n_train + n_val + n_test :],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Split Egypt satellite dataset into train/val/test/reserve"
    )
    parser.add_argument(
        "--src", required=True,
        help="Folder containing all Egypt images"
    )
    parser.add_argument(
        "--out", default="data/egypt_split",
        help="Output root folder (default: data/egypt_split)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print split summary without copying any files"
    )
    parser.add_argument(
        "--train",   type=float, default=DATASET_SPLIT_CONFIG["train_ratio"],
        help=f"Train ratio (default {DATASET_SPLIT_CONFIG['train_ratio']})"
    )
    parser.add_argument(
        "--val",     type=float, default=DATASET_SPLIT_CONFIG["val_ratio"],
        help=f"Val ratio (default {DATASET_SPLIT_CONFIG['val_ratio']})"
    )
    parser.add_argument(
        "--test",    type=float, default=DATASET_SPLIT_CONFIG["test_ratio"],
        help=f"Test ratio (default {DATASET_SPLIT_CONFIG['test_ratio']})"
    )
    parser.add_argument(
        "--seed",    type=int, default=DATASET_SPLIT_CONFIG["random_seed"],
        help=f"Random seed (default {DATASET_SPLIT_CONFIG['random_seed']})"
    )
    args = parser.parse_args()

    src = Path(args.src)
    out = Path(args.out)

    if not src.exists():
        raise FileNotFoundError(f"Source folder not found: {src}")

    images = find_images(src)
    if not images:
        raise ValueError(f"No images found in {src} (supported: {SUPPORTED_EXTENSIONS})")

    cfg = {
        "train_ratio":   args.train,
        "val_ratio":     args.val,
        "test_ratio":    args.test,
        "reserve_ratio": 1.0 - args.train - args.val - args.test,
        "random_seed":   args.seed,
    }

    splits = split_indices(len(images), cfg)

    # Print summary
    print(f"\n{'='*55}")
    print(f"  Egypt Dataset Split Summary")
    print(f"{'='*55}")
    print(f"  Source:  {src}  ({len(images)} images)")
    print(f"  Output:  {out}")
    print(f"  Seed:    {cfg['random_seed']}")
    print(f"{'─'*55}")
    for split_name, idxs in splits.items():
        pct = 100 * len(idxs) / len(images)
        print(f"  {split_name:<10}  {len(idxs):>4} images  ({pct:.1f}%)")
    print(f"{'='*55}\n")

    if args.dry_run:
        print("  [DRY RUN] No files copied.")
        return

    # Create output directories
    for split_name in splits:
        (out / split_name).mkdir(parents=True, exist_ok=True)

    # Copy files + build manifest
    manifest_rows = []
    for split_name, idxs in splits.items():
        for idx in idxs:
            src_file = images[idx]
            dst_file = out / split_name / src_file.name
            # Handle name collisions (e.g. same filename in different subfolders)
            if dst_file.exists():
                stem   = src_file.stem
                suffix = src_file.suffix
                dst_file = out / split_name / f"{stem}_{idx}{suffix}"

            shutil.copy2(src_file, dst_file)
            manifest_rows.append({
                "split":    split_name,
                "source":   str(src_file),
                "dest":     str(dst_file),
                "filename": src_file.name,
            })

    # Save manifest CSV
    manifest_path = out / "split_manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "filename", "source", "dest"])
        writer.writeheader()
        writer.writerows(sorted(manifest_rows, key=lambda r: (r["split"], r["filename"])))

    print(f"  Done! Files copied to: {out}")
    print(f"  Manifest saved: {manifest_path}\n")


if __name__ == "__main__":
    main()
