"""
prepare_delta_split.py
======================
After downloading from GEE and auto-labelling, this script organises the
Nile Delta tiles into the same train/val/test/unlabelled split format as
KEMET1 so the existing train_classifier.py and evaluate.py work unchanged.

Directory structure produced:
  data/NileDelta_split/
    train/   (60 %)
    val/     (15 %)
    test/    (15 %)
    unlabelled/ (10 %)

File naming convention (same as KEMET1):
  T{t}_{year}_tile_{id}_{label}.tif
  e.g.  T1_2021_tile_001_neg.tif
        T2_2022_tile_001_neg.tif
        T5_2025_tile_001_pos.tif

Pair construction for the classifier:
  - neg→pos pairs: first year with neg label + first year with pos label
  - All-neg tiles: consecutive year pairs all labelled neg
  - pos→pos tiles: pairs after encroachment has started (labelled pos at T1)

Usage:
  python prepare_delta_split.py \
    --src  data/NileDelta_Encroachment \
    --dst  data/NileDelta_split \
    --tiles data/delta_tiles.json
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

YEARS  = [2021, 2022, 2023, 2024, 2025]
SPLITS = {"train": 0.60, "val": 0.15, "test": 0.15, "unlabelled": 0.10}

YEAR_TO_T = {y: f"T{i+1}" for i, y in enumerate(YEARS)}  # 2021→T1, 2022→T2, …


def load_tiles(tiles_json: Path) -> dict[int, dict]:
    tiles = json.loads(tiles_json.read_text())
    return {t["id"]: t for t in tiles}


def split_tile_ids(tile_ids: list[int], seed: int = 42) -> dict[str, list[int]]:
    rng = random.Random(seed)
    ids = sorted(tile_ids)
    rng.shuffle(ids)
    n = len(ids)
    cuts = {}
    acc = 0
    for split, frac in SPLITS.items():
        size = round(n * frac)
        cuts[split] = ids[acc: acc + size]
        acc += size
    # leftover goes to train
    cuts["train"] += ids[acc:]
    return cuts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src",   type=Path, default=Path("data/NileDelta_Encroachment"))
    parser.add_argument("--dst",   type=Path, default=Path("data/NileDelta_split"))
    parser.add_argument("--tiles", type=Path, default=Path("data/delta_tiles.json"))
    args = parser.parse_args()

    tile_meta = load_tiles(args.tiles)

    # Collect all downloaded files grouped by tile_id
    src_files: dict[int, dict[int, Path]] = defaultdict(dict)  # {tile_id: {year: path}}
    for tif in args.src.rglob("*.tif"):
        parts = tif.stem.split("_")
        if len(parts) == 4 and parts[0] == "tile":
            tid  = int(parts[1])
            year = int(parts[2])
            src_files[tid][year] = tif

    tile_ids = sorted(src_files.keys())
    print(f"Found {len(tile_ids)} tiles with downloaded imagery.")

    # Only include tiles that have all 5 years
    complete = [tid for tid in tile_ids if len(src_files[tid]) == len(YEARS)]
    partial  = [tid for tid in tile_ids if len(src_files[tid]) < len(YEARS)]
    print(f"  Complete (all {len(YEARS)} years): {len(complete)}")
    print(f"  Partial (missing years):          {len(partial)}")

    if len(complete) == 0:
        print("[ERROR] No complete tiles found. Check --src path and that GEE exports finished.")
        return

    # Split
    split_map = split_tile_ids(complete)
    for split, ids in split_map.items():
        print(f"  {split:12s}: {len(ids)} tiles")

    # Copy files into split folders with correct naming
    args.dst.mkdir(parents=True, exist_ok=True)
    copied = 0
    for split, ids in split_map.items():
        out_dir = args.dst / split
        out_dir.mkdir(exist_ok=True)
        for tid in ids:
            meta  = tile_meta.get(tid, {})
            label = meta.get("label", "unk")
            for year, src_path in sorted(src_files[tid].items()):
                t_tag   = YEAR_TO_T[year]
                dst_name = f"{t_tag}_{year}_tile_{tid:03d}_{label}.tif"
                dst_path = out_dir / dst_name
                shutil.copy2(src_path, dst_path)
                copied += 1

    print(f"\nCopied {copied} files → {args.dst}")
    print("Ready to train:  python train_classifier.py --data data/NileDelta_split")


if __name__ == "__main__":
    main()
