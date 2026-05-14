"""Prepare the Roboflow dataset for YOLOv8 fine-tuning.

Reads the raw Roboflow export (all frames in one 'train' folder) and:
  1. Groups frames by source match (derived from filename prefix).
  2. Allocates whole matches to train / val / test (no cross-match leakage).
  3. Optionally merges 3-class labels (Referee / Team-1 / Team-2) → 1-class
     (player), which is the recommended mode for this pipeline.
  4. Writes a clean data.yaml compatible with `ultralytics` training.

Output structure
----------------
  <out>/
    train/images/   train/labels/
    val/images/     val/labels/
    test/images/    test/labels/
    data.yaml

Usage
-----
# Merge classes (recommended — TeamClassifier handles team separation)
python scripts/prepare_dataset.py --merge-classes

# Keep 3 original classes
python scripts/prepare_dataset.py

# Custom paths
python scripts/prepare_dataset.py \\
    --src-images "data/Rugby League Player Tracking.yolov8/train/images" \\
    --src-labels "data/Rugby League Player Tracking.yolov8/train/labels" \\
    --out data/rugby_player_detection \\
    --merge-classes \\
    --val-matches 1 --test-matches 1
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def discover_sources(img_dir: Path) -> dict[str, list[Path]]:
    """Return {source_name: [image_path, ...]} sorted by frame number."""
    sources: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for p in sorted(img_dir.glob("*.jpg")):
        m = re.match(r"^(.+?)_mp4-(\d+)_", p.name)
        if m:
            sources[m.group(1)].append((int(m.group(2)), p))
        else:
            sources["__unknown__"].append((0, p))
    # Sort each source by frame number and strip the int
    return {k: [p for _, p in sorted(v)] for k, v in sources.items()}


def remap_label(label_path: Path, dst_path: Path, merge: bool) -> None:
    """Copy label file to dst, optionally remapping all class IDs to 0."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if not label_path.exists():
        dst_path.write_text("")  # empty label = no annotations
        return
    lines = label_path.read_text().splitlines()
    if merge:
        out = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                out.append("0 " + " ".join(parts[1:]))
        dst_path.write_text("\n".join(out) + ("\n" if out else ""))
    else:
        shutil.copy2(label_path, dst_path)


def copy_split(
    image_paths: list[Path],
    src_labels: Path,
    out_root: Path,
    split: str,
    merge: bool,
) -> None:
    img_out = out_root / split / "images"
    lbl_out = out_root / split / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    for img_path in image_paths:
        # Image
        shutil.copy2(img_path, img_out / img_path.name)
        # Label — Roboflow uses same stem with .txt extension
        lbl_path = src_labels / (img_path.stem + ".txt")
        remap_label(lbl_path, lbl_out / (img_path.stem + ".txt"), merge)


def write_yaml(out_root: Path, merge: bool) -> None:
    if merge:
        names = ["player"]
    else:
        names = ["Referee", "Team 1", "Team 2"]
    config = {
        "path": str(out_root.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(names),
        "names": names,
    }
    (out_root / "data.yaml").write_text(yaml.dump(config, sort_keys=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    src_images = Path(args.src_images)
    src_labels = Path(args.src_labels)
    out_root = Path(args.out)

    if not src_images.exists():
        sys.exit(f"Images directory not found: {src_images}")

    sources = discover_sources(src_images)
    if not sources:
        sys.exit("No images found — check --src-images path.")

    # Sort sources by frame count (descending) for a balanced allocation
    sorted_sources = sorted(sources.items(), key=lambda x: -len(x[1]))

    print(f"Found {sum(len(v) for v in sources.values())} images across {len(sources)} sources:")
    for name, imgs in sorted_sources:
        print(f"  {len(imgs):>3} frames  {name}")

    n_val = args.val_matches
    n_test = args.test_matches

    if n_val + n_test >= len(sorted_sources):
        sys.exit(
            f"Cannot allocate {n_val} val + {n_test} test matches from {len(sorted_sources)} total. "
            "Reduce --val-matches or --test-matches."
        )

    # Allocate: smallest sources → test, next smallest → val, rest → train
    # (keeps most data for training)
    test_sources = [sorted_sources[-(i + 1)] for i in range(n_test)]
    val_sources = [sorted_sources[-(n_test + i + 1)] for i in range(n_val)]
    train_sources = sorted_sources[: len(sorted_sources) - n_val - n_test]

    print(f"\nSplit allocation (--merge-classes={args.merge_classes}):")
    print(f"  Train ({sum(len(v) for _, v in train_sources)} frames): "
          + ", ".join(n for n, _ in train_sources))
    print(f"  Val   ({sum(len(v) for _, v in val_sources)} frames): "
          + ", ".join(n for n, _ in val_sources))
    print(f"  Test  ({sum(len(v) for _, v in test_sources)} frames): "
          + ", ".join(n for n, _ in test_sources))

    for split, split_sources in [
        ("train", train_sources),
        ("val", val_sources),
        ("test", test_sources),
    ]:
        for _, img_paths in split_sources:
            copy_split(img_paths, src_labels, out_root, split, args.merge_classes)

    write_yaml(out_root, args.merge_classes)

    total = sum(len(v) for v in sources.values())
    print(f"\nDataset written to: {out_root}")
    print(f"  {total} images total  |  nc={'1 (player)' if args.merge_classes else '3 (Ref/T1/T2)'}")
    print(f"  Train with:  yolo train data={out_root / 'data.yaml'} model=yolov8n.pt epochs=50")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prepare Roboflow dataset for YOLOv8 training")
    p.add_argument(
        "--src-images",
        default="data/Rugby League Player Tracking.yolov8/train/images",
        help="Path to Roboflow train/images directory",
    )
    p.add_argument(
        "--src-labels",
        default="data/Rugby League Player Tracking.yolov8/train/labels",
        help="Path to Roboflow train/labels directory",
    )
    p.add_argument(
        "--out",
        default="data/rugby_player_detection",
        help="Output directory for the prepared dataset",
    )
    p.add_argument(
        "--merge-classes",
        action="store_true",
        help="Remap all 3 classes (Referee / Team-1 / Team-2) to class 0 (player)",
    )
    p.add_argument(
        "--val-matches",
        type=int,
        default=1,
        help="Number of whole matches to allocate to val split (default: 1)",
    )
    p.add_argument(
        "--test-matches",
        type=int,
        default=1,
        help="Number of whole matches to allocate to test split (default: 1)",
    )
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
