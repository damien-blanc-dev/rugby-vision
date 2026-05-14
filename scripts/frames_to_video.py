"""Assemble annotated dataset images into a playable MP4.

The Roboflow export contains frames sampled from several distinct match
videos.  This script groups them by source match, sorts by frame number,
and writes one MP4 per source (or one chosen source).

Usage
-----
# List available sources
python scripts/frames_to_video.py --images-dir data/... --list

# Build video for the largest source (default)
python scripts/frames_to_video.py --images-dir "data/Rugby League Player Tracking.yolov8/train/images"

# Build video for a specific source and draw GT boxes
python scripts/frames_to_video.py \
    --images-dir "data/Rugby League Player Tracking.yolov8/train/images" \
    --labels-dir "data/Rugby League Player Tracking.yolov8/train/labels" \
    --source "2022-Broncos-Bulldogs-H2-Video-Wide" \
    --output data/broncos_bulldogs.mp4 \
    --fps 8 \
    --draw-labels
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


# Class colours for GT overlay [B, G, R]
_CLASS_COLORS = {
    0: (200, 200, 200),  # Referee — grey
    1: (0, 100, 255),    # Team 1  — orange
    2: (255, 50, 50),    # Team 2  — blue
}
_CLASS_NAMES = {0: "Ref", 1: "T1", 2: "T2"}


def discover_sources(img_dir: Path) -> dict[str, list[tuple[int, Path]]]:
    """Return {source_name: [(frame_num, path), ...]} sorted by frame."""
    sources: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for p in img_dir.glob("*.jpg"):
        m = re.match(r"^(.+?)_mp4-(\d+)_jpg", p.name)
        if m:
            sources[m.group(1)].append((int(m.group(2)), p))
        else:
            sources["unknown"].append((0, p))
    for frames in sources.values():
        frames.sort(key=lambda x: x[0])
    return dict(sources)


def draw_gt_boxes(frame: np.ndarray, label_path: Path) -> np.ndarray:
    """Overlay YOLO-format ground-truth boxes on *frame*."""
    if not label_path.exists():
        return frame
    h, w = frame.shape[:2]
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            colour = _CLASS_COLORS.get(cls, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(
                frame,
                _CLASS_NAMES.get(cls, str(cls)),
                (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                colour,
                1,
                cv2.LINE_AA,
            )
    return frame


def build_video(
    frames: list[tuple[int, Path]],
    output: Path,
    fps: float,
    repeat: int,
    labels_dir: Path | None,
    draw_labels: bool,
) -> None:
    """Write frames to *output* MP4."""
    if not frames:
        sys.exit("No frames found for this source.")

    # Determine resolution from first frame
    first = cv2.imread(str(frames[0][1]))
    if first is None:
        sys.exit(f"Cannot read image: {frames[0][1]}")
    h, w = first.shape[:2]

    output.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output), fourcc, fps, (w, h))

    print(f"  Resolution : {w}×{h}")
    print(f"  Frames     : {len(frames)}  (×{repeat} repeat = {len(frames)*repeat} total)")
    print(f"  Duration   : ~{len(frames) * repeat / fps:.1f} s at {fps} fps")

    for frame_num, img_path in frames:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        if draw_labels and labels_dir:
            label_stem = img_path.stem + ".txt"
            label_path = labels_dir / label_stem
            # Roboflow label stems match image stems
            if not label_path.exists():
                # Try stripping the hash suffix
                label_path = labels_dir / (img_path.stem.rsplit(".", 1)[0] + ".txt")
            img = draw_gt_boxes(img, label_path)

        # Stamp frame number
        cv2.putText(
            img,
            f"frame #{frame_num}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        for _ in range(repeat):
            writer.write(img)

    writer.release()
    print(f"  Saved → {output}")


def main(args: argparse.Namespace) -> None:
    img_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir) if args.labels_dir else None

    sources = discover_sources(img_dir)

    if args.list:
        print("Available sources:")
        for name, frames in sorted(sources.items(), key=lambda x: -len(x[1])):
            print(f"  {len(frames):>3} frames  {name}")
        return

    # Choose source
    if args.source:
        if args.source not in sources:
            sys.exit(f"Source '{args.source}' not found. Use --list to see options.")
        chosen = args.source
    else:
        # Largest source
        chosen = max(sources, key=lambda k: len(sources[k]))

    frames = sources[chosen]
    output = Path(args.output) if args.output else Path(f"data/{chosen}.mp4")

    print(f"\nBuilding video for: {chosen}")
    build_video(
        frames,
        output,
        fps=args.fps,
        repeat=args.repeat,
        labels_dir=labels_dir,
        draw_labels=args.draw_labels,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Assemble Roboflow frames into MP4")
    p.add_argument(
        "--images-dir",
        default="data/Rugby League Player Tracking.yolov8/train/images",
        help="Path to images folder",
    )
    p.add_argument(
        "--labels-dir",
        default="data/Rugby League Player Tracking.yolov8/train/labels",
        help="Path to labels folder (for --draw-labels)",
    )
    p.add_argument("--source", help="Source name (use --list to see options)")
    p.add_argument("--output", help="Output MP4 path (default: data/<source>.mp4)")
    p.add_argument("--fps", type=float, default=8.0, help="Output FPS (default: 8)")
    p.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="Repeat each frame N times for smoother playback (default: 3)",
    )
    p.add_argument(
        "--draw-labels",
        action="store_true",
        help="Overlay GT bounding boxes coloured by class",
    )
    p.add_argument("--list", action="store_true", help="List available sources and exit")
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
