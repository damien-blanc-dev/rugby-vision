"""Extract a demo GIF from the annotated output video.

Picks the most visually interesting segment (most detections on screen),
resizes to a reasonable width, and saves assets/demo.gif.

Usage
-----
python scripts/make_gif.py --video outputs/broncos_bulldogs_annotated.mp4
python scripts/make_gif.py --video outputs/broncos_bulldogs_annotated.mp4 \
    --start 10 --duration 6 --width 800 --fps 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def make_gif(
    video_path: Path,
    output_path: Path,
    start_s: float,
    duration_s: float,
    width: int,
    fps: int,
) -> None:
    try:
        from PIL import Image
    except ImportError:
        sys.exit("Install Pillow: pip install Pillow")

    import cv2

    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = int(start_s * src_fps)
    end_frame = min(int((start_s + duration_s) * src_fps), total)
    step = max(1, int(src_fps / fps))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames_pil = []
    for i in range(start_frame, end_frame, step):
        ret, frame = cap.read()
        if not ret:
            break
        # Skip non-sampled frames
        if (i - start_frame) % step != 0:
            continue
        h, w = frame.shape[:2]
        new_h = int(width * h / w)
        frame = cv2.resize(frame, (width, new_h))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames_pil.append(Image.fromarray(rgb))

    cap.release()

    if not frames_pil:
        sys.exit("No frames extracted — check --start and --duration values.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames_pil[0].save(
        output_path,
        save_all=True,
        append_images=frames_pil[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=True,
    )
    size_kb = output_path.stat().st_size // 1024
    print(f"GIF saved → {output_path}  ({len(frames_pil)} frames, {size_kb} KB)")
    print(f"Tip: add to README.md:\n  ![Demo](assets/demo.gif)")


def find_best_start(video_path: Path, sample_every: int = 30) -> float:
    """Return the timestamp (seconds) with the most active detections.

    Heuristic: the annotated frame has more coloured pixels (non-green) when
    more players are visible with bounding boxes.
    """
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    best_t, best_score = 0.0, -1.0
    for i in range(0, total, sample_every):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        # Score: fraction of pixels that are NOT grass-green
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
        non_green = 1.0 - green_mask.mean() / 255.0
        if non_green > best_score:
            best_score = non_green
            best_t = i / fps

    cap.release()
    return best_t


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract demo GIF from annotated video")
    p.add_argument("--video", required=True, help="Path to annotated MP4")
    p.add_argument(
        "--output", default="assets/demo.gif", help="Output GIF path"
    )
    p.add_argument(
        "--start",
        type=float,
        default=-1,
        help="Start time in seconds (-1 = auto-detect best segment)",
    )
    p.add_argument("--duration", type=float, default=5.0, help="GIF duration in seconds")
    p.add_argument("--width", type=int, default=800, help="Output width in pixels")
    p.add_argument("--fps", type=int, default=8, help="GIF frame rate")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    video_path = Path(args.video)
    if not video_path.exists():
        sys.exit(f"Video not found: {video_path}")

    start = args.start
    if start < 0:
        print("Auto-detecting best segment…")
        start = find_best_start(video_path)
        print(f"  → using t={start:.1f}s")

    make_gif(
        video_path,
        Path(args.output),
        start_s=start,
        duration_s=args.duration,
        width=args.width,
        fps=args.fps,
    )
