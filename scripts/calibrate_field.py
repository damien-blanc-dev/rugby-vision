"""Interactive field calibration tool.

Run this script once to produce ``calibration.npz`` that the main pipeline
will load on subsequent runs.

Usage
-----
python scripts/calibrate_field.py --source video.mp4
python scripts/calibrate_field.py --source video.mp4 --frame 120 --output my_cal.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rugby_vision.homography import FieldHomography


def main(args: argparse.Namespace) -> None:
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        sys.exit(f"Cannot open: {args.source}")

    # Seek to the target frame
    if args.frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        sys.exit(f"Could not read frame {args.frame} from {args.source}")

    hom = FieldHomography(
        field_length_m=args.length,
        field_width_m=args.width,
    )

    print("=== Interactive Field Calibration ===")
    print("Click the 4 corners in this order:")
    print("  1. Top-Left")
    print("  2. Top-Right")
    print("  3. Bottom-Right")
    print("  4. Bottom-Left")
    print("Press  r  to reset  |  Enter to confirm  |  q to abort\n")

    hom.calibrate_interactive(frame)
    hom.save(args.output)
    print(f"Calibration saved to: {args.output}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RugbyVision — interactive field calibration")
    p.add_argument("--source", required=True, help="Video file path")
    p.add_argument(
        "--frame", type=int, default=0, help="Frame index to use for calibration"
    )
    p.add_argument(
        "--output", default="calibration.npz", help="Output calibration file path"
    )
    p.add_argument("--length", type=float, default=100.0, help="Field length in metres")
    p.add_argument("--width", type=float, default=68.0, help="Field width in metres")
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
