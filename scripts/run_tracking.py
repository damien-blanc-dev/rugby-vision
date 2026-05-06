"""Main CLI script — run the full RugbyVision pipeline on a video.

Usage
-----
python scripts/run_tracking.py --source video.mp4 --config config.yaml
python scripts/run_tracking.py --source "https://youtu.be/..." --config config.yaml
python scripts/run_tracking.py --source video.mp4 --calibrate  # interactive calibration
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rugby_vision.detector import PlayerDetector
from rugby_vision.tracker import PlayerTracker
from rugby_vision.team_classifier import TeamClassifier
from rugby_vision.homography import FieldHomography
from rugby_vision.visualizer import Visualizer
from rugby_vision.metrics import MetricsCollector
from rugby_vision.reidentifier import OcclusionReidentifier
from rugby_vision.analytics import MatchAnalyzer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def open_source(source: str):
    """Return a cv2.VideoCapture for a local path or YouTube URL.

    Parameters
    ----------
    source : str
        Local file path or YouTube URL.

    Returns
    -------
    cv2.VideoCapture
    """
    if source.startswith("http"):
        try:
            import yt_dlp  # noqa: F401
        except ImportError:
            sys.exit("Install yt-dlp: pip install yt-dlp")
        import subprocess, tempfile, shutil

        tmp = tempfile.mktemp(suffix=".mp4")
        cmd = ["yt-dlp", "-f", "mp4", "-o", tmp, source]
        print(f"[yt-dlp] Downloading {source} …")
        subprocess.run(cmd, check=True)
        return cv2.VideoCapture(tmp)
    return cv2.VideoCapture(source)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Warm-up: fit TeamClassifier on the first N frames
# ---------------------------------------------------------------------------

WARMUP_FRAMES = 30  # number of frames used for team clustering


def warmup_team_classifier(
    clf: TeamClassifier,
    cap: cv2.VideoCapture,
    detector: PlayerDetector,
    n_frames: int = WARMUP_FRAMES,
) -> int:
    """Run detector on the first *n_frames* frames and fit the classifier.

    Returns the position (frame index) after warm-up so the caller can seek.
    """
    frames, dets_list = [], []
    for _ in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        dets = detector.detect(frame)
        if len(dets) > 0:
            frames.append(frame)
            dets_list.append(dets)

    if frames:
        print(f"[warm-up] Fitting TeamClassifier on {len(frames)} frames …")
        clf.fit(frames, dets_list)
    else:
        print("[warm-up] No detections during warm-up — team colours not fitted.")

    pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    return pos


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)

    # --- Output dir ---
    out_dir = Path(cfg["metrics"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.source).stem if not args.source.startswith("http") else "youtube"

    # --- Initialise modules ---
    detector = PlayerDetector.from_config(cfg["model"])
    tracker = PlayerTracker.from_config(cfg["tracking"])
    clf = TeamClassifier.from_config(cfg["team_classifier"])
    hom = FieldHomography(
        field_length_m=cfg["homography"]["field_length_m"],
        field_width_m=cfg["homography"]["field_width_m"],
        calibration_path=cfg["homography"]["calibration_save_path"]
        if not args.calibrate
        else None,
    )
    vis = Visualizer.from_config(cfg)
    reid = OcclusionReidentifier.from_config(cfg.get("reidentification", {}))
    analyzer = MatchAnalyzer.from_config(cfg)

    # --- Open source ---
    print(f"[source] Opening {args.source} …")
    cap = open_source(args.source)
    if not cap.isOpened():
        sys.exit(f"Cannot open source: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[source] {width}×{height} @ {fps:.1f} fps  ({total_frames} frames)")

    metrics = MetricsCollector.from_config(cfg, fps=fps)

    # --- Calibration ---
    if args.calibrate or not hom.is_calibrated():
        ret, first_frame = cap.read()
        if not ret:
            sys.exit("Cannot read first frame for calibration.")
        print("[calibration] Starting interactive calibration …")
        hom.calibrate_interactive(first_frame)
        save_path = cfg["homography"]["calibration_save_path"]
        hom.save(save_path)
        print(f"[calibration] Saved to {save_path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # --- Team classifier warm-up (or reload frozen centres) ---
    centers_path = cfg["team_classifier"].get("centers_save_path", "")
    if centers_path and Path(centers_path).exists() and not args.recalibrate_teams:
        print(f"[team] Loading frozen cluster centres from {centers_path}")
        clf.load_centers(centers_path)
    else:
        warmup_team_classifier(clf, cap, detector)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # --- Video writer ---
    mm_w = int(cfg["visualization"]["minimap"]["width"] * (height / cfg["visualization"]["minimap"]["height"]))
    out_w = width + mm_w
    out_path = str(out_dir / f"{stem}_annotated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, height))

    field_dims = (
        cfg["homography"]["field_width_m"],
        cfg["homography"]["field_length_m"],
    )

    frame_idx = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detection + tracking
        dets = detector.detect(frame)
        dets = tracker.update(dets, frame)

        # Homography (needed before re-ID for field-coordinate matching)
        field_pts = hom.project_detections(dets) if hom.is_calibrated() and len(dets) > 0 else None

        # Re-identification after rucks/scrums
        ruck_zones = reid.active_ruck_zones()
        dets = reid.update(frame, dets, frame_idx=frame_idx, field_pts=field_pts)

        # Team classification
        team_labels = clf.predict(frame, dets) if clf.is_fitted() and len(dets) > 0 else None

        # Metrics + analytics
        if (
            field_pts is not None
            and dets.tracker_id is not None
            and len(field_pts) > 0
        ):
            metrics.update(dets.tracker_id, field_pts, team_labels)
            analyzer.update(
                frame_idx, dets.tracker_id, field_pts, team_labels,
                ruck_zones=ruck_zones,
            )

        # Visualisation
        out_frame = vis.draw(
            frame, dets, team_labels, field_pts, field_dims,
            ruck_zones=ruck_zones,
        )
        writer.write(out_frame)

        if args.display:
            cv2.imshow("RugbyVision", cv2.resize(out_frame, (min(out_w, 1280), height * min(out_w, 1280) // out_w)))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1
        if frame_idx % 50 == 0:
            elapsed = time.time() - t0
            fps_real = frame_idx / elapsed
            pct = 100 * frame_idx / total_frames if total_frames else 0
            print(f"[{pct:5.1f}%] frame {frame_idx}  {fps_real:.1f} fps")

    cap.release()
    writer.release()
    if args.display:
        cv2.destroyAllWindows()

    # --- Post-processing metrics ---
    # --- Save frozen team-colour centres if requested ---
    if centers_path and clf.is_fitted():
        clf.save_centers(centers_path)
        print(f"[team] Cluster centres saved to {centers_path}")

    print("\n[metrics] Saving outputs …")
    metrics.plot_heatmap(str(out_dir / f"{stem}_heatmap.png"))
    metrics.plot_distances(str(out_dir / f"{stem}_distances.png"))

    # --- Match intelligence report ---
    report_path = str(out_dir / f"{stem}_match_report.json")
    analyzer.export(metrics, reid, report_path)
    print(f"[analytics] Match report saved to: {report_path}")

    summary = metrics.summary()
    print("\n=== Player Summary ===")
    for tid, s in sorted(summary.items()):
        print(
            f"  ID {tid:>3} | team {s['team']} | "
            f"{s['distance_m']:6.1f} m | "
            f"{s['speed_ms']:4.1f} m/s | "
            f"avg ({s['avg_position'][0]:.1f}, {s['avg_position'][1]:.1f}) m"
        )

    print(f"\n[done] Annotated video saved to: {out_path}")

    # --- Re-ID summary ---
    if reid.reid_log:
        print(f"\n[re-ID] {len(reid.reid_log)} re-identification(s) performed:")
        for match in reid.reid_log:
            print(
                f"  frame {match.frame_idx:>5} | ruck #{match.event_id} | "
                f"ID {match.new_tracker_id} → {match.old_tracker_id}  "
                f"(conf={match.confidence:.2f})"
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="RugbyVision — player tracking and tactical analysis"
    )
    p.add_argument("--source", required=True, help="Local video path or YouTube URL")
    p.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    p.add_argument(
        "--calibrate",
        action="store_true",
        help="Force interactive field calibration even if a saved matrix exists",
    )
    p.add_argument(
        "--display",
        action="store_true",
        help="Show live preview window (press q to quit)",
    )
    p.add_argument(
        "--recalibrate-teams",
        action="store_true",
        help="Force re-fitting of team colours even if saved centres exist",
    )
    return p


if __name__ == "__main__":
    parser = build_parser()
    run(parser.parse_args())
