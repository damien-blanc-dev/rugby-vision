"""RugbyVision benchmark — MOTA / HOTA / ID Switches on MOT-format datasets.

Supports:
- MOT Challenge format (gt.txt + img1/ or video.mp4 + seqinfo.ini)
- CEA Rugby Sevens dataset (https://kalisteo.cea.fr/…/README_R7.html)
- Roboflow MOT export

Usage
-----
# Single sequence
python scripts/benchmark.py --sequences path/to/seq --output benchmark/results.md

# Batch (all subdirs under a root)
python scripts/benchmark.py --sequences benchmark/sequences/ --output benchmark/results.md

# Disable Re-ID (baseline only)
python scripts/benchmark.py --sequences benchmark/sequences/ --baseline-only

Ground-truth format (standard MOT Challenge):
    <frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,<x>,<y>,<z>
Frame ids are 1-based (converted internally to 0-based).

Roboflow MOT export uses the same format.
CEA Rugby uses the same format with class-id in column 7 (ignored).
"""

from __future__ import annotations

import argparse
import configparser
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rugby_vision.detector import PlayerDetector
from rugby_vision.tracker import PlayerTracker
from rugby_vision.team_classifier import TeamClassifier
from rugby_vision.homography import FieldHomography
from rugby_vision.reidentifier import OcclusionReidentifier


# ---------------------------------------------------------------------------
# MOT-format data loading
# ---------------------------------------------------------------------------


@dataclass
class MOTSequence:
    """One annotated video sequence in MOT Challenge format."""

    name: str
    video_path: Optional[Path]      # video.mp4 if available
    img_dir: Optional[Path]         # img1/ folder fallback
    gt_path: Path
    fps: float = 25.0
    img_width: int = 1920
    img_height: int = 1080


def discover_sequences(root: Path) -> list[MOTSequence]:
    """Walk *root* and return all valid MOT sequences found.

    Accepted layouts:
    - ``root/gt/gt.txt``          (single sequence at root)
    - ``root/seq*/gt/gt.txt``     (batch)
    - ``root/seq*/img1/*.jpg`` or ``root/seq*/video.mp4``
    """
    seqs: list[MOTSequence] = []

    # Check if root itself is a sequence
    if (root / "gt" / "gt.txt").exists():
        seqs.append(_build_sequence(root))
        return seqs

    # Otherwise look for subdirs
    for candidate in sorted(root.iterdir()):
        if candidate.is_dir() and (candidate / "gt" / "gt.txt").exists():
            seqs.append(_build_sequence(candidate))

    return seqs


def _build_sequence(seq_dir: Path) -> MOTSequence:
    gt_path = seq_dir / "gt" / "gt.txt"
    video_path = seq_dir / "video.mp4"
    img_dir = seq_dir / "img1"

    fps, w, h = 25.0, 1920, 1080

    ini_path = seq_dir / "seqinfo.ini"
    if ini_path.exists():
        cfg = configparser.ConfigParser()
        cfg.read(ini_path)
        s = cfg["Sequence"] if "Sequence" in cfg else {}
        fps = float(s.get("frameRate", fps))
        w = int(s.get("imWidth", w))
        h = int(s.get("imHeight", h))

    return MOTSequence(
        name=seq_dir.name,
        video_path=video_path if video_path.exists() else None,
        img_dir=img_dir if img_dir.exists() else None,
        gt_path=gt_path,
        fps=fps,
        img_width=w,
        img_height=h,
    )


def load_gt(gt_path: Path) -> dict[int, list[dict]]:
    """Parse a MOT gt.txt and return ``{frame_0based: [row_dict, ...]}``."""
    gt: dict[int, list[dict]] = {}
    with open(gt_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue
            frame_1 = int(parts[0])
            obj_id = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            conf = float(parts[6]) if len(parts) > 6 else 1.0
            if conf == 0:
                continue  # ignore crowd/distractor annotations
            key = frame_1 - 1  # convert to 0-based
            gt.setdefault(key, []).append(
                {"id": obj_id, "xyxy": np.array([x, y, x + w, y + h])}
            )
    return gt


def frame_generator(seq: MOTSequence):
    """Yield ``(frame_idx, bgr_frame)`` for a sequence."""
    if seq.video_path:
        cap = cv2.VideoCapture(str(seq.video_path))
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield idx, frame
            idx += 1
        cap.release()
    elif seq.img_dir:
        imgs = sorted(seq.img_dir.glob("*.jpg")) + sorted(seq.img_dir.glob("*.png"))
        for idx, img_path in enumerate(imgs):
            frame = cv2.imread(str(img_path))
            if frame is not None:
                yield idx, frame
    else:
        raise FileNotFoundError(
            f"Sequence '{seq.name}' has neither video.mp4 nor img1/ folder."
        )


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------


def iou_matrix(gt_boxes: np.ndarray, pred_boxes: np.ndarray) -> np.ndarray:
    """Compute IoU between all pairs of GT and predicted boxes.

    Parameters
    ----------
    gt_boxes : np.ndarray  shape (M, 4) xyxy
    pred_boxes : np.ndarray  shape (N, 4) xyxy

    Returns
    -------
    np.ndarray  shape (M, N)
    """
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return np.zeros((len(gt_boxes), len(pred_boxes)))

    inter_x1 = np.maximum(gt_boxes[:, 0:1], pred_boxes[:, 0])
    inter_y1 = np.maximum(gt_boxes[:, 1:2], pred_boxes[:, 1])
    inter_x2 = np.minimum(gt_boxes[:, 2:3], pred_boxes[:, 2])
    inter_y2 = np.minimum(gt_boxes[:, 3:4], pred_boxes[:, 3])

    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    gt_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    union_area = gt_area[:, None] + pred_area[None, :] - inter_area

    return np.where(union_area > 0, inter_area / union_area, 0.0)


def match_frame(
    gt_rows: list[dict],
    pred_xyxy: np.ndarray,
    pred_ids: np.ndarray,
    iou_threshold: float = 0.5,
) -> tuple[dict[int, int], list[int], list[int]]:
    """Match GT objects to predictions using greedy IoU matching.

    Returns
    -------
    gt_to_pred : dict[int, int]
        Mapping gt_id → pred_tracker_id for matched pairs.
    unmatched_gt : list[int]
        GT ids with no prediction (→ FN).
    unmatched_pred : list[int]
        Pred tracker_ids with no GT match (→ FP).
    """
    if not gt_rows:
        return {}, [], list(map(int, pred_ids))
    if len(pred_xyxy) == 0:
        return {}, [r["id"] for r in gt_rows], []

    gt_boxes = np.stack([r["xyxy"] for r in gt_rows])
    gt_ids = [r["id"] for r in gt_rows]

    iou = iou_matrix(gt_boxes, pred_xyxy)
    # Greedy: highest IoU first
    gt_to_pred = {}
    used_pred = set()
    order = np.dstack(np.unravel_index(np.argsort(-iou.ravel()), iou.shape))[0]
    for gi, pi in order:
        if iou[gi, pi] < iou_threshold:
            break
        if gt_ids[gi] in gt_to_pred or pi in used_pred:
            continue
        gt_to_pred[gt_ids[gi]] = int(pred_ids[pi])
        used_pred.add(pi)

    unmatched_gt = [gid for gid in gt_ids if gid not in gt_to_pred]
    unmatched_pred = [int(pred_ids[pi]) for pi in range(len(pred_xyxy)) if pi not in used_pred]
    return gt_to_pred, unmatched_gt, unmatched_pred


@dataclass
class FrameAccumulator:
    """Accumulate per-frame tracking stats for MOTA / HOTA computation."""

    total_gt: int = 0
    total_fp: int = 0
    total_fn: int = 0
    total_idsw: int = 0
    # For HOTA association: list of (gt_id, pred_id, frame_idx)
    tp_events: list[tuple] = field(default_factory=list)
    # Per-frame previous mapping for IDSW detection
    _prev_gt_to_pred: dict[int, int] = field(default_factory=dict)

    def update(
        self,
        gt_to_pred: dict[int, int],
        unmatched_gt: list[int],
        unmatched_pred: list[int],
        frame_idx: int,
    ) -> None:
        fp = len(unmatched_pred)
        fn = len(unmatched_gt)
        gt_count = len(gt_to_pred) + fn
        idsw = sum(
            1
            for gid, pid in gt_to_pred.items()
            if gid in self._prev_gt_to_pred and self._prev_gt_to_pred[gid] != pid
        )
        self.total_gt += gt_count
        self.total_fp += fp
        self.total_fn += fn
        self.total_idsw += idsw
        for gid, pid in gt_to_pred.items():
            self.tp_events.append((gid, pid, frame_idx))
        self._prev_gt_to_pred = dict(gt_to_pred)

    def mota(self) -> float:
        if self.total_gt == 0:
            return 0.0
        return 1.0 - (self.total_fp + self.total_fn + self.total_idsw) / self.total_gt

    def hota(self) -> float:
        """Simplified HOTA = sqrt(DetA × AssA)."""
        tp = len(self.tp_events)
        det_a = tp / max(tp + self.total_fp + self.total_fn, 1)

        # AssA: for each (gt_id, pred_id) pair, fraction of their co-appearance
        from collections import defaultdict
        gt_frames: dict[int, set] = defaultdict(set)
        pred_frames: dict[int, set] = defaultdict(set)
        pair_frames: dict[tuple, set] = defaultdict(set)
        for gid, pid, fidx in self.tp_events:
            gt_frames[gid].add(fidx)
            pred_frames[pid].add(fidx)
            pair_frames[(gid, pid)].add(fidx)

        if not pair_frames:
            return 0.0

        ass_scores = []
        for (gid, pid), frames_both in pair_frames.items():
            union = gt_frames[gid] | pred_frames[pid]
            ass_scores.append(len(frames_both) / len(union))

        ass_a = float(np.mean(ass_scores)) if ass_scores else 0.0
        return float(np.sqrt(det_a * ass_a))

    def id_switches(self) -> int:
        return self.total_idsw


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


def run_pipeline(
    seq: MOTSequence,
    cfg: dict,
    use_reid: bool = True,
    iou_threshold: float = 0.5,
    post_occlusion_window: int = 30,
) -> tuple[FrameAccumulator, FrameAccumulator, list[dict]]:
    """Run RugbyVision (or baseline) on *seq* and accumulate metrics.

    Parameters
    ----------
    seq : MOTSequence
    cfg : dict
        Loaded config.yaml dict.
    use_reid : bool
        If ``True``, include the OcclusionReidentifier in the pipeline.
    iou_threshold : float
        IoU threshold for GT↔prediction matching.
    post_occlusion_window : int
        Number of frames after a ruck end to include in post-occlusion MOTA.

    Returns
    -------
    global_acc : FrameAccumulator
        Metrics over all frames.
    post_occ_acc : FrameAccumulator
        Metrics restricted to the post-occlusion window.
    reid_log : list[dict]
        Re-ID events with GT cross-reference.
    """
    gt = load_gt(seq.gt_path)

    detector = PlayerDetector.from_config(cfg["model"])
    tracker = PlayerTracker.from_config(cfg["tracking"], frame_rate=int(seq.fps))
    tracker.reset()

    reid: Optional[OcclusionReidentifier] = None
    if use_reid:
        reid = OcclusionReidentifier.from_config(cfg.get("reidentification", {}))

    global_acc = FrameAccumulator()
    post_occ_acc = FrameAccumulator()
    post_occ_active_until: int = -1  # frame through which post-occ window is open

    # Track GT↔pred mapping at ruck start for re-ID accuracy
    ruck_pre_mapping: dict[int, dict] = {}  # ruck_event_id → {gt_id: pred_id}
    reid_accuracy_events: list[dict] = []

    for frame_idx, frame in frame_generator(seq):
        # --- Detect & track ---
        import supervision as sv

        dets = detector.detect(frame)
        dets = tracker.update(dets, frame)

        # --- Re-ID ---
        if reid is not None:
            active_rucks_before = set(r.event_id for r in reid._active_rucks if r.is_active)
            dets = reid.update(frame, dets, frame_idx=frame_idx)
            active_rucks_after = set(r.event_id for r in reid._active_rucks if r.is_active)

            # Detect newly ended rucks → open post-occlusion window
            ended = active_rucks_before - active_rucks_after
            if ended:
                post_occ_active_until = frame_idx + post_occlusion_window

        # --- GT matching for this frame ---
        gt_rows = gt.get(frame_idx, [])
        if dets.tracker_id is not None and len(dets) > 0:
            pred_xyxy = dets.xyxy
            pred_ids = dets.tracker_id
        else:
            pred_xyxy = np.empty((0, 4))
            pred_ids = np.array([])

        gt_to_pred, unmatched_gt, unmatched_pred = match_frame(
            gt_rows, pred_xyxy, pred_ids, iou_threshold
        )
        global_acc.update(gt_to_pred, unmatched_gt, unmatched_pred, frame_idx)

        if frame_idx <= post_occ_active_until:
            post_occ_acc.update(gt_to_pred, unmatched_gt, unmatched_pred, frame_idx)

        # --- Re-ID accuracy tracking ---
        if reid is not None:
            # Save pre-ruck mappings for newly opened rucks
            for r in reid._active_rucks:
                if r.event_id not in ruck_pre_mapping and r.is_active:
                    ruck_pre_mapping[r.event_id] = dict(gt_to_pred)

            # Check re-ID log for new events
            for match in reid.reid_log:
                if any(e.get("match") == match for e in reid_accuracy_events):
                    continue
                pre_map = ruck_pre_mapping.get(match.event_id, {})
                # Find GT id that was associated with old_tracker_id before ruck
                gt_for_old = next(
                    (gid for gid, pid in pre_map.items() if pid == match.old_tracker_id),
                    None,
                )
                # Check current frame: did we now assign old_tracker_id to same GT id?
                current_pred_for_gt = gt_to_pred.get(gt_for_old)
                correct = current_pred_for_gt == match.old_tracker_id if gt_for_old else None
                reid_accuracy_events.append(
                    {
                        "match": match,
                        "gt_id": gt_for_old,
                        "correct": correct,
                        "frame": frame_idx,
                    }
                )

    return global_acc, post_occ_acc, reid_accuracy_events


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def compute_reid_accuracy(events: list[dict]) -> Optional[float]:
    """Return fraction of re-ID attempts that were correct (GT-verified)."""
    verifiable = [e for e in events if e["correct"] is not None]
    if not verifiable:
        return None
    return sum(1 for e in verifiable if e["correct"]) / len(verifiable)


def plot_metrics(
    baseline_accs: list[FrameAccumulator],
    ours_accs: list[FrameAccumulator],
    seq_names: list[str],
    out_dir: Path,
) -> None:
    """Save comparison bar chart and ID-switch distribution plot."""
    x = np.arange(len(seq_names))
    w = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # MOTA
    ax = axes[0]
    ax.bar(x - w / 2, [a.mota() for a in baseline_accs], w, label="Baseline", color="#4A90D9")
    ax.bar(x + w / 2, [a.mota() for a in ours_accs], w, label="RugbyVision", color="#FF6B35")
    ax.set_title("MOTA")
    ax.set_xticks(x)
    ax.set_xticklabels(seq_names, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.legend()

    # HOTA
    ax = axes[1]
    ax.bar(x - w / 2, [a.hota() for a in baseline_accs], w, label="Baseline", color="#4A90D9")
    ax.bar(x + w / 2, [a.hota() for a in ours_accs], w, label="RugbyVision", color="#FF6B35")
    ax.set_title("HOTA (simplified)")
    ax.set_xticks(x)
    ax.set_xticklabels(seq_names, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.legend()

    # ID Switches
    ax = axes[2]
    ax.bar(x - w / 2, [a.id_switches() for a in baseline_accs], w, label="Baseline", color="#4A90D9")
    ax.bar(x + w / 2, [a.id_switches() for a in ours_accs], w, label="RugbyVision", color="#FF6B35")
    ax.set_title("ID Switches")
    ax.set_xticks(x)
    ax.set_xticklabels(seq_names, rotation=20, ha="right")
    ax.legend()

    plt.tight_layout()
    fig.savefig(out_dir / "benchmark_comparison.png", dpi=150)
    plt.close(fig)


def plot_idsw_distribution(
    baseline_accs: list[FrameAccumulator],
    ours_accs: list[FrameAccumulator],
    out_dir: Path,
) -> None:
    """Histogram of ID switch counts across sequences."""
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.arange(0, max(max(a.id_switches() for a in baseline_accs + ours_accs) + 2, 5))
    ax.hist(
        [a.id_switches() for a in baseline_accs],
        bins=bins,
        alpha=0.6,
        label="Baseline",
        color="#4A90D9",
    )
    ax.hist(
        [a.id_switches() for a in ours_accs],
        bins=bins,
        alpha=0.6,
        label="RugbyVision",
        color="#FF6B35",
    )
    ax.set_xlabel("ID Switches per sequence")
    ax.set_ylabel("Count")
    ax.set_title("ID Switch Distribution")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "idsw_distribution.png", dpi=150)
    plt.close(fig)


def write_markdown_report(
    seq_names: list[str],
    baseline_accs: list[FrameAccumulator],
    ours_accs: list[FrameAccumulator],
    post_occ_baseline: list[FrameAccumulator],
    post_occ_ours: list[FrameAccumulator],
    reid_accuracies: list[Optional[float]],
    out_path: Path,
) -> None:
    """Write results.md with tables and image references."""
    lines = [
        "# RugbyVision Benchmark Results\n",
        f"_Generated on {__import__('datetime').date.today()}_\n",
        "\n## Global Metrics\n",
        "| Sequence | Baseline MOTA | RV MOTA | Baseline HOTA | RV HOTA | "
        "Baseline IDSW | RV IDSW |",
        "|----------|--------------|---------|--------------|---------|"
        "-------------|---------|",
    ]
    for i, name in enumerate(seq_names):
        b = baseline_accs[i]
        o = ours_accs[i]
        lines.append(
            f"| {name} "
            f"| {b.mota():.3f} | **{o.mota():.3f}** "
            f"| {b.hota():.3f} | **{o.hota():.3f}** "
            f"| {b.id_switches()} | **{o.id_switches()}** |"
        )

    # Aggregate
    def _safe_mean(lst):
        return float(np.mean(lst)) if lst else float("nan")

    lines += [
        f"| **Mean** "
        f"| {_safe_mean([a.mota() for a in baseline_accs]):.3f} "
        f"| **{_safe_mean([a.mota() for a in ours_accs]):.3f}** "
        f"| {_safe_mean([a.hota() for a in baseline_accs]):.3f} "
        f"| **{_safe_mean([a.hota() for a in ours_accs]):.3f}** "
        f"| {int(np.sum([a.id_switches() for a in baseline_accs]))} "
        f"| **{int(np.sum([a.id_switches() for a in ours_accs]))}** |",
        "",
        "\n## Post-Occlusion Metrics (frames t+1…t+30 after each ruck)\n",
        "| Sequence | Baseline Post-Occ MOTA | RV Post-Occ MOTA | Re-ID Accuracy |",
        "|----------|----------------------|-----------------|----------------|",
    ]
    for i, name in enumerate(seq_names):
        b = post_occ_baseline[i]
        o = post_occ_ours[i]
        acc = reid_accuracies[i]
        acc_str = f"{acc:.1%}" if acc is not None else "N/A"
        lines.append(
            f"| {name} "
            f"| {b.mota():.3f} "
            f"| **{o.mota():.3f}** "
            f"| {acc_str} |"
        )

    lines += [
        "",
        "\n## Plots\n",
        "![Comparison](benchmark_comparison.png)\n",
        "![ID Switch Distribution](idsw_distribution.png)\n",
        "\n## Notes\n",
        "- **MOTA** = 1 − (FP + FN + IDSW) / GT  (higher is better)\n",
        "- **HOTA** = √(DetA × AssA) — simplified implementation  (higher is better)\n",
        "- **Post-Occlusion MOTA** — evaluated only on frames immediately after "
        "each ruck detection; the key metric of OcclusionReidentifier.\n",
        "- **Re-ID Accuracy** — % of re-assignments that match the GT track ID "
        "(requires GT ground truth).\n",
        "- Baseline: YOLOv8 + ByteTrack, no OcclusionReidentifier.\n",
        "- RugbyVision: YOLOv8 + ByteTrack + OcclusionReidentifier.\n",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] Written to {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="RugbyVision benchmark — MOTA / HOTA / Re-ID accuracy"
    )
    p.add_argument(
        "--sequences",
        required=True,
        help="Path to a single MOT sequence directory or a folder of sequences",
    )
    p.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    p.add_argument(
        "--output",
        default="benchmark/results.md",
        help="Output markdown report path",
    )
    p.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for GT↔prediction matching",
    )
    p.add_argument(
        "--post-occlusion-window",
        type=int,
        default=30,
        help="Frames after ruck end used for post-occlusion MOTA",
    )
    p.add_argument(
        "--baseline-only",
        action="store_true",
        help="Only run the baseline (no OcclusionReidentifier)",
    )
    return p


def main(args: argparse.Namespace) -> None:
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seqs = discover_sequences(Path(args.sequences))
    if not seqs:
        sys.exit(f"No MOT sequences found under: {args.sequences}")

    print(f"[benchmark] Found {len(seqs)} sequence(s): {[s.name for s in seqs]}")

    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    seq_names, baseline_g, ours_g = [], [], []
    baseline_po, ours_po = [], []
    reid_accs = []

    for seq in seqs:
        print(f"\n=== {seq.name} ===")

        # --- Baseline ---
        print("  [baseline] Running …")
        t0 = time.time()
        b_global, b_po, _ = run_pipeline(
            seq, cfg,
            use_reid=False,
            iou_threshold=args.iou_threshold,
            post_occlusion_window=args.post_occlusion_window,
        )
        print(
            f"  [baseline] Done in {time.time()-t0:.1f}s  "
            f"MOTA={b_global.mota():.3f}  IDSW={b_global.id_switches()}"
        )

        # --- RugbyVision (with Re-ID) ---
        if not args.baseline_only:
            print("  [RugbyVision] Running …")
            t0 = time.time()
            o_global, o_po, reid_events = run_pipeline(
                seq, cfg,
                use_reid=True,
                iou_threshold=args.iou_threshold,
                post_occlusion_window=args.post_occlusion_window,
            )
            print(
                f"  [RugbyVision] Done in {time.time()-t0:.1f}s  "
                f"MOTA={o_global.mota():.3f}  IDSW={o_global.id_switches()}"
            )
            reid_acc = compute_reid_accuracy(reid_events)
            if reid_acc is not None:
                print(f"  [Re-ID accuracy] {reid_acc:.1%}")
        else:
            o_global, o_po = b_global, b_po
            reid_acc = None
            reid_events = []

        seq_names.append(seq.name)
        baseline_g.append(b_global)
        ours_g.append(o_global)
        baseline_po.append(b_po)
        ours_po.append(o_po)
        reid_accs.append(reid_acc)

    # --- Plots ---
    if len(seq_names) > 0:
        plot_metrics(baseline_g, ours_g, seq_names, out_dir)
        if len(seq_names) > 1:
            plot_idsw_distribution(baseline_g, ours_g, out_dir)

    # --- Markdown report ---
    write_markdown_report(
        seq_names,
        baseline_g, ours_g,
        baseline_po, ours_po,
        reid_accs,
        Path(args.output),
    )

    # --- Console summary ---
    print("\n=== Summary ===")
    for i, name in enumerate(seq_names):
        print(
            f"  {name:20s}  "
            f"Baseline MOTA={baseline_g[i].mota():.3f}  "
            f"RV MOTA={ours_g[i].mota():.3f}  "
            f"IDSW: {baseline_g[i].id_switches()} → {ours_g[i].id_switches()}"
        )


if __name__ == "__main__":
    main(build_parser().parse_args())
