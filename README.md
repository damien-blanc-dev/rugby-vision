# RugbyVision

> Research-oriented computer vision pipeline for robust rugby player tracking under dense occlusion, with field-aware analytics from monocular broadcast video.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/rugby-vision/blob/main/notebooks/demo.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)P

<!-- Replace with an actual GIF or short MP4 preview once a sample sequence is available -->
<!-- ![Demo GIF](assets/demo.gif) -->

## Overview

**RugbyVision** is a monocular computer vision system for tracking rugby players in broadcast match footage, recovering identities after dense occlusions such as rucks and scrums, projecting player positions onto field coordinates, and producing match-level tactical analytics.

The project is designed not only as an engineering pipeline, but also as an exploration of a harder research question:

> **How can identity persistence be maintained in sports video when multiple visually similar players disappear inside short-lived, structured occlusions?**

That question matters because standard multi-object trackers perform well when players remain visible, but frequently break identity consistency when several athletes merge into a compact cluster and reappear a few frames later.

## Why this problem is hard

Tracking rugby players from a single broadcast camera is challenging for several reasons:

- **Dense multi-player occlusions** are common during rucks and scrums.
- **Appearance cues are weak** because players on the same team wear similar jerseys.
- **Broadcast footage is not metric by default**, so tactical analysis requires geometric calibration.
- **Team-color clustering can drift** when one team temporarily leaves the frame.
- **Identity recovery must be measurable**, not just visually plausible.

RugbyVision focuses on these failure modes rather than treating player tracking as a generic bounding-box problem.

## Current capabilities

The current implementation already supports the following outputs:

| Output | Status | Description |
|--------|--------|-------------|
| Annotated video | Implemented | Bounding boxes, persistent IDs, team-colored overlays, and side-by-side visualization |
| Top-down minimap | Implemented | Real-time 2D field projection of player positions |
| Heatmaps | Implemented | Per-player or per-team spatial occupancy summaries |
| Distance and speed estimates | Implemented | Metrics derived from projected field coordinates |
| Match intelligence report | Implemented | JSON export with player summaries, ruck events, re-ID log, and proximity matrix |
| Baseline vs. re-ID benchmark | Implemented | Comparison between ByteTrack alone and ByteTrack + custom re-identification |

## Method

The pipeline is organized as follows:

```text
VideoCapture
    │
    ▼
PlayerDetector (YOLOv8)
    │
    ▼
PlayerTracker (ByteTrack via supervision)
    │
    ▼
FieldHomography
    │
    ├──► OcclusionReidentifier
    │       • detects dense ruck/scrum-like clusters
    │       • freezes player memory before occlusion
    │       • matches resurfacing players with Hungarian assignment
    │       • restores original pre-occlusion identities
    │
    ├──► TeamClassifier
    │       • clusters jersey colors
    │       • freezes cluster centers to prevent drift
    │
    ├──► MetricsCollector
    │       • distance, speed, average position, heatmaps
    │
    ├──► MatchAnalyzer
    │       • ruck statistics, proximity matrix, team aggregates
    │
    └──► Visualizer
            • annotated video + tactical minimap
```

## Re-identification strategy

The most research-oriented part of the project is the **post-occlusion re-identification module**.

When a dense player cluster is detected:

1. A **ruck event** is opened from connected components of nearby player centroids.
2. A **memory snapshot** is frozen for each player in the cluster.
3. When the cluster dissolves, newly surfaced tracker IDs are matched to lost players.
4. Matching is solved as a **bipartite assignment** using the Hungarian algorithm.
5. Confirmed matches remap new tracker IDs back to the original pre-ruck identities.

Each frozen memory can contain:

- Last image-space bounding box
- Last field position, when homography is available
- Mean jersey color in HSV
- Bounding-box aspect ratio
- Optional OCR jersey number

The current cost function combines appearance and geometry:

```text
cost(old_i, new_j)
  = color_weight    × normalized HSV distance
  + position_weight × normalized spatial distance
  + size_weight     × normalized bbox-shape difference
  - ocr_bonus       × jersey-number agreement
```

Only assignments above a configurable confidence threshold are committed.

## Team classification stability

A common failure mode in sports footage is **cluster drift**: if one team temporarily dominates the visible set of players, online K-means can shift its centers and silently swap team labels.

To reduce this effect, RugbyVision:

- fits team-color clusters during a short warm-up phase,
- optionally reloads previously saved centers,
- freezes the cluster centers after a configurable number of prediction calls,
- switches to nearest-centroid assignment once stable.

This makes team labels more consistent across long clips and unbalanced camera views.

## Geometric reasoning

Broadcast video is not directly expressed in metres, so player analytics depend on a field calibration step.

RugbyVision uses a planar **homography** estimated from four manually clicked field corners. Once calibrated, the bottom-center point of each player bounding box is projected from image coordinates to field coordinates. This enables:

- distance covered in metres,
- average speed in m/s,
- tactical occupancy heatmaps,
- pairwise proximity analysis,
- team-level spatial summaries.

## Evaluation

The repository already includes a benchmark script for MOT-style evaluation.

It currently supports:

- MOT-format ground truth sequences,
- Roboflow MOT exports,
- CEA Rugby Sevens–style data,
- baseline comparison against ByteTrack without re-identification.

The implemented metrics include:

| Metric | Status | Purpose |
|--------|--------|---------|
| MOTA | Implemented | Global tracking quality |
| HOTA (simplified) | Implemented | Combined detection/association quality |
| ID Switches | Implemented | Identity consistency |
| Post-occlusion MOTA | Implemented | Recovery quality immediately after ruck events |
| Re-ID accuracy | Implemented | Correctness of identity restoration when GT is available |

The benchmark is intentionally focused on the most important claim of the project: **identity recovery after structured occlusion**.

## What is already implemented

### Core pipeline

- YOLO-based player detection
- ByteTrack-based multi-object tracking
- Interactive field calibration
- Homography projection to field coordinates
- Team classification from jersey colors
- Video rendering with minimap
- Distance, speed, and heatmap analytics
- Match report export to JSON
- Post-occlusion re-identification with Hungarian matching
- Benchmarking against a no-reID baseline

### Engineering choices already visible in the codebase

- Modular project structure with separate components for detection, tracking, homography, analytics, visualization, and re-identification
- Config-driven parameters rather than hardcoded thresholds
- CLI entry points for calibration, full tracking runs, and benchmark experiments
- Support for local videos and YouTube inputs
- Persisted team-cluster centers across runs
- Logging of re-identification events for offline analysis

## What is partially done

These features are present conceptually or structurally, but still look like good candidates for further strengthening:

- **Fine-tuned rugby detector support**: the README already points to using a fine-tuned YOLO model, but the project would benefit from publishing actual trained weights, training protocol, and quantitative gain over the generic detector.
- **OCR-based jersey reading**: optional support exists in the re-identification module, but it should be treated as experimental until benchmarked on real rugby footage.
- **Programmatic API exposure**: basic API usage is possible, but a more polished library-style interface and versioned examples would make reuse easier.
- **Testing**: tests are mentioned, but the project would benefit from broader unit and regression coverage across tracking, re-ID, and analytics modules.

## What is not done yet

The following directions are not yet fully implemented and represent the most promising next steps:

### Research extensions

- **Ablation study of the re-identification cost terms**  
  Measure the isolated impact of color, position, box shape, and OCR on post-occlusion recovery.

- **Learned player embeddings for re-identification**  
  Replace hand-crafted appearance cues with a trainable embedding model and compare against the current HSV-based memory.

- **Uncertainty-aware tracking and analytics**  
  Attach confidence estimates to identity recovery, field projection, and downstream tactical metrics.

- **Event understanding beyond rucks**  
  Extend the pipeline toward structured recognition of mauls, kick chases, defensive lines, or transition phases.

- **Homography sensitivity analysis**  
  Quantify how calibration noise propagates into distance, speed, and heatmap errors.

### Product and reproducibility improvements

- Publish a reproducible benchmark subset with annotations.
- Add visual failure-case galleries to the README.
- Include result tables from actual experiments, not only the evaluation script.
- Add a short technical report or mini-paper describing methodology and limitations.
- Provide demo assets directly in the repository.

## Quick start

### 1. Install

```bash
git clone https://github.com/YOUR_USERNAME/rugby-vision.git
cd rugby-vision
pip install -r requirements.txt
```

### 2. Calibrate the field

```bash
python scripts/calibrate_field.py --source match.mp4
```

An OpenCV window opens on the first frame. Click the four field corners in order:

**Top-left → top-right → bottom-right → bottom-left**

Then press **Enter**. The homography matrix is saved to `calibration.npz`.

### 3. Run the full pipeline

```bash
# Local file
python scripts/run_tracking.py --source match.mp4

# YouTube URL (requires yt-dlp)
python scripts/run_tracking.py --source "https://youtu.be/..."

# Force interactive re-calibration and enable preview
python scripts/run_tracking.py --source match.mp4 --calibrate --display
```

By default, outputs are saved to the configured output directory and include:

- `*_annotated.mp4` — annotated match video with minimap
- `*_heatmap.png` — spatial heatmap
- `*_distances.png` — per-player distance chart
- `*_match_report.json` — structured analytics report

## Benchmarking

```bash
# Full benchmark
python scripts/benchmark.py \
    --sequences benchmark/sequences/ \
    --output benchmark/results.md

# Baseline only
python scripts/benchmark.py \
    --sequences benchmark/sequences/ \
    --baseline-only
```

Expected benchmark directory layout:

```text
benchmark/sequences/
└── seq01/
    ├── gt/gt.txt
    ├── img1/*.jpg
    ├── video.mp4
    └── seqinfo.ini
```

## Example outputs

After a run, RugbyVision can produce a match report with:

- per-player distance and average speed,
- average field position,
- time spent near active ruck zones,
- team-level aggregates,
- pairwise player proximity matrix,
- list of detected ruck events,
- list of re-identification events.

## Limitations

This project is intentionally ambitious, and several limitations remain:

- The current re-identification approach is still largely **heuristic**, not learned.
- Homography assumes a roughly planar field and a usable manual calibration.
- Broadcast camera cuts, zoom changes, and strong motion can still degrade tracking.
- Team-color clustering remains sensitive when jersey colors are visually close.
- The HOTA computation is currently a simplified implementation.
- Benchmark claims depend on the quality and representativeness of available MOT annotations.

These limitations are not hidden because they define the next research and engineering steps.

## Research roadmap

The most valuable next experiments would be:

1. **Ablation benchmarking** of the re-ID module.
2. **Comparison against learned ReID embeddings**.
3. **Quantitative calibration-error study** for field analytics.
4. **Event recognition from trajectories** rather than only player tracking.
5. **Public benchmark report** with actual tables, plots, and failure cases.

## Project structure

```text
rugby-vision/
├── config.yaml
├── requirements.txt
├── rugby_vision/
│   ├── detector.py
│   ├── tracker.py
│   ├── team_classifier.py
│   ├── homography.py
│   ├── visualizer.py
│   ├── metrics.py
│   ├── reidentifier.py
│   └── analytics.py
├── scripts/
│   ├── calibrate_field.py
│   ├── run_tracking.py
│   ├── benchmark.py
│   └── frames_to_video.py
├── notebooks/
│   └── demo.ipynb
└── tests/
    └── test_homography.py
```

## Positioning

RugbyVision can be read at three levels:

- as an **engineering project** for end-to-end sports video analytics,
- as a **computer vision project** about tracking under structured occlusion,
- as a **research prototype** for studying identity persistence in monocular multi-player scenes.

That combination is the main point of the repository.

## License

MIT — see [LICENSE](LICENSE).
