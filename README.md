# RugbyVision

> Computer vision pipeline for rugby player tracking and tactical analysis from match video.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/rugby-vision/blob/main/notebooks/demo.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

<!-- Replace with an actual GIF once you have a sample output -->
<!-- ![Demo GIF](assets/demo.gif) -->

## What it does

| Output | Description |
|--------|-------------|
| **Annotated video** | Bounding boxes + persistent tracker IDs, coloured by team |
| **2-D tactical minimap** | Real-time top-down field view with player positions |
| **Heatmaps** | Per-team or per-player cumulative position density |
| **Distance & speed** | Total metres covered and average speed per player |

## Architecture

```
VideoCapture
    │
    ▼
PlayerDetector (YOLOv8)
    │
    ▼
PlayerTracker (ByteTrack via supervision)
    │
    ▼
OcclusionReidentifier  ←── ruck/scrum detected?
    │   • freezes PlayerMemory (position, HSV, bbox, OCR jersey#)
    │   • Hungarian matching on dispersion
    │   • remaps ByteTrack IDs → original pre-ruck IDs
    │
    ├──► TeamClassifier (K-means on HSV jersey colours)
    │
    ├──► FieldHomography (perspective → 2-D field coords)
    │
    ├──► MetricsCollector (heatmap, distance, speed)
    │
    └──► Visualizer (annotated feed + minimap side-by-side)
```

## Quick start

### 1. Install

```bash
git clone https://github.com/YOUR_USERNAME/rugby-vision.git
cd rugby-vision
pip install -r requirements.txt
```

### 2. Calibrate the field (first time only)

```bash
python scripts/calibrate_field.py --source match.mp4
```

An OpenCV window opens on the first frame. Click the 4 corners of the pitch in order:
**Top-Left → Top-Right → Bottom-Right → Bottom-Left**, then press **Enter**.
The matrix is saved to `calibration.npz`.

### 3. Run tracking

```bash
# Local file
python scripts/run_tracking.py --source match.mp4

# YouTube URL (requires yt-dlp)
python scripts/run_tracking.py --source "https://youtu.be/..."

# Force re-calibration + live preview
python scripts/run_tracking.py --source match.mp4 --calibrate --display
```

Outputs go to `outputs/` by default (configurable in `config.yaml`):
- `*_annotated.mp4` — side-by-side annotated video
- `*_heatmap.png` — position heatmap
- `*_distances.png` — distance per player bar chart

## Configuration

All tunable parameters live in [`config.yaml`](config.yaml) — no hardcoded values.

```yaml
model:
  weights: yolov8n.pt   # swap for a fine-tuned rugby model
  confidence: 0.35

team_classifier:
  n_clusters: 2         # set to 3 to separate referees

homography:
  field_length_m: 100
  field_width_m: 68
```

## Using a fine-tuned rugby model

The [rugby-league-player-tracking](https://universe.roboflow.com/max-hornigold-muhgz/rugby-league-player-tracking)
dataset on Roboflow Universe provides labelled match footage.

```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.train(data="path/to/dataset.yaml", epochs=50, imgsz=640)
```

Then point `config.yaml → model.weights` at your trained `best.pt`.

## Programmatic API

```python
import yaml, cv2
from rugby_vision import PlayerDetector, PlayerTracker, TeamClassifier
from rugby_vision import FieldHomography, Visualizer, MetricsCollector

cfg = yaml.safe_load(open("config.yaml"))

detector = PlayerDetector.from_config(cfg["model"])
tracker  = PlayerTracker.from_config(cfg["tracking"])
clf      = TeamClassifier.from_config(cfg["team_classifier"])
hom      = FieldHomography(calibration_path="calibration.npz")
vis      = Visualizer.from_config(cfg)
metrics  = MetricsCollector.from_config(cfg, fps=25.0)

cap = cv2.VideoCapture("match.mp4")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    dets        = detector.detect(frame)
    dets        = tracker.update(dets, frame)
    team_labels = clf.predict(frame, dets)
    field_pts   = hom.project_detections(dets)
    metrics.update(dets.tracker_id, field_pts, team_labels)
    out = vis.draw(frame, dets, team_labels, field_pts)
    cv2.imshow("RugbyVision", out)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

metrics.plot_heatmap("heatmap.png")
summary = metrics.summary()
```

## Running tests

```bash
pip install pytest
pytest tests/
```

## Post-occlusion re-identification

Rucks and scrums cause dense, multi-player occlusions that exceed ByteTrack's
built-in buffer, leading to ID switches when players resurface.
`OcclusionReidentifier` addresses this with a three-step approach:

1. **Ruck detection** — connected-component analysis on player centroids; a
   cluster of ≥ N players within `ruck_radius_px` triggers a ruck event.
2. **Player memory** — position, mean HSV jersey colour, bbox aspect ratio, and
   (optionally) jersey number via EasyOCR are frozen for each player in the zone.
3. **Hungarian matching** — when the cluster disperses, new ByteTrack IDs are
   matched against the frozen memories via a weighted cost matrix
   (`color_weight`, `position_weight`, `size_weight`) and
   `scipy.optimize.linear_sum_assignment`.  Matches above
   `reid_confidence_threshold` restore the original IDs.

All re-ID events are logged in `OcclusionReidentifier.reid_log` for benchmarking.

Enable OCR jersey-number reading (slower, requires `easyocr`):
```yaml
# config.yaml
reidentification:
  use_ocr: true
```

## Benchmarking

`scripts/benchmark.py` evaluates the pipeline on MOT-format sequences and
produces a comparison between the baseline (ByteTrack only) and RugbyVision
(ByteTrack + OcclusionReidentifier).

Compatible datasets:
- [CEA Rugby Sevens](https://kalisteo.cea.fr/wp-content/uploads/2022/04/README_R7.html)
- [Roboflow MOT export](https://universe.roboflow.com/max-hornigold-muhgz/rugby-league-player-tracking)
- Any MOT Challenge–format dataset

```
benchmark/sequences/
└── seq01/
    ├── gt/gt.txt        # ground truth (MOT format)
    ├── img1/*.jpg       # frames  (or video.mp4)
    └── seqinfo.ini      # optional metadata (fps, resolution)
```

```bash
# Run full benchmark
python scripts/benchmark.py \
    --sequences benchmark/sequences/ \
    --output benchmark/results.md

# Baseline only (skip Re-ID)
python scripts/benchmark.py --sequences benchmark/sequences/ --baseline-only
```

Metrics computed:
| Metric | Description |
|--------|-------------|
| **MOTA** | 1 − (FP + FN + IDSW) / GT |
| **HOTA** | √(DetA × AssA) — simplified |
| **ID Switches** | Total identity swaps |
| **Post-occlusion MOTA** | MOTA on frames t+1…t+30 after each ruck (key metric) |
| **Re-ID Accuracy** | % of re-assignments matching GT track ID |

## Match intelligence report

After each run, `outputs/<stem>_match_report.json` is produced:

```jsonc
{
  "meta": { "total_frames": 3750, "duration_s": 150.0, "total_rucks": 8 },
  "players": {
    "3": {
      "team": 0, "distance_m": 412.7, "avg_speed_ms": 4.1,
      "ruck_time_s": 18.4, "ruck_density": 0.123,
      "avg_position": { "x_m": 34.2, "y_m": 61.8 }
    }
  },
  "team_aggregates": {
    "0": { "total_distance_m": 3840.2, "total_ruck_time_s": 96.0 }
  },
  "proximity_matrix": {
    "players": [1, 3, 5, 7, ...],
    "matrix": [[0, 8.4, 14.1, ...], ...]   // avg metres between each pair
  },
  "ruck_events": [...],
  "reid_log": [...]
}
```

## Team colour stability

A common failure mode is **cluster drift**: if one team leaves frame during a
kick-chase, K-means re-assigns on the unbalanced sample and swaps team colours
for the rest of the clip.

RugbyVision solves this by **freezing cluster centres** after
`stable_after_frames` (default: 100) prediction calls.  After that, assignment
is done by simple nearest-centroid (O(1) per player) with fixed anchors.

Centres can be persisted across sessions:
```yaml
# config.yaml
team_classifier:
  stable_after_frames: 100
  centers_save_path: centers.json  # saved at end of run, loaded on next run
```

```bash
# Second run loads saved centres — no warm-up needed
python scripts/run_tracking.py --source game2.mp4

# Force re-calibration even if centers.json exists
python scripts/run_tracking.py --source game2.mp4 --recalibrate-teams
```

## Architecture décisionnelle

### Pourquoi le Matching Hongrois pour la ré-identification ?

Après un ruck, $N$ joueurs ont perdu leur ID ByteTrack et $M$ nouveaux IDs
sont apparus.  Le problème est une **affectation bipartite** : chaque joueur
perdu doit être associé au plus à un nouveau détection, et vice-versa.

| Approche | Complexité | Optimalité |
|----------|-----------|------------|
| Greedy (plus proche voisin) | O(N²) | Non — l'ordre d'affectation biaised |
| Recherche exhaustive | O(N!) | Optimale — impraticable au-delà de N=8 |
| **Algorithme Hongrois** (Kuhn-Munkres) | **O(N³)** | **Optimale garantie** |

Pour un ruck de 8 joueurs perdus × 8 candidats, l'algo hongrois s'exécute en
< 1 ms sur CPU.  La **matrice de coût** combine trois signaux normalisés :

```
cost[i,j] = 0.50 × ΔcouleurHSV / 441.7
           + 0.35 × distance_px / diagonale_image
           + 0.15 × |ratio_bbox_i − ratio_bbox_j| / max_ratio
           − 0.40 × [1 si numéros_OCR_concordent]
```

Seules les paires avec `confidence = 1 − cost/cost_max ≥ 0.6` sont validées.

### Comment l'homographie transforme les pixels en mètres réels ?

Une caméra fixe projette le plan du terrain sur le plan image selon une
**transformation homographique** — une bijection projective 2D→2D représentée
par une matrice 3×3 **H**.

**Calibration** : l'utilisateur clique 4 points du terrain dont les
coordonnées réelles sont connues (coins de la pelouse en mètres).
OpenCV résout le système linéaire 4×4 (8 équations, 8 inconnues) par
décomposition SVD pour trouver **H**.

**Projection** d'un pixel (u, v) vers un point terrain (x_m, y_m) :

```
[x']   [H₀₀  H₀₁  H₀₂] [u]
[y'] = [H₁₀  H₁₁  H₁₂] [v]
[w']   [H₂₀  H₂₁  H₂₂] [1]

x_m = x'/w',  y_m = y'/w'
```

Le pied du joueur (centre bas de sa bbox) est utilisé comme point de contact
au sol — c'est la projection la plus précise pour un être debout.

**Précision typique** : avec 4 points bien choisis sur un terrain de 100×68 m,
l'erreur de projection est < 0.5 m pour les zones centrales et < 1.5 m en
bord de cadre (distorsion de parallaxe).

## Project structure

```
rugby-vision/
├── config.yaml                  # all tunable parameters
├── requirements.txt
├── rugby_vision/
│   ├── detector.py              # YOLOv8 wrapper
│   ├── tracker.py               # ByteTrack via supervision
│   ├── team_classifier.py       # K-means HSV + frozen centres (anti-drift)
│   ├── homography.py            # field calibration + 2-D projection
│   ├── visualizer.py            # annotated video + minimap + trails + ruck halos
│   ├── metrics.py               # heatmap, distances, statistics
│   ├── reidentifier.py          # post-occlusion re-ID (ruck/scrum)
│   └── analytics.py             # match intelligence → match_report.json
├── scripts/
│   ├── run_tracking.py          # CLI entry point
│   ├── calibrate_field.py       # interactive calibration tool
│   └── benchmark.py             # MOTA/HOTA evaluation on MOT datasets
├── notebooks/
│   └── demo.ipynb               # Colab-friendly walkthrough
└── tests/
    └── test_homography.py
```

## License

MIT — see [LICENSE](LICENSE).
