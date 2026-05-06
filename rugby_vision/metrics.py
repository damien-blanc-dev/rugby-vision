"""Heatmaps, distances, and positional statistics for tracked players."""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend (safe for scripts & Colab)
import matplotlib.pyplot as plt
import numpy as np


class MetricsCollector:
    """Accumulate per-player positional data and compute metrics.

    Parameters
    ----------
    field_width_m : float
        Real field width in metres.
    field_length_m : float
        Real field length in metres.
    heatmap_bins : int
        Number of grid cells per axis for heatmap computation.
    speed_smoothing_frames : int
        Rolling window size for speed estimation.
    fps : float
        Video frame rate used to convert frame-distance to speed (m/s).

    Examples
    --------
    >>> mc = MetricsCollector(field_width_m=68, field_length_m=100, fps=25)
    >>> mc.update(tracker_ids, field_pts, team_labels)
    >>> mc.plot_heatmap(output_path="heatmap.png")
    >>> summary = mc.summary()
    """

    def __init__(
        self,
        field_width_m: float = 68.0,
        field_length_m: float = 100.0,
        heatmap_bins: int = 50,
        speed_smoothing_frames: int = 5,
        fps: float = 25.0,
    ) -> None:
        self.field_width_m = field_width_m
        self.field_length_m = field_length_m
        self.heatmap_bins = heatmap_bins
        self.speed_smoothing_frames = speed_smoothing_frames
        self.fps = fps

        # tracker_id → list of (x_m, y_m)
        self._positions: dict[int, list[tuple[float, float]]] = defaultdict(list)
        # tracker_id → team label
        self._team: dict[int, int] = {}

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def update(
        self,
        tracker_ids: np.ndarray,
        field_pts: np.ndarray,
        team_labels: Optional[np.ndarray] = None,
    ) -> None:
        """Record one frame's worth of positions.

        Parameters
        ----------
        tracker_ids : np.ndarray
            Shape ``(N,)`` — ByteTrack integer IDs.
        field_pts : np.ndarray
            Shape ``(N, 2)`` — field coordinates (x_m, y_m).
        team_labels : np.ndarray or None
            Shape ``(N,)`` — integer team index per player.
        """
        for i, tid in enumerate(tracker_ids):
            tid = int(tid)
            x, y = float(field_pts[i, 0]), float(field_pts[i, 1])
            self._positions[tid].append((x, y))
            if team_labels is not None:
                self._team[tid] = int(team_labels[i])

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def distance_per_player(self) -> dict[int, float]:
        """Total distance travelled (metres) per tracker ID.

        Returns
        -------
        dict[int, float]
            Mapping ``tracker_id → distance_m``.
        """
        result = {}
        for tid, pts in self._positions.items():
            if len(pts) < 2:
                result[tid] = 0.0
                continue
            arr = np.array(pts)
            diffs = np.diff(arr, axis=0)
            result[tid] = float(np.sum(np.linalg.norm(diffs, axis=1)))
        return result

    def average_position_per_player(self) -> dict[int, tuple[float, float]]:
        """Mean field position (x_m, y_m) per tracker ID.

        Returns
        -------
        dict[int, tuple[float, float]]
        """
        return {
            tid: tuple(np.mean(pts, axis=0).tolist())
            for tid, pts in self._positions.items()
        }

    def speed_per_player(self) -> dict[int, float]:
        """Estimated average speed (m/s) per tracker ID.

        Uses the last ``speed_smoothing_frames`` displacement steps.

        Returns
        -------
        dict[int, float]
        """
        result = {}
        w = self.speed_smoothing_frames
        for tid, pts in self._positions.items():
            if len(pts) < 2:
                result[tid] = 0.0
                continue
            arr = np.array(pts[-w - 1:])
            diffs = np.diff(arr, axis=0)
            distances = np.linalg.norm(diffs, axis=1)
            result[tid] = float(distances.mean() * self.fps)
        return result

    def summary(self) -> dict:
        """Return a dict with distance, average position, and speed per player.

        Returns
        -------
        dict
            Keys are tracker IDs; values contain ``distance_m``,
            ``avg_position``, ``speed_ms``, and ``team``.
        """
        dist = self.distance_per_player()
        avg_pos = self.average_position_per_player()
        speed = self.speed_per_player()
        return {
            tid: {
                "distance_m": dist[tid],
                "avg_position": avg_pos[tid],
                "speed_ms": speed[tid],
                "team": self._team.get(tid, -1),
            }
            for tid in self._positions
        }

    # ------------------------------------------------------------------
    # Visualisations
    # ------------------------------------------------------------------

    def plot_heatmap(
        self,
        output_path: str = "heatmap.png",
        team_id: Optional[int] = None,
        title: str = "Player Heatmap",
    ) -> None:
        """Save a heatmap PNG of cumulative player positions.

        Parameters
        ----------
        output_path : str
            Destination file path.
        team_id : int or None
            If given, only include players from this team.
        title : str
            Plot title.
        """
        all_pts = []
        for tid, pts in self._positions.items():
            if team_id is not None and self._team.get(tid) != team_id:
                continue
            all_pts.extend(pts)

        if not all_pts:
            return

        xs, ys = zip(*all_pts)
        xs = np.array(xs)
        ys = np.array(ys)

        fig, ax = plt.subplots(figsize=(8, 5))
        h, xedges, yedges = np.histogram2d(
            xs, ys,
            bins=self.heatmap_bins,
            range=[[0, self.field_width_m], [0, self.field_length_m]],
        )
        im = ax.imshow(
            h.T,
            origin="lower",
            extent=[0, self.field_width_m, 0, self.field_length_m],
            aspect="auto",
            cmap="hot",
            interpolation="gaussian",
        )
        plt.colorbar(im, ax=ax, label="Dwell count")
        ax.set_xlabel("Width (m)")
        ax.set_ylabel("Length (m)")
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close(fig)

    def plot_distances(self, output_path: str = "distances.png") -> None:
        """Save a bar chart of total distance per player.

        Parameters
        ----------
        output_path : str
            Destination file path.
        """
        dist = self.distance_per_player()
        if not dist:
            return

        ids = sorted(dist)
        values = [dist[i] for i in ids]
        colours = [
            ("#FF6B35" if self._team.get(i, 0) == 0 else "#4A90D9")
            for i in ids
        ]

        fig, ax = plt.subplots(figsize=(max(6, len(ids) * 0.4), 4))
        ax.bar([str(i) for i in ids], values, color=colours)
        ax.set_xlabel("Tracker ID")
        ax.set_ylabel("Distance (m)")
        ax.set_title("Distance per Player")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close(fig)

    @classmethod
    def from_config(cls, cfg: dict, fps: float = 25.0) -> "MetricsCollector":
        """Instantiate from a config dict.

        Parameters
        ----------
        cfg : dict
            Top-level config dict (``config.yaml``).
        fps : float
            Video frame rate.

        Returns
        -------
        MetricsCollector
        """
        hom_cfg = cfg.get("homography", {})
        met_cfg = cfg.get("metrics", {})
        return cls(
            field_width_m=hom_cfg.get("field_width_m", 68.0),
            field_length_m=hom_cfg.get("field_length_m", 100.0),
            heatmap_bins=met_cfg.get("heatmap_bins", 50),
            speed_smoothing_frames=met_cfg.get("speed_smoothing_frames", 5),
            fps=fps,
        )
