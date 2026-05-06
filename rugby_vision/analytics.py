"""Match intelligence — compile tracking data into a structured JSON report.

Aggregates data from :class:`MetricsCollector` and
:class:`OcclusionReidentifier` to produce a ``match_report.json`` with:

- Per-player distance, speed, and positional profile
- Time spent in ruck zones (temporal density)
- NxN proximity matrix (average distance between each pair of players)

The :class:`MatchAnalyzer` must receive one ``update()`` call per video frame
alongside the regular metrics pipeline.  The heavy computation (proximity
matrix, ruck density) happens lazily inside :meth:`export`.

Examples
--------
>>> analyzer = MatchAnalyzer(fps=25.0)
>>> # inside frame loop:
>>> analyzer.update(frame_idx, dets.tracker_id, field_pts, team_labels,
...                 ruck_zones=reid.active_ruck_zones())
>>> # at end of video:
>>> analyzer.export(metrics, reid, "outputs/match_report.json")
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from rugby_vision.metrics import MetricsCollector
    from rugby_vision.reidentifier import OcclusionReidentifier


class MatchAnalyzer:
    """Per-frame data collector that synthesises a full match performance report.

    Parameters
    ----------
    fps : float
        Video frame rate — used to convert frame counts to seconds.
    field_width_m : float
        Real field width in metres (for proximity normalisation).
    field_length_m : float
        Real field length in metres.
    ruck_proximity_m : float
        Radius in **field metres** within which a player is considered
        "in a ruck zone".  Used when pixel-space zone info is unavailable.

    Notes
    -----
    Ruck zone membership is determined each frame by comparing each player's
    field position against the list of active ruck zone centroids (if provided
    in pixel space via *ruck_zones*) or by a simpler field-space proximity
    heuristic when homography coords are available.
    """

    def __init__(
        self,
        fps: float = 25.0,
        field_width_m: float = 68.0,
        field_length_m: float = 100.0,
        ruck_proximity_m: float = 5.0,
    ) -> None:
        self.fps = fps
        self.field_width_m = field_width_m
        self.field_length_m = field_length_m
        self.ruck_proximity_m = ruck_proximity_m

        # {tracker_id: {frame_idx: (x_m, y_m)}}
        self._positions: dict[int, dict[int, tuple[float, float]]] = defaultdict(dict)
        # {tracker_id: team_label}
        self._team: dict[int, int] = {}
        # {tracker_id: total_frames_visible}
        self._frames_visible: dict[int, int] = defaultdict(int)
        # {tracker_id: frames_in_ruck}
        self._ruck_frames: dict[int, int] = defaultdict(int)
        # total frames processed
        self._total_frames: int = 0

    # ------------------------------------------------------------------
    # Per-frame ingestion
    # ------------------------------------------------------------------

    def update(
        self,
        frame_idx: int,
        tracker_ids: np.ndarray,
        field_pts: np.ndarray,
        team_labels: Optional[np.ndarray] = None,
        ruck_zones: Optional[list[tuple[float, float, float]]] = None,
    ) -> None:
        """Record positional data for a single frame.

        Parameters
        ----------
        frame_idx : int
            Monotonically increasing frame counter (0-based).
        tracker_ids : np.ndarray
            Shape ``(N,)`` — ByteTrack persistent IDs.
        field_pts : np.ndarray
            Shape ``(N, 2)`` — field coordinates (x_m, y_m).
        team_labels : np.ndarray or None
            Shape ``(N,)`` — integer team index per detection.
        ruck_zones : list of (cx_px, cy_px, radius_px) or None
            Active ruck zone descriptors from
            :meth:`OcclusionReidentifier.active_ruck_zones`.  When provided,
            ruck membership is tested in **field-metres space** using
            ``ruck_proximity_m`` if field_pts are available.  Pixel-based
            coordinates are used as a fallback signal only (not converted
            here — conversion requires the homography matrix).
        """
        self._total_frames = max(self._total_frames, frame_idx + 1)

        for i, tid in enumerate(tracker_ids):
            tid = int(tid)
            x_m, y_m = float(field_pts[i, 0]), float(field_pts[i, 1])
            self._positions[tid][frame_idx] = (x_m, y_m)
            self._frames_visible[tid] += 1
            if team_labels is not None:
                self._team[tid] = int(team_labels[i])

        # Ruck membership — use field-space proximity when field_pts available
        if ruck_zones:
            # Compute centroids of ruck zones in field space from the mean of
            # nearby players (best approximation without the homography here)
            in_ruck_ids = self._detect_ruck_members_field(
                tracker_ids, field_pts, ruck_zones
            )
            for tid in in_ruck_ids:
                self._ruck_frames[int(tid)] += 1

    # ------------------------------------------------------------------
    # Report export
    # ------------------------------------------------------------------

    def export(
        self,
        metrics: "MetricsCollector",
        reid: "OcclusionReidentifier",
        output_path: str = "outputs/match_report.json",
        indent: int = 2,
    ) -> dict:
        """Compile and write the full match report to *output_path*.

        Parameters
        ----------
        metrics : MetricsCollector
            The metrics object used throughout the pipeline (provides distance
            and speed data).
        reid : OcclusionReidentifier
            The re-identifier (provides ruck event log and re-ID events).
        output_path : str
            Destination JSON file path.
        indent : int
            JSON indentation for readability.

        Returns
        -------
        dict
            The report dictionary (also written to disk).
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        dist = metrics.distance_per_player()
        speed = metrics.speed_per_player()
        avg_pos = metrics.average_position_per_player()

        # ---- Per-player section ----
        players = {}
        for tid in self._positions:
            ruck_f = self._ruck_frames.get(tid, 0)
            visible_f = self._frames_visible.get(tid, 1)
            players[str(tid)] = {
                "tracker_id": tid,
                "team": self._team.get(tid, -1),
                "distance_m": round(dist.get(tid, 0.0), 2),
                "avg_speed_ms": round(speed.get(tid, 0.0), 2),
                "avg_position": {
                    "x_m": round(avg_pos.get(tid, (0.0, 0.0))[0], 2),
                    "y_m": round(avg_pos.get(tid, (0.0, 0.0))[1], 2),
                },
                "ruck_time_s": round(ruck_f / self.fps, 2),
                "ruck_density": round(ruck_f / max(visible_f, 1), 4),
                "frames_visible": visible_f,
            }

        # ---- Proximity matrix ----
        proximity = self._build_proximity_matrix()

        # ---- Ruck event log ----
        ruck_events = []
        for r in reid._active_rucks:
            ruck_events.append(
                {
                    "event_id": r.event_id,
                    "start_frame": r.start_frame,
                    "end_frame": r.end_frame,
                    "duration_s": round(
                        ((r.end_frame or self._total_frames) - r.start_frame) / self.fps, 2
                    ),
                    "player_ids": sorted(r.player_ids_at_start),
                    "zone_centroid_px": [
                        round(float(r.zone_centroid_px[0]), 1),
                        round(float(r.zone_centroid_px[1]), 1),
                    ],
                }
            )

        # ---- Re-ID log ----
        reid_events = [
            {
                "ruck_event_id": m.event_id,
                "frame": m.frame_idx,
                "old_id": m.old_tracker_id,
                "new_id": m.new_tracker_id,
                "confidence": round(m.confidence, 3),
            }
            for m in reid.reid_log
        ]

        # ---- Team aggregates ----
        team_stats = self._team_aggregates(dist, speed)

        report = {
            "meta": {
                "total_frames": self._total_frames,
                "duration_s": round(self._total_frames / self.fps, 2),
                "fps": self.fps,
                "field_width_m": self.field_width_m,
                "field_length_m": self.field_length_m,
                "total_players_tracked": len(players),
                "total_rucks": len(ruck_events),
                "total_reid_events": len(reid_events),
            },
            "players": players,
            "team_aggregates": team_stats,
            "proximity_matrix": proximity,
            "ruck_events": ruck_events,
            "reid_log": reid_events,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=indent, ensure_ascii=False)

        return report

    # ------------------------------------------------------------------
    # Proximity matrix
    # ------------------------------------------------------------------

    def _build_proximity_matrix(self) -> dict:
        """Compute average pairwise distance in metres between all player pairs.

        For each pair ``(i, j)``, the proximity value is the mean Euclidean
        distance in field metres across **all frames where both players are
        simultaneously visible**.  Pairs with fewer than 10 co-visible frames
        are marked ``null`` (insufficient data).

        Returns
        -------
        dict
            ``{"players": [...ids], "matrix": [[...], ...]}`` where
            ``matrix[r][c]`` is the mean proximity in metres between player
            ``players[r]`` and ``players[c]`` (or ``null``).
        """
        MIN_COFRAMES = 10

        player_ids = sorted(self._positions.keys())
        n = len(player_ids)
        matrix: list[list[Optional[float]]] = [[None] * n for _ in range(n)]

        for r in range(n):
            matrix[r][r] = 0.0
            id_r = player_ids[r]
            frames_r = self._positions[id_r]
            for c in range(r + 1, n):
                id_c = player_ids[c]
                frames_c = self._positions[id_c]
                common = set(frames_r) & set(frames_c)
                if len(common) < MIN_COFRAMES:
                    matrix[r][c] = matrix[c][r] = None
                    continue
                dists = [
                    math.hypot(
                        frames_r[f][0] - frames_c[f][0],
                        frames_r[f][1] - frames_c[f][1],
                    )
                    for f in common
                ]
                val = round(float(np.mean(dists)), 2)
                matrix[r][c] = matrix[c][r] = val

        return {"players": player_ids, "matrix": matrix}

    # ------------------------------------------------------------------
    # Team aggregates
    # ------------------------------------------------------------------

    def _team_aggregates(
        self, dist: dict[int, float], speed: dict[int, float]
    ) -> dict:
        """Compute totals and averages grouped by team.

        Returns
        -------
        dict
            Keyed by team index (as string), with ``total_distance_m``,
            ``avg_distance_m``, ``avg_speed_ms``, ``ruck_time_s``,
            ``player_ids``.
        """
        teams: dict[int, list[int]] = defaultdict(list)
        for tid, team in self._team.items():
            teams[team].append(tid)

        result = {}
        for team, tids in sorted(teams.items()):
            team_dists = [dist.get(t, 0.0) for t in tids]
            team_speeds = [speed.get(t, 0.0) for t in tids]
            team_ruck_f = sum(self._ruck_frames.get(t, 0) for t in tids)
            result[str(team)] = {
                "player_ids": sorted(tids),
                "total_distance_m": round(sum(team_dists), 2),
                "avg_distance_m": round(float(np.mean(team_dists)) if team_dists else 0.0, 2),
                "avg_speed_ms": round(float(np.mean(team_speeds)) if team_speeds else 0.0, 3),
                "total_ruck_time_s": round(team_ruck_f / self.fps, 2),
            }
        return result

    # ------------------------------------------------------------------
    # Ruck membership detection
    # ------------------------------------------------------------------

    def _detect_ruck_members_field(
        self,
        tracker_ids: np.ndarray,
        field_pts: np.ndarray,
        ruck_zones: list[tuple],
    ) -> set[int]:
        """Return tracker IDs that are within ``ruck_proximity_m`` of any active
        ruck zone centroid.

        Because ruck_zones are expressed in pixel space but field_pts are in
        metres, this method uses a cluster-density approach: any player whose
        field position is within a ball of radius ``ruck_proximity_m`` around
        the **mean field position of the ruck cluster** is considered inside.

        Parameters
        ----------
        tracker_ids : np.ndarray  shape (N,)
        field_pts : np.ndarray    shape (N, 2) metres
        ruck_zones : list         (cx_px, cy_px, radius_px) — pixel space (used
                                  only to count zones; actual matching is done
                                  in field space using clustering).

        Returns
        -------
        set[int]  tracker IDs currently in a ruck.
        """
        if len(field_pts) == 0 or not ruck_zones:
            return set()

        # Find dense clusters in field space
        r2 = self.ruck_proximity_m ** 2
        in_ruck: set[int] = set()
        n_zones = len(ruck_zones)

        # Simple approach: for each zone, pick a seed from the densest region
        # (centre of mass of all players within the ruck_proximity_m ball)
        # Iterate over all pairs to seed zones
        for zone_idx in range(min(n_zones, len(field_pts))):
            zone_seed = field_pts[zone_idx]
            # collect players within proximity
            members = []
            for i, pt in enumerate(field_pts):
                dx = pt[0] - zone_seed[0]
                dy = pt[1] - zone_seed[1]
                if dx * dx + dy * dy <= r2 * 4:  # 2× radius initial gather
                    members.append(i)
            if len(members) < 2:
                continue
            # recompute centroid
            centroid = field_pts[members].mean(axis=0)
            for i in members:
                dx = field_pts[i, 0] - centroid[0]
                dy = field_pts[i, 1] - centroid[1]
                if dx * dx + dy * dy <= r2:
                    in_ruck.add(int(tracker_ids[i]))

        return in_ruck

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: dict, fps: float = 25.0) -> "MatchAnalyzer":
        """Instantiate from a config dict.

        Parameters
        ----------
        cfg : dict
            Top-level ``config.yaml`` dict.
        fps : float
            Video frame rate.

        Returns
        -------
        MatchAnalyzer
        """
        hom = cfg.get("homography", {})
        ana = cfg.get("analytics", {})
        return cls(
            fps=fps,
            field_width_m=hom.get("field_width_m", 68.0),
            field_length_m=hom.get("field_length_m", 100.0),
            ruck_proximity_m=ana.get("ruck_proximity_m", 5.0),
        )
