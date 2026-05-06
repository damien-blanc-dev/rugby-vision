"""Post-occlusion player re-identification for rugby rucks and scrums.

Algorithm overview
------------------
1. **Ruck detection** — each frame, compute pairwise centroid distances among
   active bounding boxes.  A connected component with >= ``min_players_in_group``
   players (where "connected" means centroid distance < ``ruck_radius_px``)
   is declared an active ruck zone.
2. **Player memory** — as soon as a ruck forms, a :class:`PlayerMemory` record
   is frozen for every player entering the zone (last position, mean HSV jersey
   colour, bbox aspect ratio, optional OCR jersey number).  Records expire after
   ``memory_ttl_frames`` frames.
3. **Dispersion matching** — when a ruck zone dissolves (player count drops back
   below threshold for ``hysteresis_frames`` consecutive frames), new ByteTrack
   IDs that appear near the zone are matched against the frozen memories using a
   weighted cost matrix and the Hungarian algorithm.
4. **ID remapping** — confirmed matches are stored and applied to every
   subsequent ``sv.Detections`` object so the rest of the pipeline sees
   consistent IDs.
5. **Re-ID log** — every matching event is appended to :attr:`reid_log` for
   offline benchmarking.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import supervision as sv
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PlayerMemory:
    """Snapshot of a player's appearance taken before occlusion."""

    tracker_id: int
    last_seen_frame: int
    last_xyxy: np.ndarray          # image bbox [x1,y1,x2,y2]
    last_field_pt: Optional[np.ndarray]  # field coords (x_m, y_m) or None
    mean_hsv: np.ndarray           # mean jersey colour in HSV [H,S,V]
    aspect_ratio: float            # bbox height / width
    jersey_number: Optional[int]   # OCR result, None if unavailable


@dataclass
class RuckEvent:
    """Lifecycle record for a single ruck/scrum episode."""

    event_id: int
    start_frame: int
    zone_centroid_px: np.ndarray   # pixel centroid of the ruck zone
    zone_radius_px: float          # approximate radius in pixels
    player_ids_at_start: set       # tracker_ids that entered the ruck
    is_active: bool = True
    end_frame: Optional[int] = None
    inactive_streak: int = 0       # consecutive frames below threshold


@dataclass
class ReidMatch:
    """One successful re-identification event (written to reid_log)."""

    event_id: int                  # parent RuckEvent id
    frame_idx: int
    old_tracker_id: int
    new_tracker_id: int
    cost: float                    # matching cost (lower = more confident)
    confidence: float              # 1 - cost/max_cost


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class OcclusionReidentifier:
    """Re-identify players after ruck/scrum occlusions.

    Parameters
    ----------
    occlusion_iou_threshold : float
        *Deprecated* — kept for API compatibility.  Use ``ruck_radius_px``.
    ruck_radius_px : float
        Maximum centroid-to-centroid distance (pixels) for two players to be
        considered part of the same ruck cluster.
    min_players_in_group : int
        Minimum cluster size to declare a ruck.
    memory_ttl_frames : int
        Number of frames a :class:`PlayerMemory` entry stays valid after last
        being observed.
    reid_confidence_threshold : float
        Minimum confidence (0–1) required to commit a re-ID match.
    hysteresis_frames : int
        Consecutive frames below ``min_players_in_group`` needed to declare a
        ruck over (avoids flickering).
    color_weight : float
        Weight of the jersey-colour distance in the cost matrix.
    position_weight : float
        Weight of the spatial distance term in the cost matrix.
    size_weight : float
        Weight of the aspect-ratio difference term.
    ocr_bonus : float
        Cost reduction applied when two jersey numbers match via OCR.
    use_ocr : bool
        Whether to attempt EasyOCR jersey number reading (slower but more
        accurate).  Falls back silently if easyocr is not installed.
    ocr_crop_top_ratio : float
        Fraction of bbox height for the torso crop passed to OCR.

    Examples
    --------
    >>> reid = OcclusionReidentifier(min_players_in_group=4)
    >>> dets = reid.update(frame, dets, frame_idx=42)
    >>> print(reid.reid_log)
    """

    def __init__(
        self,
        occlusion_iou_threshold: float = 0.3,  # kept for compat
        ruck_radius_px: float = 120.0,
        min_players_in_group: int = 4,
        memory_ttl_frames: int = 60,
        reid_confidence_threshold: float = 0.6,
        hysteresis_frames: int = 5,
        color_weight: float = 0.5,
        position_weight: float = 0.35,
        size_weight: float = 0.15,
        ocr_bonus: float = 0.4,
        use_ocr: bool = False,
        ocr_crop_top_ratio: float = 0.3,
    ) -> None:
        self.ruck_radius_px = ruck_radius_px
        self.min_players_in_group = min_players_in_group
        self.memory_ttl_frames = memory_ttl_frames
        self.reid_confidence_threshold = reid_confidence_threshold
        self.hysteresis_frames = hysteresis_frames
        self.color_weight = color_weight
        self.position_weight = position_weight
        self.size_weight = size_weight
        self.ocr_bonus = ocr_bonus
        self.use_ocr = use_ocr
        self.ocr_crop_top_ratio = ocr_crop_top_ratio

        # State
        self._memories: dict[int, PlayerMemory] = {}        # tracker_id → memory
        self._known_ids: set[int] = set()                   # all IDs ever seen
        self._active_rucks: list[RuckEvent] = []
        self._id_remapping: dict[int, int] = {}             # new_id → old_id
        self._event_counter: int = 0
        self.reid_log: list[ReidMatch] = []

        # Lazy OCR reader
        self._ocr_reader = None
        if use_ocr:
            self._init_ocr()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        frame_idx: int = 0,
        field_pts: Optional[np.ndarray] = None,
    ) -> sv.Detections:
        """Process one frame and return detections with corrected tracker IDs.

        Parameters
        ----------
        frame : np.ndarray
            Current BGR frame.
        detections : sv.Detections
            Output of :class:`PlayerTracker` (must have ``tracker_id`` set).
        frame_idx : int
            Monotonically increasing frame counter.
        field_pts : np.ndarray or None
            Shape ``(N, 2)`` field coordinates from
            :meth:`FieldHomography.project_detections`.

        Returns
        -------
        sv.Detections
            Same detections with ``tracker_id`` potentially remapped.
        """
        if detections.tracker_id is None or len(detections) == 0:
            return detections

        # 1. Apply existing remappings (IDs already resolved in prior frames)
        detections = self._apply_remapping(detections)

        # 2. Update the set of all known IDs
        self._known_ids.update(int(tid) for tid in detections.tracker_id)

        # 3. Update player memories for all currently visible players
        self._update_memories(frame, detections, frame_idx, field_pts)

        # 4. Expire stale memories
        self._expire_memories(frame_idx)

        # 5. Detect ruck clusters in this frame
        clusters = self._detect_clusters(detections)

        # 6. Update ruck lifecycle (open new rucks, advance hysteresis)
        new_rucks_ending = self._update_ruck_lifecycle(clusters, detections, frame_idx)

        # 7. Trigger re-ID matching for rucks that just ended
        for ruck in new_rucks_ending:
            new_ids = self._find_candidate_new_ids(ruck, detections, frame_idx)
            if new_ids:
                matches = self._match(ruck, new_ids, detections, frame, frame_idx)
                self._commit_remapping(matches, detections, ruck, frame_idx)
                # Apply immediately to this frame
                detections = self._apply_remapping(detections)

        return detections

    @property
    def active_ruck_centroids(self) -> list[np.ndarray]:
        """Return pixel centroids of currently active rucks (for visualisation)."""
        return [r.zone_centroid_px for r in self._active_rucks if r.is_active]

    def active_ruck_zones(self) -> list[tuple[float, float, float]]:
        """Return ``(cx_px, cy_px, radius_px)`` tuples for all active rucks.

        Used by :class:`Visualizer` to draw halos and by :class:`MatchAnalyzer`
        to determine ruck-zone membership.

        Returns
        -------
        list of (float, float, float)
        """
        return [
            (
                float(r.zone_centroid_px[0]),
                float(r.zone_centroid_px[1]),
                float(r.zone_radius_px),
            )
            for r in self._active_rucks
            if r.is_active
        ]

    @classmethod
    def from_config(cls, cfg: dict) -> "OcclusionReidentifier":
        """Instantiate from a config dict (``config.yaml`` → ``reidentification`` section).

        Parameters
        ----------
        cfg : dict
            The ``reidentification`` sub-dict from ``config.yaml``.

        Returns
        -------
        OcclusionReidentifier
        """
        return cls(
            ruck_radius_px=cfg.get("ruck_radius_px", 120.0),
            min_players_in_group=cfg.get("min_players_in_group", 4),
            memory_ttl_frames=cfg.get("memory_ttl_frames", 60),
            reid_confidence_threshold=cfg.get("reid_confidence_threshold", 0.6),
            hysteresis_frames=cfg.get("hysteresis_frames", 5),
            color_weight=cfg.get("color_weight", 0.5),
            position_weight=cfg.get("position_weight", 0.35),
            size_weight=cfg.get("size_weight", 0.15),
            ocr_bonus=cfg.get("ocr_bonus", 0.4),
            use_ocr=cfg.get("use_ocr", False),
        )

    # ------------------------------------------------------------------
    # Cluster detection
    # ------------------------------------------------------------------

    def _detect_clusters(self, detections: sv.Detections) -> list[list[int]]:
        """Return connected components of players closer than ``ruck_radius_px``.

        Returns
        -------
        list[list[int]]
            Each inner list is a group of detection *indices* (not tracker IDs).
        """
        n = len(detections)
        if n == 0:
            return []

        centroids = self._centroids(detections.xyxy)  # (N, 2)

        # Build adjacency as union-find
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            parent[find(x)] = find(y)

        for i in range(n):
            for j in range(i + 1, n):
                dist = float(np.linalg.norm(centroids[i] - centroids[j]))
                if dist < self.ruck_radius_px:
                    union(i, j)

        groups: dict[int, list[int]] = defaultdict(list)
        for i in range(n):
            groups[find(i)].append(i)

        return [g for g in groups.values() if len(g) >= self.min_players_in_group]

    # ------------------------------------------------------------------
    # Ruck lifecycle
    # ------------------------------------------------------------------

    def _update_ruck_lifecycle(
        self,
        clusters: list[list[int]],
        detections: sv.Detections,
        frame_idx: int,
    ) -> list[RuckEvent]:
        """Open new rucks, advance hysteresis for existing ones.

        Returns the list of rucks that just ended this frame.
        """
        centroids = self._centroids(detections.xyxy) if len(detections) else np.empty((0, 2))
        just_ended: list[RuckEvent] = []

        # --- Match existing active rucks to current clusters ---
        matched_ruck_ids = set()
        for ruck in self._active_rucks:
            if not ruck.is_active:
                continue
            # Check if any cluster overlaps this ruck's zone
            matched = False
            for cluster in clusters:
                cluster_centroid = centroids[cluster].mean(axis=0)
                if np.linalg.norm(cluster_centroid - ruck.zone_centroid_px) < self.ruck_radius_px * 1.5:
                    # Ruck still active — reset hysteresis, update centroid
                    ruck.inactive_streak = 0
                    ruck.zone_centroid_px = cluster_centroid
                    # Accumulate new player IDs
                    for idx in cluster:
                        tid = int(detections.tracker_id[idx])
                        ruck.player_ids_at_start.add(tid)
                    matched = True
                    matched_ruck_ids.add(id(ruck))
                    break
            if not matched:
                ruck.inactive_streak += 1
                if ruck.inactive_streak >= self.hysteresis_frames:
                    ruck.is_active = False
                    ruck.end_frame = frame_idx
                    just_ended.append(ruck)
                    logger.debug(
                        "Ruck #%d ended at frame %d (%d players involved)",
                        ruck.event_id, frame_idx, len(ruck.player_ids_at_start),
                    )

        # --- Open new rucks for unmatched clusters ---
        for cluster in clusters:
            cluster_centroid = centroids[cluster].mean(axis=0)
            already_tracked = any(
                np.linalg.norm(cluster_centroid - r.zone_centroid_px) < self.ruck_radius_px * 1.5
                for r in self._active_rucks
                if r.is_active
            )
            if not already_tracked:
                player_ids = {int(detections.tracker_id[i]) for i in cluster}
                radius = float(
                    max(np.linalg.norm(centroids[i] - cluster_centroid) for i in cluster) + 20
                )
                self._event_counter += 1
                ruck = RuckEvent(
                    event_id=self._event_counter,
                    start_frame=frame_idx,
                    zone_centroid_px=cluster_centroid,
                    zone_radius_px=radius,
                    player_ids_at_start=player_ids,
                )
                self._active_rucks.append(ruck)
                logger.debug(
                    "Ruck #%d opened at frame %d, %d players",
                    ruck.event_id, frame_idx, len(player_ids),
                )

        # Prune finished rucks older than TTL
        self._active_rucks = [
            r for r in self._active_rucks
            if r.is_active or (frame_idx - (r.end_frame or frame_idx)) < self.memory_ttl_frames
        ]

        return just_ended

    # ------------------------------------------------------------------
    # Player memory
    # ------------------------------------------------------------------

    def _update_memories(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        frame_idx: int,
        field_pts: Optional[np.ndarray],
    ) -> None:
        """Refresh memory for every currently visible player."""
        h_img, w_img = frame.shape[:2]
        for i, (xyxy, tid) in enumerate(zip(detections.xyxy, detections.tracker_id)):
            tid = int(tid)
            hsv = self._mean_hsv(frame, xyxy)
            x1, y1, x2, y2 = xyxy
            bh = max(y2 - y1, 1)
            bw = max(x2 - x1, 1)
            ar = float(bh / bw)
            fp = field_pts[i] if field_pts is not None and i < len(field_pts) else None

            jersey_num = None
            if self.use_ocr and self._ocr_reader is not None:
                jersey_num = self._ocr_jersey(frame, xyxy)

            self._memories[tid] = PlayerMemory(
                tracker_id=tid,
                last_seen_frame=frame_idx,
                last_xyxy=np.array(xyxy, dtype=np.float32),
                last_field_pt=np.array(fp, dtype=np.float32) if fp is not None else None,
                mean_hsv=hsv,
                aspect_ratio=ar,
                jersey_number=jersey_num,
            )

    def _expire_memories(self, frame_idx: int) -> None:
        to_delete = [
            tid for tid, mem in self._memories.items()
            if frame_idx - mem.last_seen_frame > self.memory_ttl_frames
        ]
        for tid in to_delete:
            del self._memories[tid]

    # ------------------------------------------------------------------
    # Re-ID matching
    # ------------------------------------------------------------------

    def _find_candidate_new_ids(
        self,
        ruck: RuckEvent,
        detections: sv.Detections,
        frame_idx: int,
    ) -> list[int]:
        """Return tracker IDs that appeared after the ruck and are near its zone.

        These are IDs that were not seen *before* the ruck started, or IDs that
        appeared for the first time within ``memory_ttl_frames`` frames.
        """
        candidates = []
        centroids = self._centroids(detections.xyxy)
        search_radius = ruck.zone_radius_px * 2.5

        for i, tid in enumerate(detections.tracker_id):
            tid = int(tid)
            # Already known before ruck → not a candidate (ByteTrack kept it)
            if tid in ruck.player_ids_at_start:
                continue
            # Already remapped (resolved in a prior frame)
            if tid in self._id_remapping:
                continue
            # Check spatial proximity to ruck zone
            dist = float(np.linalg.norm(centroids[i] - ruck.zone_centroid_px))
            if dist < search_radius:
                candidates.append(tid)

        return candidates

    def _match(
        self,
        ruck: RuckEvent,
        new_ids: list[int],
        detections: sv.Detections,
        frame: np.ndarray,
        frame_idx: int,
    ) -> list[tuple[int, int, float]]:
        """Run Hungarian matching between lost players and new detections.

        Returns
        -------
        list of (old_id, new_id, cost)
            Only pairs whose normalised cost implies confidence >=
            ``reid_confidence_threshold``.
        """
        # Lost players = players that were in the ruck but are no longer visible
        visible_ids = set(int(t) for t in detections.tracker_id)
        lost_ids = [tid for tid in ruck.player_ids_at_start if tid not in visible_ids]

        if not lost_ids or not new_ids:
            return []

        # Build detection lookup: new_id → index in detections
        id_to_idx = {int(tid): i for i, tid in enumerate(detections.tracker_id)}

        # Image diagonal for position normalisation
        h, w = frame.shape[:2]
        img_diag = float(np.sqrt(h ** 2 + w ** 2))

        # Cost matrix: rows = lost memories, cols = new IDs
        R, C = len(lost_ids), len(new_ids)
        cost = np.ones((R, C), dtype=np.float64)

        for r, old_id in enumerate(lost_ids):
            mem = self._memories.get(old_id)
            if mem is None:
                continue  # no memory → leave cost=1 (worst)
            mem_centroid = self._centroid_of_xyxy(mem.last_xyxy)

            for c, new_id in enumerate(new_ids):
                idx = id_to_idx.get(new_id)
                if idx is None:
                    continue
                new_xyxy = detections.xyxy[idx]
                new_centroid = self._centroid_of_xyxy(new_xyxy)

                # Spatial term (normalised to [0,1])
                pos_dist = float(np.linalg.norm(new_centroid - mem_centroid)) / img_diag

                # Colour term
                new_hsv = self._mean_hsv(frame, new_xyxy)
                hsv_dist = float(np.linalg.norm(new_hsv - mem.mean_hsv)) / 441.7  # max possible L2

                # Aspect-ratio term
                nx1, ny1, nx2, ny2 = new_xyxy
                new_ar = (ny2 - ny1) / max(nx2 - nx1, 1)
                size_dist = abs(new_ar - mem.aspect_ratio) / max(mem.aspect_ratio, new_ar, 1)

                cell = (
                    self.position_weight * pos_dist
                    + self.color_weight * hsv_dist
                    + self.size_weight * size_dist
                )

                # OCR bonus
                if (
                    self.use_ocr
                    and mem.jersey_number is not None
                    and mem.jersey_number == self._ocr_jersey(frame, new_xyxy)
                ):
                    cell = max(0.0, cell - self.ocr_bonus)

                cost[r, c] = cell

        row_ind, col_ind = linear_sum_assignment(cost)

        matches = []
        max_cost = self.position_weight + self.color_weight + self.size_weight
        for r, c in zip(row_ind, col_ind):
            c_val = float(cost[r, c])
            confidence = max(0.0, 1.0 - c_val / max_cost)
            if confidence >= self.reid_confidence_threshold:
                matches.append((lost_ids[r], new_ids[c], c_val))

        return matches

    def _commit_remapping(
        self,
        matches: list[tuple[int, int, float]],
        detections: sv.Detections,
        ruck: RuckEvent,
        frame_idx: int,
    ) -> None:
        """Store matches in the remapping table and append to reid_log."""
        max_cost = self.position_weight + self.color_weight + self.size_weight
        for old_id, new_id, cost in matches:
            self._id_remapping[new_id] = old_id
            confidence = max(0.0, 1.0 - cost / max_cost)
            self.reid_log.append(
                ReidMatch(
                    event_id=ruck.event_id,
                    frame_idx=frame_idx,
                    old_tracker_id=old_id,
                    new_tracker_id=new_id,
                    cost=cost,
                    confidence=confidence,
                )
            )
            logger.info(
                "Re-ID: %d → %d  (conf=%.2f, frame=%d)",
                new_id, old_id, confidence, frame_idx,
            )

    # ------------------------------------------------------------------
    # ID remapping application
    # ------------------------------------------------------------------

    def _apply_remapping(self, detections: sv.Detections) -> sv.Detections:
        """Return a copy of *detections* with remapped tracker IDs."""
        if not self._id_remapping or detections.tracker_id is None:
            return detections

        new_ids = detections.tracker_id.copy()
        for i, tid in enumerate(new_ids):
            remapped = self._id_remapping.get(int(tid))
            if remapped is not None:
                new_ids[i] = remapped

        # sv.Detections is immutable — rebuild with updated tracker_id
        return sv.Detections(
            xyxy=detections.xyxy,
            confidence=detections.confidence,
            class_id=detections.class_id,
            tracker_id=new_ids,
            data=detections.data,
        )

    # ------------------------------------------------------------------
    # Feature helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _centroids(xyxy: np.ndarray) -> np.ndarray:
        """Return shape ``(N, 2)`` centroid array from xyxy boxes."""
        if len(xyxy) == 0:
            return np.empty((0, 2), dtype=np.float32)
        return np.stack(
            [(xyxy[:, 0] + xyxy[:, 2]) / 2, (xyxy[:, 1] + xyxy[:, 3]) / 2], axis=1
        )

    @staticmethod
    def _centroid_of_xyxy(xyxy: np.ndarray) -> np.ndarray:
        return np.array([(xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2])

    def _mean_hsv(self, frame: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        """Crop torso and return mean HSV colour."""
        x1, y1, x2, y2 = map(int, xyxy)
        h = y2 - y1
        y_top = y1 + int(h * self.ocr_crop_top_ratio)
        crop = frame[y_top:y2, x1:x2]
        if crop.size == 0:
            return np.zeros(3, dtype=np.float32)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        return hsv.reshape(-1, 3).mean(axis=0).astype(np.float32)

    # ------------------------------------------------------------------
    # Optional OCR
    # ------------------------------------------------------------------

    def _init_ocr(self) -> None:
        try:
            import easyocr
            self._ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            logger.info("EasyOCR loaded for jersey number detection.")
        except ImportError:
            logger.warning("easyocr not installed — jersey-number OCR disabled.")
            self.use_ocr = False

    def _ocr_jersey(self, frame: np.ndarray, xyxy: np.ndarray) -> Optional[int]:
        """Return jersey number from OCR, or None if undetectable."""
        if self._ocr_reader is None:
            return None
        x1, y1, x2, y2 = map(int, xyxy)
        h = y2 - y1
        y_top = y1 + int(h * 0.05)
        y_bot = y1 + int(h * 0.45)  # upper half → jersey number area
        crop = frame[y_top:y_bot, x1:x2]
        if crop.size == 0:
            return None
        try:
            results = self._ocr_reader.readtext(crop, detail=0, allowlist="0123456789")
            for r in results:
                text = r.strip()
                if text.isdigit() and 1 <= int(text) <= 99:
                    return int(text)
        except Exception:
            pass
        return None
