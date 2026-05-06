"""Annotated video output, tactical 2-D minimap, ruck halos, and player trails."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Optional

import cv2
import numpy as np
import supervision as sv


# BGR palette used when team_labels are unavailable
_FALLBACK_COLOUR = (200, 200, 200)

# Ruck-halo colour (semi-transparent yellow)
_RUCK_HALO_COLOUR = (0, 220, 255)
_RUCK_HALO_ALPHA = 0.30


class Visualizer:
    """Compose the final annotated output frame.

    The output is a side-by-side composition:
    ``[annotated camera feed | tactical minimap]``

    New in this version
    -------------------
    * :meth:`draw` accepts ``ruck_zones`` to overlay a coloured halo around
      active ruck clusters.
    * Player trails (fading tails over the last ``trail_length`` frames) are
      rendered automatically when ``show_trails=True``.

    Parameters
    ----------
    team_colors : list[tuple[int, int, int]]
        BGR colours for team A and team B.
    referee_color : tuple[int, int, int]
        BGR colour for players not assigned to a team.
    show_ids : bool
        Overlay tracker IDs on bounding boxes.
    show_team_color : bool
        Colour bounding boxes by team.
    annotation_thickness : int
        Bounding-box line thickness in pixels.
    label_text_scale : float
        Font scale for ID labels.
    minimap_width : int
        Width of the minimap panel in pixels.
    minimap_height : int
        Height of the minimap panel in pixels.
    player_dot_radius : int
        Radius of player dots on the minimap.
    show_trails : bool
        Draw a fading positional trail behind each player.
    trail_length : int
        Number of past frames retained for the trail.
    trail_min_alpha : float
        Opacity of the oldest trail segment (0–1).

    Examples
    --------
    >>> vis = Visualizer(team_colors=[(0,100,255),(255,50,50)], show_trails=True)
    >>> out = vis.draw(frame, detections, team_labels, field_pts,
    ...               ruck_zones=reid.active_ruck_zones())
    >>> cv2.imshow("RugbyVision", out)
    """

    def __init__(
        self,
        team_colors: Optional[list[tuple]] = None,
        referee_color: tuple = (200, 200, 200),
        show_ids: bool = True,
        show_team_color: bool = True,
        annotation_thickness: int = 2,
        label_text_scale: float = 0.5,
        minimap_width: int = 400,
        minimap_height: int = 260,
        player_dot_radius: int = 6,
        show_trails: bool = True,
        trail_length: int = 20,
        trail_min_alpha: float = 0.05,
    ) -> None:
        self.team_colors = team_colors or [(0, 100, 255), (255, 50, 50)]
        self.referee_color = referee_color
        self.show_ids = show_ids
        self.show_team_color = show_team_color
        self.annotation_thickness = annotation_thickness
        self.label_text_scale = label_text_scale
        self.minimap_width = minimap_width
        self.minimap_height = minimap_height
        self.player_dot_radius = player_dot_radius
        self.show_trails = show_trails
        self.trail_length = trail_length
        self.trail_min_alpha = trail_min_alpha

        # trail_history[tracker_id] = deque of (px, py) in image space
        self._trail_history: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=trail_length)
        )
        # team_history[tracker_id] = last known team label
        self._team_cache: dict[int, Optional[int]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def draw(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        team_labels: Optional[np.ndarray] = None,
        field_pts: Optional[np.ndarray] = None,
        field_dims: tuple[float, float] = (68.0, 100.0),
        ruck_zones: Optional[list[tuple[float, float, float]]] = None,
    ) -> np.ndarray:
        """Render the full composite output frame.

        Parameters
        ----------
        frame : np.ndarray
            Original BGR camera frame.
        detections : sv.Detections
            Tracked detections (must have ``tracker_id`` set).
        team_labels : np.ndarray or None
            Integer array of shape ``(N,)`` with team index per detection.
        field_pts : np.ndarray or None
            Shape ``(N, 2)`` field coordinates in metres.
        field_dims : tuple[float, float]
            ``(width_m, length_m)`` of the real field.
        ruck_zones : list of (cx_px, cy_px, radius_px) or None
            Active ruck zones from
            :meth:`OcclusionReidentifier.active_ruck_zones`.

        Returns
        -------
        np.ndarray
            Horizontally stacked BGR image
            ``[annotated_frame | minimap_resized]``.
        """
        # Update team cache and trail history before drawing
        self._update_state(detections, team_labels)

        canvas = frame.copy()

        # Layer order (back to front):
        # 1. Ruck halos (semi-transparent blobs)
        if ruck_zones:
            canvas = self._draw_ruck_clusters(canvas, ruck_zones)

        # 2. Player trails
        if self.show_trails:
            canvas = self._draw_trails(canvas)

        # 3. Bounding boxes + ID labels
        canvas = self._annotate_frame(canvas, detections, team_labels)

        # 4. Minimap
        minimap = self._draw_minimap(field_pts, team_labels, field_dims, ruck_zones)
        h = canvas.shape[0]
        minimap_w = int(self.minimap_width * (h / self.minimap_height))
        minimap_resized = cv2.resize(minimap, (minimap_w, h))

        return np.hstack([canvas, minimap_resized])

    def draw_ruck_clusters(
        self,
        frame: np.ndarray,
        ruck_zones: list[tuple[float, float, float]],
        colour: tuple = _RUCK_HALO_COLOUR,
        alpha: float = _RUCK_HALO_ALPHA,
    ) -> np.ndarray:
        """Overlay a semi-transparent halo on each active ruck zone.

        Can be called standalone on an existing annotated frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR frame to draw on (modified in place).
        ruck_zones : list of (cx_px, cy_px, radius_px)
            Ruck zone descriptors in pixel coordinates.
        colour : tuple
            BGR halo colour.
        alpha : float
            Halo opacity (0 = invisible, 1 = opaque).

        Returns
        -------
        np.ndarray
            Frame with halos composited.
        """
        return self._draw_ruck_clusters(frame, ruck_zones, colour, alpha)

    def draw_trails(self, frame: np.ndarray) -> np.ndarray:
        """Draw fading positional trails on *frame* using the current history.

        Can be called standalone after :meth:`draw` has been used to update
        the internal trail history.

        Parameters
        ----------
        frame : np.ndarray
            BGR frame to draw on.

        Returns
        -------
        np.ndarray
            Frame with trails composited.
        """
        return self._draw_trails(frame.copy())

    @classmethod
    def from_config(cls, cfg: dict) -> "Visualizer":
        """Instantiate from a config dict.

        Parameters
        ----------
        cfg : dict
            Top-level config dict (``config.yaml``).

        Returns
        -------
        Visualizer
        """
        vis_cfg = cfg.get("visualization", {})
        tc_cfg = cfg.get("team_classifier", {})
        mm = vis_cfg.get("minimap", {})
        tr = vis_cfg.get("trails", {})
        raw_colors = tc_cfg.get("team_colors", {})
        team_colors = [
            tuple(raw_colors.get("team_a", [0, 100, 255])),
            tuple(raw_colors.get("team_b", [255, 50, 50])),
        ]
        referee_color = tuple(tc_cfg.get("referee_color", [200, 200, 200]))
        return cls(
            team_colors=team_colors,
            referee_color=referee_color,
            show_ids=vis_cfg.get("show_ids", True),
            show_team_color=vis_cfg.get("show_team_color", True),
            annotation_thickness=vis_cfg.get("annotation_thickness", 2),
            label_text_scale=vis_cfg.get("label_text_scale", 0.5),
            minimap_width=mm.get("width", 400),
            minimap_height=mm.get("height", 260),
            player_dot_radius=mm.get("player_dot_radius", 6),
            show_trails=tr.get("enabled", True),
            trail_length=tr.get("length", 20),
            trail_min_alpha=tr.get("min_alpha", 0.05),
        )

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _update_state(
        self,
        detections: sv.Detections,
        team_labels: Optional[np.ndarray],
    ) -> None:
        """Update trail history and team cache from the current frame's detections."""
        if detections.tracker_id is None:
            return

        for i, tid in enumerate(detections.tracker_id):
            tid = int(tid)
            x1, y1, x2, y2 = detections.xyxy[i]
            # Use bottom-centre as the trail anchor point
            px = int((x1 + x2) / 2)
            py = int(y2)
            self._trail_history[tid].append((px, py))
            if team_labels is not None:
                self._team_cache[tid] = int(team_labels[i])

    # ------------------------------------------------------------------
    # Ruck cluster halos
    # ------------------------------------------------------------------

    def _draw_ruck_clusters(
        self,
        frame: np.ndarray,
        ruck_zones: list[tuple],
        colour: tuple = _RUCK_HALO_COLOUR,
        alpha: float = _RUCK_HALO_ALPHA,
    ) -> np.ndarray:
        """Render a pulsing elliptical halo for each ruck zone."""
        overlay = frame.copy()
        for zone in ruck_zones:
            cx, cy, radius = int(zone[0]), int(zone[1]), int(zone[2])
            if radius <= 0:
                radius = 60
            # Outer glow (larger, more transparent)
            cv2.ellipse(
                overlay,
                (cx, cy),
                (int(radius * 1.3), int(radius * 0.85)),
                0, 0, 360,
                colour,
                -1,
            )
            # Inner core (tighter, slightly brighter)
            inner_colour = tuple(min(255, int(c * 1.25)) for c in colour)
            cv2.ellipse(
                overlay,
                (cx, cy),
                (radius, int(radius * 0.65)),
                0, 0, 360,
                inner_colour,
                -1,
            )
            # Label
            cv2.putText(
                overlay,
                "RUCK",
                (cx - 20, cy - int(radius * 0.9)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

        return cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0)

    # ------------------------------------------------------------------
    # Player trails
    # ------------------------------------------------------------------

    def _draw_trails(self, frame: np.ndarray) -> np.ndarray:
        """Draw fading trails for every player with history.

        The trail is rendered as a series of filled circles whose opacity
        decreases linearly from ``1.0`` (most recent) to ``trail_min_alpha``
        (oldest).  Each circle is blended via a per-segment overlay so the
        fade is smooth even when the trail length is short.
        """
        if not self._trail_history:
            return frame

        for tid, history in self._trail_history.items():
            pts = list(history)
            n = len(pts)
            if n < 2:
                continue

            team_lbl = self._team_cache.get(tid)
            colour = self._colour_for(team_lbl)

            for i, pt in enumerate(pts):
                # Alpha: oldest = trail_min_alpha, newest = 1.0
                t = i / max(n - 1, 1)
                a = self.trail_min_alpha + (1.0 - self.trail_min_alpha) * t
                radius = max(1, int(3 * t + 1))  # grow towards present

                overlay = frame.copy()
                cv2.circle(overlay, pt, radius, colour, -1, cv2.LINE_AA)
                frame = cv2.addWeighted(overlay, a, frame, 1.0 - a, 0)

        return frame

    # ------------------------------------------------------------------
    # Bounding-box annotation
    # ------------------------------------------------------------------

    def _colour_for(self, label: Optional[int]) -> tuple:
        if label is None or not self.show_team_color:
            return _FALLBACK_COLOUR
        if label < len(self.team_colors):
            return self.team_colors[label]
        return self.referee_color

    def _annotate_frame(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        team_labels: Optional[np.ndarray],
    ) -> np.ndarray:
        """Draw bounding boxes and optional ID labels on *frame*."""
        if len(detections) == 0:
            return frame

        colors = [
            self._colour_for(int(team_labels[i]) if team_labels is not None else None)
            for i in range(len(detections))
        ]

        colour_palette = sv.ColorPalette.from_hex(
            [self._bgr_to_hex(c) for c in colors]
        )
        box_ann = sv.BoxAnnotator(
            color=colour_palette, thickness=self.annotation_thickness
        )
        frame = box_ann.annotate(scene=frame, detections=detections)

        if self.show_ids and detections.tracker_id is not None:
            labels = [f"#{tid}" for tid in detections.tracker_id]
            label_ann = sv.LabelAnnotator(
                color=colour_palette, text_scale=self.label_text_scale
            )
            frame = label_ann.annotate(scene=frame, detections=detections, labels=labels)

        return frame

    # ------------------------------------------------------------------
    # Minimap
    # ------------------------------------------------------------------

    def _draw_minimap(
        self,
        field_pts: Optional[np.ndarray],
        team_labels: Optional[np.ndarray],
        field_dims: tuple[float, float],
        ruck_zones_px: Optional[list[tuple]] = None,
    ) -> np.ndarray:
        """Render a top-down 2-D tactical minimap."""
        W, H = self.minimap_width, self.minimap_height
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        canvas[:] = (40, 120, 40)  # grass green
        self._draw_field_markings(canvas, field_dims)

        fw, fl = field_dims

        # Draw ruck zone halos on minimap (approximate — no homography here)
        # We shade the ruck zones as circles proportional to minimap size
        if ruck_zones_px:
            for zone in ruck_zones_px:
                # Scale pixel centroid to minimap coords using field aspect
                # (rough approximation — real mapping requires the homography)
                pass  # Requires homography inversion; skipped here

        if field_pts is None or len(field_pts) == 0:
            return canvas

        # Trail on minimap (last N field points per player)
        for tid, history in self._trail_history.items():
            pts_px = list(history)
            # We have pixel history, not field history — minimap trails are
            # drawn in a separate pass below using field_pts only
            pass

        # Current positions
        for i, (x_m, y_m) in enumerate(field_pts):
            x_m = float(np.clip(x_m, 0, fw))
            y_m = float(np.clip(y_m, 0, fl))
            px = int(x_m / fw * W)
            py = int(y_m / fl * H)
            lbl = int(team_labels[i]) if team_labels is not None else None
            colour = self._colour_for(lbl)
            cv2.circle(canvas, (px, py), self.player_dot_radius, colour, -1)
            cv2.circle(canvas, (px, py), self.player_dot_radius, (0, 0, 0), 1)

        return canvas

    @staticmethod
    def _draw_field_markings(canvas: np.ndarray, field_dims: tuple) -> None:
        """Draw simplified rugby pitch lines on *canvas*."""
        H, W = canvas.shape[:2]
        fw, fl = field_dims
        lc = (255, 255, 255)  # line colour

        cv2.rectangle(canvas, (0, 0), (W - 1, H - 1), lc, 1)
        cv2.line(canvas, (0, H // 2), (W, H // 2), lc, 1)
        y22a = int(22 / fl * H)
        y22b = H - y22a
        cv2.line(canvas, (0, y22a), (W, y22a), lc, 1)
        cv2.line(canvas, (0, y22b), (W, y22b), lc, 1)
        # Try-lines (5 m in from each end of the field image — visual only)
        y_try_a = int(5 / fl * H)
        y_try_b = H - y_try_a
        cv2.line(canvas, (0, y_try_a), (W, y_try_a), (180, 180, 255), 1)
        cv2.line(canvas, (0, y_try_b), (W, y_try_b), (180, 180, 255), 1)

    @staticmethod
    def _bgr_to_hex(bgr: tuple) -> str:
        b, g, r = bgr
        return f"#{r:02x}{g:02x}{b:02x}"
