"""ByteTrack multi-object tracker via the supervision library."""

from __future__ import annotations

import numpy as np
import supervision as sv


class PlayerTracker:
    """Wrap supervision's ByteTrack to assign persistent IDs to detections.

    Parameters
    ----------
    track_thresh : float
        Confidence threshold for track activation.
    track_buffer : int
        Number of frames to keep a lost track alive before deleting it.
    match_thresh : float
        IoU matching threshold between detections and existing tracks.
    frame_rate : int
        Video frame rate used internally by ByteTrack.

    Examples
    --------
    >>> tracker = PlayerTracker()
    >>> detections = tracker.update(detections, frame)
    >>> # detections.tracker_id now contains persistent integer IDs
    """

    def __init__(
        self,
        track_thresh: float = 0.45,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        frame_rate: int = 25,
    ) -> None:
        self._tracker = sv.ByteTrack(
            track_thresh=track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            frame_rate=frame_rate,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self, detections: sv.Detections, frame: np.ndarray | None = None
    ) -> sv.Detections:
        """Update tracker state and return detections with ``tracker_id`` set.

        Parameters
        ----------
        detections : sv.Detections
            Raw detections from :class:`PlayerDetector`.
        frame : np.ndarray or None
            Current frame (unused by ByteTrack but kept for API symmetry).

        Returns
        -------
        sv.Detections
            Detections with ``tracker_id`` array populated.
        """
        return self._tracker.update_with_detections(detections)

    def reset(self) -> None:
        """Reset internal tracker state (call between clips/videos)."""
        self._tracker.reset()

    @classmethod
    def from_config(cls, cfg: dict, frame_rate: int = 25) -> "PlayerTracker":
        """Instantiate from a config dict (``config.yaml`` → ``tracking`` section).

        Parameters
        ----------
        cfg : dict
            Dictionary with keys: ``track_thresh``, ``track_buffer``,
            ``match_thresh``.
        frame_rate : int
            Video frame rate.

        Returns
        -------
        PlayerTracker
        """
        return cls(
            track_thresh=cfg.get("track_thresh", 0.45),
            track_buffer=cfg.get("track_buffer", 30),
            match_thresh=cfg.get("match_thresh", 0.8),
            frame_rate=frame_rate,
        )
