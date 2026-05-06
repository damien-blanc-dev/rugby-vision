"""Team separation by K-means clustering on HSV jersey colours.

Stability mechanism
-------------------
A common failure mode in live tracking is *cluster drift*: if a large fraction
of players from one team temporarily leave the camera frame (e.g. during a
kick-chase), the K-means assignments on that skewed population shift, causing
previously orange players to be labelled blue and vice-versa for the rest of
the clip.

This implementation mitigates drift by **freezing cluster centres** after
``stable_after_frames`` successful prediction calls.  Once frozen, every
subsequent :meth:`predict` call uses a simple nearest-centroid rule (O(1) per
player, no K-means inference) rather than the sklearn model.  The frozen
centres can also be saved and reloaded across runs via :meth:`save_centers` /
:meth:`load_centers`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from sklearn.cluster import KMeans


class TeamClassifier:
    """Assign a team label to each tracked player based on jersey colour.

    The classifier crops the torso region of each bounding box, converts it to
    the chosen colour space, and clusters the mean colours with K-means.  After
    fitting on a batch of frames it can predict team membership for any new
    detection in O(1).

    After ``stable_after_frames`` prediction calls the cluster centres are
    **frozen**: subsequent predictions use fixed nearest-centroid assignment,
    preventing drift when the visible player distribution becomes unbalanced.

    Parameters
    ----------
    n_clusters : int
        Number of distinct jersey colours (teams).  Typically 2; use 3 to
        separate referees.
    color_space : str
        ``"hsv"``, ``"rgb"``, or ``"lab"``.
    crop_top_ratio : float
        Fraction of bounding-box height removed from the top (avoids the head).
    crop_bottom_ratio : float
        Fraction of bounding-box height removed from the bottom (avoids legs).
    stable_after_frames : int
        Number of :meth:`predict` calls before the cluster centres are frozen.
        Set to ``0`` to disable freezing (always use live K-means).

    Examples
    --------
    >>> clf = TeamClassifier(n_clusters=2, stable_after_frames=100)
    >>> clf.fit(frame_list, detections_list)
    >>> labels = clf.predict(frame, detections)  # live K-means
    >>> # after 100 calls, centres freeze automatically
    >>> labels = clf.predict(frame, detections)  # fixed nearest-centroid
    >>> clf.save_centers("centers.json")
    """

    _CONVERSIONS = {
        "hsv": cv2.COLOR_BGR2HSV,
        "lab": cv2.COLOR_BGR2Lab,
        "rgb": cv2.COLOR_BGR2RGB,
    }

    def __init__(
        self,
        n_clusters: int = 2,
        color_space: str = "hsv",
        crop_top_ratio: float = 0.15,
        crop_bottom_ratio: float = 0.15,
        stable_after_frames: int = 100,
    ) -> None:
        if color_space not in self._CONVERSIONS:
            raise ValueError(f"color_space must be one of {list(self._CONVERSIONS)}")
        self.n_clusters = n_clusters
        self.color_space = color_space
        self.crop_top_ratio = crop_top_ratio
        self.crop_bottom_ratio = crop_bottom_ratio
        self.stable_after_frames = stable_after_frames

        self._kmeans: Optional[KMeans] = None
        # Fixed centres set after stability threshold is reached
        self._frozen_centers: Optional[np.ndarray] = None
        self._predict_call_count: int = 0

    # ------------------------------------------------------------------
    # Public API — fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        frames: list[np.ndarray],
        detections_list: list,
        max_samples: int = 500,
    ) -> "TeamClassifier":
        """Fit K-means on jersey colours sampled from several frames.

        Calling :meth:`fit` after the centres have been frozen will reopen
        the live-K-means mode and reset the stability counter.

        Parameters
        ----------
        frames : list[np.ndarray]
            BGR frames (same length as *detections_list*).
        detections_list : list[sv.Detections]
            Corresponding detections for each frame.
        max_samples : int
            Maximum number of player crops to use for fitting (random sample).

        Returns
        -------
        TeamClassifier
            ``self``, to allow chaining.
        """
        colours: list[np.ndarray] = []
        for frame, dets in zip(frames, detections_list):
            for xyxy in dets.xyxy:
                crop = self._extract_torso(frame, xyxy)
                if crop is not None:
                    colours.append(self._mean_colour(crop))

        if not colours:
            raise RuntimeError("No valid crops found — check detection results.")

        X = np.array(colours)
        if len(X) > max_samples:
            idx = np.random.choice(len(X), max_samples, replace=False)
            X = X[idx]

        self._kmeans = KMeans(n_clusters=self.n_clusters, n_init="auto", random_state=0)
        self._kmeans.fit(X)

        # Reset stability state (a new fit invalidates a previous freeze)
        self._frozen_centers = None
        self._predict_call_count = 0
        return self

    # ------------------------------------------------------------------
    # Public API — prediction
    # ------------------------------------------------------------------

    def predict(self, frame: np.ndarray, detections) -> np.ndarray:
        """Predict team label (0 … n_clusters-1) for each detection.

        The first ``stable_after_frames`` calls use the live K-means model.
        Subsequent calls use the frozen nearest-centroid rule.

        Parameters
        ----------
        frame : np.ndarray
            Current BGR frame.
        detections : sv.Detections
            Detections whose bounding boxes will be used for cropping.

        Returns
        -------
        np.ndarray
            Integer array of shape ``(N,)`` with cluster labels.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called yet.
        """
        if self._kmeans is None:
            raise RuntimeError("Call fit() before predict().")

        colours: list[np.ndarray] = []
        for xyxy in detections.xyxy:
            crop = self._extract_torso(frame, xyxy)
            colours.append(self._mean_colour(crop) if crop is not None else np.zeros(3))

        if not colours:
            return np.array([], dtype=int)

        X = np.array(colours)
        self._predict_call_count += 1

        # Freeze centres after the stability threshold
        if (
            self.stable_after_frames > 0
            and self._predict_call_count == self.stable_after_frames
            and self._frozen_centers is None
        ):
            self._freeze()

        if self._frozen_centers is not None:
            return self._nearest_centroid(X)

        return self._kmeans.predict(X)

    def is_fitted(self) -> bool:
        """Return ``True`` if :meth:`fit` has been called."""
        return self._kmeans is not None

    def is_frozen(self) -> bool:
        """Return ``True`` if cluster centres have been frozen."""
        return self._frozen_centers is not None

    def freeze(self) -> None:
        """Manually freeze cluster centres at the current K-means state.

        Useful to trigger freezing before the automatic threshold is reached
        (e.g. after a clean initial warm-up).
        """
        if self._kmeans is None:
            raise RuntimeError("Call fit() before freeze().")
        self._freeze()

    # ------------------------------------------------------------------
    # Centre persistence
    # ------------------------------------------------------------------

    def save_centers(self, path: str) -> None:
        """Persist cluster centres (and metadata) to a JSON file.

        Parameters
        ----------
        path : str
            Destination file path (typically ``*.json``).

        Raises
        ------
        RuntimeError
            If no centres are available yet.
        """
        centers = self._frozen_centers if self._frozen_centers is not None else (
            self._kmeans.cluster_centers_ if self._kmeans else None
        )
        if centers is None:
            raise RuntimeError("No centres to save — call fit() first.")
        data = {
            "color_space": self.color_space,
            "n_clusters": self.n_clusters,
            "centers": centers.tolist(),
            "frozen": self._frozen_centers is not None,
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load_centers(self, path: str) -> None:
        """Load and immediately freeze cluster centres from a JSON file.

        This skips the warm-up phase entirely — useful when resuming analysis
        on a sequence from a previously calibrated session.

        Parameters
        ----------
        path : str
            Source JSON file (saved by :meth:`save_centers`).
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        centers = np.array(data["centers"], dtype=np.float64)
        self._frozen_centers = centers
        # Bootstrap a dummy KMeans so is_fitted() returns True
        if self._kmeans is None:
            self._kmeans = _DummyKMeans(centers)

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: dict) -> "TeamClassifier":
        """Instantiate from a config dict (``config.yaml`` → ``team_classifier`` section).

        Parameters
        ----------
        cfg : dict
            Dictionary with keys ``n_clusters``, ``color_space``,
            ``crop_top_ratio``, ``crop_bottom_ratio``,
            ``stable_after_frames``.

        Returns
        -------
        TeamClassifier
        """
        return cls(
            n_clusters=cfg.get("n_clusters", 2),
            color_space=cfg.get("color_space", "hsv"),
            crop_top_ratio=cfg.get("crop_top_ratio", 0.15),
            crop_bottom_ratio=cfg.get("crop_bottom_ratio", 0.15),
            stable_after_frames=cfg.get("stable_after_frames", 100),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _freeze(self) -> None:
        """Copy current K-means centres into the frozen array."""
        self._frozen_centers = self._kmeans.cluster_centers_.copy()

    def _nearest_centroid(self, X: np.ndarray) -> np.ndarray:
        """Assign each colour vector in *X* to its nearest frozen centre.

        Parameters
        ----------
        X : np.ndarray  shape (N, D)

        Returns
        -------
        np.ndarray  shape (N,)  integer labels 0 … n_clusters-1
        """
        # Squared L2 distance: (N, K)
        diff = X[:, np.newaxis, :] - self._frozen_centers[np.newaxis, :, :]  # (N,K,D)
        sq_dist = (diff ** 2).sum(axis=2)  # (N, K)
        return sq_dist.argmin(axis=1).astype(int)

    def _extract_torso(
        self, frame: np.ndarray, xyxy: np.ndarray
    ) -> Optional[np.ndarray]:
        """Return a BGR crop of the torso region, or ``None`` if too small."""
        x1, y1, x2, y2 = map(int, xyxy)
        h = y2 - y1
        top = y1 + int(h * self.crop_top_ratio)
        bottom = y2 - int(h * self.crop_bottom_ratio)
        if bottom <= top or x2 <= x1:
            return None
        crop = frame[top:bottom, x1:x2]
        return crop if crop.size > 0 else None

    def _mean_colour(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Convert *crop_bgr* and return its mean colour vector."""
        converted = cv2.cvtColor(crop_bgr, self._CONVERSIONS[self.color_space])
        return converted.reshape(-1, 3).mean(axis=0)


# ---------------------------------------------------------------------------
# Minimal stub to satisfy is_fitted() after load_centers()
# ---------------------------------------------------------------------------


class _DummyKMeans:
    """Minimal sklearn-compatible stub used after :meth:`load_centers`."""

    def __init__(self, centers: np.ndarray) -> None:
        self.cluster_centers_ = centers

    def predict(self, X: np.ndarray) -> np.ndarray:
        diff = X[:, np.newaxis, :] - self.cluster_centers_[np.newaxis, :, :]
        return (diff ** 2).sum(axis=2).argmin(axis=1).astype(int)
