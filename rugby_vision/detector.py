"""YOLOv8 player detection wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from ultralytics import YOLO
import supervision as sv


class PlayerDetector:
    """Detect players in a video frame using a YOLOv8 model.

    Parameters
    ----------
    weights : str
        Path to a local ``.pt`` file or a Roboflow/HuggingFace model ID.
    confidence : float
        Minimum detection confidence to keep a box.
    iou : float
        NMS IoU threshold.
    device : str
        Inference device: ``"cuda"``, ``"cpu"``, or ``""`` (auto-select).
    classes : list[int] or None
        COCO class IDs to retain.  ``None`` keeps all classes.  Ignored when
        using a custom single-class model.

    Examples
    --------
    >>> detector = PlayerDetector("yolov8n.pt", confidence=0.35)
    >>> detections = detector.detect(frame)
    """

    def __init__(
        self,
        weights: str = "yolov8n.pt",
        confidence: float = 0.35,
        iou: float = 0.45,
        device: str = "",
        classes: Optional[list[int]] = None,
    ) -> None:
        self.confidence = confidence
        self.iou = iou
        self.device = device or None
        self.classes = classes
        self._model = YOLO(weights)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """Run inference on a single BGR frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image as returned by ``cv2.VideoCapture.read()``.

        Returns
        -------
        sv.Detections
            supervision Detections object (xyxy, confidence, class_id).
        """
        results = self._model.predict(
            source=frame,
            conf=self.confidence,
            iou=self.iou,
            device=self.device,
            classes=self.classes,
            verbose=False,
        )[0]
        return sv.Detections.from_ultralytics(results)

    @classmethod
    def from_config(cls, cfg: dict) -> "PlayerDetector":
        """Instantiate from a config dict (``config.yaml`` → ``model`` section).

        Parameters
        ----------
        cfg : dict
            Dictionary with keys: ``weights``, ``confidence``, ``iou``,
            ``device``, ``classes``.

        Returns
        -------
        PlayerDetector
        """
        return cls(
            weights=cfg.get("weights", "yolov8n.pt"),
            confidence=cfg.get("confidence", 0.35),
            iou=cfg.get("iou", 0.45),
            device=cfg.get("device", ""),
            classes=cfg.get("classes") or None,
        )
