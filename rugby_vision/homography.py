"""Field calibration and homographic projection to a 2-D top-down view."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# Real-world corners of a standard rugby pitch (top-left origin, metres)
#   TL=(0,0)  TR=(W,0)  BR=(W,L)  BL=(0,L)
_DEFAULT_FIELD_CORNERS_M = np.array(
    [[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32
)  # normalised — scaled later


class FieldHomography:
    """Compute and apply a homography from camera view to a 2-D field plane.

    Usage
    -----
    Calibration can be done in two ways:

    1. **Interactive** — call :meth:`calibrate_interactive` with the first
       video frame.  The user clicks 4 field corners in the displayed window.
    2. **From file** — call :meth:`load` with a previously saved ``.npz``.

    Once calibrated, call :meth:`project_point` or :meth:`project_detections`
    to map image coordinates to field coordinates.

    Parameters
    ----------
    field_length_m : float
        Real field length in metres (touch-line to touch-line).
    field_width_m : float
        Real field width in metres (try-line to try-line).
    calibration_path : str or Path or None
        If provided, attempt to load a saved calibration on construction.

    Examples
    --------
    >>> hom = FieldHomography(field_length_m=100, field_width_m=68)
    >>> hom.calibrate_interactive(first_frame)
    >>> field_pts = hom.project_detections(detections)
    """

    # Corner indices for click order: TL, TR, BR, BL
    _CORNER_NAMES = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]

    def __init__(
        self,
        field_length_m: float = 100.0,
        field_width_m: float = 68.0,
        calibration_path: Optional[str | Path] = None,
    ) -> None:
        self.field_length_m = field_length_m
        self.field_width_m = field_width_m
        self._H: Optional[np.ndarray] = None  # 3×3 homography matrix
        self._dst_corners: Optional[np.ndarray] = None

        if calibration_path and Path(calibration_path).exists():
            self.load(calibration_path)

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate_interactive(self, frame: np.ndarray) -> None:
        """Open an OpenCV window and let the user click the 4 field corners.

        Click order: **Top-Left → Top-Right → Bottom-Right → Bottom-Left**.
        Press ``r`` to reset clicks, ``q`` to abort.

        Parameters
        ----------
        frame : np.ndarray
            BGR image (first video frame is recommended).

        Raises
        ------
        RuntimeError
            If the user closes the window before selecting 4 points.
        """
        src_pts: list[list[float]] = []
        clone = frame.copy()

        def _on_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(src_pts) < 4:
                src_pts.append([x, y])
                idx = len(src_pts) - 1
                cv2.circle(clone, (x, y), 6, (0, 255, 0), -1)
                cv2.putText(
                    clone,
                    self._CORNER_NAMES[idx],
                    (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("Calibration — click 4 corners", clone)

        win = "Calibration — click 4 corners"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win, _on_click)

        instructions = (
            "Click: TL → TR → BR → BL  |  r = reset  |  Enter = confirm  |  q = quit"
        )
        cv2.putText(
            clone, instructions, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
        )
        cv2.imshow(win, clone)

        while True:
            key = cv2.waitKey(20) & 0xFF
            if key == ord("r"):
                src_pts.clear()
                clone = frame.copy()
                cv2.putText(
                    clone, instructions, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
                )
                cv2.imshow(win, clone)
            elif key in (13, ord("\r")) and len(src_pts) == 4:  # Enter
                break
            elif key == ord("q"):
                cv2.destroyWindow(win)
                raise RuntimeError("Calibration aborted by user.")
            elif cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                raise RuntimeError("Calibration window closed before 4 points were selected.")

        cv2.destroyWindow(win)
        self._compute_homography(np.array(src_pts, dtype=np.float32))

    def calibrate_from_points(self, src_pts: np.ndarray) -> None:
        """Set the homography from pre-selected source points.

        Parameters
        ----------
        src_pts : np.ndarray
            Shape ``(4, 2)`` — pixel coordinates of TL, TR, BR, BL corners.
        """
        self._compute_homography(src_pts.astype(np.float32))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save the homography matrix to a ``.npz`` file.

        Parameters
        ----------
        path : str or Path
            Destination file path.

        Raises
        ------
        RuntimeError
            If the homography has not been computed yet.
        """
        if self._H is None:
            raise RuntimeError("No homography to save — call calibrate first.")
        np.savez(
            path,
            H=self._H,
            dst_corners=self._dst_corners,
            field_dims=np.array([self.field_length_m, self.field_width_m]),
        )

    def load(self, path: str | Path) -> None:
        """Load a previously saved homography from a ``.npz`` file.

        Parameters
        ----------
        path : str or Path
            Source file path.
        """
        data = np.load(path)
        self._H = data["H"]
        self._dst_corners = data["dst_corners"]
        dims = data["field_dims"]
        self.field_length_m, self.field_width_m = float(dims[0]), float(dims[1])

    # ------------------------------------------------------------------
    # Projection
    # ------------------------------------------------------------------

    def project_point(self, pt: np.ndarray) -> np.ndarray:
        """Project a single image point to field coordinates (metres).

        Parameters
        ----------
        pt : np.ndarray
            Shape ``(2,)`` — pixel (x, y).

        Returns
        -------
        np.ndarray
            Shape ``(2,)`` — field coordinates (x_m, y_m).

        Raises
        ------
        RuntimeError
            If the homography has not been computed yet.
        """
        if self._H is None:
            raise RuntimeError("Homography not computed — call calibrate first.")
        pt_h = np.array([[pt]], dtype=np.float32)
        projected = cv2.perspectiveTransform(pt_h, self._H)
        return projected[0, 0]

    def project_detections(self, detections) -> np.ndarray:
        """Project bottom-centre of each bounding box to field coordinates.

        Parameters
        ----------
        detections : sv.Detections
            Detections with ``xyxy`` attribute.

        Returns
        -------
        np.ndarray
            Shape ``(N, 2)`` — field (x_m, y_m) for each detection.
        """
        if self._H is None:
            raise RuntimeError("Homography not computed — call calibrate first.")

        pts = []
        for xyxy in detections.xyxy:
            x1, y1, x2, y2 = xyxy
            foot = np.array([[(x1 + x2) / 2, y2]], dtype=np.float32)
            pts.append(foot)

        if not pts:
            return np.empty((0, 2), dtype=np.float32)

        src = np.array(pts, dtype=np.float32)  # (N, 1, 2)
        dst = cv2.perspectiveTransform(src, self._H)  # (N, 1, 2)
        return dst[:, 0, :]

    def is_calibrated(self) -> bool:
        """Return ``True`` if a valid homography matrix is available."""
        return self._H is not None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_homography(self, src_pts: np.ndarray) -> None:
        """Compute homography from *src_pts* (pixel) to field coords (metres)."""
        W, L = self.field_width_m, self.field_length_m
        # Destination: TL, TR, BR, BL in metres
        dst_pts = np.array(
            [[0, 0], [W, 0], [W, L], [0, L]], dtype=np.float32
        )
        self._H, _ = cv2.findHomography(src_pts, dst_pts)
        self._dst_corners = dst_pts
