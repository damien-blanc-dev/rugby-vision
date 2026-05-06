"""Unit tests for FieldHomography."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from rugby_vision.homography import FieldHomography


# Canonical 4 corners (pixels) for a 640×480 image where the field
# occupies the inner rectangle.
SRC_CORNERS = np.array(
    [[80, 60], [560, 60], [560, 420], [80, 420]], dtype=np.float32
)

FIELD_W = 68.0  # metres
FIELD_L = 100.0  # metres


@pytest.fixture()
def calibrated_hom() -> FieldHomography:
    hom = FieldHomography(field_length_m=FIELD_L, field_width_m=FIELD_W)
    hom.calibrate_from_points(SRC_CORNERS)
    return hom


class TestCalibration:
    def test_is_calibrated_after_calibrate_from_points(self, calibrated_hom):
        assert calibrated_hom.is_calibrated()

    def test_not_calibrated_on_init(self):
        hom = FieldHomography()
        assert not hom.is_calibrated()

    def test_project_point_top_left_corner(self, calibrated_hom):
        """Top-left pixel corner should map to (0, 0) in field coords."""
        pt = calibrated_hom.project_point(SRC_CORNERS[0])
        np.testing.assert_allclose(pt, [0.0, 0.0], atol=1e-3)

    def test_project_point_top_right_corner(self, calibrated_hom):
        """Top-right pixel corner should map to (FIELD_W, 0)."""
        pt = calibrated_hom.project_point(SRC_CORNERS[1])
        np.testing.assert_allclose(pt, [FIELD_W, 0.0], atol=1e-3)

    def test_project_point_bottom_right_corner(self, calibrated_hom):
        pt = calibrated_hom.project_point(SRC_CORNERS[2])
        np.testing.assert_allclose(pt, [FIELD_W, FIELD_L], atol=1e-3)

    def test_project_point_bottom_left_corner(self, calibrated_hom):
        pt = calibrated_hom.project_point(SRC_CORNERS[3])
        np.testing.assert_allclose(pt, [0.0, FIELD_L], atol=1e-3)

    def test_project_point_centre(self, calibrated_hom):
        """Image centre should map close to field centre."""
        cx = (SRC_CORNERS[0, 0] + SRC_CORNERS[1, 0]) / 2
        cy = (SRC_CORNERS[0, 1] + SRC_CORNERS[2, 1]) / 2
        pt = calibrated_hom.project_point(np.array([cx, cy]))
        np.testing.assert_allclose(pt, [FIELD_W / 2, FIELD_L / 2], atol=1.0)


class TestProjectDetections:
    def _make_detections(self, xyxy_list):
        """Build a minimal object with a .xyxy attribute."""
        import types
        dets = types.SimpleNamespace()
        dets.xyxy = np.array(xyxy_list, dtype=np.float32)
        return dets

    def test_empty_detections(self, calibrated_hom):
        dets = self._make_detections([])
        result = calibrated_hom.project_detections(dets)
        assert result.shape == (0, 2)

    def test_single_detection_foot_maps_to_field(self, calibrated_hom):
        # Bottom-centre of a box whose foot is at the TL pixel corner
        x1, y1, x2, y2 = SRC_CORNERS[0, 0] - 10, SRC_CORNERS[0, 1], \
                          SRC_CORNERS[0, 0] + 10, SRC_CORNERS[0, 1]
        dets = self._make_detections([[x1, y1, x2, y2]])
        result = calibrated_hom.project_detections(dets)
        assert result.shape == (1, 2)
        # foot midpoint x is SRC_CORNERS[0,0], y is SRC_CORNERS[0,1]
        np.testing.assert_allclose(result[0], [0.0, 0.0], atol=0.5)

    def test_multiple_detections(self, calibrated_hom):
        # Two boxes at TL and TR corners
        boxes = [
            [SRC_CORNERS[0, 0] - 5, SRC_CORNERS[0, 1], SRC_CORNERS[0, 0] + 5, SRC_CORNERS[0, 1]],
            [SRC_CORNERS[1, 0] - 5, SRC_CORNERS[1, 1], SRC_CORNERS[1, 0] + 5, SRC_CORNERS[1, 1]],
        ]
        dets = self._make_detections(boxes)
        result = calibrated_hom.project_detections(dets)
        assert result.shape == (2, 2)


class TestPersistence:
    def test_save_and_load(self, calibrated_hom):
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            calibrated_hom.save(path)
            hom2 = FieldHomography()
            hom2.load(path)
            assert hom2.is_calibrated()
            np.testing.assert_array_almost_equal(hom2._H, calibrated_hom._H)
            assert hom2.field_length_m == calibrated_hom.field_length_m
            assert hom2.field_width_m == calibrated_hom.field_width_m
        finally:
            Path(path).unlink(missing_ok=True)

    def test_save_without_calibration_raises(self):
        hom = FieldHomography()
        with pytest.raises(RuntimeError, match="No homography"):
            hom.save("nowhere.npz")

    def test_project_without_calibration_raises(self):
        hom = FieldHomography()
        with pytest.raises(RuntimeError, match="not computed"):
            hom.project_point(np.array([0.0, 0.0]))

    def test_load_on_init(self, calibrated_hom):
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            calibrated_hom.save(path)
            hom2 = FieldHomography(calibration_path=path)
            assert hom2.is_calibrated()
        finally:
            Path(path).unlink(missing_ok=True)
