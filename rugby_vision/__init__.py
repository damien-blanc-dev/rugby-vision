"""
RugbyVision — Computer vision pipeline for rugby player tracking and tactical analysis.

Modules
-------
detector       : YOLOv8 player detection wrapper
tracker        : ByteTrack multi-object tracking via supervision
team_classifier: K-means HSV jersey-colour team separation
homography     : Field calibration and 2-D projection
visualizer     : Annotated video output + tactical minimap
metrics        : Heatmaps, distances, positional statistics
reidentifier   : Post-occlusion player re-ID after rucks/scrums
analytics      : Match intelligence report (distance, ruck time, proximity matrix)
"""

from rugby_vision.detector import PlayerDetector
from rugby_vision.tracker import PlayerTracker
from rugby_vision.team_classifier import TeamClassifier
from rugby_vision.homography import FieldHomography
from rugby_vision.visualizer import Visualizer
from rugby_vision.metrics import MetricsCollector
from rugby_vision.reidentifier import OcclusionReidentifier
from rugby_vision.analytics import MatchAnalyzer

__all__ = [
    "PlayerDetector",
    "PlayerTracker",
    "TeamClassifier",
    "FieldHomography",
    "Visualizer",
    "MetricsCollector",
    "OcclusionReidentifier",
    "MatchAnalyzer",
]

__version__ = "0.1.0"
