"""Configuration dataclasses for the CV pipeline.

All thresholds live here so they can be tuned on real recordings without touching
analyzer code. Defaults were chosen from a small number of still images and are
expected to need adjustment once real webcam sessions are available.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# <repo>/cv_module/models is the default model directory (gitignored).
_DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "models"

MODEL_SOURCES: dict[str, str] = {
    # MediaPipe Face Landmarker (478 landmarks incl. iris, blendshapes, transformation matrix).
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task"
    ),
    # MediaPipe Pose Landmarker, lite variant (33 body landmarks).
    "pose_landmarker_lite.task": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    ),
    # HSEmotion EfficientNet-B0, 8 AffectNet classes, ONNX export (Apache-2.0 code; AffectNet weights).
    "enet_b0_8_best_vgaf.onnx": (
        "https://github.com/HSE-asavchenko/face-emotion-recognition/raw/main/"
        "models/affectnet_emotions/onnx/enet_b0_8_best_vgaf.onnx"
    ),
}

HSEMOTION_LABELS: tuple[str, ...] = (
    "Anger",
    "Contempt",
    "Disgust",
    "Fear",
    "Happiness",
    "Neutral",
    "Sadness",
    "Surprise",
)

# Broad, cautious grouping used for suggestions. Fine-grained negative classes are
# unreliable on webcam footage, so they are only ever reported as "other".
EXPRESSION_GROUPS: dict[str, str] = {
    "Neutral": "neutral",
    "Happiness": "positive",
    "Surprise": "other",
    "Anger": "other",
    "Contempt": "other",
    "Disgust": "other",
    "Fear": "other",
    "Sadness": "other",
}


def default_model_dir() -> Path:
    """Model directory: ``$INTERVIEW_CV_MODEL_DIR`` if set, else ``<repo>/cv_module/models``."""
    env = os.environ.get("INTERVIEW_CV_MODEL_DIR")
    return Path(env) if env else _DEFAULT_MODEL_DIR


@dataclass
class ModelPaths:
    model_dir: Path = field(default_factory=default_model_dir)
    face_landmarker: str = "face_landmarker.task"
    pose_landmarker: str = "pose_landmarker_lite.task"
    expression_onnx: str = "enet_b0_8_best_vgaf.onnx"

    def path(self, name: str) -> Path:
        return Path(self.model_dir) / name

    @property
    def face_landmarker_path(self) -> Path:
        return self.path(self.face_landmarker)

    @property
    def pose_landmarker_path(self) -> Path:
        return self.path(self.pose_landmarker)

    @property
    def expression_onnx_path(self) -> Path:
        return self.path(self.expression_onnx)

    def missing(self) -> list[Path]:
        return [p for p in (self.face_landmarker_path, self.pose_landmarker_path, self.expression_onnx_path) if not p.is_file()]


@dataclass
class FaceConfig:
    min_detection_confidence: float = 0.5
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    # Keep per-frame landmark arrays in memory (needed for visualization; not written to the log).
    output_blendshapes: bool = False


@dataclass
class GazeConfig:
    """Approximate "camera-facing vs looking away" classification.

    Head angles come from MediaPipe's facial transformation matrix. Iris offsets
    (0 = inner/left edge, 1 = outer/right edge of the eye opening) are converted to
    an approximate angle with ``iris_gain_deg`` and added to the head angle so that
    a head turn compensated by the eyes still counts as camera-facing.
    """

    yaw_threshold_deg: float = 20.0
    pitch_up_threshold_deg: float = 15.0
    pitch_down_threshold_deg: float = 20.0
    iris_x_center: float = 0.5
    iris_y_center: float = 0.45
    iris_gain_deg: float = 60.0
    # Eye aspect ratio (opening / width) below which the eye is treated as closed and
    # the vertical iris signal is ignored.
    blink_aspect_ratio: float = 0.12


@dataclass
class ExpressionConfig:
    # Relative margin added around the landmark bounding box before cropping, and upward
    # extension (fraction of box height) so the crop includes the forehead like HSEmotion's
    # training crops. Chosen from two still images; validate on real recordings.
    crop_margin: float = 0.3
    forehead_extension: float = 0.25
    input_size: int = 224
    labels: tuple[str, ...] = HSEMOTION_LABELS
    providers: tuple[str, ...] = ("CPUExecutionProvider",)


@dataclass
class PostureConfig:
    visibility_threshold: float = 0.5
    shoulder_tilt_threshold_deg: float = 8.0
    # Head offset from shoulder midpoint, as a fraction of shoulder width.
    head_offset_threshold: float = 0.15
    # Relative drop of head height (nose above shoulder line / shoulder width) vs. baseline.
    head_drop_threshold: float = 0.15
    # Relative change of apparent shoulder width vs. baseline (closer / further from camera).
    distance_change_threshold: float = 0.25
    # Seconds of visible posture used to build the per-session baseline.
    baseline_seconds: float = 5.0
    min_detection_confidence: float = 0.5
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


@dataclass
class ReportConfig:
    # Minimum duration (s) for a contiguous run of a state to count as an "episode".
    min_episode_seconds: float = 2.0
    # Re-classify gaze relative to the session's median head direction, to cancel a
    # constant bias from camera placement (laptop camera below eye level, etc.).
    # The correction is capped so a session spent mostly looking away is not "normalised".
    recenter_gaze: bool = True
    max_recenter_deg: float = 15.0
    gaze: GazeConfig = field(default_factory=GazeConfig)
    face_presence_warn_below: float = 0.9
    camera_focus_warn_below: float = 0.6
    camera_focus_strength_above: float = 0.8
    long_look_away_seconds: float = 5.0
    posture_flag_warn_above: float = 0.3
    upper_body_visible_warn_below: float = 0.7
    positive_expression_low_below: float = 0.05
    other_expression_high_above: float = 0.4


@dataclass
class PipelineConfig:
    sample_fps: float = 5.0
    enable_face: bool = True
    enable_expression: bool = True
    enable_gaze: bool = True
    enable_posture: bool = True
    # Stop after this many seconds of source time (None = whole video / until webcam stop).
    max_seconds: float | None = None
    models: ModelPaths = field(default_factory=ModelPaths)
    face: FaceConfig = field(default_factory=FaceConfig)
    gaze: GazeConfig = field(default_factory=GazeConfig)
    expression: ExpressionConfig = field(default_factory=ExpressionConfig)
    posture: PostureConfig = field(default_factory=PostureConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
