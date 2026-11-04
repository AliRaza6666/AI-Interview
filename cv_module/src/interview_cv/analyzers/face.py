"""Face presence, landmarks and head pose via MediaPipe Face Landmarker.

The analyzer returns a :class:`FaceResult` that carries both the serializable
:class:`FaceObservation` and the raw landmark array (478 x 3, normalized), which
the gaze and expression analyzers consume but which is not written to the log.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ..config import FaceConfig
from ..schemas import BBox, FaceObservation, HeadPose
from .base import FrameContext, MonotonicTimestamps


@dataclass
class FaceResult:
    observation: FaceObservation
    landmarks: Optional[np.ndarray] = None  # (478, 3) normalized x, y, z
    transform: Optional[np.ndarray] = None  # 4x4 facial transformation matrix
    blendshapes: dict[str, float] = field(default_factory=dict)


def head_pose_from_matrix(matrix: np.ndarray) -> HeadPose:
    """Extract yaw / pitch / roll (degrees) from MediaPipe's 4x4 facial transformation matrix.

    Conventions (verified by horizontally flipping test images: yaw and roll flip sign,
    pitch does not):

    * yaw   > 0 : face turned toward the image's right side
    * pitch > 0 : chin down (looking toward the bottom of the image)
    * roll  > 0 : head tilted so the top of the head moves toward the image's right side
    """
    r = np.asarray(matrix, dtype=np.float64)[:3, :3]
    sy = math.hypot(r[0, 0], r[1, 0])
    if sy < 1e-6:  # gimbal lock, roll is undefined
        x = math.atan2(-r[1, 2], r[1, 1])
        y = math.atan2(-r[2, 0], sy)
        z = 0.0
    else:
        x = math.atan2(r[2, 1], r[2, 2])
        y = math.atan2(-r[2, 0], sy)
        z = math.atan2(r[1, 0], r[0, 0])
    pitch = math.degrees(x)
    yaw = math.degrees(y)
    roll = -math.degrees(z)
    return HeadPose(yaw_deg=round(yaw, 3), pitch_deg=round(pitch, 3), roll_deg=round(roll, 3))


def landmarks_bbox(landmarks: np.ndarray) -> BBox:
    xs = landmarks[:, 0]
    ys = landmarks[:, 1]
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    return BBox(x=round(x0, 4), y=round(y0, 4), w=round(x1 - x0, 4), h=round(y1 - y0, 4))


class FaceAnalyzer:
    def __init__(self, model_path: str | Path, config: Optional[FaceConfig] = None):
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        self.config = config or FaceConfig()
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Face landmarker model not found: {model_path}")
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=self.config.min_detection_confidence,
            min_face_presence_confidence=self.config.min_presence_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
            output_face_blendshapes=self.config.output_blendshapes,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._clock = MonotonicTimestamps()

    def process(self, ctx: FrameContext) -> FaceResult:
        result = self._landmarker.detect_for_video(ctx.mp_image, self._clock.next(ctx.timestamp_ms))
        if not result.face_landmarks:
            return FaceResult(observation=FaceObservation(present=False))
        lm = result.face_landmarks[0]
        landmarks = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
        transform = None
        head_pose = None
        if result.facial_transformation_matrixes:
            transform = np.asarray(result.facial_transformation_matrixes[0], dtype=np.float64)
            head_pose = head_pose_from_matrix(transform)
        blendshapes = {}
        if result.face_blendshapes:
            blendshapes = {c.category_name: float(c.score) for c in result.face_blendshapes[0]}
        obs = FaceObservation(present=True, bbox=landmarks_bbox(landmarks), head_pose=head_pose)
        return FaceResult(observation=obs, landmarks=landmarks, transform=transform, blendshapes=blendshapes)

    def close(self) -> None:
        self._landmarker.close()
