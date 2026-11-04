"""Upper-body posture observations via MediaPipe Pose Landmarker.

A single front-facing webcam cannot measure spine angle, so this module only
derives simple, observable quantities from the shoulders and nose:

* ``shoulder_tilt_deg``   - angle of the shoulder line vs. horizontal.
* ``head_offset_ratio``   - nose x-offset from the shoulder midpoint / shoulder width (+ = image right).
* ``head_height_ratio``   - nose height above the shoulder line / shoulder width.
* ``shoulder_width_norm`` - shoulder width / image width (proxy for distance to the camera).

Flags are raised either against fixed thresholds (tilt, offset) or against a
per-session baseline built from the first few seconds (head lowered, moved
closer / further). ``derive_posture`` is a pure function for unit tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ..config import PostureConfig
from ..schemas import PostureObservation
from .base import FrameContext, MonotonicTimestamps

# MediaPipe Pose landmark indices.
NOSE = 0
LEFT_SHOULDER = 11  # subject's left (image right in a non-mirrored frame)
RIGHT_SHOULDER = 12
LEFT_EAR = 7
RIGHT_EAR = 8
UPPER_BODY_INDICES = (NOSE, LEFT_EAR, RIGHT_EAR, LEFT_SHOULDER, RIGHT_SHOULDER)

FLAG_SHOULDERS_TILTED = "shoulders_tilted"
FLAG_HEAD_OFF_CENTER = "head_off_center"
FLAG_HEAD_LOWERED = "head_lowered"
FLAG_MOVED_CLOSER = "moved_closer"
FLAG_MOVED_AWAY = "moved_away"
ALL_FLAGS = (FLAG_SHOULDERS_TILTED, FLAG_HEAD_OFF_CENTER, FLAG_HEAD_LOWERED, FLAG_MOVED_CLOSER, FLAG_MOVED_AWAY)


@dataclass
class PostureBaseline:
    """Median of early-session measurements, used for relative flags."""

    seconds: float = 5.0
    head_height_samples: list[float] = field(default_factory=list)
    shoulder_width_samples: list[float] = field(default_factory=list)
    first_timestamp: Optional[float] = None
    head_height_ratio: Optional[float] = None
    shoulder_width_norm: Optional[float] = None

    @property
    def ready(self) -> bool:
        return self.head_height_ratio is not None

    def update(self, timestamp_s: float, head_height_ratio: float, shoulder_width_norm: float) -> None:
        if self.ready:
            return
        if self.first_timestamp is None:
            self.first_timestamp = timestamp_s
        self.head_height_samples.append(head_height_ratio)
        self.shoulder_width_samples.append(shoulder_width_norm)
        if timestamp_s - self.first_timestamp >= self.seconds and len(self.head_height_samples) >= 3:
            self.finalize()

    def finalize(self) -> None:
        if self.head_height_samples:
            self.head_height_ratio = float(np.median(self.head_height_samples))
            self.shoulder_width_norm = float(np.median(self.shoulder_width_samples))


def derive_posture(
    landmarks: Optional[np.ndarray],
    visibility: Optional[np.ndarray],
    image_width: int,
    image_height: int,
    config: Optional[PostureConfig] = None,
    baseline: Optional[PostureBaseline] = None,
) -> PostureObservation:
    """``landmarks``: (33, 2+) normalized pose landmarks; ``visibility``: (33,) scores. Either may be None."""
    cfg = config or PostureConfig()
    if landmarks is None or len(landmarks) <= LEFT_SHOULDER + 1:
        return PostureObservation(upper_body_visible=False)
    if visibility is not None:
        needed = [NOSE, LEFT_SHOULDER, RIGHT_SHOULDER]
        if any(float(visibility[i]) < cfg.visibility_threshold for i in needed):
            return PostureObservation(upper_body_visible=False)

    px = landmarks[:, 0].astype(np.float64) * image_width
    py = landmarks[:, 1].astype(np.float64) * image_height
    ls = np.array([px[LEFT_SHOULDER], py[LEFT_SHOULDER]])
    rs = np.array([px[RIGHT_SHOULDER], py[RIGHT_SHOULDER]])
    nose = np.array([px[NOSE], py[NOSE]])

    # Order shoulders by image x so the tilt sign is "right side lower = positive".
    left_pt, right_pt = (ls, rs) if ls[0] <= rs[0] else (rs, ls)
    dx = right_pt[0] - left_pt[0]
    dy = right_pt[1] - left_pt[1]
    shoulder_width = math.hypot(dx, dy)
    if shoulder_width < 1e-6:
        return PostureObservation(upper_body_visible=False)
    tilt = math.degrees(math.atan2(dy, dx))
    mid = (left_pt + right_pt) / 2.0
    head_offset = (nose[0] - mid[0]) / shoulder_width
    head_height = (mid[1] - nose[1]) / shoulder_width
    width_norm = shoulder_width / float(image_width)

    flags: list[str] = []
    if abs(tilt) > cfg.shoulder_tilt_threshold_deg:
        flags.append(FLAG_SHOULDERS_TILTED)
    if abs(head_offset) > cfg.head_offset_threshold:
        flags.append(FLAG_HEAD_OFF_CENTER)
    baseline_ready = bool(baseline and baseline.ready)
    if baseline_ready:
        assert baseline is not None and baseline.head_height_ratio is not None and baseline.shoulder_width_norm is not None
        if head_height < baseline.head_height_ratio * (1.0 - cfg.head_drop_threshold):
            flags.append(FLAG_HEAD_LOWERED)
        if width_norm > baseline.shoulder_width_norm * (1.0 + cfg.distance_change_threshold):
            flags.append(FLAG_MOVED_CLOSER)
        elif width_norm < baseline.shoulder_width_norm * (1.0 - cfg.distance_change_threshold):
            flags.append(FLAG_MOVED_AWAY)

    return PostureObservation(
        upper_body_visible=True,
        shoulder_tilt_deg=round(tilt, 2),
        head_offset_ratio=round(head_offset, 3),
        head_height_ratio=round(head_height, 3),
        shoulder_width_norm=round(width_norm, 4),
        baseline_ready=baseline_ready,
        flags=flags,
    )


class PostureAnalyzer:
    def __init__(self, model_path: str | Path, config: Optional[PostureConfig] = None):
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        self.config = config or PostureConfig()
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Pose landmarker model not found: {model_path}")
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=self.config.min_detection_confidence,
            min_pose_presence_confidence=self.config.min_presence_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._clock = MonotonicTimestamps()
        self.baseline = PostureBaseline(seconds=self.config.baseline_seconds)
        self.last_landmarks: Optional[np.ndarray] = None

    def process(self, ctx: FrameContext) -> PostureObservation:
        result = self._landmarker.detect_for_video(ctx.mp_image, self._clock.next(ctx.timestamp_ms))
        if not result.pose_landmarks:
            self.last_landmarks = None
            return PostureObservation(upper_body_visible=False)
        lm = result.pose_landmarks[0]
        landmarks = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
        visibility = np.array([p.visibility if p.visibility is not None else 1.0 for p in lm], dtype=np.float32)
        self.last_landmarks = landmarks
        obs = derive_posture(landmarks, visibility, ctx.width, ctx.height, self.config, self.baseline)
        if obs.upper_body_visible and obs.head_height_ratio is not None and obs.shoulder_width_norm is not None:
            self.baseline.update(ctx.timestamp_s, obs.head_height_ratio, obs.shoulder_width_norm)
        return obs

    def close(self) -> None:
        self._landmarker.close()
