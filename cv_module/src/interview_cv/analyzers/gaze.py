"""Approximate gaze estimation from face landmarks and head pose.

This is deliberately simple geometry, not eye tracking:

1. Head yaw / pitch come from the facial transformation matrix.
2. The iris centre position inside each eye opening (0..1 horizontally and
   vertically) is turned into an approximate eye angle with a fixed gain.
3. Head angle + eye angle = effective angle. If both effective angles are inside
   the thresholds the person is counted as "camera-facing", otherwise "away" with
   a coarse direction in *image* coordinates ("left" = toward the left of the image).

Pure numpy; unit-testable with synthetic landmarks.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..config import GazeConfig
from ..schemas import GazeObservation, HeadPose

# MediaPipe Face Mesh indices. "A" is the eye on the image's left, "B" on the image's right
# for a non-mirrored frame.
EYE_A = {"corner_outer": 33, "corner_inner": 133, "lid_top": 159, "lid_bottom": 145, "iris": 468}
EYE_B = {"corner_inner": 362, "corner_outer": 263, "lid_top": 386, "lid_bottom": 374, "iris": 473}
IRIS_LANDMARK_COUNT = 478


def _eye_ratios(landmarks: np.ndarray, eye: dict[str, int], aspect: float) -> Optional[tuple[float, float, float]]:
    """Return (iris_x_ratio, iris_y_ratio, eye_aspect_ratio) for one eye, or None if degenerate."""
    pts = landmarks[:, :2].astype(np.float64).copy()
    pts[:, 0] *= aspect  # make x/y comparable (image may not be square)
    c1 = pts[eye["corner_outer"]]
    c2 = pts[eye["corner_inner"]]
    top = pts[eye["lid_top"]]
    bottom = pts[eye["lid_bottom"]]
    iris = pts[eye["iris"]]
    x_min, x_max = min(c1[0], c2[0]), max(c1[0], c2[0])
    width = x_max - x_min
    opening = bottom[1] - top[1]
    if width <= 1e-6:
        return None
    rx = float((iris[0] - x_min) / width)
    ry = float((iris[1] - top[1]) / opening) if opening > 1e-6 else 0.5
    return rx, ry, float(opening / width)


def classify_angles(eff_yaw: float, eff_pitch: float, cfg: Optional[GazeConfig] = None) -> tuple[str, str]:
    """Map effective (head + eye) angles to ``(state, direction)`` using the configured thresholds.

    Shared by the per-frame analyzer and by the report's optional re-centering pass.
    """
    cfg = cfg or GazeConfig()
    horizontal_excess = abs(eff_yaw) - cfg.yaw_threshold_deg
    if eff_pitch >= 0:
        vertical_excess = eff_pitch - cfg.pitch_down_threshold_deg
    else:
        vertical_excess = -eff_pitch - cfg.pitch_up_threshold_deg

    if horizontal_excess <= 0 and vertical_excess <= 0:
        return "camera", "center"
    if horizontal_excess >= vertical_excess:
        return "away", ("right" if eff_yaw > 0 else "left")
    return "away", ("down" if eff_pitch > 0 else "up")


def estimate_gaze(
    landmarks: Optional[np.ndarray],
    head_pose: Optional[HeadPose],
    config: Optional[GazeConfig] = None,
    aspect: float = 1.0,
) -> GazeObservation:
    """Classify gaze as camera / away / unknown.

    ``landmarks``: (N, 2+) normalized face landmarks (N >= 478 for iris data), or None.
    ``aspect``: image width / height, used so horizontal and vertical ratios are not distorted.
    """
    cfg = config or GazeConfig()
    if head_pose is None and (landmarks is None or len(landmarks) < IRIS_LANDMARK_COUNT):
        return GazeObservation(state="unknown", direction="center")

    iris_x: Optional[float] = None
    iris_y: Optional[float] = None
    blinking = False
    if landmarks is not None and len(landmarks) >= IRIS_LANDMARK_COUNT:
        ratios = [r for r in (_eye_ratios(landmarks, EYE_A, aspect), _eye_ratios(landmarks, EYE_B, aspect)) if r]
        if ratios:
            iris_x = float(np.mean([r[0] for r in ratios]))
            iris_y = float(np.mean([r[1] for r in ratios]))
            blinking = all(r[2] < cfg.blink_aspect_ratio for r in ratios)

    head_yaw = head_pose.yaw_deg if head_pose else 0.0
    head_pitch = head_pose.pitch_deg if head_pose else 0.0

    eye_yaw = (iris_x - cfg.iris_x_center) * cfg.iris_gain_deg if iris_x is not None else 0.0
    eye_pitch = (iris_y - cfg.iris_y_center) * cfg.iris_gain_deg if (iris_y is not None and not blinking) else 0.0
    eff_yaw = head_yaw + eye_yaw
    eff_pitch = head_pitch + eye_pitch
    state, direction = classify_angles(eff_yaw, eff_pitch, cfg)

    return GazeObservation(
        state=state,
        direction=direction,
        head_yaw_deg=round(head_yaw, 2),
        head_pitch_deg=round(head_pitch, 2),
        iris_x=round(iris_x, 3) if iris_x is not None else None,
        iris_y=round(iris_y, 3) if iris_y is not None else None,
        effective_yaw_deg=round(eff_yaw, 2),
        effective_pitch_deg=round(eff_pitch, 2),
    )
