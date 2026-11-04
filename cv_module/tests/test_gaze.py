import math

import numpy as np
import pytest

from interview_cv.analyzers.face import head_pose_from_matrix, landmarks_bbox
from interview_cv.analyzers.gaze import EYE_A, EYE_B, IRIS_LANDMARK_COUNT, classify_angles, estimate_gaze
from interview_cv.config import GazeConfig
from interview_cv.schemas import HeadPose


def _rot(axis: str, deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    m = np.eye(4)
    if axis == "x":
        m[:3, :3] = [[1, 0, 0], [0, c, -s], [0, s, c]]
    elif axis == "y":
        m[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    else:
        m[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    return m


def test_head_pose_identity():
    hp = head_pose_from_matrix(np.eye(4))
    assert (hp.yaw_deg, hp.pitch_deg, hp.roll_deg) == (0.0, 0.0, 0.0)


def test_head_pose_single_axis_rotations():
    assert head_pose_from_matrix(_rot("x", 15)).pitch_deg == pytest.approx(15, abs=1e-6)
    # Sign conventions were fixed empirically against MediaPipe output (see face.py docstring).
    assert abs(head_pose_from_matrix(_rot("y", 25)).yaw_deg) == pytest.approx(25, abs=1e-6)
    assert abs(head_pose_from_matrix(_rot("z", 10)).roll_deg) == pytest.approx(10, abs=1e-6)
    # Flipping the image mirrors yaw and roll.
    hp = head_pose_from_matrix(_rot("y", 25))
    hp_m = head_pose_from_matrix(_rot("y", -25))
    assert hp.yaw_deg == pytest.approx(-hp_m.yaw_deg)


def _synthetic_landmarks(iris_x_ratio: float = 0.5, iris_y_ratio: float = 0.45, opening: float = 0.03) -> np.ndarray:
    """478 landmarks where only the eye-related indices matter; others are placed on a face oval."""
    lm = np.zeros((IRIS_LANDMARK_COUNT, 3), dtype=np.float32)
    lm[:, 0] = 0.5
    lm[:, 1] = 0.5
    for eye, cx in ((EYE_A, 0.42), (EYE_B, 0.58)):
        width = 0.06
        x0, x1 = cx - width / 2, cx + width / 2
        top, bottom = 0.40 - opening / 2, 0.40 + opening / 2
        lm[eye["corner_outer"]] = [x0 if eye is EYE_A else x1, 0.40, 0]
        lm[eye["corner_inner"]] = [x1 if eye is EYE_A else x0, 0.40, 0]
        lm[eye["lid_top"]] = [cx, top, 0]
        lm[eye["lid_bottom"]] = [cx, bottom, 0]
        lm[eye["iris"]] = [x0 + iris_x_ratio * width, top + iris_y_ratio * opening, 0]
    return lm


def test_gaze_unknown_without_data():
    g = estimate_gaze(None, None)
    assert g.state == "unknown"
    g = estimate_gaze(np.zeros((10, 3)), None)
    assert g.state == "unknown"


def test_gaze_camera_when_centered():
    g = estimate_gaze(_synthetic_landmarks(), HeadPose(0, 0, 0))
    assert g.state == "camera" and g.direction == "center"
    assert g.iris_x == pytest.approx(0.5, abs=0.01)


def test_gaze_away_from_head_yaw():
    g = estimate_gaze(_synthetic_landmarks(), HeadPose(yaw_deg=35, pitch_deg=0, roll_deg=0))
    assert g.state == "away" and g.direction == "right"
    g = estimate_gaze(_synthetic_landmarks(), HeadPose(yaw_deg=-35, pitch_deg=0, roll_deg=0))
    assert g.direction == "left"


def test_gaze_away_from_head_pitch():
    g = estimate_gaze(_synthetic_landmarks(), HeadPose(0, pitch_deg=30, roll_deg=0))
    assert g.state == "away" and g.direction == "down"
    g = estimate_gaze(_synthetic_landmarks(), HeadPose(0, pitch_deg=-25, roll_deg=0))
    assert g.direction == "up"


def test_gaze_away_from_iris_only():
    g = estimate_gaze(_synthetic_landmarks(iris_x_ratio=0.1), HeadPose(0, 0, 0))
    assert g.state == "away" and g.direction == "left"
    g = estimate_gaze(_synthetic_landmarks(iris_x_ratio=0.9), HeadPose(0, 0, 0))
    assert g.direction == "right"


def test_eyes_compensate_head_turn():
    # Head turned 15 deg right but eyes shifted left -> effective angle small -> camera.
    g = estimate_gaze(_synthetic_landmarks(iris_x_ratio=0.3), HeadPose(yaw_deg=15, pitch_deg=0, roll_deg=0))
    assert g.state == "camera"
    assert abs(g.effective_yaw_deg) < 10


def test_blink_ignores_vertical_iris():
    cfg = GazeConfig()
    g = estimate_gaze(_synthetic_landmarks(iris_y_ratio=1.0, opening=0.002), HeadPose(0, 0, 0), cfg)
    assert g.state == "camera"


def test_classify_angles_thresholds():
    cfg = GazeConfig(yaw_threshold_deg=20, pitch_up_threshold_deg=15, pitch_down_threshold_deg=20)
    assert classify_angles(19.9, 0, cfg) == ("camera", "center")
    assert classify_angles(20.1, 0, cfg) == ("away", "right")
    assert classify_angles(0, -15.1, cfg) == ("away", "up")
    assert classify_angles(0, 20.1, cfg) == ("away", "down")
    # The dominant excess decides the direction.
    assert classify_angles(30, 25, cfg) == ("away", "right")
    assert classify_angles(21, 40, cfg) == ("away", "down")


def test_landmarks_bbox():
    lm = np.array([[0.2, 0.3, 0], [0.6, 0.9, 0]])
    bb = landmarks_bbox(lm)
    assert (bb.x, bb.y, bb.w, bb.h) == pytest.approx((0.2, 0.3, 0.4, 0.6))
    assert bb.to_pixels(100, 100) == (20, 30, 60, 90)
