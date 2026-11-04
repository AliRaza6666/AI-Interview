import numpy as np
import pytest

from interview_cv.analyzers.posture import (
    FLAG_HEAD_LOWERED,
    FLAG_HEAD_OFF_CENTER,
    FLAG_MOVED_CLOSER,
    FLAG_SHOULDERS_TILTED,
    LEFT_SHOULDER,
    NOSE,
    RIGHT_SHOULDER,
    PostureBaseline,
    derive_posture,
)
from interview_cv.config import PostureConfig

W, H = 640, 480


def _pose(nose=(0.5, 0.3), left=(0.7, 0.6), right=(0.3, 0.6), vis=1.0):
    lm = np.full((33, 3), 0.5, dtype=np.float32)
    lm[NOSE, :2] = nose
    lm[LEFT_SHOULDER, :2] = left
    lm[RIGHT_SHOULDER, :2] = right
    visibility = np.full(33, vis, dtype=np.float32)
    return lm, visibility


def test_missing_landmarks():
    assert derive_posture(None, None, W, H).upper_body_visible is False
    assert derive_posture(np.zeros((5, 3)), None, W, H).upper_body_visible is False


def test_low_visibility_is_not_visible():
    lm, vis = _pose()
    vis[LEFT_SHOULDER] = 0.2
    assert derive_posture(lm, vis, W, H).upper_body_visible is False


def test_level_centered_posture_has_no_flags():
    lm, vis = _pose()
    obs = derive_posture(lm, vis, W, H)
    assert obs.upper_body_visible
    assert obs.flags == []
    assert obs.shoulder_tilt_deg == pytest.approx(0.0)
    assert obs.head_offset_ratio == pytest.approx(0.0)
    assert obs.head_height_ratio > 0
    assert obs.shoulder_width_norm == pytest.approx(0.4, abs=1e-3)
    assert obs.baseline_ready is False


def test_tilted_shoulders_flag():
    lm, vis = _pose(left=(0.7, 0.66), right=(0.3, 0.54))  # right side (image) lower
    obs = derive_posture(lm, vis, W, H)
    assert FLAG_SHOULDERS_TILTED in obs.flags
    assert obs.shoulder_tilt_deg > 8


def test_head_off_center_flag():
    lm, vis = _pose(nose=(0.62, 0.3))
    obs = derive_posture(lm, vis, W, H)
    assert FLAG_HEAD_OFF_CENTER in obs.flags
    assert obs.head_offset_ratio > 0.15


def test_baseline_relative_flags():
    cfg = PostureConfig(baseline_seconds=1.0)
    baseline = PostureBaseline(seconds=cfg.baseline_seconds)
    lm, vis = _pose()
    # Build the baseline from identical frames over > 1 s.
    for t in (0.0, 0.5, 1.0, 1.5):
        obs = derive_posture(lm, vis, W, H, cfg, baseline)
        baseline.update(t, obs.head_height_ratio, obs.shoulder_width_norm)
    assert baseline.ready
    assert derive_posture(lm, vis, W, H, cfg, baseline).flags == []
    # Head drops relative to shoulders.
    low, _ = _pose(nose=(0.5, 0.45))
    obs = derive_posture(low, vis, W, H, cfg, baseline)
    assert obs.baseline_ready and FLAG_HEAD_LOWERED in obs.flags
    # Shoulders appear much wider -> moved closer.
    close, _ = _pose(nose=(0.5, 0.2), left=(0.85, 0.6), right=(0.15, 0.6))
    obs = derive_posture(close, vis, W, H, cfg, baseline)
    assert FLAG_MOVED_CLOSER in obs.flags


def test_baseline_finalize_with_few_samples():
    b = PostureBaseline(seconds=5.0)
    b.update(0.0, 0.5, 0.4)
    assert not b.ready
    b.finalize()
    assert b.ready and b.head_height_ratio == 0.5
