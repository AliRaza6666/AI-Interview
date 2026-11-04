import json

import pytest

from interview_cv.config import ReportConfig
from interview_cv.report import build_report, episodes, format_report_text
from interview_cv.schemas import (
    ExpressionObservation,
    FaceObservation,
    FrameObservation,
    GazeObservation,
    ObservationLog,
    PostureObservation,
    SessionMeta,
)

FPS = 5.0
DT = 1.0 / FPS


def _log(n_seconds: float = 20.0) -> ObservationLog:
    return ObservationLog(meta=SessionMeta(source="synthetic", width=640, height=480, source_fps=30.0, sample_fps=FPS))


def _add(log: ObservationLog, t: float, present=True, gaze="camera", direction="center", yaw=0.0, pitch=0.0,
         label="Neutral", posture_flags=(), visible=True):
    f = FrameObservation(timestamp_s=t, frame_index=int(t * 30))
    f.face = FaceObservation(present=present)
    if present:
        f.gaze = GazeObservation(gaze, direction, yaw, pitch, 0.5, 0.45, yaw, pitch)
        group = {"Neutral": "neutral", "Happiness": "positive"}.get(label, "other")
        f.expression = ExpressionObservation(label, group, {label: 1.0})
    f.posture = PostureObservation(visible, 0.0, 0.0, 0.4, 0.3, True, list(posture_flags))
    log.append(f)


def test_episodes_grouping():
    frames = []
    for i, state in enumerate("aab.bb"):  # '.' = None
        f = FrameObservation(timestamp_s=i * DT, frame_index=i)
        f.errors = [state]
        frames.append(f)
    eps = episodes(frames, lambda f: None if f.errors[0] == "." else f.errors[0], DT)
    assert [(e.state, round(e.duration_s, 2)) for e in eps] == [("a", 0.4), ("b", 0.2), ("b", 0.4)]


def test_empty_log_report():
    report = build_report(_log())
    assert report.face.presence_rate is None
    assert report.scores["overall"] is None
    assert report.suggestions == []
    json.loads(report.to_json())
    assert "CV feedback report" in format_report_text(report)


def test_rates_and_episodes():
    log = _log()
    cfg = ReportConfig(recenter_gaze=False)
    # 10 s camera-facing, then 6 s away-left, then 4 s face absent
    for i in range(100):
        t = i * DT
        if t < 10.0:
            _add(log, t)
        elif t < 16.0:
            _add(log, t, gaze="away", direction="left", yaw=-40.0)
        else:
            _add(log, t, present=False)
    report = build_report(log, cfg)
    assert report.face.frames == 100 and report.face.present_frames == 80
    assert report.face.presence_rate == pytest.approx(0.8)
    assert report.face.absence_episodes == 1 and report.face.longest_absence_s == pytest.approx(4.0, abs=DT)
    assert report.gaze.camera_rate == pytest.approx(50 / 80)
    assert report.gaze.away_rate == pytest.approx(30 / 80)
    assert report.gaze.direction_rates == {"left": pytest.approx(30 / 80)}
    assert report.gaze.away_episodes == 1 and report.gaze.longest_away_s == pytest.approx(6.0, abs=DT)
    assert report.scores["face_presence"] == 80.0
    assert report.scores["camera_focus"] == pytest.approx(62.5)
    titles = {s.title for s in report.suggestions}
    assert "Stay in frame" in titles
    assert "Avoid long stretches of looking away" in titles
    assert "Look toward the camera more often" not in titles  # 62.5% is above the 60% warning threshold
    assert report.expression.group_rates["neutral"] == 1.0
    assert any(s.category == "expression" for s in report.suggestions)  # no smiles at all
    assert report.disclaimer


def test_posture_suggestions_and_strengths():
    log = _log()
    for i in range(50):
        _add(log, i * DT, label="Happiness" if i % 4 == 0 else "Neutral", posture_flags=("shoulders_tilted",) if i < 25 else ())
    report = build_report(log, ReportConfig(recenter_gaze=False))
    assert report.posture.flag_rates["shoulders_tilted"] == pytest.approx(0.5)
    assert report.posture.flag_episodes["shoulders_tilted"] == 1
    assert report.scores["posture_stability"] == pytest.approx(50.0)
    assert any(s.title == "Level your shoulders" for s in report.suggestions)
    assert any("Smile-like" in s for s in report.strengths)
    assert any("Mostly camera-facing" in s for s in report.strengths)


def test_upper_body_not_visible_suggestion():
    log = _log()
    for i in range(20):
        _add(log, i * DT, visible=False)
    report = build_report(log)
    assert report.posture.upper_body_visible_rate == 0.0
    assert report.posture.flag_rates == {}
    assert report.scores["posture_stability"] is None
    assert any(s.title == "Frame your head and shoulders" for s in report.suggestions)


def test_recentering_cancels_camera_placement_bias():
    """A constant 18 deg chin-down bias must not be reported as 'looking down' when re-centering is on."""
    log = _log()
    for i in range(50):
        # Raw per-frame classification (as the analyzer would produce it) says 'away/down'.
        _add(log, i * DT, gaze="away", direction="down", yaw=0.0, pitch=22.0)
    raw = build_report(log, ReportConfig(recenter_gaze=False))
    assert raw.gaze.away_rate == 1.0
    fixed = build_report(log, ReportConfig(recenter_gaze=True, max_recenter_deg=15.0))
    # Offset is capped at 15 deg, so 22 - 15 = 7 deg -> within the 20 deg 'down' threshold.
    assert fixed.gaze.recenter_pitch_deg == 15.0
    assert fixed.gaze.camera_rate == 1.0


def test_recentering_cap_keeps_real_look_away():
    log = _log()
    for i in range(50):
        _add(log, i * DT, gaze="away", direction="right", yaw=60.0, pitch=0.0)
    fixed = build_report(log, ReportConfig(recenter_gaze=True, max_recenter_deg=15.0))
    assert fixed.gaze.away_rate == 1.0  # 60 - 15 = 45 deg is still away
