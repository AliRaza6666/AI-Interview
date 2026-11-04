"""End-to-end tests feeding in-memory synthetic frames through a host-driven CVSession.
Need model files + sample image; no webcam, no video files."""

from pathlib import Path

import pytest

from interview_cv.config import PipelineConfig
from interview_cv.pipeline import InterviewCVPipeline
from interview_cv.report import build_report
from interview_cv.schemas import InterviewEvent, ObservationLog
from interview_cv.session import CVSession
from interview_cv.visualize import draw_overlay

pytestmark = pytest.mark.integration

FPS = 10.0


@pytest.fixture(scope="module")
def sample_log(model_paths, sample_frames) -> ObservationLog:
    """Replays the 8 s clip at 5 fps with q1 answered during 0-3 s and q2 during 5-8 s."""
    cfg = PipelineConfig(sample_fps=5.0, models=model_paths)
    t0 = 1_700_000_000.0
    with CVSession("itest", cfg, camera=None, clock=lambda: t0) as s:
        s.handle_event(InterviewEvent("interview_started", "itest", t0))
        s.handle_event(InterviewEvent("answer_started", "itest", t0, "q1"))
        for i, img in enumerate(sample_frames):
            t = i / FPS
            if i % 2:  # 10 fps source -> 5 fps analysis
                continue
            if t == 3.0:
                s.handle_event(InterviewEvent("answer_ended", "itest", t0 + 3.0, "q1"))
            if t == 5.0:
                s.handle_event(InterviewEvent("answer_started", "itest", t0 + 5.0, "q2"))
            analysis = s.process_frame(img, timestamp_s=t, frame_index=i)
            overlay = draw_overlay(img, analysis)
            assert overlay.shape == img.shape
        s.handle_event(InterviewEvent("answer_ended", "itest", t0 + 8.0, "q2"))
        s.handle_event(InterviewEvent("interview_ended", "itest", t0 + 8.0))
        return s.log


def test_pipeline_detects_face_and_absence(sample_log: ObservationLog):
    frames = sample_log.frames
    assert len(frames) == 40
    assert all(f.errors == [] for f in frames)
    present = [f for f in frames if f.face and f.face.present]
    absent = [f for f in frames if f.face and not f.face.present]
    assert len(present) >= 25
    assert absent, "the blank segment should yield no-face frames"
    assert all(3.0 - 0.01 <= f.timestamp_s < 5.0 for f in absent)
    # Frames without a face carry no gaze / expression but still have a posture record.
    assert all(f.gaze is None and f.expression is None and f.posture is not None for f in absent)
    f0 = present[0]
    assert f0.face.bbox is not None and f0.face.head_pose is not None
    assert f0.gaze is not None and f0.gaze.state in {"camera", "away"}
    assert f0.expression is not None and f0.expression.label
    assert f0.posture is not None and f0.posture.upper_body_visible


def test_report_session_and_per_question(sample_log: ObservationLog):
    report = build_report(sample_log)
    assert report.face.presence_rate == pytest.approx(0.75, abs=0.1)
    assert report.face.absence_episodes == 1
    assert report.gaze.camera_rate is not None and report.gaze.camera_rate >= 0.9
    assert report.posture.upper_body_visible_rate == pytest.approx(report.face.presence_rate, abs=0.1)
    assert report.posture.flag_rates["shoulders_tilted"] == 0.0
    assert report.expression.frames_evaluated > 0
    assert report.scores["overall"] is not None
    qs = {q.question_id: q for q in report.questions}
    assert set(qs) == {"q1", "q2"}
    assert qs["q1"].frames == 15 and qs["q1"].answer_duration_s == 3.0 and qs["q1"].face_presence_rate == 1.0
    assert qs["q2"].frames == 15 and qs["q2"].camera_focus_rate >= 0.9
    report.to_json()


def test_log_round_trip_keeps_events_and_questions(sample_log: ObservationLog, tmp_path: Path):
    p = tmp_path / "log.jsonl"
    sample_log.save_jsonl(p)
    loaded = ObservationLog.load_jsonl(p)
    assert len(loaded.frames) == len(sample_log.frames) and len(loaded.events) == 6
    assert build_report(loaded).to_dict() == build_report(sample_log).to_dict()


def test_blank_frames_and_disabled_analyzers(model_paths, blank_frames_list):
    cfg = PipelineConfig(sample_fps=5.0, models=model_paths)
    with InterviewCVPipeline(cfg) as pipeline:
        observations = [pipeline.analyze(img, i / FPS, i).observation for i, img in enumerate(blank_frames_list)]
    assert all(o.face is not None and not o.face.present for o in observations)
    assert all(o.posture is not None and not o.posture.upper_body_visible for o in observations)

    cfg2 = PipelineConfig(models=model_paths, enable_expression=False, enable_posture=False, enable_gaze=False)
    with InterviewCVPipeline(cfg2) as pipeline:
        o = pipeline.analyze(blank_frames_list[0], 0.0).observation
    assert o.posture is None and o.gaze is None and o.expression is None
