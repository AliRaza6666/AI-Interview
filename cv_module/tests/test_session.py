"""CVSession tests with a fake pipeline: no models, no webcam."""

from __future__ import annotations

import time

import numpy as np
import pytest

from interview_cv.config import PipelineConfig
from interview_cv.pipeline import FrameAnalysis
from interview_cv.schemas import (
    PHASE_ANSWER,
    PHASE_IDLE,
    PHASE_QUESTION,
    ExpressionObservation,
    FaceObservation,
    FrameObservation,
    GazeObservation,
    InterviewEvent,
    PostureObservation,
)
from interview_cv.session import CVSession
from interview_cv.video import FrameSequenceSource


class FakePipeline:
    """Returns a fixed 'face present, camera-facing' observation; records calls."""

    def __init__(self):
        self.calls = 0
        self.finalized = False
        self.closed = False
        self.config = PipelineConfig()

    def analyze(self, image_bgr, timestamp_s, frame_index=0) -> FrameAnalysis:
        self.calls += 1
        obs = FrameObservation(
            timestamp_s=timestamp_s,
            frame_index=frame_index,
            face=FaceObservation(True),
            gaze=GazeObservation("camera", "center", 0.0, 0.0, 0.5, 0.45, 0.0, 0.0),
            expression=ExpressionObservation("Neutral", "neutral", {"Neutral": 1.0}),
            posture=PostureObservation(True, 0.0, 0.0, 0.4, 0.3, True, []),
        )
        return FrameAnalysis(observation=obs)

    def finalize(self):
        self.finalized = True

    def model_names(self):
        return {"fake": "fake"}

    def close(self):
        self.closed = True


def _frames(n: int):
    return [np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(n)]


def _event(sid: str, type_: str, t: float, qid=None) -> InterviewEvent:
    return InterviewEvent(type=type_, session_id=sid, timestamp=t, question_id=qid)


def test_host_driven_session_tags_frames_with_question_context():
    fake = FakePipeline()
    clock_t = [1000.0]
    s = CVSession("s1", camera=None, pipeline=fake, clock=lambda: clock_t[0])
    s.start()
    assert s.log is not None and s.log.meta.session_id == "s1" and s.log.meta.started_at_epoch == 1000.0
    assert s.log.meta.source == "host-driven"

    s.handle_event(_event("s1", "interview_started", 1000.0))
    s.process_frame(_frames(1)[0], timestamp_s=0.0)  # idle
    s.handle_event(_event("s1", "question_started", 1001.0, "q1"))
    s.process_frame(_frames(1)[0], timestamp_s=1.0)  # question
    s.handle_event(_event("s1", "answer_started", 1002.0, "q1"))
    s.process_frame(_frames(1)[0], timestamp_s=2.0)
    s.process_frame(_frames(1)[0], timestamp_s=2.2)
    s.handle_event(_event("s1", "answer_ended", 1003.0, "q1"))
    s.process_frame(_frames(1)[0], timestamp_s=3.0)  # idle again
    s.handle_event(_event("s1", "interview_ended", 1004.0))

    assert not s.is_running and fake.finalized
    phases = [(f.phase, f.question_id) for f in s.log.frames]
    assert phases == [(PHASE_IDLE, None), (PHASE_QUESTION, "q1"), (PHASE_ANSWER, "q1"), (PHASE_ANSWER, "q1"), (PHASE_IDLE, None)]
    assert len(s.log.events) == 5
    assert fake.calls == 5

    report = s.build_report()
    assert report.session["session_id"] == "s1" and report.session["events"] == 5
    assert len(report.questions) == 1
    q = report.questions[0]
    assert q.question_id == "q1" and q.frames == 2
    assert q.answer_start_s == 2.0 and q.answer_end_s == 3.0 and q.answer_duration_s == 1.0
    assert q.face_presence_rate == 1.0 and q.camera_focus_rate == 1.0 and q.dominant_expression_group == "neutral"
    s.close()
    assert not fake.closed  # the session does not own an injected pipeline


def test_event_validation():
    s = CVSession("s1", camera=None, pipeline=FakePipeline())
    s.start()
    with pytest.raises(ValueError):
        s.handle_event({"type": "coffee_break", "session_id": "s1", "timestamp": 1.0})
    with pytest.raises(ValueError):
        s.handle_event({"type": "question_started", "session_id": "s1", "timestamp": 1.0})  # missing question_id
    with pytest.raises(ValueError):
        s.handle_event(_event("other", "interview_started", 1.0))  # wrong session
    s.handle_event({"type": "answer_started", "session_id": "s1", "timestamp": 1.0, "question_id": "q7", "payload": {"text": "..."}})
    assert s.phase == PHASE_ANSWER and s.question_id == "q7"
    s.close()


def test_process_frame_before_start_raises():
    s = CVSession("s1", camera=None, pipeline=FakePipeline())
    with pytest.raises(RuntimeError):
        s.process_frame(_frames(1)[0])


def _wait_capture_end(s: CVSession, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not s.capture_ended and time.monotonic() < deadline:
        time.sleep(0.01)
    assert s.capture_ended, "capture loop did not finish"


def test_self_capturing_session_samples_and_releases_camera():
    fake = FakePipeline()
    src = FrameSequenceSource(_frames(20), fps=20.0, realtime=True)  # 1 s of frames at camera pace
    cfg = PipelineConfig(sample_fps=10.0)
    s = CVSession("cam", cfg, camera=src, pipeline=fake)
    s.start()
    assert src.opened and s.is_running
    _wait_capture_end(s)
    assert src.closed and not s.is_running  # end of stream releases the camera by itself
    s.stop()
    assert s.log is not None
    assert s.log.meta.frames_total == 20
    assert 7 <= len(s.log.frames) <= 11  # ~10 fps over 1 s on the session clock
    assert s.log.meta.source == "frames" and s.log.meta.width == 64
    s.stop()  # idempotent
    s.close()


# --------------------------------------------------------------------------- regression: fix 1


class _OffsetClockSource(FrameSequenceSource):
    """Mimics a camera whose own clock is unrelated to the session clock (e.g. slow warm-up)."""

    def read(self):
        frame = super().read()
        if frame is not None:
            frame.timestamp_s += 100.0
        return frame


def test_webcam_frames_are_stamped_with_the_session_clock():
    src = _OffsetClockSource(_frames(6), fps=20.0, realtime=True)
    s = CVSession("clock", PipelineConfig(sample_fps=0), camera=src, pipeline=FakePipeline())
    t_start = time.monotonic()
    s.start()
    _wait_capture_end(s)
    elapsed = time.monotonic() - t_start
    s.close()
    ts = [f.timestamp_s for f in s.log.frames]
    assert len(ts) == 6
    assert ts == sorted(ts)
    assert all(0.0 <= t <= elapsed + 0.05 for t in ts), ts  # session-relative, not the camera's +100 s clock
    assert ts[-1] >= 0.2  # frames were paced (5 intervals at 20 fps)


# --------------------------------------------------------------------------- regression: fix 2


class _EventDuringAnalysisPipeline(FakePipeline):
    """Delivers answer_ended while the frame is being analysed, like an async audio event would."""

    def __init__(self):
        super().__init__()
        self.session: CVSession | None = None
        self.fired = False

    def analyze(self, image_bgr, timestamp_s, frame_index=0):
        if self.session is not None and not self.fired:
            self.fired = True
            self.session.handle_event(_event(self.session.session_id, "answer_ended", 2.5, "q1"))
        return super().analyze(image_bgr, timestamp_s, frame_index)


def test_phase_is_snapshotted_before_analysis():
    fake = _EventDuringAnalysisPipeline()
    s = CVSession("snap", camera=None, pipeline=fake)
    fake.session = s
    s.start()
    s.handle_event(_event("snap", "answer_started", 2.0, "q1"))
    boundary = s.process_frame(_frames(1)[0], timestamp_s=2.4)  # captured during the answer
    after = s.process_frame(_frames(1)[0], timestamp_s=2.6)  # captured after answer_ended
    s.close()
    assert (boundary.observation.phase, boundary.observation.question_id) == (PHASE_ANSWER, "q1")
    assert (after.observation.phase, after.observation.question_id) == (PHASE_IDLE, None)
    assert s.build_report().questions[0].frames == 1


# --------------------------------------------------------------------------- regression: fix 3


class _FailingOpenSource(FrameSequenceSource):
    def open(self):
        raise RuntimeError("Could not open webcam device 0")


class _FailingReadSource(FrameSequenceSource):
    def __init__(self, frames, fail_at: int, **kw):
        super().__init__(frames, **kw)
        self.fail_at = fail_at

    def read(self):
        if self._index == self.fail_at:
            raise RuntimeError("device disconnected")
        return super().read()


def test_start_failure_resets_session_state_and_allows_retry():
    bad = _FailingOpenSource(_frames(2), fps=10.0)
    s = CVSession("boot", PipelineConfig(sample_fps=0), camera=bad, pipeline=FakePipeline())
    with pytest.raises(RuntimeError, match="Could not open"):
        s.start()
    assert not s.is_running and s.log is None and s._thread is None
    with pytest.raises(RuntimeError):
        s.process_frame(_frames(1)[0])
    s.stop()  # no-op, must not raise
    # The host can retry with a working camera on the same session object.
    good = FrameSequenceSource(_frames(3), fps=20.0, realtime=True)
    s.source = good
    s.start()
    assert s.is_running and s.log is not None
    _wait_capture_end(s)
    s.close()
    assert good.closed and len(s.log.frames) == 3


def test_capture_loop_exception_releases_camera_and_marks_capture_ended():
    src = _FailingReadSource(_frames(5), fail_at=2, fps=20.0, realtime=True)
    s = CVSession("crash", PipelineConfig(sample_fps=0), camera=src, pipeline=FakePipeline())
    s.start()
    _wait_capture_end(s)
    assert src.closed, "camera must be released by the capture loop itself"
    assert not s.is_running and s.capture_ended
    assert isinstance(s.last_error, RuntimeError) and "disconnected" in str(s.last_error)
    assert len(s.log.frames) == 2  # frames before the failure are kept
    assert s.log.meta.frames_total == 2
    s.stop()  # still safe after the loop died
    report = s.build_report()
    assert report.face.frames == 2
    s.close()


def test_interview_ended_stops_capture_and_report_has_no_questions_without_events():
    fake = FakePipeline()
    src = FrameSequenceSource(_frames(100), fps=20.0, realtime=True)  # 5 s: still streaming when the event arrives
    with CVSession("x", PipelineConfig(sample_fps=0), camera=src, pipeline=fake) as s:
        time.sleep(0.2)
        assert s.is_running
        s.handle_event(_event("x", "interview_ended", time.time()))
        assert src.closed and not s.is_running
        assert 0 < len(s.log.frames) < 100
        assert s.build_report().questions == []


def test_on_frame_callback_receives_analysis():
    seen = []
    src = FrameSequenceSource(_frames(3), fps=20.0, realtime=True)
    s = CVSession("cb", PipelineConfig(sample_fps=0), camera=src, pipeline=FakePipeline(), on_frame=lambda fr, an: seen.append((fr.index, an.observation.phase)))
    s.start()
    _wait_capture_end(s)
    s.close()
    assert seen == [(0, PHASE_IDLE), (1, PHASE_IDLE), (2, PHASE_IDLE)]
