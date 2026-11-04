import json
from pathlib import Path

import pytest

from interview_cv.schemas import (
    BBox,
    ExpressionObservation,
    FaceObservation,
    FrameObservation,
    GazeObservation,
    HeadPose,
    InterviewEvent,
    ObservationLog,
    PostureObservation,
    SessionMeta,
)


def _frame(t: float, present: bool = True) -> FrameObservation:
    if not present:
        return FrameObservation(timestamp_s=t, frame_index=int(t * 10), face=FaceObservation(present=False), posture=PostureObservation(False))
    return FrameObservation(
        timestamp_s=t,
        frame_index=int(t * 10),
        face=FaceObservation(True, BBox(0.1, 0.2, 0.3, 0.4), HeadPose(1.0, -2.0, 0.5)),
        expression=ExpressionObservation("Neutral", "neutral", {"Neutral": 0.9, "Happiness": 0.1}),
        gaze=GazeObservation("camera", "center", 1.0, -2.0, 0.5, 0.45, 1.0, -2.0),
        posture=PostureObservation(True, 1.0, 0.02, 0.4, 0.3, True, []),
        errors=[],
    )


def test_frame_round_trip():
    f = _frame(1.5)
    d = json.loads(json.dumps(f.to_dict()))
    assert FrameObservation.from_dict(d) == f
    g = _frame(2.0, present=False)
    assert FrameObservation.from_dict(json.loads(json.dumps(g.to_dict()))) == g


def test_event_round_trip_and_validation():
    e = InterviewEvent("answer_started", "s1", 1700000000.5, "q1", {"text": "Tell me about yourself"})
    assert InterviewEvent.from_dict(json.loads(json.dumps(e.to_dict()))) == e
    e.validate()
    with pytest.raises(ValueError):
        InterviewEvent("answer_started", "s1", 1.0).validate()
    with pytest.raises(ValueError):
        InterviewEvent("nope", "s1", 1.0).validate()


def test_old_frame_records_without_context_fields_still_load():
    d = _frame(1.0).to_dict()
    d.pop("question_id")
    d.pop("phase")
    f = FrameObservation.from_dict(d)
    assert f.question_id is None and f.phase == "idle"


def test_log_json_and_jsonl_round_trip(tmp_path: Path):
    log = ObservationLog(meta=SessionMeta(source="webcam:0", session_id="s1", width=640, height=480, source_fps=30.0, sample_fps=5.0))
    log.events.append(InterviewEvent("interview_started", "s1", 1.0))
    log.events.append(InterviewEvent("answer_started", "s1", 2.0, "q1"))
    log.append(_frame(0.0))
    log.append(_frame(0.2, present=False))
    assert log.meta.frames_sampled == 2 and log.meta.duration_s == 0.2

    p = tmp_path / "log.json"
    log.save_json(p)
    assert ObservationLog.load_json(p) == log

    q = tmp_path / "log.jsonl"
    log.save_jsonl(q)
    assert ObservationLog.load_jsonl(q) == log
    assert len(q.read_text(encoding="utf-8").strip().splitlines()) == 5  # meta + 2 events + 2 frames
