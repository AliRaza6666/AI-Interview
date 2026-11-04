"""Data structures for observations and the CV feedback report.

Everything here is a plain dataclass that round-trips through ``to_dict`` /
``from_dict`` so it can be stored as JSON and returned by a web API unchanged.
A missing detection is represented explicitly (``present=False`` or ``None``)
rather than by dropping the frame, so downstream code can compute rates.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

GazeState = str  # "camera" | "away" | "unknown"
GazeDirection = str  # "center" | "left" | "right" | "up" | "down"


# --------------------------------------------------------------------------- observations


@dataclass
class BBox:
    """Normalized bounding box (fractions of image width/height)."""

    x: float
    y: float
    w: float
    h: float

    def to_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        x0 = max(0, int(round(self.x * width)))
        y0 = max(0, int(round(self.y * height)))
        x1 = min(width, int(round((self.x + self.w) * width)))
        y1 = min(height, int(round((self.y + self.h) * height)))
        return x0, y0, x1, y1


@dataclass
class HeadPose:
    """Head rotation in degrees. yaw: + = turned toward image right; pitch: + = chin down; roll: + = tilted toward image right."""

    yaw_deg: float
    pitch_deg: float
    roll_deg: float


@dataclass
class FaceObservation:
    present: bool
    bbox: Optional[BBox] = None
    head_pose: Optional[HeadPose] = None


@dataclass
class ExpressionObservation:
    label: str
    group: str  # neutral | positive | other
    probabilities: dict[str, float] = field(default_factory=dict)


@dataclass
class GazeObservation:
    state: GazeState
    direction: GazeDirection = "center"
    head_yaw_deg: Optional[float] = None
    head_pitch_deg: Optional[float] = None
    iris_x: Optional[float] = None
    iris_y: Optional[float] = None
    effective_yaw_deg: Optional[float] = None
    effective_pitch_deg: Optional[float] = None


@dataclass
class PostureObservation:
    upper_body_visible: bool
    shoulder_tilt_deg: Optional[float] = None
    head_offset_ratio: Optional[float] = None
    head_height_ratio: Optional[float] = None
    shoulder_width_norm: Optional[float] = None
    baseline_ready: bool = False
    flags: list[str] = field(default_factory=list)


PHASE_IDLE = "idle"  # no question active (before the first question, between answers, after the end)
PHASE_QUESTION = "question"  # question is being presented (question_started .. answer_started)
PHASE_ANSWER = "answer"  # user is answering (answer_started .. answer_ended)


@dataclass
class FrameObservation:
    timestamp_s: float  # seconds since the CV session started
    frame_index: int
    face: Optional[FaceObservation] = None
    expression: Optional[ExpressionObservation] = None
    gaze: Optional[GazeObservation] = None
    posture: Optional[PostureObservation] = None
    errors: list[str] = field(default_factory=list)
    # Interview context at the time of the frame, filled from lifecycle events (see InterviewEvent).
    question_id: Optional[str] = None
    phase: str = PHASE_IDLE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FrameObservation":
        face = d.get("face")
        expr = d.get("expression")
        gaze = d.get("gaze")
        posture = d.get("posture")
        return cls(
            timestamp_s=float(d["timestamp_s"]),
            frame_index=int(d["frame_index"]),
            face=_face_from_dict(face) if face else None,
            expression=ExpressionObservation(**expr) if expr else None,
            gaze=GazeObservation(**gaze) if gaze else None,
            posture=PostureObservation(**posture) if posture else None,
            errors=list(d.get("errors", [])),
            question_id=d.get("question_id"),
            phase=d.get("phase", PHASE_IDLE),
        )


# --------------------------------------------------------------------------- interview events

EVENT_INTERVIEW_STARTED = "interview_started"
EVENT_QUESTION_STARTED = "question_started"
EVENT_ANSWER_STARTED = "answer_started"
EVENT_ANSWER_ENDED = "answer_ended"
EVENT_INTERVIEW_ENDED = "interview_ended"
EVENT_TYPES = (
    EVENT_INTERVIEW_STARTED,
    EVENT_QUESTION_STARTED,
    EVENT_ANSWER_STARTED,
    EVENT_ANSWER_ENDED,
    EVENT_INTERVIEW_ENDED,
)


@dataclass
class InterviewEvent:
    """Proposed cross-module contract for interview lifecycle events.

    * ``type``: one of EVENT_TYPES.
    * ``session_id``: shared interview session id, generated once by the host application.
    * ``timestamp``: UTC epoch seconds (``time.time()``), the common time base for all modules.
    * ``question_id``: required for question_started / answer_started / answer_ended.
    * ``payload``: free-form extras (e.g. question text); ignored by the CV module.
    """

    type: str
    session_id: str
    timestamp: float
    question_id: Optional[str] = None
    payload: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.type not in EVENT_TYPES:
            raise ValueError(f"unknown event type {self.type!r}; expected one of {EVENT_TYPES}")
        if self.type in (EVENT_QUESTION_STARTED, EVENT_ANSWER_STARTED, EVENT_ANSWER_ENDED) and not self.question_id:
            raise ValueError(f"{self.type} requires question_id")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "InterviewEvent":
        return cls(
            type=str(d["type"]),
            session_id=str(d["session_id"]),
            timestamp=float(d["timestamp"]),
            question_id=d.get("question_id"),
            payload=dict(d.get("payload") or {}),
        )


def _face_from_dict(d: dict[str, Any]) -> FaceObservation:
    bbox = d.get("bbox")
    hp = d.get("head_pose")
    return FaceObservation(
        present=bool(d["present"]),
        bbox=BBox(**bbox) if bbox else None,
        head_pose=HeadPose(**hp) if hp else None,
    )


@dataclass
class SessionMeta:
    source: str
    session_id: str = ""  # shared interview session id (from the host application)
    width: int = 0
    height: int = 0
    source_fps: float = 0.0
    sample_fps: float = 0.0
    frames_total: int = 0
    frames_sampled: int = 0
    duration_s: float = 0.0
    started_at: Optional[str] = None  # ISO-8601 UTC wall-clock time
    started_at_epoch: Optional[float] = None  # UTC epoch seconds; timestamp_s values are relative to this
    models: dict[str, str] = field(default_factory=dict)
    library_versions: dict[str, str] = field(default_factory=dict)


@dataclass
class ObservationLog:
    meta: SessionMeta
    frames: list[FrameObservation] = field(default_factory=list)
    events: list[InterviewEvent] = field(default_factory=list)

    def append(self, obs: FrameObservation) -> None:
        self.frames.append(obs)
        self.meta.frames_sampled = len(self.frames)
        self.meta.duration_s = max(self.meta.duration_s, obs.timestamp_s)

    def to_dict(self) -> dict[str, Any]:
        return {
            "meta": asdict(self.meta),
            "events": [e.to_dict() for e in self.events],
            "frames": [f.to_dict() for f in self.frames],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ObservationLog":
        return cls(
            meta=SessionMeta(**d["meta"]),
            frames=[FrameObservation.from_dict(f) for f in d.get("frames", [])],
            events=[InterviewEvent.from_dict(e) for e in d.get("events", [])],
        )

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "ObservationLog":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save_jsonl(self, path: str | Path) -> None:
        """One JSON object per line: the meta record, then ``{"event": ...}`` records, then one record per frame."""
        with Path(path).open("w", encoding="utf-8") as fh:
            fh.write(json.dumps({"meta": asdict(self.meta)}) + "\n")
            for e in self.events:
                fh.write(json.dumps({"event": e.to_dict()}) + "\n")
            for f in self.frames:
                fh.write(json.dumps(f.to_dict()) + "\n")

    @classmethod
    def load_jsonl(cls, path: str | Path) -> "ObservationLog":
        meta: Optional[SessionMeta] = None
        frames: list[FrameObservation] = []
        events: list[InterviewEvent] = []
        with Path(path).open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if "meta" in rec and meta is None:
                    meta = SessionMeta(**rec["meta"])
                elif "event" in rec:
                    events.append(InterviewEvent.from_dict(rec["event"]))
                else:
                    frames.append(FrameObservation.from_dict(rec))
        if meta is None:
            raise ValueError(f"{path}: no meta record found")
        return cls(meta=meta, frames=frames, events=events)


# --------------------------------------------------------------------------- report


@dataclass
class Episode:
    """A contiguous run of sampled frames sharing a state."""

    state: str
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass
class FaceMetrics:
    frames: int = 0
    present_frames: int = 0
    presence_rate: Optional[float] = None
    longest_absence_s: float = 0.0
    absence_episodes: int = 0


@dataclass
class GazeMetrics:
    frames_evaluated: int = 0
    camera_rate: Optional[float] = None
    away_rate: Optional[float] = None
    unknown_rate: Optional[float] = None
    direction_rates: dict[str, float] = field(default_factory=dict)
    away_episodes: int = 0
    longest_away_s: float = 0.0
    mean_abs_head_yaw_deg: Optional[float] = None
    mean_abs_head_pitch_deg: Optional[float] = None
    # Offsets subtracted before classification when ReportConfig.recenter_gaze is on.
    recenter_yaw_deg: float = 0.0
    recenter_pitch_deg: float = 0.0


@dataclass
class ExpressionMetrics:
    frames_evaluated: int = 0
    label_rates: dict[str, float] = field(default_factory=dict)
    group_rates: dict[str, float] = field(default_factory=dict)
    dominant_label: Optional[str] = None
    dominant_group: Optional[str] = None


@dataclass
class PostureMetrics:
    frames_evaluated: int = 0
    upper_body_visible_rate: Optional[float] = None
    flag_rates: dict[str, float] = field(default_factory=dict)
    mean_abs_shoulder_tilt_deg: Optional[float] = None
    mean_abs_head_offset_ratio: Optional[float] = None
    flag_episodes: dict[str, int] = field(default_factory=dict)


@dataclass
class Suggestion:
    category: str  # face | gaze | expression | posture
    title: str
    message: str
    evidence: str


@dataclass
class QuestionSummary:
    """CV metrics over one question's *answer* segment (answer_started .. answer_ended)."""

    question_id: str
    answer_start_s: Optional[float]
    answer_end_s: Optional[float]
    answer_duration_s: float
    frames: int
    face_presence_rate: Optional[float]
    camera_focus_rate: Optional[float]
    dominant_expression_group: Optional[str]
    posture_flag_rates: dict[str, float] = field(default_factory=dict)


@dataclass
class CVReport:
    session: dict[str, Any]
    face: FaceMetrics
    gaze: GazeMetrics
    expression: ExpressionMetrics
    posture: PostureMetrics
    scores: dict[str, Optional[float]]
    strengths: list[str]
    suggestions: list[Suggestion]
    disclaimer: str
    questions: list[QuestionSummary] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")


def iter_states(frames: Iterable[FrameObservation], key) -> Iterable[tuple[float, Optional[str]]]:
    """Yield ``(timestamp, state)`` pairs where ``key(frame)`` returns a state string or None."""
    for f in frames:
        yield f.timestamp_s, key(f)
