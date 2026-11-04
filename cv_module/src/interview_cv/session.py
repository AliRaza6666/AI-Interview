"""CV session manager: owns the webcam, the pipeline, the observation log and the
interview context (current question / phase) derived from lifecycle events.

Two ways to drive it:

* **Self-capturing** (default): ``start()`` opens the webcam and runs a background
  capture thread; ``stop()`` (or an ``interview_ended`` event) releases the camera.
* **Host-driven**: construct with ``camera=None``; the host calls ``process_frame``
  from its own capture loop. Useful for tests and for a host that already owns the
  camera.

The analyzers never see events; they only get frames. Events only tag the
resulting observations with ``question_id`` / ``phase`` so the report can
summarise each answer separately.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Optional, Union

import numpy as np

from .config import PipelineConfig
from .pipeline import FrameAnalysis, InterviewCVPipeline, library_versions
from .report import build_report
from .schemas import (
    EVENT_ANSWER_ENDED,
    EVENT_ANSWER_STARTED,
    EVENT_INTERVIEW_ENDED,
    EVENT_INTERVIEW_STARTED,
    EVENT_QUESTION_STARTED,
    PHASE_ANSWER,
    PHASE_IDLE,
    PHASE_QUESTION,
    CVReport,
    InterviewEvent,
    ObservationLog,
    SessionMeta,
)
from .video import Frame, FrameSampler, VideoSource, WebcamVideoSource

log = logging.getLogger(__name__)

FrameCallback = Callable[[Frame, FrameAnalysis], None]


class CVSession:
    """Lifecycle: ``start()`` -> frames are processed -> ``handle_event()`` as the
    interview progresses -> ``stop()`` -> ``build_report()`` -> ``close()``."""

    def __init__(
        self,
        session_id: str,
        config: Optional[PipelineConfig] = None,
        camera: Union[int, VideoSource, None] = 0,
        pipeline: Optional[InterviewCVPipeline] = None,
        on_frame: Optional[FrameCallback] = None,
        clock: Callable[[], float] = time.time,
    ):
        self.session_id = str(session_id)
        self.config = config or PipelineConfig()
        self.source: Optional[VideoSource]
        if camera is None:
            self.source = None
        elif isinstance(camera, VideoSource):
            self.source = camera
        else:
            self.source = WebcamVideoSource(int(camera), max_seconds=self.config.max_seconds)
        self._pipeline = pipeline
        self._owns_pipeline = pipeline is None
        self.on_frame = on_frame
        self._clock = clock

        self.log: Optional[ObservationLog] = None
        self._lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._started = False
        self._stopped = False
        self._t0_epoch: Optional[float] = None
        self._t0_mono: Optional[float] = None
        self._frame_counter = 0
        self._sampler: Optional[FrameSampler] = None
        self._capture_ended = False

        self.phase: str = PHASE_IDLE
        self.question_id: Optional[str] = None
        self.last_error: Optional[BaseException] = None

    # ------------------------------------------------------------------ properties
    @property
    def pipeline(self) -> InterviewCVPipeline:
        if self._pipeline is None:
            self._pipeline = InterviewCVPipeline(self.config)
        return self._pipeline

    @property
    def is_running(self) -> bool:
        """True while the session is started, not stopped, and (in webcam mode) capture is still alive."""
        return self._started and not self._stopped and not self._capture_ended

    @property
    def capture_ended(self) -> bool:
        """True once the background capture loop has finished (normally or after an error)."""
        return self._capture_ended

    @property
    def elapsed_s(self) -> float:
        if self._t0_mono is None:
            return 0.0
        return time.monotonic() - self._t0_mono

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        """Open the camera (if any), create the log, and begin processing frames."""
        with self._lock:
            if self._started:
                return
            self._started = True
            self._capture_ended = False
            self._t0_epoch = float(self._clock())
            self._t0_mono = time.monotonic()
            try:
                pipeline = self.pipeline  # load models before opening the camera
                source_desc = "host-driven"
                width = height = 0
                fps = 0.0
                if self.source is not None:
                    self.source.open()
                    source_desc = self.source.description
                    width, height, fps = self.source.width, self.source.height, self.source.fps
                self.log = ObservationLog(
                    meta=SessionMeta(
                        source=source_desc,
                        session_id=self.session_id,
                        width=width,
                        height=height,
                        source_fps=fps,
                        sample_fps=self.config.sample_fps,
                        started_at=datetime.fromtimestamp(self._t0_epoch, tz=timezone.utc).isoformat(timespec="seconds"),
                        started_at_epoch=self._t0_epoch,
                        models=pipeline.model_names(),
                        library_versions=library_versions(),
                    )
                )
                self._sampler = FrameSampler(self.source, self.config.sample_fps) if self.source is not None else None
                if self.source is not None:
                    self._stop_event.clear()
                    self._thread = threading.Thread(target=self._capture_loop, name=f"cv-session-{self.session_id}", daemon=True)
                    self._thread.start()
            except Exception:
                # Startup failed (camera unavailable, model missing, ...): leave the session unstarted
                # so is_running is False, process_frame refuses frames and start() can be retried.
                self._started = False
                self.log = None
                self._sampler = None
                self._thread = None
                self._t0_epoch = None
                self._t0_mono = None
                if self.source is not None:
                    try:
                        self.source.close()
                    except Exception:  # noqa: BLE001
                        pass
                raise

    def _capture_loop(self) -> None:
        assert self.source is not None and self._sampler is not None
        try:
            while not self._stop_event.is_set():
                frame = self.source.read()
                if frame is None:
                    break
                # Fix 1: timestamps come from the session clock (seconds since start()), not from the
                # camera's own clock, so started_at_epoch + timestamp_s is a true wall-clock time.
                timestamp_s = self.elapsed_s
                if self.config.max_seconds is not None and timestamp_s > self.config.max_seconds:
                    break
                if not self._sampler.should_sample(timestamp_s):
                    continue
                analysis = self.process_frame(frame.image, timestamp_s=timestamp_s, frame_index=frame.index)
                if self.on_frame is not None:
                    self.on_frame(frame, analysis)
        except Exception as exc:  # noqa: BLE001 - surface the error to the host instead of dying silently
            log.exception("capture loop failed: %s", exc)
            self.last_error = exc
        finally:
            # Fix 3: always release the camera and mark capture as ended, even after an exception.
            try:
                self.source.close()
            except Exception as exc:  # noqa: BLE001
                log.warning("closing the camera failed: %s", exc)
            with self._lock:
                if self.log is not None and self._sampler is not None:
                    self.log.meta.frames_total = self._sampler.frames_seen
                self._capture_ended = True

    def process_frame(self, image_bgr: np.ndarray, timestamp_s: Optional[float] = None, frame_index: Optional[int] = None) -> FrameAnalysis:
        """Analyse one frame and record the observation with the interview context at capture time."""
        if not self._started or self.log is None:
            raise RuntimeError("CVSession.start() must be called before processing frames")
        if timestamp_s is None:
            timestamp_s = self.elapsed_s
        if frame_index is None:
            frame_index = self._frame_counter
        self._frame_counter = frame_index + 1
        # Fix 2: snapshot the context before analysis (which takes ~100 ms) so an event arriving
        # during analysis does not re-label a frame that was captured in the previous segment.
        with self._lock:
            phase, question_id = self.phase, self.question_id
        analysis = self.pipeline.analyze(image_bgr, timestamp_s, frame_index)
        analysis.observation.phase = phase
        analysis.observation.question_id = question_id
        with self._lock:
            self.log.append(analysis.observation)
        return analysis

    def handle_event(self, event: Union[InterviewEvent, dict[str, Any]]) -> None:
        """Apply an interview lifecycle event (see ``InterviewEvent`` for the contract)."""
        if isinstance(event, dict):
            event = InterviewEvent.from_dict(event)
        event.validate()
        if event.session_id != self.session_id:
            raise ValueError(f"event session_id {event.session_id!r} does not match this session {self.session_id!r}")
        with self._lock:
            if self.log is not None:
                self.log.events.append(event)
            if event.type == EVENT_INTERVIEW_STARTED:
                self.phase, self.question_id = PHASE_IDLE, None
            elif event.type == EVENT_QUESTION_STARTED:
                self.phase, self.question_id = PHASE_QUESTION, event.question_id
            elif event.type == EVENT_ANSWER_STARTED:
                self.phase, self.question_id = PHASE_ANSWER, event.question_id
            elif event.type == EVENT_ANSWER_ENDED:
                self.phase, self.question_id = PHASE_IDLE, None
            elif event.type == EVENT_INTERVIEW_ENDED:
                self.phase, self.question_id = PHASE_IDLE, None
        if event.type == EVENT_INTERVIEW_ENDED:
            self.stop()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop capture, release the camera and finalize the log. Idempotent."""
        with self._lock:
            if not self._started or self._stopped:
                return
            self._stopped = True
        self._stop_event.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout)
        if self.source is not None:
            self.source.close()
        if self._pipeline is not None:
            self._pipeline.finalize()
        with self._lock:
            if self.log is not None:
                if self._sampler is not None:
                    self.log.meta.frames_total = self._sampler.frames_seen
                else:
                    self.log.meta.frames_total = self._frame_counter

    def build_report(self) -> CVReport:
        if self.log is None:
            raise RuntimeError("session has not been started")
        return build_report(self.log, self.config.report)

    def close(self) -> None:
        self.stop()
        if self._pipeline is not None and self._owns_pipeline:
            self._pipeline.close()
            self._pipeline = None

    def __enter__(self) -> "CVSession":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()
