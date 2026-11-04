"""Live webcam capture, separated from frame processing.

The CV module works on the live webcam only. ``WebcamVideoSource`` wraps
``cv2.VideoCapture`` with explicit ``open`` / ``read`` / ``close`` steps,
``FrameSampler`` throttles it to the analysis rate, and ``FrameSequenceSource``
replays in-memory frames so the processing code can be tested without a camera.
"""

from __future__ import annotations

import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

import cv2
import numpy as np


@dataclass
class Frame:
    index: int
    timestamp_s: float  # seconds since the source was opened
    image: np.ndarray  # BGR, HxWx3, uint8

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])


class VideoSource(ABC):
    """Abstract live frame producer. Use as a context manager or call ``open``/``close``."""

    description: str = "source"

    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def read(self) -> Optional[Frame]:
        """Return the next frame or ``None`` when the stream has ended."""

    @abstractmethod
    def close(self) -> None: ...

    @property
    @abstractmethod
    def fps(self) -> float: ...

    @property
    @abstractmethod
    def width(self) -> int: ...

    @property
    @abstractmethod
    def height(self) -> int: ...

    def __enter__(self) -> "VideoSource":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __iter__(self) -> Iterator[Frame]:
        while True:
            frame = self.read()
            if frame is None:
                return
            yield frame


class WebcamVideoSource(VideoSource):
    """Live webcam. Timestamps are monotonic seconds since the first frame.

    NOTE: not yet exercised on real hardware in this repository (the development
    machine had no working webcam). It only wraps ``cv2.VideoCapture``.
    """

    def __init__(self, device: int = 0, width: Optional[int] = None, height: Optional[int] = None, max_seconds: Optional[float] = None):
        self.device = device
        self.requested_width = width
        self.requested_height = height
        self.max_seconds = max_seconds
        self.description = f"webcam:{device}"
        self._cap: Optional[cv2.VideoCapture] = None
        self._index = 0
        self._t0: Optional[float] = None

    # ------------------------------------------------------------------ lifecycle
    def open(self) -> None:
        if self._cap is not None:
            return
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.device, backend)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(self.device)  # retry with the default backend
        if not cap.isOpened():
            raise RuntimeError(f"Could not open webcam device {self.device}")
        if self.requested_width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.requested_width)
        if self.requested_height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.requested_height)
        self._cap = cap
        self._index = 0
        self._t0 = None

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def is_open(self) -> bool:
        return self._cap is not None

    # ------------------------------------------------------------------ properties
    def _ensure_open(self) -> cv2.VideoCapture:
        if self._cap is None:
            self.open()
        assert self._cap is not None
        return self._cap

    @property
    def fps(self) -> float:
        fps = float(self._ensure_open().get(cv2.CAP_PROP_FPS) or 0.0)
        return fps if 0 < fps < 1000 else 0.0

    @property
    def width(self) -> int:
        return int(self._ensure_open().get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        return int(self._ensure_open().get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ------------------------------------------------------------------ frames
    def read(self) -> Optional[Frame]:
        cap = self._ensure_open()
        ok, img = cap.read()
        if not ok or img is None:
            return None
        now = time.monotonic()
        if self._t0 is None:
            self._t0 = now
        ts = now - self._t0
        if self.max_seconds is not None and ts > self.max_seconds:
            return None
        frame = Frame(index=self._index, timestamp_s=ts, image=img)
        self._index += 1
        return frame


class FrameSequenceSource(VideoSource):
    """Replays in-memory frames at a nominal rate. For tests and for hosts that already
    own a capture loop; it never touches a file or a camera."""

    def __init__(self, frames: Iterable[np.ndarray], fps: float = 10.0, description: str = "frames", realtime: bool = False):
        """``realtime=True`` paces ``read()`` to ``fps`` with the wall clock, like a camera would."""
        self._frames = list(frames)
        self._fps = float(fps)
        self.description = description
        self.realtime = realtime
        self._index = 0
        self._t_open: Optional[float] = None
        self.opened = False
        self.closed = False

    def open(self) -> None:
        self.opened = True
        self._index = 0
        self._t_open = time.monotonic()

    def close(self) -> None:
        self.closed = True

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def width(self) -> int:
        return int(self._frames[0].shape[1]) if self._frames else 0

    @property
    def height(self) -> int:
        return int(self._frames[0].shape[0]) if self._frames else 0

    def read(self) -> Optional[Frame]:
        if self._index >= len(self._frames):
            return None
        if self.realtime and self._t_open is not None:
            due = self._t_open + self._index / self._fps
            delay = due - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        frame = Frame(index=self._index, timestamp_s=self._index / self._fps, image=self._frames[self._index])
        self._index += 1
        return frame


class FrameSampler:
    """Yield frames from a source at approximately ``sample_fps`` based on frame timestamps.

    ``sample_fps=None`` or ``<= 0`` passes every frame through.
    """

    def __init__(self, source: VideoSource, sample_fps: Optional[float], max_seconds: Optional[float] = None):
        self.source = source
        self.interval = (1.0 / sample_fps) if sample_fps and sample_fps > 0 else 0.0
        self.max_seconds = max_seconds
        self.frames_seen = 0
        self._next_ts = 0.0

    def should_sample(self, timestamp_s: float) -> bool:
        """Stateful decision for one frame timestamp (used by the session's capture loop)."""
        self.frames_seen += 1
        if self.interval <= 0 or timestamp_s + 1e-9 >= self._next_ts:
            if self.interval > 0:
                if timestamp_s >= self._next_ts + self.interval:
                    # Slots were missed (stall / slow webcam): realign to the frame just seen.
                    self._next_ts = timestamp_s + self.interval
                else:
                    self._next_ts += self.interval
            return True
        return False

    def __iter__(self) -> Iterator[Frame]:
        for frame in self.source:
            if self.max_seconds is not None and frame.timestamp_s > self.max_seconds:
                return
            if self.should_sample(frame.timestamp_s):
                yield frame


def open_webcam(spec: str | int = 0, max_seconds: Optional[float] = None) -> WebcamVideoSource:
    """``0`` / ``"0"`` / ``"webcam:1"`` -> WebcamVideoSource. Files are not supported."""
    if isinstance(spec, int):
        return WebcamVideoSource(spec, max_seconds=max_seconds)
    s = str(spec).strip().lower()
    if s.startswith("webcam:"):
        s = s.split(":", 1)[1] or "0"
    if not s.isdigit():
        raise ValueError(f"Expected a webcam index such as 0 or 'webcam:1', got {spec!r}. File input is not supported.")
    return WebcamVideoSource(int(s), max_seconds=max_seconds)


class VideoWriter:
    """Optional debug recording of the annotated live feed (``cv2.VideoWriter`` wrapper)."""

    def __init__(self, path: str | Path, fps: float, size: tuple[int, int], codec: str = "mp4v"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*codec), max(fps, 1.0), size)
        if not self._writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {self.path}")
        self.size = size
        self.frames_written = 0

    def write(self, image: np.ndarray) -> None:
        if (image.shape[1], image.shape[0]) != self.size:
            image = cv2.resize(image, self.size)
        self._writer.write(image)
        self.frames_written += 1

    def close(self) -> None:
        self._writer.release()

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
