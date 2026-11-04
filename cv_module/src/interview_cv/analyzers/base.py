"""Shared per-frame context passed to every analyzer."""

from __future__ import annotations

from typing import Any, Optional

import cv2
import numpy as np


class FrameContext:
    """Holds one frame in the formats the analyzers need, converting lazily and once.

    * ``bgr``: original OpenCV frame.
    * ``rgb``: contiguous RGB copy (HSEmotion, MediaPipe).
    * ``mp_image``: MediaPipe ``Image`` wrapper, shared by the face and pose landmarkers.
    * ``timestamp_ms``: integer milliseconds for MediaPipe VIDEO mode.
    """

    def __init__(self, bgr: np.ndarray, timestamp_s: float, frame_index: int = 0):
        self.bgr = bgr
        self.timestamp_s = float(timestamp_s)
        self.timestamp_ms = int(round(self.timestamp_s * 1000.0))
        self.frame_index = frame_index
        self.height, self.width = bgr.shape[:2]
        self._rgb: Optional[np.ndarray] = None
        self._mp_image: Optional[Any] = None

    @property
    def rgb(self) -> np.ndarray:
        if self._rgb is None:
            self._rgb = np.ascontiguousarray(cv2.cvtColor(self.bgr, cv2.COLOR_BGR2RGB))
        return self._rgb

    @property
    def mp_image(self) -> Any:
        if self._mp_image is None:
            import mediapipe as mp  # imported lazily so pure-numpy tests do not need MediaPipe loaded

            self._mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=self.rgb)
        return self._mp_image


class MonotonicTimestamps:
    """MediaPipe VIDEO mode requires strictly increasing timestamps per detector instance."""

    def __init__(self) -> None:
        self._last: Optional[int] = None

    def next(self, timestamp_ms: int) -> int:
        if self._last is not None and timestamp_ms <= self._last:
            timestamp_ms = self._last + 1
        self._last = timestamp_ms
        return timestamp_ms
