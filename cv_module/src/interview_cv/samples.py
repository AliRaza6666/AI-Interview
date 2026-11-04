"""In-memory synthetic frames so the processing code can be exercised without a webcam.

Nothing here reads or writes video files. ``animate_image`` gently pans/zooms/rotates
a still photo; ``blank_frames`` produces frames with no person. Real head turns cannot
be fabricated from a 2-D image (MediaPipe fits a 3-D face model), so these are only
smoke-test inputs; threshold tuning needs real webcam sessions.
"""

from __future__ import annotations

import math
from typing import Iterator, Optional

import cv2
import numpy as np

SAMPLE_IMAGE_SOURCES: dict[str, str] = {
    # Image used in Google's MediaPipe Face Landmarker example notebook (hosted by Google).
    "business-person.png": "https://storage.googleapis.com/mediapipe-assets/business-person.png",
}


def animate_image(
    image_bgr: np.ndarray,
    seconds: float = 6.0,
    fps: float = 10.0,
    absent_from: Optional[float] = None,
    absent_until: Optional[float] = None,
) -> Iterator[np.ndarray]:
    """Yield ``seconds * fps`` BGR frames derived from ``image_bgr``.

    ``absent_from``..``absent_until``: frames are replaced with a plain background (no face).
    """
    h, w = image_bgr.shape[:2]
    n = int(round(seconds * fps))
    for i in range(n):
        t = i / fps
        if absent_from is not None and absent_until is not None and absent_from <= t < absent_until:
            yield np.full((h, w, 3), 90, dtype=np.uint8)
            continue
        scale = 1.0 + 0.03 * math.sin(2 * math.pi * t / 4.0)
        angle = 2.0 * math.sin(2 * math.pi * t / 5.0)
        tx = 0.01 * w * math.sin(2 * math.pi * t / 3.0)
        m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
        m[0, 2] += tx
        yield cv2.warpAffine(image_bgr, m, (w, h), borderMode=cv2.BORDER_REPLICATE)


def blank_frames(seconds: float = 2.0, fps: float = 10.0, size: tuple[int, int] = (320, 240)) -> Iterator[np.ndarray]:
    """Frames with no face at all (a moving rectangle on gray); for 'nothing detected' paths."""
    n = int(round(seconds * fps))
    for i in range(n):
        frame = np.full((size[1], size[0], 3), 120, dtype=np.uint8)
        x = int((i / max(n - 1, 1)) * (size[0] - 40))
        cv2.rectangle(frame, (x, 50), (x + 40, 90), (200, 100, 50), -1)
        yield frame
