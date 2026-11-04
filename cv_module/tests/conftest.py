"""Shared fixtures. No test here needs a webcam or any video file.

* Unit tests use synthetic landmarks / observations / in-memory frames and run anywhere.
* Integration tests (marked ``integration``) need the downloaded model files and the
  sample image (``interview-cv download-models --samples``) and are skipped otherwise.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from interview_cv.config import ModelPaths

CV_MODULE_DIR = Path(__file__).resolve().parents[1]
SAMPLE_DIR = Path(os.environ.get("INTERVIEW_CV_SAMPLE_DIR", CV_MODULE_DIR / "samples"))


@pytest.fixture(scope="session")
def model_paths() -> ModelPaths:
    mp = ModelPaths()
    missing = mp.missing()
    if missing:
        pytest.skip("model files missing (run `interview-cv download-models`): " + ", ".join(m.name for m in missing))
    return mp


@pytest.fixture(scope="session")
def sample_image_path() -> Path:
    p = SAMPLE_DIR / "business-person.png"
    if not p.is_file():
        pytest.skip(f"sample image missing: {p} (run `interview-cv download-models --samples`)")
    return p


@pytest.fixture(scope="session")
def sample_image(sample_image_path: Path) -> np.ndarray:
    import cv2

    img = cv2.imread(str(sample_image_path))
    assert img is not None
    return img


@pytest.fixture(scope="session")
def sample_frames(sample_image: np.ndarray) -> list[np.ndarray]:
    """8 s of synthetic frames at 10 fps from the sample photo, face absent between 3 s and 5 s."""
    from interview_cv.samples import animate_image

    return list(animate_image(sample_image, seconds=8.0, fps=10.0, absent_from=3.0, absent_until=5.0))


@pytest.fixture
def blank_frames_list() -> list[np.ndarray]:
    from interview_cv.samples import blank_frames

    return list(blank_frames(seconds=2.0, fps=10.0))
