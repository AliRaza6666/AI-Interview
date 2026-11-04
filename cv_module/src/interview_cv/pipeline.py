"""Frame processing pipeline: runs the analyzers on one frame at a time.

Capture (webcam) and session bookkeeping live elsewhere (``video`` and ``session``);
this module only turns a BGR frame plus a timestamp into a ``FrameObservation``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .analyzers.base import FrameContext
from .analyzers.expression import ExpressionClassifier
from .analyzers.face import FaceAnalyzer, FaceResult
from .analyzers.gaze import estimate_gaze
from .analyzers.posture import PostureAnalyzer
from .config import PipelineConfig
from .schemas import FaceObservation, FrameObservation

log = logging.getLogger(__name__)


@dataclass
class FrameAnalysis:
    """Observation plus the raw landmark arrays, for visualization. Not serialized."""

    observation: FrameObservation
    face_landmarks: Optional[np.ndarray] = None
    pose_landmarks: Optional[np.ndarray] = None


def library_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in ("mediapipe", "onnxruntime", "cv2", "numpy"):
        try:
            mod = __import__(name)
            versions[name] = str(getattr(mod, "__version__", "?"))
        except Exception:  # pragma: no cover
            pass
    return versions


class InterviewCVPipeline:
    """Processes frames and produces :class:`FrameObservation` records.

    Each analyzer is isolated: an exception inside one is logged, recorded in
    ``FrameObservation.errors`` and leaves that field ``None`` for the frame.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        cfg = self.config
        self.face: Optional[FaceAnalyzer] = None
        self.expression: Optional[ExpressionClassifier] = None
        self.posture: Optional[PostureAnalyzer] = None
        if cfg.enable_face or cfg.enable_gaze or cfg.enable_expression:
            self.face = FaceAnalyzer(cfg.models.face_landmarker_path, cfg.face)
        if cfg.enable_expression:
            self.expression = ExpressionClassifier(cfg.models.expression_onnx_path, cfg.expression)
        if cfg.enable_posture:
            self.posture = PostureAnalyzer(cfg.models.pose_landmarker_path, cfg.posture)

    # ------------------------------------------------------------------ per frame
    def analyze(self, image_bgr: np.ndarray, timestamp_s: float, frame_index: int = 0) -> FrameAnalysis:
        ctx = FrameContext(image_bgr, timestamp_s, frame_index)
        obs = FrameObservation(timestamp_s=round(ctx.timestamp_s, 4), frame_index=frame_index)
        face_result: Optional[FaceResult] = None

        if self.face is not None:
            try:
                face_result = self.face.process(ctx)
                obs.face = face_result.observation
            except Exception as exc:  # noqa: BLE001 - keep the pipeline alive
                log.warning("face analyzer failed at %.2fs: %s", ctx.timestamp_s, exc)
                obs.errors.append(f"face: {exc}")
                obs.face = FaceObservation(present=False)

        face_present = bool(face_result and face_result.observation.present)

        if self.config.enable_gaze and face_result is not None:
            if face_present:
                try:
                    aspect = ctx.width / float(ctx.height) if ctx.height else 1.0
                    obs.gaze = estimate_gaze(face_result.landmarks, face_result.observation.head_pose, self.config.gaze, aspect)
                except Exception as exc:  # noqa: BLE001
                    log.warning("gaze estimation failed at %.2fs: %s", ctx.timestamp_s, exc)
                    obs.errors.append(f"gaze: {exc}")
            # No face -> gaze stays None (unknown), reported as "no face" rather than "away".

        if self.expression is not None and face_present and face_result and face_result.observation.bbox:
            try:
                obs.expression = self.expression.classify(ctx.rgb, face_result.observation.bbox)
            except Exception as exc:  # noqa: BLE001
                log.warning("expression classifier failed at %.2fs: %s", ctx.timestamp_s, exc)
                obs.errors.append(f"expression: {exc}")

        pose_landmarks = None
        if self.posture is not None:
            try:
                obs.posture = self.posture.process(ctx)
                pose_landmarks = self.posture.last_landmarks
            except Exception as exc:  # noqa: BLE001
                log.warning("posture analyzer failed at %.2fs: %s", ctx.timestamp_s, exc)
                obs.errors.append(f"posture: {exc}")

        return FrameAnalysis(
            observation=obs,
            face_landmarks=face_result.landmarks if face_result else None,
            pose_landmarks=pose_landmarks,
        )

    def finalize(self) -> None:
        """Call once at the end of a session: finalizes the posture baseline for short sessions."""
        if self.posture is not None and not self.posture.baseline.ready:
            self.posture.baseline.finalize()

    def model_names(self) -> dict[str, str]:
        cfg = self.config
        return {
            "face_landmarker": cfg.models.face_landmarker,
            "pose_landmarker": cfg.models.pose_landmarker,
            "expression": cfg.models.expression_onnx,
        }

    def close(self) -> None:
        for a in (self.face, self.expression, self.posture):
            if a is not None:
                try:
                    a.close()
                except Exception:  # pragma: no cover
                    pass

    def __enter__(self) -> "InterviewCVPipeline":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
