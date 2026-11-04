"""Facial expression classification with HSEmotion weights running on ONNX Runtime.

Only the ONNX file from the HSEmotion project is used; preprocessing is
re-implemented here (ImageNet mean/std normalisation on an RGB 224x224 crop,
identical to the ``hsemotion-onnx`` package) so the module does not depend on
that package or its unused ``onnx`` dependency.

Expression classification is independent from landmark detection: it only needs
an RGB frame and a face bounding box.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..config import EXPRESSION_GROUPS, ExpressionConfig
from ..schemas import BBox, ExpressionObservation

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def crop_face(rgb: np.ndarray, bbox: BBox, margin: float = 0.2, top_extension: float = 0.0) -> Optional[np.ndarray]:
    """Crop the face region with a relative margin, clamped to the image. Returns None if empty.

    ``top_extension`` grows the box upward by that fraction of its height. The landmark
    box stops at the eyebrows, whereas the detector boxes HSEmotion was trained on include
    the forehead; extending upward brings the crop closer to the training distribution.
    """
    h, w = rgb.shape[:2]
    if top_extension:
        bbox = BBox(bbox.x, bbox.y - top_extension * bbox.h, bbox.w, bbox.h * (1.0 + top_extension))
    cx = bbox.x + bbox.w / 2.0
    cy = bbox.y + bbox.h / 2.0
    # Use a square crop based on the larger side so the face is not stretched by resizing.
    side = max(bbox.w * w, bbox.h * h) * (1.0 + margin)
    x0 = int(round(cx * w - side / 2.0))
    y0 = int(round(cy * h - side / 2.0))
    x1 = int(round(cx * w + side / 2.0))
    y1 = int(round(cy * h + side / 2.0))
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return rgb[y0:y1, x0:x1]


def preprocess(face_rgb: np.ndarray, size: int = 224) -> np.ndarray:
    """RGB uint8 crop -> float32 NCHW tensor normalised with ImageNet statistics."""
    x = cv2.resize(face_rgb, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    x = (x - _MEAN) / _STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[np.newaxis, ...], dtype=np.float32)


class ExpressionClassifier:
    def __init__(self, model_path: str | Path, config: Optional[ExpressionConfig] = None):
        import onnxruntime as ort

        self.config = config or ExpressionConfig()
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Expression ONNX model not found: {model_path}")
        self._session = ort.InferenceSession(str(model_path), providers=list(self.config.providers))
        self._input_name = self._session.get_inputs()[0].name
        n_out = self._session.get_outputs()[0].shape[-1]
        if isinstance(n_out, int) and n_out != len(self.config.labels):
            raise ValueError(f"Model outputs {n_out} classes but {len(self.config.labels)} labels are configured")

    def classify_crop(self, face_rgb: np.ndarray) -> ExpressionObservation:
        logits = self._session.run(None, {self._input_name: preprocess(face_rgb, self.config.input_size)})[0][0]
        probs = softmax(logits)
        labels = self.config.labels
        idx = int(np.argmax(probs))
        label = labels[idx]
        return ExpressionObservation(
            label=label,
            group=EXPRESSION_GROUPS.get(label, "other"),
            probabilities={lab: round(float(p), 4) for lab, p in zip(labels, probs)},
        )

    def classify(self, rgb: np.ndarray, bbox: BBox) -> Optional[ExpressionObservation]:
        crop = crop_face(rgb, bbox, self.config.crop_margin, self.config.forehead_extension)
        if crop is None:
            return None
        return self.classify_crop(crop)

    def close(self) -> None:  # symmetry with the other analyzers
        self._session = None
