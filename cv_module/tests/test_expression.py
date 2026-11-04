import numpy as np
import pytest

from interview_cv.analyzers.expression import ExpressionClassifier, crop_face, preprocess, softmax
from interview_cv.config import EXPRESSION_GROUPS, HSEMOTION_LABELS
from interview_cv.schemas import BBox


def test_softmax_sums_to_one():
    p = softmax(np.array([1.0, 2.0, 3.0]))
    assert p.sum() == pytest.approx(1.0)
    assert np.argmax(p) == 2
    assert np.isfinite(softmax(np.array([1000.0, 1000.0]))).all()


def test_preprocess_shape_and_normalisation():
    img = np.full((50, 80, 3), 255, dtype=np.uint8)
    x = preprocess(img, 224)
    assert x.shape == (1, 3, 224, 224) and x.dtype == np.float32
    # white pixel -> (1 - mean) / std per channel
    assert x[0, 0, 0, 0] == pytest.approx((1 - 0.485) / 0.229, abs=1e-4)
    assert x[0, 2, 0, 0] == pytest.approx((1 - 0.406) / 0.225, abs=1e-4)


def test_crop_face_square_with_margin_and_clamping():
    rgb = np.zeros((200, 300, 3), dtype=np.uint8)
    crop = crop_face(rgb, BBox(0.4, 0.3, 0.2, 0.3), margin=0.0)
    assert crop.shape[0] == crop.shape[1] == 60  # square on the larger side (0.3 * 200)
    crop = crop_face(rgb, BBox(0.0, 0.0, 0.1, 0.1), margin=1.0)
    assert crop is not None and crop.shape[0] > 0  # clamped at the image border
    assert crop_face(rgb, BBox(0.99, 0.99, 0.001, 0.001), margin=0.0) is None


def test_groups_cover_all_labels():
    assert set(HSEMOTION_LABELS) == set(EXPRESSION_GROUPS)
    assert set(EXPRESSION_GROUPS.values()) == {"neutral", "positive", "other"}


@pytest.mark.integration
def test_classifier_on_sample_image(model_paths, sample_image):
    import cv2

    clf = ExpressionClassifier(model_paths.expression_onnx_path)
    rgb = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)
    # Approximate face box for the sample photo (face occupies the upper-middle region).
    obs = clf.classify(rgb, BBox(0.38, 0.13, 0.27, 0.21))
    assert obs is not None
    assert obs.label in HSEMOTION_LABELS
    assert obs.group in {"neutral", "positive", "other"}
    assert sum(obs.probabilities.values()) == pytest.approx(1.0, abs=1e-2)
