import numpy as np
import pytest

from interview_cv.video import FrameSampler, FrameSequenceSource, WebcamVideoSource, open_webcam


def _frames(n: int, size=(320, 240)):
    return [np.full((size[1], size[0], 3), i, dtype=np.uint8) for i in range(n)]


def test_frame_sequence_source_lifecycle():
    src = FrameSequenceSource(_frames(20), fps=10.0)
    with src:
        assert src.opened and src.width == 320 and src.height == 240 and src.fps == 10.0
        frames = list(src)
    assert src.closed
    assert len(frames) == 20
    assert [f.index for f in frames] == list(range(20))
    ts = [f.timestamp_s for f in frames]
    assert ts == sorted(ts) and ts[1] - ts[0] == pytest.approx(0.1)
    assert frames[0].image.shape == (240, 320, 3)


def test_sampler_reduces_rate():
    with FrameSequenceSource(_frames(20), fps=10.0) as src:
        sampled = list(FrameSampler(src, sample_fps=2.0))
    # 2 s at 2 fps -> 4 frames (t = 0, 0.5, 1.0, 1.5)
    assert [round(f.timestamp_s, 1) for f in sampled] == [0.0, 0.5, 1.0, 1.5]


def test_sampler_passthrough_and_max_seconds():
    with FrameSequenceSource(_frames(20), fps=10.0) as src:
        sampler = FrameSampler(src, sample_fps=None, max_seconds=0.95)
        sampled = list(sampler)
    assert len(sampled) == 10  # t = 0.0 .. 0.9
    assert sampler.frames_seen == 10


def test_sampler_should_sample_skips_missed_slots():
    sampler = FrameSampler(FrameSequenceSource([], fps=1.0), sample_fps=5.0)
    assert sampler.should_sample(0.0)
    assert not sampler.should_sample(0.1)
    assert sampler.should_sample(0.2)
    # A long stall: the next frame after it must be sampled immediately, not several times.
    assert sampler.should_sample(2.0)
    assert not sampler.should_sample(2.1)
    assert sampler.should_sample(2.2)


def test_open_webcam_dispatch():
    assert isinstance(open_webcam(0), WebcamVideoSource)
    assert open_webcam("1").device == 1
    assert open_webcam("webcam:2").device == 2
    with pytest.raises(ValueError):
        open_webcam("interview.mp4")  # file input is not supported


def test_webcam_source_is_closed_without_opening():
    src = WebcamVideoSource(0)
    assert not src.is_open
    src.close()  # no-op, must not raise
