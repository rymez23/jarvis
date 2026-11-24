"""
Tests for the wake word, using a fake model (no microphone or openWakeWord needed).
Run with:  python -m pytest
"""
import numpy as np

from jarvis.wake import FRAME_SAMPLES, WakeWord, to_int16


class FakeModel:
    """Scores frames from a list, and records what it was fed."""
    def __init__(self, scores):
        self.scores = list(scores)
        self.fed = []
        self.resets = 0

    def predict(self, frame):
        self.fed.append(frame)
        return {"hey_jarvis": self.scores.pop(0) if self.scores else 0.0}

    def reset(self):
        self.resets += 1


def frames(n, rate=44_100):
    return (np.zeros(int(rate * 0.08), np.float32) for _ in range(n))


def test_to_int16_clips_and_scales():
    out = to_int16(np.array([0.0, 0.5, 1.0, -1.0, 2.0], np.float32))
    assert out.dtype == np.int16
    assert list(out) == [0, 16383, 32767, -32767, 32767]


def test_wakes_when_score_crosses_threshold():
    model = FakeModel([0.1, 0.3, 0.8, 0.9])
    score = WakeWord(threshold=0.5, model=model).wait(frames(10), rate=44_100)
    assert score == 0.8
    assert len(model.fed) == 3                     # stopped at the first score over 0.5
    assert model.resets == 1                       # cleared so it won't fire twice


def test_frames_are_resampled_to_what_the_model_expects():
    model = FakeModel([0.9])
    WakeWord(model=model).wait(frames(1, rate=44_100), rate=44_100)
    assert model.fed[0].dtype == np.int16 and model.fed[0].size == FRAME_SAMPLES   # 1280


def test_higher_threshold_ignores_weak_matches():
    model = FakeModel([0.4, 0.45, 0.6])
    assert WakeWord(threshold=0.7, model=model).wait(frames(3), rate=44_100) == 0.0
