"""
Tests for the ears, using made-up audio (no microphone or Whisper model needed).
Run with:  python -m pytest
"""
import numpy as np

from jarvis.ears import find_speech, resample, rms

CHUNK = 0.1                                       # seconds per chunk, as in record_utterance
quiet = lambda: np.full(1600, 0.001, np.float32)   # background hiss
loud = lambda: np.full(1600, 0.2, np.float32)      # someone talking


def test_rms():
    assert rms(np.zeros(10)) == 0.0
    assert abs(rms(np.full(10, 0.5)) - 0.5) < 1e-9


def test_keeps_listening_before_anyone_speaks():
    chunks = [quiet() for _ in range(30)]          # 3 s of silence
    assert find_speech(chunks, CHUNK) is None


def test_keeps_listening_while_talking():
    chunks = [quiet() for _ in range(3)] + [loud() for _ in range(10)]
    assert find_speech(chunks, CHUNK) is None


def test_stops_after_a_second_of_quiet():
    chunks = [quiet() for _ in range(3)] + [loud() for _ in range(10)]
    chunks += [quiet() for _ in range(9)]           # 0.9 s quiet: not yet
    assert find_speech(chunks, CHUNK) is None
    chunks.append(quiet())                          # 1.0 s quiet: done
    keep = find_speech(chunks, CHUNK)
    assert keep == 13 + 5                           # all speech + half a second of tail


def test_noisy_room_raises_the_bar():
    hum = lambda: np.full(1600, 0.05, np.float32)  # e.g. a fan
    chunks = [hum() for _ in range(3)] + [np.full(1600, 0.08, np.float32) for _ in range(5)]
    assert find_speech(chunks, CHUNK) is None       # 0.08 isn't 3x louder than the fan


def test_resample_44k_to_16k():
    one_second = np.sin(np.linspace(0, 2 * np.pi * 440, 44_100)).astype(np.float32)
    out = resample(one_second, 44_100)
    assert out.size == 16_000 and out.dtype == np.float32
