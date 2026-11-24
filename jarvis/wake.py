"""
THE WAKE WORD (stage 5): waits for "Hey Jarvis" so you don't need the keyboard.

How it works:
  openWakeWord runs a tiny neural network (about 1 MB) on the microphone
  stream, every 80 milliseconds. For each slice of audio it outputs a score
  from 0 to 1: how much the last couple of seconds sounded like "hey jarvis".
  When the score goes over `threshold`, Jarvis wakes up.

  It's cheap enough to run all the time, unlike Whisper, which would be far too
  slow to transcribe everything you say just to spot one phrase.

  Threshold trade-off (set "wake" -> "threshold" in settings.json):
    lower  (e.g. 0.3) = wakes more easily, but more false alarms from the TV
    higher (e.g. 0.7) = fewer false alarms, but you may need to repeat yourself

The model needs 16 kHz, 16-bit audio in chunks of 1280 samples (80 ms). Your
mic records at 44.1 kHz, so each chunk is resampled first (resample() from ears.py).
"""
from __future__ import annotations

import sys
from typing import Iterable

import numpy as np

from .ears import WHISPER_RATE, resample

FRAME_SECONDS = 0.08                      # 80 ms, what openWakeWord expects
FRAME_SAMPLES = int(WHISPER_RATE * FRAME_SECONDS)   # 1280 samples at 16 kHz


def to_int16(audio: np.ndarray) -> np.ndarray:
    """Float audio (-1.0..1.0) -> 16-bit whole numbers (-32768..32767), the format the model wants."""
    return (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)


def beep() -> None:
    """A short tone so you know Jarvis is listening."""
    if sys.platform == "win32":
        import winsound
        winsound.Beep(880, 120)        # 880 Hz for 120 ms


class WakeWord:
    def __init__(self, threshold: float = 0.5, name: str = "hey_jarvis", model=None):
        if model is None:
            from openwakeword.model import Model
            # inference_framework="onnx": the Windows-friendly engine (Piper uses it too)
            model = Model(wakeword_models=[name], inference_framework="onnx")
        self.model = model
        self.name = name
        self.threshold = threshold

    def heard(self, frame16: np.ndarray) -> float:
        """Score one 80 ms frame of 16 kHz int16 audio."""
        return float(self.model.predict(frame16).get(self.name, 0.0))

    def wait(self, frames: Iterable[np.ndarray], rate: int) -> float:
        """Read frames (float audio at `rate`) until the wake word is heard; return its score.

        Takes any stream of frames, so tests can feed it made-up audio instead of a mic.
        """
        for frame in frames:
            score = self.heard(to_int16(resample(frame, rate)))
            if score >= self.threshold:
                self.model.reset()          # clear its memory so the same "hey jarvis" doesn't fire twice
                return score
        return 0.0

    def wait_for_mic(self) -> float:
        """Listen on the default microphone until "Hey Jarvis" is heard."""
        import sounddevice as sd

        rate = int(sd.query_devices(kind="input")["default_samplerate"])
        frame_len = int(rate * FRAME_SECONDS)

        def mic_frames():
            with sd.InputStream(samplerate=rate, channels=1, dtype="float32") as stream:
                while True:
                    data, _overflowed = stream.read(frame_len)
                    yield data[:, 0]

        # The stream closes when wait() returns, freeing the mic for ears.py to record.
        return self.wait(mic_frames(), rate)
