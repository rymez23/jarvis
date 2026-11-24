"""
THE EARS (stage 3): records your voice and turns it into text with faster-whisper.

Two steps:

1. RECORD (record_utterance)
   The microphone delivers sound as a stream of numbers ("samples"), thousands
   per second. We read it in small chunks (0.1 s each) and measure how loud
   each chunk is (RMS = root mean square, a standard loudness measure).
     - The first 0.3 s measures the room's background noise.
     - A chunk noticeably louder than that counts as speech.
     - Once you've started talking, about 1 second of quiet means you're done.
     - It also stops after max_seconds, in case the room never goes quiet.

2. TRANSCRIBE (Ears.transcribe)
   Whisper is a speech-recognition neural network. "base.en" is a small
   English-only version (~145 MB) that runs on the CPU. faster-whisper runs it
   with int8 math (small whole numbers instead of decimals), which is several
   times faster on a laptop with very little accuracy loss.

The model lives in  C:\\Users\\<you>\\.jarvis\\whisper\\  and works offline.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

WHISPER_DIR = Path.home() / ".jarvis" / "whisper"
WHISPER_RATE = 16_000            # Whisper expects 16,000 samples per second


def rms(chunk: np.ndarray) -> float:
    """Loudness of a chunk of audio (0 = silence)."""
    return float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size else 0.0


def resample(audio: np.ndarray, from_rate: int, to_rate: int = WHISPER_RATE) -> np.ndarray:
    """Convert audio to a different sample rate by straight-line interpolation.
    Good enough for speech; avoids needing a big audio library."""
    if from_rate == to_rate or audio.size == 0:
        return audio.astype(np.float32)
    duration = audio.size / from_rate
    new_times = np.linspace(0, duration, int(duration * to_rate), endpoint=False)
    old_times = np.arange(audio.size) / from_rate
    return np.interp(new_times, old_times, audio).astype(np.float32)


def find_speech(chunks: list[np.ndarray], chunk_seconds: float, calibrate_seconds: float = 0.3,
                silence_seconds: float = 1.0, sensitivity: float = 3.0, min_level: float = 0.01) -> int | None:
    """Look at recorded chunks and decide whether the speaker has finished.

    Returns the number of chunks to keep once speech has started and then
    stopped for `silence_seconds`; returns None if we should keep listening.
    Kept separate from the microphone code so it can be tested with fake audio.
    """
    n_cal = max(1, int(calibrate_seconds / chunk_seconds))
    if len(chunks) <= n_cal:
        return None
    noise = np.median([rms(c) for c in chunks[:n_cal]])
    threshold = max(noise * sensitivity, min_level)          # "speech" = clearly above the room noise
    loud = [rms(c) > threshold for c in chunks]
    if not any(loud[n_cal:]):
        return None                                          # haven't heard any speech yet
    quiet_needed = int(silence_seconds / chunk_seconds)
    last_loud = max(i for i, is_loud in enumerate(loud) if is_loud)
    if len(chunks) - 1 - last_loud >= quiet_needed:
        return last_loud + 1 + quiet_needed // 2               # keep a little trailing quiet
    return None


def record_utterance(max_seconds: float = 15, chunk_seconds: float = 0.1, on_start=None) -> np.ndarray:
    """Record from the default microphone until you stop talking. Returns 16 kHz audio."""
    import sounddevice as sd    # needs the PortAudio sound system; imported only when recording

    rate = int(sd.query_devices(kind="input")["default_samplerate"])
    chunk_frames = int(rate * chunk_seconds)
    chunks: list[np.ndarray] = []
    with sd.InputStream(samplerate=rate, channels=1, dtype="float32") as stream:
        if on_start:
            on_start()
        while len(chunks) * chunk_seconds < max_seconds:
            data, _overflowed = stream.read(chunk_frames)
            chunks.append(data[:, 0].copy())
            keep = find_speech(chunks, chunk_seconds)
            if keep is not None:
                chunks = chunks[:keep]
                break
    return resample(np.concatenate(chunks) if chunks else np.zeros(0), rate)


class Ears:
    def __init__(self, model: str = "base.en", folder: Path = WHISPER_DIR):
        from faster_whisper import WhisperModel
        # local_files_only: never go online; the model was downloaded during setup
        self.model = WhisperModel(model, device="cpu", compute_type="int8",
                                  download_root=str(folder), local_files_only=True)

    def transcribe(self, audio: np.ndarray) -> str:
        if audio.size < WHISPER_RATE * 0.3:          # under 0.3 s: nothing useful was said
            return ""
        segments, _info = self.model.transcribe(
            audio, language="en",
            beam_size=1,          # 1 = fastest: take the single best guess at each step
            vad_filter=True)      # skip stretches with no voice in them
        return " ".join(s.text.strip() for s in segments).strip()

    def listen(self, on_start=None) -> str:
        """Record one thing you say and return it as text."""
        return self.transcribe(record_utterance(on_start=on_start))
