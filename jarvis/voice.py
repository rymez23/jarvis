"""
THE VOICE (stage 2): turns Jarvis's text replies into speech with Piper.

How it works:
  Piper is a small neural network (an .onnx file, about 60 MB) trained on
  one person's recorded voice. Give it text, and it produces audio samples
  that sound like that person. It runs on the CPU, faster than real time.

  We ask Piper to write a WAV file into memory (io.BytesIO, a "file" that
  lives in RAM instead of on disk), then hand those bytes to winsound,
  Windows' built-in sound player. Nothing is saved to disk.

Voices live in  C:\\Users\\<you>\\.jarvis\\voices\\  as pairs of files:
    en_GB-alan-medium.onnx        the voice model
    en_GB-alan-medium.onnx.json   its settings (sample rate, etc.)
Get more with:  python -m piper.download_voices <voice name> --download-dir <that folder>
Browse voices:  https://rhasspy.github.io/piper-samples/
"""
from __future__ import annotations

import io
import re
import sys
import wave
from pathlib import Path

VOICES_DIR = Path.home() / ".jarvis" / "voices"


def installed_voices(folder: Path = VOICES_DIR) -> list[str]:
    """Names of the downloaded voices, e.g. ['en_GB-alan-medium', ...]."""
    return sorted(p.name.removesuffix(".onnx") for p in folder.glob("*.onnx"))


def clean_for_speech(text: str) -> str:
    """Remove things that sound silly when read aloud: Markdown symbols, emoji, links."""
    text = re.sub(r"https?://\S+", "a link", text)
    text = re.sub(r"[*_#`>|~]", "", text)                     # Markdown symbols
    text = re.sub(r"[\U0001F000-\U0001FAFF\u2600-\u27BF]", "", text)   # emoji
    return re.sub(r"\s+", " ", text).strip()


def play_wav(data: bytes) -> None:
    """Play WAV bytes through the speakers and wait until finished (Windows only)."""
    if sys.platform != "win32":
        return
    import winsound
    winsound.PlaySound(data, winsound.SND_MEMORY)


class Voice:
    def __init__(self, name: str, speed: float = 1.0, folder: Path = VOICES_DIR, player=play_wav):
        """speed: 1.0 is normal; 0.9 is a bit faster, 1.2 slower (Piper calls it length_scale)."""
        from piper import PiperVoice, SynthesisConfig   # imported here: only needed when speaking

        model = folder / f"{name}.onnx"
        if not model.exists():
            have = ", ".join(installed_voices(folder)) or "none"
            raise FileNotFoundError(f"Voice '{name}' not found in {folder}. Installed: {have}")
        self.name = name
        self.piper = PiperVoice.load(model)      # loads the neural network into memory (~1 s)
        self.config = SynthesisConfig(length_scale=speed)
        self.player = player

    def to_wav(self, text: str) -> bytes:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            self.piper.synthesize_wav(text, wav, syn_config=self.config)
        return buffer.getvalue()

    def say(self, text: str) -> None:
        text = clean_for_speech(text)
        if text:
            self.player(self.to_wav(text))
