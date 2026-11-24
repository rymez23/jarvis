"""
Tests for the voice. A fake Piper stands in for the real one, so these run
anywhere (no voice download, no speakers). Run with:  python -m pytest
"""
import io
import sys
import types
import wave

import pytest

from jarvis import voice


def test_clean_for_speech():
    assert voice.clean_for_speech("**Hello** _PJ_ 👋  see https://x.com/a") == "Hello PJ see a link"


def test_installed_voices(tmp_path):
    (tmp_path / "en_GB-alan-medium.onnx").write_bytes(b"")
    (tmp_path / "en_GB-alan-medium.onnx.json").write_text("{}")
    assert voice.installed_voices(tmp_path) == ["en_GB-alan-medium"]


@pytest.fixture
def fake_piper(monkeypatch):
    """Pretend piper is installed: synthesize_wav writes 0.1 s of silence."""
    class FakeVoice:
        @classmethod
        def load(cls, path):
            return cls()

        def synthesize_wav(self, text, wav, syn_config=None):
            wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(22050)
            wav.writeframes(b"\x00\x00" * 2205)
            FakeVoice.last_text = text

    mod = types.SimpleNamespace(PiperVoice=FakeVoice, SynthesisConfig=lambda **k: k)
    monkeypatch.setitem(sys.modules, "piper", mod)
    return FakeVoice


def test_say_cleans_text_and_plays_a_real_wav(tmp_path, fake_piper):
    (tmp_path / "v.onnx").write_bytes(b"")
    played = []
    v = voice.Voice("v", folder=tmp_path, player=played.append)
    v.say("*Good* morning 🌞")
    assert fake_piper.last_text == "Good morning"
    with wave.open(io.BytesIO(played[0])) as w:          # the bytes are a valid WAV file
        assert w.getframerate() == 22050 and w.getnframes() == 2205


def test_missing_voice_lists_what_is_installed(tmp_path, fake_piper):
    (tmp_path / "en_GB-alan-medium.onnx").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="Installed: en_GB-alan-medium"):
        voice.Voice("nope", folder=tmp_path)


def test_empty_text_plays_nothing(tmp_path, fake_piper):
    (tmp_path / "v.onnx").write_bytes(b"")
    played = []
    voice.Voice("v", folder=tmp_path, player=played.append).say("  ** ")
    assert played == []
