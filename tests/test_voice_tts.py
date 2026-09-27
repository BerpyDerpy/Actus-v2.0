"""Tests for the TTS backend factory and fallback logic (T-VOICE).

``create_backend`` accepts injectable backend classes precisely so this can be tested without
the real ``kokoro-onnx`` / ``pyttsx3`` packages installed (they're optional, heavy deps under
``uv sync --extra voice``)."""

from __future__ import annotations

import pytest

from actus.voice.tts import NullTTS, create_backend


class _FakeOK:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs


class _FakeFails:
    def __init__(self, *args, **kwargs) -> None:
        raise FileNotFoundError("no weights here")


_CFG = {
    "engine": "kokoro",
    "kokoro_model": "models/voice/kokoro-v1.0.onnx",
    "kokoro_voices": "models/voice/voices-v1.0.bin",
    "voice": "af_heart",
    "speed": 1.0,
}


def test_kokoro_engine_uses_kokoro_backend_when_it_succeeds():
    backend = create_backend(_CFG, kokoro_cls=_FakeOK, pyttsx3_cls=_FakeFails)
    assert isinstance(backend, _FakeOK)


def test_kokoro_engine_falls_back_to_pyttsx3_on_failure():
    backend = create_backend(_CFG, kokoro_cls=_FakeFails, pyttsx3_cls=_FakeOK)
    assert isinstance(backend, _FakeOK)


def test_pyttsx3_engine_never_touches_kokoro():
    cfg = dict(_CFG, engine="pyttsx3")

    class _ExplodesIfCalled:
        def __init__(self, *a, **k) -> None:
            raise AssertionError("kokoro_cls should not be constructed")

    backend = create_backend(cfg, kokoro_cls=_ExplodesIfCalled, pyttsx3_cls=_FakeOK)
    assert isinstance(backend, _FakeOK)


def test_unknown_engine_is_rejected():
    with pytest.raises(ValueError):
        create_backend(dict(_CFG, engine="carrier_pigeon"), kokoro_cls=_FakeOK, pyttsx3_cls=_FakeOK)


def test_null_tts_records_spoken_text_and_stop_calls():
    backend = NullTTS()
    backend.speak("hello")
    backend.speak("")  # NullTTS doesn't filter; real backends skip empty text
    backend.stop()
    backend.close()
    assert backend.spoken == ["hello", ""]
    assert backend.stop_calls == 1


# -- real backends: only run where the optional package is installed (`uv sync --extra voice`) --


def test_real_kokoro_backend_raises_on_missing_weights():
    pytest.importorskip("kokoro_onnx")
    from actus.voice.tts import KokoroTTS

    with pytest.raises(FileNotFoundError):
        KokoroTTS("models/voice/does-not-exist.onnx", "models/voice/does-not-exist.bin")


def test_real_pyttsx3_backend_constructs_and_speaks_nothing_for_empty_text():
    pytest.importorskip("pyttsx3")
    from actus.voice.tts import Pyttsx3TTS

    backend = Pyttsx3TTS()
    try:
        backend.speak("")  # must not raise / must not block on an empty utterance
    finally:
        backend.close()
