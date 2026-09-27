"""Text-to-speech: Kokoro-82M via ``kokoro-onnx`` (offline, natural voice), falling back to
``pyttsx3`` (a native OS engine) when Kokoro's weights or package are unavailable. Both run
entirely offline, in-process; ``kokoro-onnx``/``pyttsx3``/``sounddevice`` are heavy/native
dependencies so their imports stay inside the backend constructors.

``TTSBackend.speak`` renders and plays audio synchronously, blocking the caller until playback
ends or another thread calls ``stop()``. ``VoicePlayer`` (in ``voice_queue.py``) relies on this
to serialise speech with one worker thread and to interrupt the current utterance for an alert.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

from actus.core.config import resolve_path

logger = logging.getLogger(__name__)


@runtime_checkable
class TTSBackend(Protocol):
    def speak(self, text: str) -> None: ...  # blocks until spoken or stop() is called
    def stop(self) -> None: ...  # interrupts an in-progress speak(); a no-op if idle
    def close(self) -> None: ...  # releases the engine / audio device


class KokoroTTS:
    """Kokoro-82M offline TTS, played back through ``sounddevice``."""

    def __init__(
        self,
        model_path: str | Path,
        voices_path: str | Path,
        *,
        voice: str = "af_heart",
        speed: float = 1.0,
    ) -> None:
        import sounddevice as sd
        from kokoro_onnx import Kokoro

        model_path, voices_path = resolve_path(model_path), resolve_path(voices_path)
        if not model_path.exists() or not voices_path.exists():
            raise FileNotFoundError(f"Kokoro weights not found: {model_path}, {voices_path}")
        self._kokoro = Kokoro(str(model_path), str(voices_path))
        self._sd = sd
        self.voice = voice
        self.speed = float(speed)

    def speak(self, text: str) -> None:
        if not text.strip():
            return
        samples, sample_rate = self._kokoro.create(
            text, voice=self.voice, speed=self.speed, lang="en-us"
        )
        self._sd.play(samples, sample_rate)
        self._sd.wait()

    def stop(self) -> None:
        self._sd.stop()

    def close(self) -> None:
        self.stop()


class Pyttsx3TTS:
    """``pyttsx3`` fallback: SAPI5 / NSSpeechSynthesizer / espeak, whichever the OS provides."""

    def __init__(self, *, voice: str | None = None, speed: float = 1.0) -> None:
        import pyttsx3

        self._engine = pyttsx3.init()
        if speed and speed != 1.0:
            rate = self._engine.getProperty("rate")
            self._engine.setProperty("rate", int(rate * speed))
        if voice:
            for v in self._engine.getProperty("voices") or []:
                name = f"{v.id or ''} {v.name or ''}".lower()
                if voice.lower() in name:
                    self._engine.setProperty("voice", v.id)
                    break

    def speak(self, text: str) -> None:
        if not text.strip():
            return
        self._engine.say(text)
        self._engine.runAndWait()

    def stop(self) -> None:
        self._engine.stop()

    def close(self) -> None:
        self.stop()


class NullTTS:
    """No-op backend: records spoken text instead of producing audio. Used by tests, and for a
    headless run with ``voice.enabled: false``."""

    def __init__(self) -> None:
        self.spoken: list[str] = []
        self.stop_calls = 0

    def speak(self, text: str) -> None:
        self.spoken.append(text)

    def stop(self) -> None:
        self.stop_calls += 1

    def close(self) -> None:
        pass


def create_backend(
    tts_cfg,
    *,
    kokoro_cls: type = KokoroTTS,
    pyttsx3_cls: type = Pyttsx3TTS,
) -> TTSBackend:
    """Build the backend named by ``tts_cfg["engine"]`` ("kokoro" | "pyttsx3").

    "kokoro" falls back to pyttsx3 if Kokoro's package, weights, or audio device are
    unavailable, so the voice lane never hard-fails setup. ``kokoro_cls``/``pyttsx3_cls`` are
    injectable so tests can exercise the fallback without the real packages installed.
    """
    engine = tts_cfg.get("engine", "kokoro")
    voice = tts_cfg.get("voice", "af_heart")
    speed = tts_cfg.get("speed", 1.0)
    if engine == "kokoro":
        try:
            return kokoro_cls(
                tts_cfg["kokoro_model"], tts_cfg["kokoro_voices"], voice=voice, speed=speed
            )
        except Exception as exc:  # missing package, missing weights, no audio device, ...
            logger.warning("Kokoro TTS unavailable (%s); falling back to pyttsx3", exc)
    elif engine != "pyttsx3":
        raise ValueError(f"unknown tts engine {engine!r}")
    return pyttsx3_cls(voice=voice, speed=speed)
