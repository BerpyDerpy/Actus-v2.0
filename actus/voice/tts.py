"""Text-to-speech: Kokoro-82M via ``kokoro-onnx`` (offline, natural voice), falling back to
``pyttsx3`` (a native OS engine) when Kokoro's weights or package are unavailable. Both run
entirely offline, in-process; ``kokoro-onnx``/``pyttsx3``/``sounddevice`` are heavy/native
dependencies so their imports stay inside the backend constructors.

``TTSBackend.speak`` renders and plays audio synchronously, blocking the caller until playback
ends or another thread calls ``stop()``. ``VoicePlayer`` (in ``voice_queue.py``) relies on this
to serialise speech with one worker thread and to interrupt the current utterance for an alert.
``stop()`` also cancels an utterance that is still being synthesised, so it never starts playing.

``pyttsx3`` (SAPI5/COM on Windows) is not safe to drive from several threads, so
``Pyttsx3TTS`` owns one thread that creates the engine and runs every ``say``/``runAndWait``;
``stop()`` only sets a flag, which the engine's own ``started-word`` callback acts on.
"""

from __future__ import annotations

import contextlib
import logging
import queue
import threading
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
        self._lock = threading.Lock()
        self._cancel = threading.Event()

    def speak(self, text: str) -> None:
        if not text.strip():
            return
        self._cancel.clear()
        samples, sample_rate = self._kokoro.create(
            text, voice=self.voice, speed=self.speed, lang="en-us"
        )
        with self._lock:
            # stop() during synthesis cancels this utterance before it plays.
            if self._cancel.is_set():
                return
            self._sd.play(samples, sample_rate)
        self._sd.wait()

    def stop(self) -> None:
        with self._lock:
            self._cancel.set()
            self._sd.stop()

    def close(self) -> None:
        self.stop()


class Pyttsx3TTS:
    """``pyttsx3`` fallback: SAPI5 / NSSpeechSynthesizer / espeak, whichever the OS provides.

    The engine lives on its own thread (created there, used only there). Construction waits for
    it to initialise, so a broken engine still fails in ``create_backend`` as before."""

    def __init__(
        self, *, voice: str | None = None, speed: float = 1.0, init_timeout_s: float = 10.0
    ) -> None:
        import pyttsx3  # noqa: F401  (fail fast here if the package is missing)

        self._voice = voice
        self._speed = speed
        self._engine = None
        self._requests: queue.Queue = queue.Queue()
        self._stop_flag = threading.Event()
        self._ready = threading.Event()
        self._init_error: BaseException | None = None
        self._thread = threading.Thread(target=self._loop, name="pyttsx3", daemon=True)
        self._thread.start()
        if not self._ready.wait(init_timeout_s):
            raise RuntimeError("pyttsx3 engine did not initialise")
        if self._init_error is not None:
            raise self._init_error

    def _init_engine(self):
        import pyttsx3

        engine = pyttsx3.init()
        if self._speed and self._speed != 1.0:
            rate = engine.getProperty("rate")
            engine.setProperty("rate", int(rate * self._speed))
        if self._voice:
            for v in engine.getProperty("voices") or []:
                name = f"{v.id or ''} {v.name or ''}".lower()
                if self._voice.lower() in name:
                    engine.setProperty("voice", v.id)
                    break
        engine.connect("started-word", self._on_word)
        return engine

    def _on_word(self, name, location, length) -> None:
        # Runs on the engine's own thread, inside runAndWait: the safe place to stop it.
        if self._stop_flag.is_set() and self._engine is not None:
            self._engine.stop()

    def _loop(self) -> None:
        try:
            self._engine = self._init_engine()
        except BaseException as exc:
            self._init_error = exc
            self._ready.set()
            return
        self._ready.set()
        while True:
            req = self._requests.get()
            if req is None:
                break
            text, done, errors = req
            try:
                if not self._stop_flag.is_set():
                    self._engine.say(text)
                    self._engine.runAndWait()
            except Exception as exc:
                errors.append(exc)
            finally:
                done.set()
        with contextlib.suppress(Exception):
            self._engine.stop()

    def speak(self, text: str) -> None:
        if not text.strip():
            return
        if not self._thread.is_alive():
            raise RuntimeError("pyttsx3 engine thread is not running")
        self._stop_flag.clear()
        done = threading.Event()
        errors: list[BaseException] = []
        self._requests.put((text, done, errors))
        done.wait()
        if errors:
            raise errors[0]

    def stop(self) -> None:
        self._stop_flag.set()

    def close(self) -> None:
        self.stop()
        self._requests.put(None)
        self._thread.join(timeout=2.0)


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
