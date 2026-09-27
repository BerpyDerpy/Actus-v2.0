"""Offline speech recognition: Vosk small English model, restricted to a fixed command grammar.

The microphone is gated: recognition runs only while an ``Ask`` is open (a question is pending)
or while push-to-talk is held — never continuously. That keeps CPU/mic use near zero the rest of
the time and means recognition needs no network at all, ever (Vosk's grammar mode never touches
the internet regardless).

Recognised speech becomes an ``Answer`` (only while a question is open) or a ``Command`` (works
any time the gate is open), both published to the bus for the engine / GUI to consume.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from pathlib import Path

from actus.core.bus import Bus, BusClosed
from actus.core.config import resolve_path
from actus.core.contracts import (
    Answer,
    AnswerValue,
    Ask,
    Command,
    CommandName,
    InputVia,
    Topic,
)

GRAMMAR_WORDS: tuple[str, ...] = (
    "yes",
    "no",
    "next",
    "repeat",
    "what's next",
    "confirm",
    "cancel",
)

_YES_WORDS = frozenset({"yes", "confirm"})
_NO_WORDS = frozenset({"no", "cancel"})
_COMMAND_WORDS: dict[str, CommandName] = {
    "next": CommandName.NEXT,
    "repeat": CommandName.REPEAT,
    "what's next": CommandName.WHATS_NEXT,
    "whats next": CommandName.WHATS_NEXT,  # Vosk's grammar output drops the apostrophe
    "confirm": CommandName.CONFIRM,
    "cancel": CommandName.CANCEL,
}


class AskGate:
    """Whether the ASR should be capturing: open while a question is pending, or while
    push-to-talk is held. Thread-safe; fed by ``Ask``/``Answer`` from the bus and by a key
    handler for push-to-talk."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._open_question_id: str | None = None
        self._ptt_held = False

    def on_ask(self, ask: Ask) -> None:
        with self._lock:
            self._open_question_id = ask.question_id

    def on_answer(self, answer: Answer) -> None:
        with self._lock:
            if self._open_question_id == answer.question_id:
                self._open_question_id = None

    def cancel(self) -> None:
        """Close the gate without an Answer, e.g. the question timed out."""
        with self._lock:
            self._open_question_id = None

    def set_push_to_talk(self, held: bool) -> None:
        with self._lock:
            self._ptt_held = held

    @property
    def open_question_id(self) -> str | None:
        with self._lock:
            return self._open_question_id

    @property
    def listening(self) -> bool:
        with self._lock:
            return self._open_question_id is not None or self._ptt_held


def parse_utterance(
    text: str, *, open_question_id: str | None, t: float
) -> Answer | Command | None:
    """Map recognised text to an ``Answer`` (only while a question is open) or a ``Command``."""
    words = text.strip().lower()
    if not words:
        return None
    if open_question_id is not None:
        if words in _YES_WORDS:
            return Answer(open_question_id, AnswerValue.YES, InputVia.VOICE, t=t)
        if words in _NO_WORDS:
            return Answer(open_question_id, AnswerValue.NO, InputVia.VOICE, t=t)
    cmd = _COMMAND_WORDS.get(words)
    if cmd is not None:
        return Command(cmd, InputVia.VOICE, t=t)
    return None


class VoskRecognizer:
    """Thin wrapper around ``vosk.KaldiRecognizer``, restricted to ``GRAMMAR_WORDS``."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        sample_rate: int = 16000,
        grammar: tuple[str, ...] = GRAMMAR_WORDS,
    ) -> None:
        from vosk import KaldiRecognizer, Model

        model_path = resolve_path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Vosk model not found: {model_path}")
        self._model = Model(str(model_path))
        self.sample_rate = sample_rate
        self._grammar_json = json.dumps([*grammar, "[unk]"])
        self._new_recognizer_ctor = lambda: KaldiRecognizer(
            self._model, self.sample_rate, self._grammar_json
        )
        self._rec = self._new_recognizer_ctor()

    def accept(self, pcm16_bytes: bytes) -> str | None:
        """Feed one chunk of 16-bit mono PCM. Returns the utterance's text once it ends."""
        if self._rec.AcceptWaveform(pcm16_bytes):
            return json.loads(self._rec.Result()).get("text", "")
        return None

    def flush(self) -> str:
        """Force-end the current utterance (e.g. the gate just closed) and reset for the next."""
        text = json.loads(self._rec.FinalResult()).get("text", "")
        self._rec = self._new_recognizer_ctor()
        return text


class AsrService:
    """Gated microphone listener: feeds audio to a recognizer only while ``gate.listening``, and
    publishes the resulting ``Answer``/``Command`` to the bus.

    ``feed`` is the hardware-independent core (unit-testable with a fake recognizer); real
    microphone capture is wired up separately by ``start_mic`` (``sounddevice``, real hardware).
    """

    def __init__(
        self,
        recognizer,
        gate: AskGate,
        *,
        bus: Bus | None = None,
        on_result: Callable[[str], None] | None = None,
        block_ms: int = 100,
    ) -> None:
        self.recognizer = recognizer
        self.gate = gate
        self.bus = bus
        self.on_result = on_result
        self.block_ms = block_ms
        self._stream = None
        self._gate_thread: threading.Thread | None = None
        self._was_listening = False

    def feed(self, pcm16_bytes: bytes) -> None:
        """Feed one chunk of mic audio. Dropped (and any in-progress utterance discarded)
        unless the gate is open."""
        if not self.gate.listening:
            if self._was_listening:
                self.recognizer.flush()
            self._was_listening = False
            return
        self._was_listening = True
        text = self.recognizer.accept(pcm16_bytes)
        if text is not None:
            self._handle(text)

    def _handle(self, text: str) -> None:
        if self.on_result is not None:
            self.on_result(text)
        msg = parse_utterance(text, open_question_id=self.gate.open_question_id, t=time.monotonic())
        if msg is None:
            return
        if isinstance(msg, Answer):
            self.gate.on_answer(msg)
        if self.bus is not None:
            topic = Topic.ANSWERS if isinstance(msg, Answer) else Topic.COMMANDS
            self.bus.publish(topic, msg)

    def start_bus_gate(self) -> None:
        """Subscribe ``gate`` to the bus's ``asks``/``answers`` topics, so e.g. a GUI-answered
        Ask also closes the gate."""
        if self.bus is None:
            raise ValueError("start_bus_gate needs a bus")
        sub = self.bus.subscribe([Topic.ASKS, Topic.ANSWERS], maxsize=32, name="asr_gate")

        def _run() -> None:
            try:
                for env in sub:
                    if isinstance(env.payload, Ask):
                        self.gate.on_ask(env.payload)
                    elif isinstance(env.payload, Answer):
                        self.gate.on_answer(env.payload)
            except BusClosed:
                pass

        self._gate_thread = threading.Thread(target=_run, name="asr-gate", daemon=True)
        self._gate_thread.start()

    def start_mic(self, *, device: int | str | None = None) -> None:
        """Open the microphone with ``sounddevice`` and feed frames to ``feed`` continuously.
        Real hardware I/O; not exercised by unit tests."""
        import sounddevice as sd

        def _callback(indata, frames, time_info, status) -> None:
            self.feed(bytes(indata))

        self._stream = sd.RawInputStream(
            samplerate=self.recognizer.sample_rate,
            blocksize=int(self.recognizer.sample_rate * self.block_ms / 1000),
            dtype="int16",
            channels=1,
            callback=_callback,
            device=device,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
