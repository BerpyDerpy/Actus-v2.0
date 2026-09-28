"""Offline speech recognition: Vosk small English model, restricted to a fixed command grammar.

The microphone is gated: recognition runs only while an ``Ask`` is open (a question is pending)
or while push-to-talk is held — never continuously. That keeps CPU/mic use near zero the rest of
the time and means recognition needs no network at all, ever (Vosk's grammar mode never touches
the internet regardless).

Recognised speech becomes an ``Answer`` (only while a question is open) or a ``Command`` (works
any time the gate is open), both published to the bus for the engine / GUI to consume.

Gate rules (``AsrService.feed``):

* **Muted while Actus speaks.** ``AskGate.mute_while`` (e.g. ``lambda: player.speaking``) closes
  the gate during TTS playback, and whatever was heard then is discarded, so the grammar-limited
  recogniser never turns Actus's own question into a "yes" or "no".
* **An answer belongs to the question it was spoken to.** If the open question changes while an
  utterance is in progress, that partial utterance is discarded, never attributed to the new one.
* **Closing flushes, it does not lose.** When the gate closes otherwise (push-to-talk released,
  the question answered elsewhere), the recogniser's final text is still handled, as a
  ``Command`` only: the question it would have answered is already closed.

``start_mic`` only copies audio in the PortAudio callback; decoding runs on a worker thread, so a
slow decode never overruns the real-time audio thread, and an exception never kills the stream.
"""

from __future__ import annotations

import contextlib
import json
import logging
import queue
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
    StepEvent,
    Topic,
)

logger = logging.getLogger(__name__)

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

    def __init__(self, *, mute_while: Callable[[], bool] | None = None) -> None:
        self._lock = threading.Lock()
        self._open_question_id: str | None = None
        self._open_step_id: str | None = None
        self._ptt_held = False
        # e.g. ``lambda: player.speaking``: the gate stays shut while this returns True.
        self.mute_while = mute_while

    def on_ask(self, ask: Ask) -> None:
        with self._lock:
            self._open_question_id = ask.question_id
            self._open_step_id = ask.step_id

    def on_answer(self, answer: Answer) -> None:
        with self._lock:
            if self._open_question_id == answer.question_id:
                self._open_question_id = None
                self._open_step_id = None

    def on_step_event(self, event: StepEvent) -> None:
        """A later ``StepEvent`` for the asked step means the engine resolved it by evidence
        (``Vault/T-ENGINE.md`` §5), so the question is closed."""
        with self._lock:
            if self._open_step_id is not None and event.step_id == self._open_step_id:
                self._open_question_id = None
                self._open_step_id = None

    def cancel(self) -> None:
        """Close the gate without an Answer, e.g. the question timed out."""
        with self._lock:
            self._open_question_id = None
            self._open_step_id = None

    def set_push_to_talk(self, held: bool) -> None:
        with self._lock:
            self._ptt_held = held

    @property
    def open_question_id(self) -> str | None:
        with self._lock:
            return self._open_question_id

    @property
    def muted(self) -> bool:
        fn = self.mute_while
        if fn is None:
            return False
        try:
            return bool(fn())
        except Exception:
            return False

    @property
    def listening(self) -> bool:
        with self._lock:
            wanted = self._open_question_id is not None or self._ptt_held
        return wanted and not self.muted


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
        reset = getattr(self._rec, "Reset", None)
        if callable(reset):
            reset()  # cheap; rebuilding the recognizer recompiles the grammar
        else:
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
        audio_queue_blocks: int = 50,
    ) -> None:
        self.recognizer = recognizer
        self.gate = gate
        self.bus = bus
        self.on_result = on_result
        self.block_ms = block_ms
        self._stream = None
        self._gate_thread: threading.Thread | None = None
        self._gate_sub = None
        self._was_listening = False
        self._qid: str | None = None  # the question the in-progress utterance was spoken to
        self._audio: queue.Queue = queue.Queue(maxsize=max(1, audio_queue_blocks))
        self._decode_thread: threading.Thread | None = None
        self.audio_dropped = 0
        self.errors = 0

    def feed(self, pcm16_bytes: bytes) -> None:
        """Feed one chunk of mic audio. Dropped unless the gate is open; see the module
        docstring for what happens to an in-progress utterance when the gate changes."""
        muted = self.gate.muted
        listening = self.gate.listening
        qid = self.gate.open_question_id
        if self._was_listening and (not listening or qid != self._qid):
            text = self.recognizer.flush()
            # Echo of our own voice, or speech meant for a different question: discard. A plain
            # close (push-to-talk released, answered elsewhere): keep it, as a Command only.
            if text and not muted and qid == self._qid:
                self._handle(text, question_id=None)
            self._was_listening = False
        if not listening:
            return
        if not self._was_listening:
            self._qid = qid
        self._was_listening = True
        text = self.recognizer.accept(pcm16_bytes)
        if text is not None:
            self._handle(text, question_id=self._qid)

    def _handle(self, text: str, *, question_id: str | None) -> None:
        if self.on_result is not None:
            self.on_result(text)
        msg = parse_utterance(text, open_question_id=question_id, t=time.monotonic())
        if msg is None:
            return
        if isinstance(msg, Answer):
            self.gate.on_answer(msg)
        if self.bus is not None:
            topic = Topic.ANSWERS if isinstance(msg, Answer) else Topic.COMMANDS
            self.bus.publish(topic, msg)

    def start_bus_gate(self) -> None:
        """Subscribe ``gate`` to the bus's ``asks``/``answers``/``events`` topics, so e.g. a
        GUI-answered Ask, or a step resolved by evidence, also closes the gate."""
        if self.bus is None:
            raise ValueError("start_bus_gate needs a bus")
        sub = self.bus.subscribe(
            [Topic.ASKS, Topic.ANSWERS, Topic.EVENTS], maxsize=32, name="asr_gate"
        )
        self._gate_sub = sub

        def _run() -> None:
            try:
                for env in sub:
                    if isinstance(env.payload, Ask):
                        self.gate.on_ask(env.payload)
                    elif isinstance(env.payload, Answer):
                        self.gate.on_answer(env.payload)
                    elif isinstance(env.payload, StepEvent):
                        self.gate.on_step_event(env.payload)
            except BusClosed:
                pass

        self._gate_thread = threading.Thread(target=_run, name="asr-gate", daemon=True)
        self._gate_thread.start()

    def _decode_loop(self) -> None:
        while True:
            chunk = self._audio.get()
            if chunk is None:
                return
            try:
                self.feed(chunk)
            except Exception:
                self.errors += 1
                logger.exception("ASR decode failed; continuing")

    def start_mic(self, *, device: int | str | None = None) -> None:
        """Open the microphone with ``sounddevice``. The audio callback only queues the raw
        blocks; a worker thread decodes them. Real hardware I/O; not exercised by unit tests."""
        import sounddevice as sd

        self._decode_thread = threading.Thread(
            target=self._decode_loop, name="asr-decode", daemon=True
        )
        self._decode_thread.start()

        def _callback(indata, frames, time_info, status) -> None:
            try:
                self._audio.put_nowait(bytes(indata))
            except queue.Full:
                self.audio_dropped += 1

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
        if self._decode_thread is not None:
            with contextlib.suppress(queue.Full):
                self._audio.put(None, timeout=1.0)
            self._decode_thread.join(timeout=2.0)
            self._decode_thread = None
        if self._gate_sub is not None:
            self._gate_sub.unsubscribe()
            self._gate_sub = None
        if self._gate_thread is not None:
            self._gate_thread.join(timeout=2.0)
            self._gate_thread = None
