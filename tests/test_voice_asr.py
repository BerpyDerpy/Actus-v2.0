"""Tests for the ASR gate, grammar parsing, and service wiring (T-VOICE).

The gate/parsing logic is pure Python and always runs. Recognition against a real ``vosk`` model
is exercised only when the optional package *and* a downloaded model are present (neither is
part of the base install; the model is a setup-time download), matching the "heavy imports stay
behind an importorskip" convention.
"""

from __future__ import annotations

import time

import pytest

from actus.core.bus import Bus
from actus.core.contracts import Answer, AnswerValue, Ask, Command, CommandName, InputVia, Topic
from actus.voice.asr import AskGate, AsrService, parse_utterance

# -- AskGate --------------------------------------------------------------------------------------


def test_gate_closed_by_default():
    gate = AskGate()
    assert not gate.listening
    assert gate.open_question_id is None


def test_ask_opens_the_gate_and_matching_answer_closes_it():
    gate = AskGate()
    gate.on_ask(Ask("q1", "S3", "Did you place it?", frame_idx=0, t=0.0))
    assert gate.listening and gate.open_question_id == "q1"
    gate.on_answer(Answer("q1", AnswerValue.YES, InputVia.VOICE, t=1.0))
    assert not gate.listening


def test_answer_for_a_different_question_does_not_close_the_gate():
    gate = AskGate()
    gate.on_ask(Ask("q1", "S3", "Did you place it?", frame_idx=0, t=0.0))
    gate.on_answer(Answer("q2", AnswerValue.YES, InputVia.VOICE, t=1.0))
    assert gate.listening


def test_push_to_talk_opens_the_gate_independent_of_any_ask():
    gate = AskGate()
    gate.set_push_to_talk(True)
    assert gate.listening
    gate.set_push_to_talk(False)
    assert not gate.listening


def test_cancel_closes_the_gate_without_an_answer():
    gate = AskGate()
    gate.on_ask(Ask("q1", "S3", "Did you place it?", frame_idx=0, t=0.0))
    gate.cancel()
    assert not gate.listening


# -- parse_utterance --------------------------------------------------------------------------


@pytest.mark.parametrize("word", ["yes", "confirm", "Yes", "  yes  "])
def test_yes_words_answer_yes_when_a_question_is_open(word):
    msg = parse_utterance(word, open_question_id="q1", t=0.0)
    assert isinstance(msg, Answer) and msg.value is AnswerValue.YES and msg.question_id == "q1"


@pytest.mark.parametrize("word", ["no", "cancel"])
def test_no_words_answer_no_when_a_question_is_open(word):
    msg = parse_utterance(word, open_question_id="q1", t=0.0)
    assert isinstance(msg, Answer) and msg.value is AnswerValue.NO


def test_yes_no_words_are_not_answers_without_an_open_question():
    # "cancel" is still a valid Command even with nothing open; "yes"/"no" alone are not.
    assert parse_utterance("yes", open_question_id=None, t=0.0) is None
    msg = parse_utterance("cancel", open_question_id=None, t=0.0)
    assert isinstance(msg, Command) and msg.name is CommandName.CANCEL


@pytest.mark.parametrize(
    ("word", "name"),
    [
        ("next", CommandName.NEXT),
        ("repeat", CommandName.REPEAT),
        ("what's next", CommandName.WHATS_NEXT),
    ],
)
def test_command_words_are_recognised_with_or_without_an_open_question(word, name):
    for qid in (None, "q1"):
        msg = parse_utterance(word, open_question_id=qid, t=0.0)
        assert isinstance(msg, Command) and msg.name is name


def test_unrecognised_text_returns_none():
    assert parse_utterance("banana", open_question_id="q1", t=0.0) is None
    assert parse_utterance("", open_question_id="q1", t=0.0) is None
    assert parse_utterance("   ", open_question_id=None, t=0.0) is None


# -- AsrService.feed (fake recognizer; no audio hardware) ----------------------------------------


class _FakeRecognizer:
    """Feeds back a scripted list of "utterance" texts, one per non-None ``accept`` call."""

    def __init__(self, script: list[str | None]) -> None:
        self.script = list(script)
        self.flush_calls = 0

    def accept(self, pcm: bytes) -> str | None:
        return self.script.pop(0) if self.script else None

    def flush(self) -> str:
        self.flush_calls += 1
        return ""


def test_feed_drops_audio_and_flushes_when_gate_is_closed():
    rec = _FakeRecognizer(["yes"])
    gate = AskGate()  # closed
    results: list[str] = []
    svc = AsrService(rec, gate, on_result=results.append)
    svc.feed(b"\x00\x00")
    assert results == []  # never reached the recognizer
    assert rec.script == ["yes"]  # untouched


def test_feed_recognizes_and_publishes_answer_only_while_gate_open():
    bus = Bus()
    sub = bus.subscribe([Topic.ANSWERS, Topic.COMMANDS], name="test")
    rec = _FakeRecognizer(["yes"])
    gate = AskGate()
    gate.on_ask(Ask("q1", "S3", "Did you place it?", frame_idx=0, t=0.0))
    svc = AsrService(rec, gate, bus=bus)

    svc.feed(b"\x00\x00")

    env = sub.get(timeout=0.1)
    assert env is not None
    assert isinstance(env.payload, Answer)
    assert env.payload.question_id == "q1" and env.payload.value is AnswerValue.YES
    assert not gate.listening  # the answer closed the gate it came from


def test_feed_flushes_the_recognizer_when_the_gate_closes_mid_utterance():
    rec = _FakeRecognizer([None])  # mid-utterance: no final result yet
    gate = AskGate()
    gate.set_push_to_talk(True)
    svc = AsrService(rec, gate)
    svc.feed(b"\x00\x00")  # gate open, nothing final yet
    gate.set_push_to_talk(False)  # push-to-talk released mid-utterance
    svc.feed(b"\x00\x00")  # gate now closed: this call must not reach the recognizer
    assert rec.flush_calls == 1


def test_command_recognized_without_an_open_question_is_published_as_command():
    bus = Bus()
    sub = bus.subscribe(Topic.COMMANDS, name="test")
    rec = _FakeRecognizer(["next"])
    gate = AskGate()
    gate.set_push_to_talk(True)
    svc = AsrService(rec, gate, bus=bus)
    svc.feed(b"\x00\x00")
    env = sub.get(timeout=0.1)
    assert isinstance(env.payload, Command) and env.payload.name is CommandName.NEXT


def test_start_bus_gate_opens_on_ask_and_closes_on_answer_from_elsewhere():
    """An Ask/Answer published by someone else (the engine, the GUI) still drives the gate."""
    bus = Bus()
    rec = _FakeRecognizer([])
    gate = AskGate()
    svc = AsrService(rec, gate, bus=bus)
    svc.start_bus_gate()
    try:
        bus.publish(Topic.ASKS, Ask("q1", "S3", "Did you place it?", frame_idx=0, t=0.0))
        deadline = time.monotonic() + 1.0
        while not gate.listening and time.monotonic() < deadline:
            time.sleep(0.01)
        assert gate.listening

        bus.publish(Topic.ANSWERS, Answer("q1", AnswerValue.NO, InputVia.GUI, t=1.0))
        deadline = time.monotonic() + 1.0
        while gate.listening and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not gate.listening
    finally:
        bus.close()


# -- real Vosk recognizer: only where the package and a downloaded model are present -------------


def test_real_vosk_recognizer_raises_on_missing_model():
    pytest.importorskip("vosk")
    from actus.voice.asr import VoskRecognizer

    with pytest.raises(FileNotFoundError):
        VoskRecognizer("models/voice/does-not-exist")
