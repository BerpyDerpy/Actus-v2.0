"""Tests for the priority voice queue and player (T-VOICE). No audio hardware or heavy optional
deps are needed: FakeTTS stands in for a real backend, since the queue/player logic is the same
whichever backend actually renders speech."""

from __future__ import annotations

import math
import threading
import time

import pytest

from actus.core.contracts import Alert, AlertKind, Ask, Prompt
from actus.voice.voice_queue import VoicePlayer, VoiceQueue, VoiceQueueClosed


class FakeTTS:
    """Records spoken text; ``speak`` takes ``speak_s`` unless interrupted by ``stop``."""

    def __init__(self, speak_s: float = 0.05) -> None:
        self.spoken: list[str] = []
        self.speak_s = speak_s
        self.stop_calls = 0
        self._interrupt = threading.Event()

    def speak(self, text: str) -> None:
        self._interrupt.clear()
        self._interrupt.wait(self.speak_s)
        self.spoken.append(text)

    def stop(self) -> None:
        self.stop_calls += 1
        self._interrupt.set()

    def close(self) -> None:
        pass


# -- VoiceQueue ordering / dedup / ttl / capacity ------------------------------------------------


def test_higher_priority_kinds_come_out_first_regardless_of_arrival_order():
    q = VoiceQueue()
    q.put("info", "i1")
    q.put("prompt", "p1")
    q.put("ask", "a1")
    q.put("alert", "x1")
    order = [q.get(timeout=0.1).kind for _ in range(4)]
    assert order == ["alert", "ask", "prompt", "info"]


def test_same_priority_items_are_fifo():
    q = VoiceQueue()
    q.put("prompt", "first")
    q.put("prompt", "second")
    assert q.get(timeout=0.1).text == "first"
    assert q.get(timeout=0.1).text == "second"


def test_identical_text_within_dedup_window_is_dropped():
    q = VoiceQueue(dedup_s=5.0)
    t0 = time.monotonic()
    assert q.put("prompt", "same", now=t0) is True
    assert q.put("prompt", "same", now=t0 + 1.0) is False
    assert q.dropped_dedup == 1
    assert q.put("prompt", "same", now=t0 + 6.0) is True  # outside the window


def test_expired_item_is_dropped_instead_of_spoken():
    q = VoiceQueue(ttl_s={"ask": 1.0})
    old = time.monotonic() - 10.0  # enqueued "long ago"
    q.put("ask", "stale question", now=old)
    assert q.get(timeout=0.05) is None
    assert q.dropped_ttl == 1


def test_full_queue_evicts_lower_priority_item_for_a_higher_priority_arrival():
    q = VoiceQueue(maxsize=2)
    q.put("prompt", "p1")
    q.put("prompt", "p2")
    assert q.put("alert", "urgent") is True
    assert q.dropped_full == 1
    kinds = {q.get(timeout=0.1).kind, q.get(timeout=0.1).kind}
    assert kinds == {"alert", "prompt"}


def test_full_queue_rejects_arrival_that_outranks_nothing_queued():
    q = VoiceQueue(maxsize=1)
    q.put("alert", "a1")
    assert q.put("prompt", "p1") is False
    assert q.dropped_full == 1
    assert q.get(timeout=0.1).text == "a1"


def test_close_drains_then_raises():
    q = VoiceQueue()
    q.put("info", "last")
    q.close()
    assert q.get().text == "last"
    with pytest.raises(VoiceQueueClosed):
        q.get()
    assert q.put("info", "too late") is False


def test_put_contract_routes_by_type():
    q = VoiceQueue()
    q.put_contract(Prompt("S2", "Take out the red box", frame_idx=0, t=0.0))
    q.put_contract(Alert(AlertKind.SKIP, "Step skipped", ("S2",), frame_idx=1, t=0.1))
    q.put_contract(Ask("q1", "S3", "Did you place it?", frame_idx=2, t=0.2))
    kinds = {q.get(timeout=0.1).kind for _ in range(3)}
    assert kinds == {"prompt", "alert", "ask"}
    with pytest.raises(TypeError):
        q.put_contract(object())


# -- VoicePlayer / interruption -------------------------------------------------------------------


def _drain(q: VoiceQueue, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        stats = q.stats()
        if stats["depth"] == 0 and stats["current"] is None:
            return
        time.sleep(0.01)
    raise TimeoutError("queue did not drain in time")


def test_alert_interrupts_a_prompt_that_is_playing():
    backend = FakeTTS(speak_s=2.0)
    q = VoiceQueue()
    player = VoicePlayer(backend, q)
    player.start()
    try:
        q.put("prompt", "a very long prompt")
        time.sleep(0.1)  # let the player start speaking the prompt
        q.put("alert", "urgent alert")
        _drain(q, timeout=5.0)
    finally:
        player.stop()
    assert backend.stop_calls >= 1
    assert backend.spoken == ["a very long prompt", "urgent alert"]


def test_burst_of_prompts_and_alerts_plays_quickly_and_speaks_every_alert():
    backend = FakeTTS(speak_s=0.05)
    kinds = ("alert", "ask", "prompt", "info")
    q = VoiceQueue(maxsize=16, dedup_s=0.0, ttl_s=dict.fromkeys(kinds, math.inf))
    player = VoicePlayer(backend, q)
    player.start()
    start = time.monotonic()
    try:
        for i in range(20):
            q.put("prompt", f"Next: step {i}")
        for i in range(3):
            q.put("alert", f"Alert {i}")
        _drain(q, timeout=15.0)
    finally:
        player.stop()
    elapsed = time.monotonic() - start
    assert elapsed < 15.0
    spoken_alerts = [s for s in backend.spoken if s.startswith("Alert")]
    assert len(spoken_alerts) == 3  # alerts are never evicted in favour of prompts
