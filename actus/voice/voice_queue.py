"""Bounded priority voice queue and the player that drains it through a TTS backend.

Ordering is fixed: ``alert > ask > prompt > info``. The queue never lets the protocol prior
influence anything here — it only orders *already-decided* engine outputs (``Alert``, ``Ask``,
``Prompt``) for speech, the same way a phone call takes priority over a notification.

* **Dedup**: the same text enqueued again within ``dedup_s`` of the last time is dropped
  silently (e.g. a re-published, unchanged ``Prompt`` while nothing has changed).
* **TTL**: each kind has its own time-to-live. An item still queued once its TTL has elapsed is
  dropped instead of spoken late.
* **Capacity**: when full, the lowest-priority (and, among ties, oldest) queued item is evicted
  to make room for an arrival that outranks it; an arrival that outranks nothing queued is
  itself dropped.
* **Interruption**: an item that outranks whatever is currently playing calls the queue's
  registered interrupt hook (``VoicePlayer`` wires this to ``TTSBackend.stop``), so an alert cuts
  off a prompt mid-sentence.
"""

from __future__ import annotations

import itertools
import math
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from actus.core.contracts import Alert, Ask, Prompt

_PRIORITY: dict[str, int] = {"alert": 0, "ask": 1, "prompt": 2, "info": 3}
_POLL_S = 0.1  # cap on how long `get` waits between TTL re-checks


class VoiceQueueClosed(Exception):
    """Raised by ``VoiceQueue.get`` once the queue is closed and drained."""


@dataclass(frozen=True, slots=True)
class VoiceItem:
    kind: str  # "alert" | "ask" | "prompt" | "info"
    text: str
    enqueued_t: float  # time.monotonic() at put()
    ttl_s: float
    seq: int
    ref: object | None = None  # the originating Alert / Ask / Prompt, if any

    @property
    def priority(self) -> int:
        return _PRIORITY[self.kind]

    def expired(self, now: float) -> bool:
        return math.isfinite(self.ttl_s) and (now - self.enqueued_t) > self.ttl_s


class VoiceQueue:
    def __init__(
        self,
        *,
        maxsize: int = 16,
        dedup_s: float = 5.0,
        ttl_s: Mapping[str, float] | None = None,
    ) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        if dedup_s < 0:
            raise ValueError("dedup_s must be >= 0")
        self.maxsize = maxsize
        self.dedup_s = float(dedup_s)
        self.ttl_s: dict[str, float] = dict.fromkeys(_PRIORITY, math.inf)
        for k, v in (ttl_s or {}).items():
            if k not in _PRIORITY:
                raise ValueError(f"unknown voice item kind {k!r} in ttl_s")
            self.ttl_s[k] = float(v)

        self._cv = threading.Condition()
        self._items: list[VoiceItem] = []
        self._recent: dict[str, float] = {}
        self._seq = itertools.count()
        self._closed = False
        self._current: VoiceItem | None = None
        self._interrupt: Callable[[], None] | None = None

        self.dropped_dedup = 0
        self.dropped_full = 0
        self.dropped_ttl = 0

    # -- wiring ---------------------------------------------------------------------------------

    def set_interrupt(self, fn: Callable[[], None] | None) -> None:
        """``fn`` is called (outside the lock) when a new item outranks what's playing now."""
        with self._cv:
            self._interrupt = fn

    # -- producer side ----------------------------------------------------------------------------

    def put(
        self, kind: str, text: str, *, ref: object | None = None, now: float | None = None
    ) -> bool:
        """Enqueue speech. Returns False if it was dropped (dedup / full / closed)."""
        if kind not in _PRIORITY:
            raise ValueError(f"unknown voice item kind {kind!r}")
        now = time.monotonic() if now is None else now
        interrupt = None
        with self._cv:
            if self._closed:
                return False
            self._evict_expired(now)
            last = self._recent.get(text)
            if last is not None and now - last < self.dedup_s:
                self.dropped_dedup += 1
                return False

            item = VoiceItem(kind, text, now, self.ttl_s[kind], next(self._seq), ref)
            if len(self._items) >= self.maxsize:
                worst = max(self._items, key=lambda it: (it.priority, it.seq))
                if worst.priority > item.priority:
                    self._items.remove(worst)
                    self.dropped_full += 1
                else:
                    self.dropped_full += 1
                    return False

            self._recent[text] = now
            self._items.append(item)
            self._items.sort(key=lambda it: (it.priority, it.seq))
            if self._current is not None and item.priority < self._current.priority:
                interrupt = self._interrupt
            self._cv.notify()
        if interrupt is not None:
            interrupt()
        return True

    def put_alert(self, alert: Alert, *, now: float | None = None) -> bool:
        return self.put("alert", alert.message, ref=alert, now=now)

    def put_ask(self, ask: Ask, *, now: float | None = None) -> bool:
        return self.put("ask", ask.text, ref=ask, now=now)

    def put_prompt(self, prompt: Prompt, *, now: float | None = None) -> bool:
        return self.put("prompt", prompt.text, ref=prompt, now=now)

    def put_contract(self, payload: object, *, now: float | None = None) -> bool:
        """Route an ``Alert``/``Ask``/``Prompt`` onto the queue with the matching priority kind."""
        if isinstance(payload, Alert):
            return self.put_alert(payload, now=now)
        if isinstance(payload, Ask):
            return self.put_ask(payload, now=now)
        if isinstance(payload, Prompt):
            return self.put_prompt(payload, now=now)
        raise TypeError(f"cannot speak a {type(payload).__name__}")

    # -- consumer side ----------------------------------------------------------------------------

    def get(self, timeout: float | None = None) -> VoiceItem | None:
        """Highest-priority item (oldest among ties); None on timeout; raises once closed and
        drained. Marks the returned item as ``current`` until ``mark_done`` is called."""
        with self._cv:
            deadline = None if timeout is None else time.monotonic() + timeout
            while True:
                now = time.monotonic()
                self._evict_expired(now)
                if self._items:
                    item = self._items.pop(0)
                    self._current = item
                    return item
                if self._closed:
                    raise VoiceQueueClosed
                remaining = None if deadline is None else deadline - now
                if remaining is not None and remaining <= 0:
                    return None
                wait = _POLL_S if remaining is None else min(remaining, _POLL_S)
                self._cv.wait(wait)

    def mark_done(self) -> None:
        """Called by the player once it has finished (or been interrupted) speaking `current`."""
        with self._cv:
            self._current = None

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()

    # -- introspection ----------------------------------------------------------------------------

    def stats(self) -> dict[str, object]:
        with self._cv:
            return {
                "depth": len(self._items),
                "current": self._current.kind if self._current else None,
                "dropped_dedup": self.dropped_dedup,
                "dropped_full": self.dropped_full,
                "dropped_ttl": self.dropped_ttl,
            }

    def _evict_expired(self, now: float) -> None:
        kept = []
        for it in self._items:
            if it.expired(now):
                self.dropped_ttl += 1
            else:
                kept.append(it)
        self._items = kept


class VoicePlayer:
    """A single worker thread that drains a ``VoiceQueue`` through a ``TTSBackend``.

    Wires the queue's interrupt hook to ``backend.stop`` so a higher-priority arrival (an alert
    while a prompt is playing) cuts the current utterance off immediately.
    """

    def __init__(self, backend, vqueue: VoiceQueue) -> None:
        self.backend = backend
        self.queue = vqueue
        self.queue.set_interrupt(backend.stop)
        self._thread: threading.Thread | None = None
        self._stop_evt = threading.Event()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run, name="voice-player", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop_evt.is_set():
            try:
                item = self.queue.get(timeout=_POLL_S)
            except VoiceQueueClosed:
                return
            if item is None:
                continue
            try:
                self.backend.speak(item.text)
            finally:
                self.queue.mark_done()

    def stop(self, *, wait: bool = True, timeout: float = 2.0) -> None:
        self._stop_evt.set()
        self.queue.close()
        self.backend.stop()
        if wait and self._thread is not None:
            self._thread.join(timeout=timeout)
        self._thread = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()
