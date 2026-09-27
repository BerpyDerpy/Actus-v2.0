"""In-process pub/sub with bounded per-subscriber queues and explicit drop counting.

* ``publish`` never blocks. When a subscriber's queue is full the configured policy drops one
  message (``drop_oldest`` keeps the freshest data, right for frames; ``drop_newest`` keeps
  history, right for logs) and increments that subscriber's ``dropped`` counter.
* A subscription may cover several topics; it receives ``Envelope`` objects in publish order.
* ``Bus.close()`` ends every subscription: ``get`` drains what is queued, then raises
  ``BusClosed``; iteration simply stops.

Example::

    bus = Bus()
    sub = bus.subscribe(Topic.FRAMES, maxsize=2, name="detector")
    bus.publish(Topic.FRAMES, packet)
    env = sub.get(timeout=0.1)       # Envelope | None on timeout
    for env in sub: ...              # until bus.close()
"""

from __future__ import annotations

import collections
import itertools
import threading
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Literal

from actus.core.contracts import Topic

DropPolicy = Literal["drop_oldest", "drop_newest"]


class BusClosed(Exception):
    """Raised by ``Subscription.get`` once the bus is closed and the queue is empty."""


@dataclass(frozen=True, slots=True)
class Envelope:
    topic: Topic
    seq: int  # bus-wide publish sequence number (gaps for a subscriber = drops or filtering)
    t_pub: float  # time.monotonic() at publish
    payload: Any


class Subscription:
    def __init__(
        self,
        bus: Bus,
        topics: frozenset[Topic],
        maxsize: int,
        policy: DropPolicy,
        name: str,
    ) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        if policy not in ("drop_oldest", "drop_newest"):
            raise ValueError(f"unknown drop policy {policy!r}")
        self._bus = bus
        self.topics = topics
        self.maxsize = maxsize
        self.policy: DropPolicy = policy
        self.name = name
        self._q: collections.deque[Envelope] = collections.deque()
        self._cv = threading.Condition()
        self._closed = False
        self.delivered = 0
        self.dropped = 0

    # -- producer side (called by Bus) --------------------------------------------------------

    def _offer(self, env: Envelope) -> None:
        with self._cv:
            if self._closed:
                return
            if len(self._q) >= self.maxsize:
                self.dropped += 1
                if self.policy == "drop_newest":
                    return
                self._q.popleft()
            self._q.append(env)
            self.delivered += 1
            self._cv.notify()

    def _close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()

    # -- consumer side ------------------------------------------------------------------------

    def get(self, timeout: float | None = None) -> Envelope | None:
        """Next envelope; None on timeout; ``BusClosed`` once closed and drained."""
        with self._cv:
            deadline = None if timeout is None else time.monotonic() + timeout
            while not self._q:
                if self._closed:
                    raise BusClosed(self.name)
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return None
                self._cv.wait(remaining)
            return self._q.popleft()

    def get_nowait(self) -> Envelope | None:
        with self._cv:
            if self._q:
                return self._q.popleft()
            if self._closed:
                raise BusClosed(self.name)
            return None

    def drain(self) -> list[Envelope]:
        """Everything queued right now (possibly empty); never blocks or raises."""
        with self._cv:
            items = list(self._q)
            self._q.clear()
            return items

    def __iter__(self) -> Iterator[Envelope]:
        while True:
            try:
                env = self.get()
            except BusClosed:
                return
            if env is not None:
                yield env

    @property
    def depth(self) -> int:
        with self._cv:
            return len(self._q)

    @property
    def closed(self) -> bool:
        return self._closed

    def unsubscribe(self) -> None:
        self._bus._remove(self)
        self._close()

    def stats(self) -> dict[str, Any]:
        with self._cv:
            return {
                "name": self.name,
                "topics": sorted(t.value for t in self.topics),
                "depth": len(self._q),
                "maxsize": self.maxsize,
                "policy": self.policy,
                "delivered": self.delivered,
                "dropped": self.dropped,
            }


class Bus:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subs: dict[Topic, list[Subscription]] = {t: [] for t in Topic}
        self._all: list[Subscription] = []
        self._seq = itertools.count()
        self._published: collections.Counter[Topic] = collections.Counter()
        self._closed = False

    def subscribe(
        self,
        topics: Topic | str | Iterable[Topic | str],
        *,
        maxsize: int = 64,
        policy: DropPolicy = "drop_oldest",
        name: str = "",
    ) -> Subscription:
        if isinstance(topics, (Topic, str)):
            topics = [topics]
        topic_set = frozenset(Topic(t) for t in topics)
        if not topic_set:
            raise ValueError("subscribe needs at least one topic")
        sub = Subscription(self, topic_set, maxsize, policy, name or f"sub{len(self._all)}")
        with self._lock:
            if self._closed:
                sub._close()
                return sub
            for t in topic_set:
                self._subs[t].append(sub)
            self._all.append(sub)
        return sub

    def publish(self, topic: Topic | str, payload: Any) -> int:
        """Deliver ``payload`` to every subscriber of ``topic``. Returns the sequence number."""
        topic = Topic(topic)
        # Offer under the bus lock so every subscriber sees one global order. _offer never
        # blocks (it only takes the subscriber's own short lock), so this stays cheap.
        with self._lock:
            if self._closed:
                raise BusClosed("publish on a closed bus")
            seq = next(self._seq)
            self._published[topic] += 1
            env = Envelope(topic, seq, time.monotonic(), payload)
            for sub in self._subs[topic]:
                sub._offer(env)
        return seq

    def close(self) -> None:
        with self._lock:
            self._closed = True
            subs = list(self._all)
        for sub in subs:
            sub._close()

    @property
    def closed(self) -> bool:
        return self._closed

    def _remove(self, sub: Subscription) -> None:
        with self._lock:
            for t in sub.topics:
                if sub in self._subs[t]:
                    self._subs[t].remove(sub)
            if sub in self._all:
                self._all.remove(sub)

    def stats(self) -> dict[str, Any]:
        """Per-topic publish counts and per-subscriber depth / delivered / dropped."""
        with self._lock:
            subs = list(self._all)
            published = {t.value: n for t, n in self._published.items()}
        return {
            "published": published,
            "subscribers": [s.stats() for s in subs],
            "dropped_total": sum(s.dropped for s in subs),
        }
