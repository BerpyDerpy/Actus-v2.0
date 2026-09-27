import threading

import pytest

from actus.core.bus import Bus, BusClosed
from actus.core.contracts import Topic


def test_publish_reaches_only_subscribers_of_that_topic():
    bus = Bus()
    a = bus.subscribe(Topic.ALERTS, name="a")
    b = bus.subscribe(Topic.PROMPTS, name="b")
    bus.publish(Topic.ALERTS, "x")
    env = a.get(timeout=0.1)
    assert env is not None and env.topic is Topic.ALERTS and env.payload == "x"
    assert b.get(timeout=0.01) is None


def test_multi_topic_subscription_preserves_order():
    bus = Bus()
    sub = bus.subscribe([Topic.EVENTS, "alerts"], maxsize=10)
    bus.publish(Topic.EVENTS, 1)
    bus.publish(Topic.ALERTS, 2)
    bus.publish(Topic.EVENTS, 3)
    envs = sub.drain()
    assert [e.payload for e in envs] == [1, 2, 3]
    assert [e.seq for e in envs] == sorted(e.seq for e in envs)


def test_drop_oldest_counts_drops_and_keeps_latest():
    bus = Bus()
    sub = bus.subscribe(Topic.FRAMES, maxsize=2, policy="drop_oldest")
    for i in range(5):
        bus.publish(Topic.FRAMES, i)
    assert [e.payload for e in sub.drain()] == [3, 4]
    assert sub.dropped == 3 and sub.delivered == 5


def test_drop_newest_keeps_history():
    bus = Bus()
    sub = bus.subscribe(Topic.EVENTS, maxsize=2, policy="drop_newest")
    for i in range(5):
        bus.publish(Topic.EVENTS, i)
    assert [e.payload for e in sub.drain()] == [0, 1]
    assert sub.dropped == 3


def test_slow_subscriber_does_not_affect_others():
    bus = Bus()
    slow = bus.subscribe(Topic.FRAMES, maxsize=1)
    fast = bus.subscribe(Topic.FRAMES, maxsize=100)
    for i in range(10):
        bus.publish(Topic.FRAMES, i)
    assert fast.dropped == 0 and len(fast.drain()) == 10
    assert slow.dropped == 9
    stats = bus.stats()
    assert stats["published"]["frames"] == 10 and stats["dropped_total"] == 9


def test_close_drains_then_raises():
    bus = Bus()
    sub = bus.subscribe(Topic.STATS)
    bus.publish(Topic.STATS, "last")
    bus.close()
    assert sub.get().payload == "last"
    with pytest.raises(BusClosed):
        sub.get()
    with pytest.raises(BusClosed):
        bus.publish(Topic.STATS, "late")


def test_iteration_ends_on_close_across_threads():
    bus = Bus()
    sub = bus.subscribe(Topic.PREDICATES, maxsize=1000)
    got = []
    th = threading.Thread(target=lambda: got.extend(e.payload for e in sub))
    th.start()
    for i in range(100):
        bus.publish(Topic.PREDICATES, i)
    bus.close()
    th.join(timeout=2)
    assert not th.is_alive()
    assert got == list(range(100))


def test_unsubscribe_stops_delivery():
    bus = Bus()
    sub = bus.subscribe(Topic.ASKS)
    sub.unsubscribe()
    bus.publish(Topic.ASKS, 1)
    assert sub.closed
    assert bus.stats()["subscribers"] == []


def test_unknown_topic_and_bad_args_rejected():
    bus = Bus()
    with pytest.raises(ValueError):
        bus.publish("nope", 1)
    with pytest.raises(ValueError):
        bus.subscribe(Topic.ASKS, maxsize=0)
    with pytest.raises(ValueError):
        bus.subscribe(Topic.ASKS, policy="block")  # type: ignore[arg-type]
