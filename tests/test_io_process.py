import sys
import time

from actus.io.process import BackoffPolicy, ManagedProcess, resolve_binary


def test_backoff_delay_grows_and_caps():
    b = BackoffPolicy(initial_s=1.0, max_s=5.0, factor=2.0)
    assert b.delay(0) == 0.0
    assert b.delay(1) == 1.0
    assert b.delay(2) == 2.0
    assert b.delay(3) == 4.0
    assert b.delay(4) == 5.0  # capped


def test_resolve_binary_prefers_an_existing_file(tmp_path):
    f = tmp_path / "thing"
    f.write_text("x")
    assert resolve_binary(f) == str(f)


def test_resolve_binary_falls_back_unchanged_when_not_found():
    assert resolve_binary("definitely-not-a-real-binary-xyz") == "definitely-not-a-real-binary-xyz"


def test_managed_process_restarts_after_exit():
    cmd = [sys.executable, "-c", "import time; time.sleep(0.05)"]
    mp = ManagedProcess(cmd, name="t-restart", backoff=BackoffPolicy(initial_s=0.05, max_s=0.1))
    mp.start()
    try:
        deadline = time.monotonic() + 5.0
        while mp.restarts < 2 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert mp.restarts >= 2
    finally:
        mp.stop()


def test_managed_process_stop_ends_the_watchdog():
    cmd = [sys.executable, "-c", "import time; time.sleep(30)"]
    mp = ManagedProcess(cmd, name="t-stop", backoff=BackoffPolicy(initial_s=0.05))
    mp.start()
    time.sleep(0.3)
    assert mp.running
    mp.stop(timeout=3.0)
    assert not mp.running
    restarts_after_stop = mp.restarts
    time.sleep(0.3)
    assert mp.restarts == restarts_after_stop  # no restart once stopped


def test_managed_process_write_stdin_reaches_the_child(tmp_path):
    out_file = tmp_path / "out.txt"
    script = (
        f"import sys\ndata = sys.stdin.buffer.read()\nopen({str(out_file)!r}, 'wb').write(data)\n"
    )
    cmd = [sys.executable, "-c", script]
    mp = ManagedProcess(cmd, name="t-stdin", want_stdin=True)
    mp.start()
    time.sleep(0.3)
    assert mp.write_stdin(b"hello-world")
    mp.stop(timeout=3.0)
    assert out_file.exists()
    assert out_file.read_bytes() == b"hello-world"


def test_managed_process_reports_stats():
    cmd = [sys.executable, "-c", "import time; time.sleep(30)"]
    mp = ManagedProcess(cmd, name="t-stats")
    mp.start()
    time.sleep(0.3)
    try:
        stats = mp.stats()
        assert stats["name"] == "t-stats"
        assert stats["running"] is True
        assert stats["restarts"] == 0
        assert stats["uptime_s"] is not None and stats["uptime_s"] >= 0
    finally:
        mp.stop()
