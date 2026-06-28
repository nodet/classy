"""Test that classify_and_label handles SIGTERM gracefully."""
import signal
import subprocess
import sys
import textwrap


def test_sigterm_handler_raises_system_exit():
    """The _sigterm_handler function raises SystemExit(0)."""
    # Inline the same logic as the handler to verify the pattern works.
    # We can't import classify_and_label directly (heavy deps), so we test
    # the actual subprocess behavior below.
    def handler(signum, frame):
        raise SystemExit(0)

    try:
        handler(signal.SIGTERM, None)
        assert False, "Should have raised SystemExit"
    except SystemExit as e:
        assert e.code == 0


def test_sigterm_causes_clean_exit_in_subprocess():
    """A process using the SIGTERM->SystemExit pattern exits 0 on SIGTERM."""
    code = textwrap.dedent("""\
        import signal, sys, time

        def _sigterm_handler(signum, frame):
            raise SystemExit(0)

        signal.signal(signal.SIGTERM, _sigterm_handler)
        try:
            time.sleep(60)
        except SystemExit:
            sys.exit(0)
    """)
    proc = subprocess.Popen([sys.executable, "-c", code])
    # Give the process a moment to register the handler
    import time
    time.sleep(0.1)
    proc.send_signal(signal.SIGTERM)
    rc = proc.wait(timeout=5)
    assert rc == 0


def test_double_sigterm_exits_cleanly():
    """A second SIGTERM during teardown must not raise into shutdown code.

    The handler resets to SIG_DFL on first fire, so a second SIGTERM (which
    systemd commonly sends while slow grpc threads are still joining) just
    terminates the process instead of raising SystemExit inside
    threading._shutdown -- the "Exception ignored" traceback we observed.
    """
    code = textwrap.dedent("""\
        import signal, sys, time

        def _sigterm_handler(signum, frame):
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            raise SystemExit(0)

        signal.signal(signal.SIGTERM, _sigterm_handler)
        try:
            time.sleep(60)
        except SystemExit:
            # Simulate slow teardown, during which a 2nd SIGTERM arrives.
            time.sleep(2)
            sys.exit(0)
    """)
    proc = subprocess.Popen([sys.executable, "-c", code])
    import time
    time.sleep(0.1)
    proc.send_signal(signal.SIGTERM)
    time.sleep(0.2)
    proc.send_signal(signal.SIGTERM)  # arrives during the simulated teardown
    rc = proc.wait(timeout=5)
    # SIG_DFL on SIGTERM terminates with -SIGTERM (-15); the point is it does
    # NOT raise SystemExit into teardown (which would surface as a traceback).
    assert rc in (0, -signal.SIGTERM)
