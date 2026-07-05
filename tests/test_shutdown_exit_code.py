"""Tests that fail-closed exits preserve their non-zero process exit code.

The __main__ wrapper used to catch every SystemExit, print "Stopped.", and
sys.exit(0) -- flattening _validate_storage_mode's sys.exit(1) and the state
cursor's raise SystemExit(...) into a clean success. In deployment that means
systemd/cron/CI reads a rejected config or fail-closed startup as a healthy run
and never restarts or alerts. The wrapper must re-raise non-zero exits and only
treat a zero/None code (Ctrl-C, clean SIGTERM) as a normal stop.
"""
import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "classify_and_label",
    Path(__file__).resolve().parent.parent / "scripts" / "classify_and_label.py",
)
cal = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cal)


def test_nonzero_sys_exit_propagates():
    def boom():
        raise SystemExit(1)
    with pytest.raises(SystemExit) as exc:
        cal._run_with_shutdown_handling(boom)
    assert exc.value.code == 1  # not flattened to 0


def test_raised_systemexit_with_message_propagates_code():
    """A bare `raise SystemExit("msg")` (the state cursor guard) carries a
    string code -> a non-zero exit, and must not be swallowed as success."""
    def boom():
        raise SystemExit("fail-closed reason")
    with pytest.raises(SystemExit) as exc:
        cal._run_with_shutdown_handling(boom)
    assert exc.value.code == "fail-closed reason"


def test_clean_sys_exit_zero_is_normal_stop():
    def clean():
        raise SystemExit(0)
    with pytest.raises(SystemExit) as exc:
        cal._run_with_shutdown_handling(clean)
    assert exc.value.code in (0, None)


def test_keyboard_interrupt_exits_zero():
    def interrupted():
        raise KeyboardInterrupt
    with pytest.raises(SystemExit) as exc:
        cal._run_with_shutdown_handling(interrupted)
    assert exc.value.code == 0
