"""Lightweight RSS memory checkpoints for startup/runtime profiling.

Uses stdlib ``resource`` only (no psutil dependency). Output matches the
``[trace]`` style used in training.py so memory and timing lines interleave
naturally in the service log.
"""
import resource
import sys
import time


def _max_rss_mb() -> float:
    """Peak resident set size of this process, in MiB.

    ``ru_maxrss`` is in kilobytes on Linux and in bytes on macOS, so we
    normalize by platform.
    """
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024 * 1024)  # bytes -> MiB
    return rss / 1024  # kilobytes -> MiB


def _current_rss_mb() -> float | None:
    """Current (not peak) RSS in MiB, or None if unavailable.

    Linux: reads /proc/self/statm. macOS (no /proc): shells out to ``ps``,
    which reports current RSS in kilobytes. Returns None if both fail.
    """
    try:
        with open("/proc/self/statm") as f:
            pages = int(f.read().split()[1])  # resident pages
        return pages * resource.getpagesize() / (1024 * 1024)
    except (OSError, IndexError, ValueError):
        pass

    try:
        import os
        import subprocess
        out = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            capture_output=True, text=True, timeout=2,
        )
        return int(out.stdout.strip()) / 1024  # kilobytes -> MiB
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def rss_mb() -> float | None:
    """Current (not peak) RSS in MiB, or None if unavailable.

    Thin public wrapper over the platform probe, for callers that want to
    prefix log lines with live memory rather than print a full checkpoint.
    """
    return _current_rss_mb()


def trim_memory() -> None:
    """Return freed heap pages to the OS (glibc ``malloc_trim``).

    glibc keeps freed pages on its free-list so our own process can reuse
    them, but on a swapless VM it will not yield them to other processes and
    the kernel cannot reclaim dirty anonymous pages -- so RSS ratchets to the
    high-water mark of the costliest message. Calling this after processing
    hands those pages back. No-op off Linux (macOS has no ``malloc_trim``).
    """
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def log_prefix() -> str:
    """The ``HH:MM:SS <rss>MB`` prefix shared by per-message log lines and
    ``[mem]`` checkpoints, so memory lines align with mail activity in the log.

    The leading value is *current* RSS (right-aligned in 5 cols, like the
    per-message lines), or ``  n/a`` where unavailable.
    """
    rss = _current_rss_mb()
    mem = f"{rss:5.0f}MB" if rss is not None else "  n/a"
    return f"{time.strftime('%H:%M:%S')} {mem}"


def log_mem(stage: str) -> None:
    """Print an RSS checkpoint for ``stage``, formatted like the per-message
    log lines for a uniform log.

    The leading number is current RSS; the peak follows so a drop after
    del/malloc_trim stays visible (current falls; peak does not).
    """
    print(f"{log_prefix()} [mem] {stage} ({_max_rss_mb():.0f}MB peak)",
          flush=True)
