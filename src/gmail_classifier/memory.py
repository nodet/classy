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


def log_mem(stage: str) -> None:
    """Print an RSS checkpoint for ``stage`` in the [trace] format.

    Shows current RSS where available (Linux) plus the process peak, so a
    drop after del/malloc_trim is visible (current falls; peak does not).
    """
    current = _current_rss_mb()
    cur = f"{current:.1f} MB" if current is not None else "n/a"
    print(f"  [mem] {time.strftime('%H:%M:%S')} {stage}: {cur} RSS "
          f"({_max_rss_mb():.1f} MB peak)", flush=True)
