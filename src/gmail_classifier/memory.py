"""Live RSS prefix for log lines + the malloc_trim heap-return helper.

Uses stdlib ``resource`` only (no psutil dependency).
"""
import resource
import time


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
    """The ``HH:MM:SS <rss>MB`` prefix for per-message log lines.

    The leading value is *current* RSS (right-aligned in 5 cols), or ``  n/a``
    where unavailable.
    """
    rss = _current_rss_mb()
    mem = f"{rss:5.0f}MB" if rss is not None else "  n/a"
    return f"{time.strftime('%H:%M:%S')} {mem}"
