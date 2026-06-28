"""Project configuration loaded from ``config.toml``.

A single source of truth for tunables that a new user would otherwise have to
discover scattered across Makefile arguments. Uses stdlib ``tomllib`` (Python
3.11+), so it adds no dependency.

The only setting today is the list of excluded label names; scripts read it as
the *default* for their ``--exclude-labels`` flag, so an explicit flag still
overrides the file for one-off runs.
"""
from __future__ import annotations

import tomllib
from pathlib import Path
from typing import List

# config.toml lives at the repo root, two parents up from this file
# (src/gmail_classifier/config.py -> src/gmail_classifier -> src -> root).
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.toml"


def load_config(path: Path | None = None) -> dict:
    """Parse ``config.toml`` into a dict. Returns ``{}`` if the file is absent."""
    cfg_path = path or _DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        return {}
    with open(cfg_path, "rb") as f:
        return tomllib.load(f)


def excluded_labels(path: Path | None = None) -> List[str]:
    """Label names to exclude everywhere (fetch, train, predict).

    Reads ``[labels].excluded`` from the config; empty list if unset.
    """
    cfg = load_config(path)
    return list(cfg.get("labels", {}).get("excluded", []))
