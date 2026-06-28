"""Tests for config.toml loading."""
from pathlib import Path

from gmail_classifier.config import excluded_labels, load_config


def _write(tmp_path: Path, body: str) -> Path:
    cfg = tmp_path / "config.toml"
    cfg.write_text(body)
    return cfg


def test_excluded_labels_reads_list(tmp_path):
    cfg = _write(tmp_path, '[labels]\nexcluded = ["A", "B"]\n')
    assert excluded_labels(cfg) == ["A", "B"]


def test_excluded_labels_empty_when_unset(tmp_path):
    cfg = _write(tmp_path, "[labels]\n")
    assert excluded_labels(cfg) == []


def test_missing_file_returns_defaults(tmp_path):
    missing = tmp_path / "nope.toml"
    assert load_config(missing) == {}
    assert excluded_labels(missing) == []


def test_repo_config_is_parseable():
    """The committed config.toml at the repo root loads without error."""
    cfg = excluded_labels()  # default path
    assert isinstance(cfg, list)
