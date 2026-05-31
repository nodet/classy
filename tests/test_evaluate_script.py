"""Test the evaluate module (the logic behind train_and_evaluate.py)."""
import numpy as np
import pytest
from unittest.mock import patch

from gmail_classifier.models import Message
from gmail_classifier.evaluate import run_evaluation


def _make_messages():
    """Create a small dataset with two clear clusters."""
    messages = []
    for i in range(6):
        messages.append(Message(
            id=f"tech_{i}", subject=f"Python tutorial {i}",
            from_address="dev@py.org", from_name="Dev",
            body_html=f"<p>Learn programming {i}</p>",
            labels=["Tech"], list_id="", date="",
        ))
    for i in range(6):
        messages.append(Message(
            id=f"travel_{i}", subject=f"Flight booking {i}",
            from_address="air@fly.com", from_name="Air",
            body_html=f"<p>Your trip to Paris {i}</p>",
            labels=["Travel"], list_id="", date="",
        ))
    return messages


@pytest.mark.slow
def test_run_evaluation_returns_metrics():
    """run_evaluation should return a metrics table and results."""
    messages = _make_messages()
    table, results = run_evaluation(messages, k=5)
    # Table should have entries for multiple thresholds
    assert len(table) >= 2
    # Each row is (threshold, precision, coverage)
    for threshold, precision, coverage in table:
        assert 0.0 <= precision <= 1.0
        assert 0.0 <= coverage <= 1.0
    # Results should have one entry per message
    assert len(results) == 12


@pytest.mark.slow
def test_run_evaluation_good_precision_at_high_threshold():
    """With distinct clusters, precision at 0.95 should be high."""
    messages = _make_messages()
    table, results = run_evaluation(messages, k=5)
    # Find the 0.95 threshold row
    high_row = next(row for row in table if row[0] == 0.95)
    # With well-separated topics, precision should be perfect or near-perfect
    assert high_row[1] >= 0.9
