"""Tests for get_credentials branch selection (auth.py).

Covers the three non-interactive branches: a valid stored token is reused
as-is, an expired token with a refresh_token is refreshed and re-saved, and a
missing token + missing client secret raises. The interactive browser flow
(run_local_server) is deliberately not exercised.
"""
from unittest.mock import MagicMock, patch

import pytest

from gmail_classifier import auth


def test_valid_token_reused_without_refresh_or_rewrite(tmp_path):
    token_path = tmp_path / "token.json"
    token_path.write_text('{"token": "existing"}')

    creds = MagicMock()
    creds.valid = True

    with patch.object(auth.Credentials, "from_authorized_user_file",
                      return_value=creds) as from_file, \
         patch.object(auth, "Request") as request:
        result = auth.get_credentials(tmp_path)

    assert result is creds
    from_file.assert_called_once()
    creds.refresh.assert_not_called()
    request.assert_not_called()
    # A valid token must not be rewritten.
    assert token_path.read_text() == '{"token": "existing"}'


def test_expired_token_is_refreshed_and_resaved(tmp_path):
    token_path = tmp_path / "token.json"
    token_path.write_text('{"token": "stale"}')

    creds = MagicMock()
    creds.valid = False
    creds.expired = True
    creds.refresh_token = "refresh-abc"
    creds.to_json.return_value = '{"token": "refreshed"}'

    with patch.object(auth.Credentials, "from_authorized_user_file",
                      return_value=creds), \
         patch.object(auth, "Request") as request:
        result = auth.get_credentials(tmp_path)

    assert result is creds
    creds.refresh.assert_called_once_with(request.return_value)
    # Refreshed token written back for next run.
    assert token_path.read_text() == '{"token": "refreshed"}'


def test_missing_token_and_secret_raises(tmp_path):
    # Neither token.json nor client_secret.json exists in tmp_path.
    with pytest.raises(FileNotFoundError):
        auth.get_credentials(tmp_path)
