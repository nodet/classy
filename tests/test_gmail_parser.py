import base64

from gmail_classifier.gmail_parser import (
    decode_body,
    extract_headers,
    parse_gmail_message,
    parse_sender,
)


def _b64url(text: str) -> str:
    """Helper to base64url-encode a string like Gmail API does."""
    return base64.urlsafe_b64encode(text.encode()).decode().rstrip("=")
from gmail_classifier.models import Message


def test_message_model_creation():
    msg = Message(
        id="msg123",
        subject="Hello",
        from_name="Alice",
        from_address="alice@example.com",
        body_html="<p>Hi</p>",
        labels=["Tech", "Newsletters"],
        list_id="dev.lists.example.com",
        date="2025-01-15T10:00:00Z",
    )
    assert msg.id == "msg123"
    assert msg.subject == "Hello"
    assert msg.from_name == "Alice"
    assert msg.from_address == "alice@example.com"
    assert msg.body_html == "<p>Hi</p>"
    assert msg.labels == ["Tech", "Newsletters"]
    assert msg.list_id == "dev.lists.example.com"
    assert msg.date == "2025-01-15T10:00:00Z"


def test_message_model_defaults():
    msg = Message(id="msg456", subject="Test", from_address="bob@example.com")
    assert msg.from_name == ""
    assert msg.body_html == ""
    assert msg.labels == []
    assert msg.list_id == ""
    assert msg.date == ""


def test_parse_sender_name_and_email():
    name, address = parse_sender("John Doe <john@example.com>")
    assert name == "John Doe"
    assert address == "john@example.com"


def test_parse_sender_email_only():
    name, address = parse_sender("john@example.com")
    assert name == ""
    assert address == "john@example.com"


def test_parse_sender_quoted_name():
    name, address = parse_sender('"Doe, John" <john@example.com>')
    assert name == "Doe, John"
    assert address == "john@example.com"


def test_parse_sender_angle_brackets_only():
    name, address = parse_sender("<bot@system.com>")
    assert name == ""
    assert address == "bot@system.com"


def test_parse_sender_empty():
    name, address = parse_sender("")
    assert name == ""
    assert address == ""


def test_extract_headers_subject():
    headers = [{"name": "Subject", "value": "Hello"}]
    result = extract_headers(headers)
    assert result["subject"] == "Hello"


def test_extract_headers_from():
    headers = [{"name": "From", "value": "Alice <alice@example.com>"}]
    result = extract_headers(headers)
    assert result["from"] == "Alice <alice@example.com>"


def test_extract_headers_list_id():
    headers = [{"name": "List-Id", "value": "<dev.lists.example.com>"}]
    result = extract_headers(headers)
    assert result["list_id"] == "dev.lists.example.com"


def test_extract_headers_list_id_missing():
    headers = [{"name": "Subject", "value": "Hi"}]
    result = extract_headers(headers)
    assert result["list_id"] == ""


def test_extract_headers_case_insensitive():
    headers = [{"name": "subject", "value": "Hello"}]
    result = extract_headers(headers)
    assert result["subject"] == "Hello"


def test_decode_body_simple_text_plain():
    payload = {"mimeType": "text/plain", "body": {"data": _b64url("Hello world")}}
    assert decode_body(payload) == "Hello world"


def test_decode_body_simple_html():
    payload = {"mimeType": "text/html", "body": {"data": _b64url("<p>Hello</p>")}}
    assert decode_body(payload) == "<p>Hello</p>"


def test_decode_body_multipart_prefers_html():
    payload = {
        "mimeType": "multipart/alternative",
        "parts": [
            {"mimeType": "text/plain", "body": {"data": _b64url("Hello")}},
            {"mimeType": "text/html", "body": {"data": _b64url("<p>Hello</p>")}},
        ],
    }
    assert decode_body(payload) == "<p>Hello</p>"


def test_decode_body_multipart_falls_back_to_plain():
    payload = {
        "mimeType": "multipart/alternative",
        "parts": [
            {"mimeType": "text/plain", "body": {"data": _b64url("Hello")}},
        ],
    }
    assert decode_body(payload) == "Hello"


def test_decode_body_nested_multipart():
    payload = {
        "mimeType": "multipart/mixed",
        "parts": [
            {
                "mimeType": "multipart/alternative",
                "parts": [
                    {"mimeType": "text/plain", "body": {"data": _b64url("plain")}},
                    {"mimeType": "text/html", "body": {"data": _b64url("<b>html</b>")}},
                ],
            },
            {"mimeType": "application/pdf", "body": {"size": 12345}},
        ],
    }
    assert decode_body(payload) == "<b>html</b>"


def test_decode_body_empty():
    payload = {"mimeType": "text/plain", "body": {}}
    assert decode_body(payload) == ""


def test_parse_gmail_message_full():
    raw = {
        "id": "abc123",
        "labelIds": ["Label_1", "Label_2"],
        "payload": {
            "mimeType": "text/html",
            "headers": [
                {"name": "From", "value": "Alice Smith <alice@example.com>"},
                {"name": "Subject", "value": "Project update"},
                {"name": "Date", "value": "Mon, 15 Jan 2025 10:00:00 +0000"},
                {"name": "List-Id", "value": "<dev.lists.example.com>"},
            ],
            "body": {"data": _b64url("<p>Here is the update.</p>")},
        },
    }
    msg = parse_gmail_message(raw)
    assert msg.id == "abc123"
    assert msg.from_name == "Alice Smith"
    assert msg.from_address == "alice@example.com"
    assert msg.subject == "Project update"
    assert msg.body_html == "<p>Here is the update.</p>"
    assert msg.labels == ["Label_1", "Label_2"]
    assert msg.list_id == "dev.lists.example.com"
    assert msg.date == "Mon, 15 Jan 2025 10:00:00 +0000"


def test_parse_gmail_message_minimal():
    raw = {
        "id": "xyz789",
        "payload": {
            "mimeType": "text/plain",
            "headers": [],
            "body": {},
        },
    }
    msg = parse_gmail_message(raw)
    assert msg.id == "xyz789"
    assert msg.from_name == ""
    assert msg.from_address == ""
    assert msg.subject == ""
    assert msg.body_html == ""
    assert msg.labels == []
