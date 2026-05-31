import base64
import email.utils
from typing import Dict, List, Tuple


def parse_sender(from_header: str) -> Tuple[str, str]:
    """Parse a From header into (name, address)."""
    if not from_header:
        return ("", "")
    name, address = email.utils.parseaddr(from_header)
    return (name, address)


def extract_headers(headers: List[Dict[str, str]]) -> Dict[str, str]:
    """Extract relevant fields from Gmail API headers list.

    Returns a dict with keys: subject, from, list_id, date.
    """
    result = {"subject": "", "from": "", "list_id": "", "date": ""}
    for h in headers:
        name = h.get("name", "").lower()
        value = h.get("value", "")
        if name == "subject":
            result["subject"] = value
        elif name == "from":
            result["from"] = value
        elif name == "list-id":
            # Strip angle brackets: "<list.example.com>" -> "list.example.com"
            result["list_id"] = value.strip("<>")
        elif name == "date":
            result["date"] = value
    return result


def decode_body(payload: dict) -> str:
    """Decode the email body from a Gmail API payload.

    Handles simple and multipart messages. Prefers text/html over text/plain.
    """
    mime_type = payload.get("mimeType", "")

    # Simple message with direct body
    if mime_type.startswith("text/"):
        data = payload.get("body", {}).get("data", "")
        if not data:
            return ""
        # Gmail uses base64url encoding (no padding)
        padded = data + "=" * (4 - len(data) % 4)
        return base64.urlsafe_b64decode(padded).decode("utf-8", errors="replace")

    # Multipart message — recurse into parts
    parts = payload.get("parts", [])
    html_body = ""
    plain_body = ""

    for part in parts:
        part_mime = part.get("mimeType", "")
        if part_mime == "text/html":
            html_body = decode_body(part)
        elif part_mime == "text/plain":
            plain_body = decode_body(part)
        elif part_mime.startswith("multipart/"):
            # Recurse into nested multipart
            nested = decode_body(part)
            if nested:
                if not html_body:
                    html_body = nested

    return html_body or plain_body
