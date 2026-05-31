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
