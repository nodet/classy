import email.utils
from typing import Tuple


def parse_sender(from_header: str) -> Tuple[str, str]:
    """Parse a From header into (name, address)."""
    if not from_header:
        return ("", "")
    name, address = email.utils.parseaddr(from_header)
    return (name, address)
