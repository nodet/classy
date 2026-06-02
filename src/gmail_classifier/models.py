from dataclasses import dataclass, field
from typing import List


@dataclass
class Message:
    id: str
    subject: str
    from_address: str
    from_name: str = ""
    body_html: str = ""
    labels: List[str] = field(default_factory=list)
    list_id: str = ""
    date: str = ""


@dataclass
class HistoryEvent:
    """A single change from Gmail's history API."""
    type: str  # "messagesAdded", "labelsAdded", "labelsRemoved"
    message_id: str
    label_ids: List[str] = field(default_factory=list)


class HistoryExpiredError(Exception):
    """Raised when the history ID is too old and Gmail returns 404."""
    pass
