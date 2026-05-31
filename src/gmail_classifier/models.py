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
