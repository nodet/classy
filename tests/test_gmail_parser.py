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
