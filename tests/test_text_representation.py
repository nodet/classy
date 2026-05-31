from gmail_classifier.preprocessing import build_text_representation


def test_build_text_representation_full():
    result = build_text_representation(
        from_name="Alice",
        from_address="alice@example.com",
        subject="Q3 Planning",
        body="Let's discuss OKRs",
    )
    assert result == "Alice <alice@example.com> | Q3 Planning | Let's discuss OKRs"


def test_build_text_representation_missing_name():
    result = build_text_representation(
        from_name="",
        from_address="bot@system.com",
        subject="Alert",
        body="Server down",
    )
    assert result == "<bot@system.com> | Alert | Server down"


def test_build_text_representation_with_list_header():
    result = build_text_representation(
        from_name="Alice",
        from_address="alice@example.com",
        subject="Re: topic",
        body="discussion",
        list_id="dev-discuss.lists.example.com",
    )
    assert "dev-discuss.lists.example.com" in result
    assert "Alice" in result
    assert "Re: topic" in result
