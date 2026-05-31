from gmail_classifier.preprocessing import remove_quoted_replies, strip_html


def test_strip_html_basic():
    assert strip_html("<p>Hello <b>world</b></p>") == "Hello world"


def test_strip_html_preserves_plain_text():
    assert strip_html("Already plain text") == "Already plain text"


def test_strip_html_handles_empty():
    assert strip_html("") == ""


def test_strip_html_extracts_from_full_email_html():
    html = """<html>
<head><style>body { font-family: Arial; }</style></head>
<body>
<div class="wrapper">
  <p>Hi team,</p>
  <p>Here are the OKRs for next quarter.</p>
</div>
</body>
</html>"""
    result = strip_html(html)
    assert "Hi team," in result
    assert "Here are the OKRs for next quarter." in result
    assert "<" not in result
    assert "font-family" not in result


def test_remove_quoted_lines():
    text = "My reply\n> Original message\n> continues here"
    assert remove_quoted_replies(text) == "My reply"


def test_remove_nested_quotes():
    text = "Reply\n> Quote\n>> Nested quote"
    assert remove_quoted_replies(text) == "Reply"


def test_remove_outlook_style_quote():
    text = "My reply\n\nOn Mon, Jan 1, 2025, John wrote:\n> old text"
    assert remove_quoted_replies(text) == "My reply"


def test_no_quotes_unchanged():
    text = "Normal email with no quoting"
    assert remove_quoted_replies(text) == "Normal email with no quoting"
