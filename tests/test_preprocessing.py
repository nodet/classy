from gmail_classifier.preprocessing import (
    preprocess_email_body,
    remove_forwarded,
    remove_quoted_replies,
    strip_html,
    trim_signature,
    truncate,
)


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


def test_remove_forwarded_header_block():
    text = (
        "FYI see below\n\n"
        "---------- Forwarded message ----------\n"
        "From: someone@example.com\n"
        "To: me@example.com\n"
        "Subject: Original\n\n"
        "Original body"
    )
    assert remove_forwarded(text) == "FYI see below"


def test_no_forward_unchanged():
    text = "Normal email"
    assert remove_forwarded(text) == "Normal email"


def test_trim_signature_dash_separator():
    text = "Main content\n\n-- \nJohn Doe\nSenior Engineer"
    assert trim_signature(text) == "Main content"


def test_trim_signature_no_separator():
    text = "Email with no signature"
    assert trim_signature(text) == "Email with no signature"


def test_trim_signature_sent_from_iphone():
    text = "Quick reply\n\nSent from my iPhone"
    assert trim_signature(text) == "Quick reply"


def test_truncate_long_text():
    # ~2000 words, well over 512 token limit
    words = ["word"] * 2000
    text = " ".join(words)
    result = truncate(text, max_words=400)
    assert len(result.split()) <= 400


def test_truncate_short_text_unchanged():
    text = "Short email"
    assert truncate(text, max_words=400) == "Short email"


def test_preprocess_combines_all_steps():
    html = """<html><head><style>h1{color:red}</style></head><body>
<p>Important update about the project.</p>
<p>We need to discuss next steps.</p>
<p>On Tue, Feb 1, 2025, Alice wrote:</p>
<blockquote>> Old message content here</blockquote>
<p>-- </p>
<p>Bob Smith<br>VP Engineering</p>
</body></html>"""
    result = preprocess_email_body(html)
    assert "Important update" in result
    assert "discuss next steps" in result
    assert "Old message" not in result
    assert "Bob Smith" not in result
    assert "<" not in result
    assert "color:red" not in result
