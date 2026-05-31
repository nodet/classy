from gmail_classifier.preprocessing import strip_html


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
