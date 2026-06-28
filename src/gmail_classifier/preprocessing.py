import re
from typing import Optional

from bs4 import BeautifulSoup

# Cap the HTML handed to BeautifulSoup. Real readable text in an email is small;
# anything past this is overwhelmingly inline base64 images, tracking pixels, and
# CSS -- all of which get_text() discards anyway. Parsing the *full* HTML builds a
# tag tree many times the byte size in RAM (a few MB of inline images -> hundreds
# of MB transient), so we cut before the parse, not after (truncate() runs on the
# extracted text, far too late to bound the parse peak). The body is truncated to
# ~400 words downstream regardless, so this cap costs no real text.
MAX_HTML_CHARS = 200_000


def html_cap_note(html: str) -> Optional[str]:
    """If ``html`` would be reduced before parsing, return a short
    ``before->after`` note for logging; otherwise None. Pure (no I/O), so the
    per-message paths can decide whether to print it without coupling
    preprocessing to logging. Mirrors the reduction strip_html applies.
    """
    if not html:
        return None
    original = len(html)
    reduced = min(len(_strip_data_uris(html)), MAX_HTML_CHARS)
    if reduced >= original:
        return None
    return f"html {original // 1024}KB -> {reduced // 1024}KB before parse"


def _strip_data_uris(html: str) -> str:
    """Drop inline ``data:`` URIs (base64 images) -- they carry no text but can
    be megabytes each, dominating the parse cost. Cheap regex, pre-parse."""
    return re.sub(r"data:[^;]+;base64,[A-Za-z0-9+/=]+", "", html)


def strip_html(html: str) -> str:
    """Extract visible text from HTML, stripping all tags and style/script content.

    Inline base64 images are removed and the input is capped at MAX_HTML_CHARS
    *before* parsing, to bound the transient memory of the BeautifulSoup tree.
    """
    if not html:
        return ""
    html = _strip_data_uris(html)
    if len(html) > MAX_HTML_CHARS:
        html = html[:MAX_HTML_CHARS]
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["style", "script", "head"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    # Collapse multiple spaces into one
    text = re.sub(r" +", " ", text)
    # Clean up lines
    lines = (line.strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line)


def remove_quoted_replies(text: str) -> str:
    """Remove quoted reply lines and 'On ... wrote:' attribution lines."""
    lines = text.splitlines()
    result = []
    for line in lines:
        # Stop at "On ... wrote:" attribution line
        if re.match(r"On .+wrote:$", line):
            break
        # Skip lines starting with >
        if line.startswith(">"):
            break
        result.append(line)
    # Strip trailing blank lines
    while result and not result[-1].strip():
        result.pop()
    return "\n".join(result)


def remove_forwarded(text: str) -> str:
    """Remove forwarded message blocks (everything after the forward separator)."""
    # Match common forward separators
    pattern = r"\n*-{5,}\s*Forwarded message\s*-{5,}"
    parts = re.split(pattern, text, maxsplit=1)
    result = parts[0].rstrip()
    return result


_MOBILE_SIGNATURES = [
    "Sent from my iPhone",
    "Sent from my iPad",
    "Sent from my Galaxy",
    "Sent from my Android",
]


def trim_signature(text: str) -> str:
    """Remove email signatures (-- separator and common mobile signatures)."""
    lines = text.splitlines()

    # Look for "-- " separator (standard sig delimiter)
    for i, line in enumerate(lines):
        if line.rstrip() == "--":
            result = "\n".join(lines[:i]).rstrip()
            return result

    # Look for mobile signatures
    for i, line in enumerate(lines):
        stripped = line.strip()
        if any(stripped.startswith(sig) for sig in _MOBILE_SIGNATURES):
            result = "\n".join(lines[:i]).rstrip()
            return result

    return text


def truncate(text: str, max_words: int = 400) -> str:
    """Truncate text to approximately max_words words."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def preprocess_email_body(html: str) -> str:
    """Full preprocessing pipeline: HTML strip, quotes, forwards, signature, truncate."""
    text = strip_html(html)
    text = remove_forwarded(text)
    text = remove_quoted_replies(text)
    text = trim_signature(text)
    text = truncate(text)
    return text


def build_text_representation(
    from_name: str,
    from_address: str,
    subject: str,
    body: str,
    list_id: str = "",
) -> str:
    """Build the text string that will be embedded for classification."""
    if from_name:
        sender = f"{from_name} <{from_address}>"
    else:
        sender = f"<{from_address}>"

    parts = [sender, subject, body]
    if list_id:
        parts.append(f"[list: {list_id}]")

    return " | ".join(parts)
