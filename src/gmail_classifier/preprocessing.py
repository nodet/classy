import re

from bs4 import BeautifulSoup


def strip_html(html: str) -> str:
    """Extract visible text from HTML, stripping all tags and style/script content."""
    if not html:
        return ""
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
