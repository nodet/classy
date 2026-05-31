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
