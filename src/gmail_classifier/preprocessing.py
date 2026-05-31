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
