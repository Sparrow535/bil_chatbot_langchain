import re
from typing import List

_FORM_KEYWORDS = [
  "form", "forms", "pdf", "download",
  "application form", "proposal form", "claim form",
  "authorization letter", "letter", "document", "documents", "application",
]


def clean_text(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "")).strip()

def chunk_text(text: str, max_chars: int = 1800, overlap: int = 250) -> List[str]:
    text = clean_text(text)
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks

def looks_like_form_request(q: str) -> bool:
    q = (q or "").lower().strip()
    if not q:
        return False

    # Require explicit form intent words
    if any(k in q for k in _FORM_KEYWORDS):
        return True

    # If user only says "loan" or "insurance" -> NOT a form request
    return False