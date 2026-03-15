import re
from typing import List

_FORM_KEYWORDS = [
  "form", "forms",
  "application form", "proposal form", "claim form",
  "authorization letter", "nomination form", "proxy form",
  "registration form", "contribution form", "refund form",
]

_DOWNLOAD_VERBS = [
  "download", "give", "send", "share", "provide", "get", "show", "need", "want",
]

_FILE_REQUEST_TERMS = [
  "download", "pdf", "file", "files", "link", "copy", "attachment", "attachments",
]

_FILE_SHARE_VERBS = [
  "send", "share", "attach", "upload",
]

_DOC_NOUNS = [
  "report", "reports", "annual report", "handbook", "guide", "manual",
  "document", "documents", "pdf", "publication", "publications",
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

    if any(k in q for k in _FORM_KEYWORDS):
        return True

    return False


def looks_like_form_download_request(q: str) -> bool:
    q = (q or "").lower().strip()
    if not q:
        return False

    formish = looks_like_form_request(q)
    has_download_verb = any(v in q for v in _DOWNLOAD_VERBS)

    # Direct patterns like "form for ppf" / "ppf form"
    if re.search(r"\bform\b\s+(for|of)\s+\w+", q):
        return True
    # "give/download/send ... form"
    if formish and has_download_verb:
        return True

    # Short direct asks like "form", "need form", "give me form"
    if "form" in q and len(q.split()) <= 4:
        return True

    return False


def looks_like_document_download_request(q: str) -> bool:
    q = (q or "").lower().strip()
    if not q:
        return False

    has_doc_noun = any(n in q for n in _DOC_NOUNS)
    if not has_doc_noun:
        return False

    has_file_term = any(t in q for t in _FILE_REQUEST_TERMS)
    has_share_verb = any(v in q for v in _FILE_SHARE_VERBS)

    # Explicit file-oriented asks only. "give me the annual report" should stay a content request.
    if has_file_term:
        return True
    if has_share_verb and has_doc_noun:
        return True

    if re.search(r"\b(annual\s+report|report|handbook|guide|manual)\b.*\b(pdf|download|file|link|attachment)\b", q):
        return True

    return False
