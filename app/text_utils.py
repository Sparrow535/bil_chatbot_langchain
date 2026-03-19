import difflib
import re
from functools import lru_cache
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

_BIL_DOMAIN_TERMS = sorted({
  "annual", "application", "applications", "auto", "authorization", "aviation",
  "benefit", "benefits", "bil", "bhutan", "branch", "branches", "burglary",
  "care", "claim", "claims", "commerce", "contact", "contacts", "contractor",
  "contractors", "contribution", "coverage", "document", "documents", "download",
  "downloads", "eligibility", "employee", "engineering", "fire", "fidelity",
  "form", "forms", "fund", "funds", "general", "gf", "gfm", "gratuity",
  "handbook", "hotel", "housing", "income", "insurance", "insurances", "interest",
  "liability", "livestock", "loan", "loans", "logging", "loss", "machinery",
  "manual", "marine", "mining", "miscellaneous", "money", "motor", "nominee",
  "personal", "policies", "policy", "premium", "premiums", "private", "product",
  "products", "project", "proposal", "provident", "publication", "publications",
  "quarrying", "rate", "rates", "refund", "registration", "report", "reports",
  "requirement", "requirements", "securities", "service", "services", "settlement",
  "shares", "student", "tenure", "tourism", "trade", "transport", "travel",
  "workmen", "accident", "loan", "reports", "branches", "branch", "forms", "contact",
  "annual", "ppf",
})

_TYPO_PROTECT_TERMS = {
  "about", "apply", "applying", "can", "criteria", "detailed", "details", "do", "download",
  "eligibility", "explain", "for", "from", "get", "give", "help", "how", "i", "is", "know",
  "like", "limit", "limits", "main", "me", "more", "need", "next", "offer", "offered",
  "offering", "options", "please", "provide", "provided", "provides", "purchase",
  "purchasing", "quick", "show", "specific", "summary", "tell", "typical", "using", "used",
  "what", "which", "would", "you", "your",
}


def clean_text(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "")).strip()


def _match_case(source: str, target: str) -> str:
    if source.isupper():
        return target.upper()
    if source[:1].isupper() and source[1:].islower():
        return target.capitalize()
    return target


@lru_cache(maxsize=1024)
def _best_domain_correction(token: str) -> str:
    t = (token or "").lower().strip()
    if not t or len(t) < 4 or any(ch.isdigit() for ch in t) or t in _BIL_DOMAIN_TERMS or t in _TYPO_PROTECT_TERMS:
        return t

    matches = difflib.get_close_matches(t, _BIL_DOMAIN_TERMS, n=2, cutoff=0.74)
    if not matches:
        return t

    best = matches[0]
    ratio = difflib.SequenceMatcher(None, t, best).ratio()
    diffs = [i for i, (a, b) in enumerate(zip(t, best)) if a != b]
    simple_transposition = (
        len(t) == len(best)
        and len(diffs) == 2
        and t[diffs[0]] == best[diffs[1]]
        and t[diffs[1]] == best[diffs[0]]
    )

    safe = False
    if simple_transposition:
        safe = True
    elif len(t) >= 7 and ratio >= 0.84 and t[0] == best[0]:
        safe = True
    elif len(t) >= 5 and ratio >= 0.88 and t[0] == best[0]:
        safe = True
    elif len(t) == len(best) and ratio >= 0.92 and t[:1] == best[:1] and t[-1:] == best[-1:]:
        safe = True

    if not safe:
        return t

    if len(matches) > 1:
        second = matches[1]
        second_ratio = difflib.SequenceMatcher(None, t, second).ratio()
        if second != best and (ratio - second_ratio) < 0.04:
            return t

    return best


@lru_cache(maxsize=2048)
def normalize_bil_query_text(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return raw

    parts = re.split(r"(\W+)", raw)
    out = []
    for part in parts:
        if not part or re.fullmatch(r"\W+", part):
            out.append(part)
            continue
        corrected = _best_domain_correction(part)
        out.append(_match_case(part, corrected))
    return "".join(out)


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
    q = normalize_bil_query_text(q).lower().strip()
    if not q:
        return False

    if any(k in q for k in _FORM_KEYWORDS):
        return True

    return False



def looks_like_form_download_request(q: str) -> bool:
    q = normalize_bil_query_text(q).lower().strip()
    if not q:
        return False

    formish = looks_like_form_request(q)
    has_download_verb = any(v in q for v in _DOWNLOAD_VERBS)

    if re.search(r"\bform\b\s+(for|of)\s+\w+", q):
        return True
    if formish and has_download_verb:
        return True

    if "form" in q and len(q.split()) <= 4:
        return True

    return False



def looks_like_document_download_request(q: str) -> bool:
    q = normalize_bil_query_text(q).lower().strip()
    if not q:
        return False

    has_doc_noun = any(n in q for n in _DOC_NOUNS)
    if not has_doc_noun:
        return False

    has_file_term = any(t in q for t in _FILE_REQUEST_TERMS)
    has_share_verb = any(v in q for v in _FILE_SHARE_VERBS)

    if has_file_term:
        return True
    if has_share_verb and has_doc_noun:
        return True

    if re.search(r"\b(annual\s+report|report|handbook|guide|manual)\b.*\b(pdf|download|file|link|attachment)\b", q):
        return True

    return False
