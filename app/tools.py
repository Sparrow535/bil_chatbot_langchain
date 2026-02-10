import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import re
from urllib.parse import unquote
import random
import threading

from langchain.tools import tool
from app.config import settings
from app.vectorstore import load_vectorstore
from app.text_utils import looks_like_form_request, clean_text

FORMS_PATH = Path("data/forms.json")


# ---- IMPORTANT: Tune these ----
TOP_K = 4
MIN_SCORE = 0.30  # raise to reduce irrelevant forms

# Words that cause "job application form" to match "loan application form"
_FORM_STOPWORDS = {
    "form", "forms", "pdf", "download", "downloads",
    "application", "proposal", "claim", "letter", "authorization",
    "bil", "bhutan", "insurance", "limited", "company", "ltd",
    "document", "documents", "doc", "file", "files",
    "the", "a", "an", "for", "of", "to", "me", "give", "send", "this", "that",
}

def _tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    text = unquote(text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    toks = [t for t in text.split(" ") if len(t) > 2 and t not in _FORM_STOPWORDS]
    return toks

def _title_from_url(url: str) -> str:
    if not url:
        return "Download form"
    name = url.split("?")[0].split("/")[-1]
    name = re.sub(r"\.(pdf|docx|doc)$", "", name, flags=re.IGNORECASE)
    name = name.replace("-", " ").replace("_", " ").strip()
    name = re.sub(r"\s+", " ", name)
    return name.title() if name else "Download form"

def _load_forms() -> List[Dict[str, Any]]:
    if not FORMS_PATH.exists():
        return []
    return json.loads(FORMS_PATH.read_text(encoding="utf-8"))

def _score_form(query: str, form: Dict[str, Any]) -> float:
    """Score a form by overlap of meaningful tokens + small bonus for filename/url matches."""
    q_tokens = set(_tokenize(query))
    if not q_tokens:
        return 0.0

    title = form.get("title") or ""
    url = form.get("url") or ""
    cat = form.get("category") or ""

    hay = f"{title} {url} {cat}"
    h_tokens = set(_tokenize(hay))

    overlap = len(q_tokens & h_tokens)
    if overlap == 0:
        return 0.0

    base = overlap / max(1, len(q_tokens))

    url_l = unquote((url or "").lower())
    bonus = 0.0
    for t in q_tokens:
        if t in url_l:
            bonus += 0.08

    return min(1.0, base + bonus)

def _best_matches(query: str, forms: List[Dict[str, Any]], top_k: int = TOP_K) -> List[Dict[str, Any]]:
    scored: List[tuple[float, Dict[str, Any]]] = []
    for f in forms:
        s = _score_form(query, f)
        if s >= MIN_SCORE:
            scored.append((s, f))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [f for _, f in scored[:top_k]]

_vectorstore = None
_forms = None
_resource_lock = threading.RLock()

def init_resources():
    global _forms, _vectorstore
    with _resource_lock:
        if _forms is None:
            _forms = _load_forms()
        if _vectorstore is None:
            _vectorstore = load_vectorstore()

def refresh_vectorstore() -> None:
    """Reload vectorstore + forms safely after incremental indexing."""
    global _vectorstore, _forms
    with _resource_lock:
        _vectorstore = load_vectorstore()
        _forms = _load_forms()

@tool
def bil_get_forms(query: str) -> str:
    """
    Use when user asks for downloadable forms/documents.
    Returns JSON string with matching forms and URLs (from forms.json only).
    """
    init_resources()
    if not looks_like_form_request(query):
        return json.dumps({"matches": [], "found": False}, ensure_ascii=False)

    with _resource_lock:
        forms = list(_forms or [])
    matches = _best_matches(query, forms, top_k=TOP_K)

    return json.dumps(
        {
            "matches": [
                {
                    "title": (m.get("title") or _title_from_url(m.get("url", ""))).strip(),
                    "url": (m.get("url") or "").strip(),
                }
                for m in matches
                if (m.get("url") or "").strip()
            ],
            "found": bool(matches),
        },
        ensure_ascii=False,
    )



@tool
def bil_retrieve_context(query: str) -> str:
    """
    Use when user asks BIL-related questions that require knowledge from bil.bt pages/docs.
    Returns JSON string:
      { "chunks": [{"content": "...", "source": "...", "title": "...", "score": 0.0..1.0}], "found": true/false }
    """
    init_resources()
    with _resource_lock:
        vectorstore = _vectorstore

    # Use raw scores and normalize to 0..1 to avoid invalid relevance warnings
    docs = vectorstore.similarity_search_with_score(query, k=settings.top_k)

    def _to_similarity(score: float) -> float:
        try:
            s = float(score)
        except Exception:
            return 0.0
        if 0.0 <= s <= 1.0:
            return s
        # Treat as distance or unbounded score -> map to (0,1]
        return 1.0 / (1.0 + abs(s))

    strong = []
    all_scored = []
    for d, score in docs:
        sim = _to_similarity(score)
        all_scored.append((d, sim))
        if sim >= settings.min_relevance:
            strong.append({
                "content": clean_text(d.page_content),
                "source": d.metadata.get("source", ""),
                "title": d.metadata.get("title", ""),
                "score": sim,
            })

    # If nothing meets threshold, fall back to top results
    if not strong and all_scored:
        for d, sim in all_scored[: max(1, min(2, len(all_scored)))]:
            strong.append({
                "content": clean_text(d.page_content),
                "source": d.metadata.get("source", ""),
                "title": d.metadata.get("title", ""),
                "score": sim,
            })

    return json.dumps({"chunks": strong, "found": bool(strong)}, ensure_ascii=False)

@tool
def bil_unrelated_reply(user_query: str) -> str:
    """
    Use when the user query is unrelated to Bhutan Insurance Limited (BIL).
    Returns a short redirection message.
    """
    options = [
        "Sorry, I can’t help with that. I can help with BIL services like insurance, claims, loans, contact info, and forms.",
        "I’m not able to answer that, but I can help with Bhutan Insurance Limited (BIL) services such as insurance, claims, loans, and forms.",
        "I'm only able to help with BIL services and products like insurance, claims, loans, contacts, and forms.",
        "I don’t have information on that topic. If you need BIL support, I can help with insurance, claims, loans, and forms.",
    ]
    return random.choice(options)

@tool
def bil_intent_hint(query: str) -> str:
    """
    Lightweight intent hint to route quickly.
    Returns one of: form_request | bil_query | unrelated
    """
    q = (query or "").lower()

    if looks_like_form_request(q):
        return "form_request"

    bil_terms = [
        "bhutan insurance", "bil", "insurance", "policy", "premium", "claim",
        "motor", "health", "travel", "fire", "branch", "contact", "loan",
        "machinery", "machinery breakdown", "contractor", "contractors", "plant",
        "engineering", "burglary", "liability", "aviation", "marine", "fidelity",
        "money", "personal accident", "workmen", "student care"
    ]
    if any(t in q for t in bil_terms) or "bil.bt" in q:
        return "bil_query"

    return "unrelated"
