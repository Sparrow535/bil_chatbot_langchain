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
from app.text_utils import (
    looks_like_form_download_request,
    looks_like_document_download_request,
    clean_text,
)

FORMS_PATH = Path("data/forms.json")
RAW_PATH = Path("data/raw.jsonl")


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
    "tell", "about", "explain", "describe", "details", "information", "info",
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

def _load_documents() -> List[Dict[str, Any]]:
    if not RAW_PATH.exists():
        return []

    out: Dict[str, Dict[str, Any]] = {}
    with RAW_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            source = (obj.get("source") or "").strip()
            if not source:
                continue

            lower_src = source.lower().split("?")[0]
            doc_type = (obj.get("type") or "").strip().lower()
            if doc_type != "document" and not lower_src.endswith((".pdf", ".doc", ".docx")):
                continue

            if source in out:
                continue

            title = (obj.get("title") or _title_from_url(source)).strip()
            text = (obj.get("text") or "").strip()
            out[source] = {
                "title": title or _title_from_url(source),
                "url": source,
                "snippet": clean_text(text[:1200]) if text else "",
            }

    return list(out.values())

def _score_form(query: str, form: Dict[str, Any]) -> float:
    """Score a form by overlap of meaningful tokens + small bonus for filename/url matches."""
    q_tokens = set(_tokenize(query))
    if not q_tokens:
        return 0.0

    title = form.get("title") or ""
    url = form.get("url") or ""
    cat = form.get("category") or ""

    hay = f"{title} {url} {cat}"
    hay_l = hay.lower()
    h_tokens = set(_tokenize(hay))

    overlap = len(q_tokens & h_tokens)
    if overlap == 0:
        return 0.0

    base = overlap / max(1, len(q_tokens))

    url_l = unquote((url or "").lower())
    ql = (query or "").lower()
    bonus = 0.0
    for t in q_tokens:
        if t in url_l:
            bonus += 0.08
    if "proposal" in ql and "proposal" in hay_l:
        bonus += 0.18
    if "application" in ql and "application" in hay_l:
        bonus += 0.18
    if "claim" in ql and ("claim" in hay_l or "intimation" in hay_l):
        bonus += 0.18
    if "form" in ql and "form" in hay_l:
        bonus += 0.08

    return min(1.0, base + bonus)

def _best_matches(query: str, forms: List[Dict[str, Any]], top_k: int = TOP_K) -> List[Dict[str, Any]]:
    ql = (query or "").lower()

    def _specificity_rank(form: Dict[str, Any]) -> int:
        hay = f"{form.get('title', '')} {form.get('url', '')}".lower()
        rank = 0
        if "proposal" in ql and "proposal" in hay:
            rank += 3
        if "application" in ql and "application" in hay:
            rank += 3
        if "claim" in ql and ("claim" in hay or "intimation" in hay):
            rank += 3
        if "form" in ql and "form" in hay:
            rank += 1
        return rank

    scored: List[tuple[float, int, Dict[str, Any]]] = []
    for f in forms:
        s = _score_form(query, f)
        if s >= MIN_SCORE:
            scored.append((s, _specificity_rank(f), f))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [f for _, _, f in scored[:top_k]]

_vectorstore = None
_forms = None
_docs = None
_resource_lock = threading.RLock()

def init_resources():
    global _forms, _docs, _vectorstore
    with _resource_lock:
        if _forms is None:
            _forms = _load_forms()
        if _docs is None:
            _docs = _load_documents()
        if _vectorstore is None:
            _vectorstore = load_vectorstore()

def refresh_vectorstore() -> None:
    """Reload vectorstore + forms safely after incremental indexing."""
    global _vectorstore, _forms, _docs
    with _resource_lock:
        _vectorstore = load_vectorstore()
        _forms = _load_forms()
        _docs = _load_documents()

@tool
def bil_get_forms(query: str) -> str:
    """
    Use when user asks for downloadable forms.
    Returns JSON string with matching forms and URLs (from forms.json only).
    """
    init_resources()

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

def _score_document(query: str, doc: Dict[str, Any]) -> float:
    q_tokens = set(_tokenize(query))
    if not q_tokens:
        return 0.0

    title = doc.get("title") or ""
    url = doc.get("url") or ""
    snippet = doc.get("snippet") or ""
    hay = f"{title} {url} {snippet}"
    h_tokens = set(_tokenize(hay))

    overlap = len(q_tokens & h_tokens)
    if overlap == 0:
        return 0.0

    base = overlap / max(1, len(q_tokens))
    bonus = 0.0

    years = re.findall(r"\b(20\d{2})\b", query or "")
    if years:
        for y in years:
            if y in title or y in url or y in snippet:
                bonus += 0.25

    ql = (query or "").lower()
    title_l = title.lower()
    url_l = url.lower()
    if "annual report" in ql and ("annual report" in title_l or "annual-report" in url_l):
        bonus += 0.20
    if any(w in ql for w in ["handbook", "guide", "manual"]):
        if any(w in title_l for w in ["handbook", "guide", "manual"]):
            bonus += 0.15

    return min(1.0, base + bonus)

@tool
def bil_get_documents(query: str) -> str:
    """
    Use when user asks to download non-form documents
    (annual reports, handbooks, guides, publications, PDFs).
    Returns JSON: {"matches":[{"title","url"}], "found": bool}
    """
    init_resources()
    if not looks_like_document_download_request(query):
        return json.dumps({"matches": [], "found": False}, ensure_ascii=False)

    with _resource_lock:
        docs = list(_docs or [])

    ql = (query or "").lower()
    want_report = ("annual report" in ql) or bool(re.search(r"\breport(s)?\b", ql))
    want_handbook = "handbook" in ql
    want_guide = "guide" in ql or "manual" in ql

    scored: List[Tuple[float, Dict[str, Any]]] = []
    for d in docs:
        title_l = (d.get("title") or "").lower()
        url_l = (d.get("url") or "").lower()

        if want_report and not (
            "annual report" in title_l
            or "annual-report" in url_l
            or re.search(r"\breport\b", title_l)
        ):
            continue
        if want_handbook and "handbook" not in title_l and "handbook" not in url_l:
            continue
        if want_guide and not (
            "guide" in title_l or "manual" in title_l or "guide" in url_l or "manual" in url_l
        ):
            continue

        s = _score_document(query, d)
        if s >= 0.22:
            scored.append((s, d))
    scored.sort(key=lambda x: x[0], reverse=True)

    years = re.findall(r"\b(20\d{2})\b", query or "")
    if years:
        year_hits_url: List[Tuple[float, Dict[str, Any]]] = []
        year_hits_title: List[Tuple[float, Dict[str, Any]]] = []
        for s, d in scored:
            title_l = (d.get("title") or "").lower()
            url_l = (d.get("url") or "").lower()
            if any(y in url_l for y in years):
                year_hits_url.append((s, d))
                continue
            if any(re.search(rf"\b{y}\b", title_l) for y in years):
                year_hits_title.append((s, d))
        if year_hits_url:
            scored = year_hits_url
        elif year_hits_title:
            scored = year_hits_title

    top = []
    for _, d in scored[:6]:
        title = (d.get("title") or _title_from_url(d.get("url", ""))).strip()
        url = (d.get("url") or "").strip()
        if not url:
            continue
        top.append({"title": title, "url": url})

    return json.dumps({"matches": top, "found": bool(top)}, ensure_ascii=False)



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

    qn = (query or "").lower()
    fund_terms = {"ppf", "provident", "gratuity", "gf", "gfm", "private provident"}
    k = settings.top_k
    if any(t in qn for t in fund_terms):
        k = max(settings.top_k, 8)

    # Use raw scores and normalize to 0..1 to avoid invalid relevance warnings
    docs = vectorstore.similarity_search_with_score(query, k=k)

    def _to_similarity(score: float) -> float:
        try:
            s = float(score)
        except Exception:
            return 0.0
        if 0.0 <= s <= 1.0:
            return s
        # Treat as distance or unbounded score -> map to (0,1]
        return 1.0 / (1.0 + abs(s))

    q_tokens = set(_tokenize(query))
    strong = []
    all_scored = []
    for d, score in docs:
        sim = _to_similarity(score)
        hay = (
            f"{d.metadata.get('title', '')} "
            f"{d.metadata.get('source', '')} "
            f"{(d.page_content or '')[:2200]}"
        ).lower()
        lexical_hits = sum(1 for t in q_tokens if t and t in hay)
        boosted = min(1.0, sim + (0.08 * min(3, lexical_hits)))

        all_scored.append((d, boosted, lexical_hits))
        if boosted >= settings.min_relevance:
            strong.append({
                "content": clean_text(d.page_content),
                "source": d.metadata.get("source", ""),
                "title": d.metadata.get("title", ""),
                "score": boosted,
            })

    strong.sort(key=lambda x: x.get("score", 0.0), reverse=True)

    # If nothing clears the main threshold, keep only weak-but-relevant lexical matches.
    if not strong and all_scored:
        all_scored.sort(key=lambda x: x[1], reverse=True)
        fallback_threshold = max(0.14, settings.min_relevance - 0.04)
        for d, sim, lexical_hits in all_scored[: max(1, min(3, len(all_scored)))]:
            if lexical_hits < 1 or sim < fallback_threshold:
                continue
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
    q_clean = re.sub(r"[^\w\s/+-]", " ", q)
    q_clean = re.sub(r"\s+", " ", q_clean).strip()

    if looks_like_form_download_request(q):
        return "form_request"
    if looks_like_document_download_request(q):
        return "bil_query"

    # Short finance shorthand used by users
    short_bil_finance = {
        "pf",
        "ppf",
        "gf",
        "gfm",
        "fund",
        "funds",
        "provident",
        "gratuity",
        "provident fund",
        "private provident fund",
        "private provident and gratuity fund",
        "ppf gf",
        "ppf/gf",
    }
    if q_clean in short_bil_finance:
        return "bil_query"

    bil_terms = [
        "bhutan insurance", "bil", "insurance", "policy", "premium", "claim",
        "motor", "health", "travel", "fire", "branch", "contact", "loan",
        "machinery", "machinery breakdown", "contractor", "contractors", "plant",
        "engineering", "burglary", "liability", "aviation", "marine", "fidelity",
        "money", "personal accident", "workmen", "student care",
        "ppf", "gf", "gfm", "provident", "gratuity", "private provident",
        "private provident fund", "private provident and gratuity fund",
        "loan against ppf", "fund management", "investment department",
        "ppf refund", "ppf employee registration", "ppf contribution",
        "ppf change of nominee", "mou gratuity fund", "mou ppf",
        "agriculture", "livestock", "tourism", "hotel", "hospitality",
        "service sector", "trade and commerce", "industrial", "housing",
        "loan for shares", "shares and securities", "vehicle loan",
        "annual report", "annual reports", "report", "reports",
        "insurance handbook", "handbook", "publications", "publication",
        "download forms", "claim intimation", "authorization letter"
    ]
    if any(t in q_clean for t in bil_terms) or "bil.bt" in q_clean:
        return "bil_query"

    return "unrelated"
