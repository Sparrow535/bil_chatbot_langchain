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
_annual_reports = None
_resource_lock = threading.RLock()

_FINANCIAL_QUERY_TERMS = {
    "annual report", "annual report summary", "report highlights", "financial highlights",
    "annual income", "income", "revenue", "profit", "earnings", "gross premium",
    "premium written", "loan assets", "total assets", "shareholders equity",
    "shareholders' equity", "equity", "npl", "profit after tax", "profit before tax",
    "operating profit", "earnings per share", "eps",
}
_FINANCIAL_METRIC_ALIASES = {
    "profit_after_tax": ["annual income", "net income", "annual earnings", "earnings after tax", "profit after tax", "profit"],
    "profit_before_tax": ["profit before tax", "pre tax profit", "pretax profit"],
    "gross_premium": ["gross premium written", "gross premium", "gross written premium", "premium written"],
    "loan_assets": ["loan assets", "loan asset"],
    "total_assets": ["total assets", "overall assets", "assets"],
    "shareholders_equity": ["shareholders equity", "shareholders' equity", "shareholder equity", "equity"],
    "net_npl_ratio": ["net npl ratio", "npl ratio", "npl"],
    "earnings_per_share": ["basic earnings per share", "earnings per share", "earning per share", "eps"],
    "general_insurance_operating_profit": ["general insurance department operating profit", "general insurance operating profit", "insurance department operating profit", "general insurance profit"],
    "financing_investment_operating_profit": ["financing and investment department operating profit", "financing and investment operating profit", "investment department operating profit", "investment operating profit", "financing operating profit"],
}
_FINANCIAL_METRIC_LABELS = {
    "profit_after_tax": "profit after tax",
    "profit_before_tax": "profit before tax",
    "gross_premium": "gross premium written",
    "loan_assets": "loan assets",
    "total_assets": "total assets",
    "shareholders_equity": "shareholders' equity",
    "net_npl_ratio": "net NPL ratio",
    "earnings_per_share": "earnings per share",
    "general_insurance_operating_profit": "general insurance operating profit",
    "financing_investment_operating_profit": "financing and investment operating profit",
}
_FINANCIAL_VALUE_FORMATS = {
    "earnings_per_share": "Nu. {value} per share",
    "general_insurance_operating_profit": "Nu. {value} million",
    "financing_investment_operating_profit": "Nu. {value} million",
}
_METRICS_MILLION_FROM_CONTEXT = {
    "profit_after_tax",
    "profit_before_tax",
    "gross_premium",
    "total_assets",
    "shareholders_equity",
    "loan_assets",
}
_FINANCIAL_PATTERNS = {
    "profit_after_tax": [
        re.compile(r"business performance highlights.{0,700}?profit after tax(?: for the year)?\s*(?P<value>[0-9]+(?:\.[0-9]+)?)", re.IGNORECASE | re.DOTALL),
        re.compile(r"profit after tax(?: for the year)?(?: attributable to [^.]+?)?(?: of| was| to)?\s*Nu\.?\s*(?P<value>[0-9][0-9,]*(?:\.[0-9]+)?)\s*(?P<unit>million|billion)?", re.IGNORECASE),
        re.compile(r"earnings after tax(?: attributable to [^.]+?)?(?: of| was| to)?\s*Nu\.?\s*(?P<value>[0-9][0-9,]*(?:\.[0-9]+)?)\s*(?P<unit>million|billion)?", re.IGNORECASE),
        re.compile(r"profit attributable to ordinary shareholders\s*(?P<value>[0-9][0-9,]*(?:\.[0-9]+)?)", re.IGNORECASE),
    ],
    "profit_before_tax": [
        re.compile(r"financial profitability.{0,320}?profit before\s*t\s*ax\s*(?P<value>[0-9]+(?:\.[0-9]+)?)", re.IGNORECASE | re.DOTALL),
        re.compile(r"profit before\s*t\s*ax(?: of| was| to)?\s*Nu\.?\s*(?P<value>[0-9][0-9,]*(?:\.[0-9]+)?)\s*(?P<unit>million|billion)?", re.IGNORECASE),
        re.compile(r"profit before\s*t\s*ax\s*(?P<value>[0-9][0-9,]*(?:\.[0-9]+)?)", re.IGNORECASE),
    ],
    "gross_premium": [
        re.compile(r"gross premium(?: written)?(?: of| was| to)?\s*Nu\.?\s*(?P<value>[0-9][0-9,]*(?:\.[0-9]+)?)\s*(?P<unit>million|billion)?", re.IGNORECASE),
        re.compile(r"business performance highlights.{0,520}?gross (?:written )?premium(?:\s+of)?(?:\s*nu\.?)?\s*(?P<value>[0-9]+(?:\.[0-9]+)?)", re.IGNORECASE | re.DOTALL),
    ],
    "loan_assets": [
        re.compile(r"loan assets.{0,260}? to\s*Nu\.?\s*(?P<value>[0-9][0-9,]*(?:\.[0-9]+)?)\s*(?P<unit>million|billion)?", re.IGNORECASE | re.DOTALL),
        re.compile(r"loan assets(?: as on [^.]+?)?(?: has increased| increased| stood at| stands? at| were| was)?.{0,200}?\bNu\.?\s*(?P<value>[0-9][0-9,]*(?:\.[0-9]+)?)\s*(?P<unit>million|billion)?", re.IGNORECASE | re.DOTALL),
    ],
    "total_assets": [
        re.compile(r"business performance highlights.{0,320}?t\s*otal assets\s*(?P<value>[0-9][0-9,]*(?:\.[0-9]+)?)", re.IGNORECASE | re.DOTALL),
        re.compile(r"(?:overall assets of the company|t\s*otal assets(?: of the company)?).*? to\s*Nu\.?\s*(?P<value>[0-9][0-9,]*(?:\.[0-9]+)?)\s*(?P<unit>million|billion)?", re.IGNORECASE),
        re.compile(r"(?:overall assets of the company|t\s*otal assets(?: of the company)?) [^.]*?Nu\.?\s*(?P<value>[0-9][0-9,]*(?:\.[0-9]+)?)\s*(?P<unit>million|billion)?", re.IGNORECASE),
    ],
    "shareholders_equity": [
        re.compile(r"shareholders? equity.*? to\s*Nu\.?\s*(?P<value>[0-9][0-9,]*(?:\.[0-9]+)?)\s*(?P<unit>million|billion)?", re.IGNORECASE),
        re.compile(r"shareholders? equity [^.]*?Nu\.?\s*(?P<value>[0-9][0-9,]*(?:\.[0-9]+)?)\s*(?P<unit>million|billion)?", re.IGNORECASE),
    ],
    "net_npl_ratio": [
        re.compile(r"net npl ratio [^.]*?(?P<value>[0-9]+(?:\.[0-9]+)?)\s*(?P<unit>%)", re.IGNORECASE),
    ],
    "earnings_per_share": [
        re.compile(r"(?:basic\s+)?earnings?\s+per\s+share\s*\(?(?:nu\.?)?\)?\s*(?:\d+\s+)?(?P<value>[0-9]+\.[0-9]+)", re.IGNORECASE),
        re.compile(r"(?:basic\s+)?earning\s+per\s+share\s*(?:\d+\s+)?(?P<value>[0-9]+\.[0-9]+)", re.IGNORECASE),
    ],
    "general_insurance_operating_profit": [
        re.compile(r"business performance highlights.{0,560}?operating profit(?: insurance| general insurance) department\s*\(?(?P<value>-?[0-9]+(?:\.[0-9]+)?)\)?", re.IGNORECASE | re.DOTALL),
        re.compile(r"general insurance department\s*\(revenue a/c\)\s*\(?(?P<value>-?[0-9][0-9,]*(?:\.[0-9]+)?)\)?", re.IGNORECASE),
    ],
    "financing_investment_operating_profit": [
        re.compile(r"business performance highlights.{0,640}?investment department\s*\(?(?P<value>-?[0-9]+(?:\.[0-9]+)?)\)?", re.IGNORECASE | re.DOTALL),
        re.compile(r"business performance highlights.{0,640}?operating profit(?: investment| financing(?:\s+and|\s*&)?\s*investment) department\s*\(?(?P<value>-?[0-9]+(?:\.[0-9]+)?)\)?", re.IGNORECASE | re.DOTALL),
        re.compile(r"financing\s*(?:&|and)\s*investment department\s*\(revenue a/c\)\s*\(?(?P<value>-?[0-9][0-9,]*(?:\.[0-9]+)?)\)?", re.IGNORECASE),
    ],
}
_FINANCIAL_HIGHLIGHT_ORDER = [
    "profit_after_tax",
    "profit_before_tax",
    "gross_premium",
    "earnings_per_share",
    "loan_assets",
    "total_assets",
    "shareholders_equity",
    "net_npl_ratio",
    "general_insurance_operating_profit",
    "financing_investment_operating_profit",
]
_FINANCIAL_SECTION_ORDER = [
    ("At a glance", ["profit_after_tax", "profit_before_tax", "gross_premium", "earnings_per_share"]),
    ("Balance sheet and portfolio", ["loan_assets", "total_assets", "shareholders_equity", "net_npl_ratio"]),
    ("Business segments", ["general_insurance_operating_profit", "financing_investment_operating_profit"]),
]
_FINANCIAL_SUMMARY_LEAD_ORDER = [
    "profit_after_tax",
    "gross_premium",
    "total_assets",
    "loan_assets",
]

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
    global _vectorstore, _forms, _docs, _annual_reports
    with _resource_lock:
        _vectorstore = load_vectorstore()
        _forms = _load_forms()
        _docs = _load_documents()
        _annual_reports = None

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
    loan_rate_terms = {"interest rate", "interest rates", "loan rate", "loan rates", "rates"}
    loan_scope_terms = {
        "loan", "loans", "housing", "personal", "transport", "shares", "securities",
        "contractor", "tourism", "hotel", "agriculture", "livestock", "service sector",
        "trade", "commerce", "production", "manufacturing", "forestry", "logging",
        "mining", "quarrying",
    }
    overview_terms = {"products", "policies", "types", "options", "offered", "available", "various", "different"}
    k = settings.top_k
    if any(t in qn for t in fund_terms):
        k = max(settings.top_k, 8)
    if any(t in qn for t in loan_rate_terms) and any(t in qn for t in loan_scope_terms):
        k = max(k, 14)
    if (("insurance" in qn or "policies" in qn) and any(t in qn for t in overview_terms)) or qn.strip() in {"insurance", "policies"}:
        k = max(k, 12)
    if (("loan" in qn or "loans" in qn) and any(t in qn for t in overview_terms)) or qn.strip() in {"loan", "loans"}:
        k = max(k, 14)
    if (("claim" in qn or "claims" in qn) and any(t in qn for t in overview_terms)) or qn.strip() in {"claim", "claims"}:
        k = max(k, 10)

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


def _load_annual_reports() -> Dict[str, Dict[str, Any]]:
    global _annual_reports
    with _resource_lock:
        if _annual_reports is not None:
            return _annual_reports

        reports: Dict[str, Dict[str, Any]] = {}
        if RAW_PATH.exists():
            with RAW_PATH.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if (obj.get("type") or "").strip().lower() != "document":
                        continue
                    title = (obj.get("title") or "").strip()
                    source = (obj.get("source") or "").strip()
                    blob = f"{title} {source}".lower()
                    if "annual report" not in blob and "annual-report" not in blob:
                        continue
                    m = re.search(r"(?:annual report|annual-report)[^0-9]*(20\d{2})", blob)
                    if not m:
                        m = re.search(r"(20\d{2})[^0-9]*(?:annual report|annual-report)", blob)
                    if not m:
                        continue
                    year = m.group(1)
                    reports[year] = {
                        "title": title,
                        "source": source,
                        "text": clean_text(obj.get("text") or ""),
                    }
        _annual_reports = reports
        return _annual_reports


def _looks_like_annual_report_summary_query(query: str) -> bool:
    ql = (query or "").lower()
    if not re.search(r"\b20\d{2}\b", ql):
        return False
    if "annual report" in ql:
        return True
    summary_cues = {"report", "summary", "highlights", "overview", "tell me about", "about the report"}
    return any(cue in ql for cue in summary_cues) and "report" in ql


def _looks_like_financial_fact_query(query: str) -> bool:
    ql = (query or "").lower()
    if not re.search(r"\b20\d{2}\b", ql):
        return False
    return _looks_like_annual_report_summary_query(query) or any(term in ql for term in _FINANCIAL_QUERY_TERMS)


def _normalize_financial_phrase(text: str) -> str:
    normalized = (text or "").lower().replace("&", " and ")
    normalized = normalized.replace("'", " ")
    normalized = re.sub(r"[^a-z0-9.%]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _resolve_financial_metric(query: str) -> str:
    ql = _normalize_financial_phrase(query)
    if not ql:
        return ""

    qn = f" {ql} "
    best_metric = ""
    best_len = 0
    for metric, aliases in _FINANCIAL_METRIC_ALIASES.items():
        for alias in aliases:
            alias_n = _normalize_financial_phrase(alias)
            if not alias_n:
                continue
            if f" {alias_n} " not in qn:
                continue
            if len(alias_n) > best_len:
                best_metric = metric
                best_len = len(alias_n)
    return best_metric


def _wants_annual_report_summary(query: str, metric: str) -> bool:
    ql = (query or "").lower()
    if _looks_like_annual_report_summary_query(query) and not metric:
        return True
    if not metric and any(cue in ql for cue in ["highlights", "summary", "overview"]):
        return True
    return False


def _extract_financial_sentence(text: str, start: int, end: int) -> str:
    left = max(text.rfind(". ", 0, start), text.rfind("; ", 0, start), text.rfind(": ", 0, start))
    if left < 0:
        left = max(text.rfind(".", 0, start), text.rfind(";", 0, start), text.rfind(":", 0, start))
    right_candidates = [idx for idx in [text.find(". ", end), text.find("; ", end)] if idx != -1]
    if right_candidates:
        right = min(right_candidates) + 1
    else:
        right = text.find(".", end)
        if right == -1:
            right = min(len(text), end + 220)
    return clean_text(text[left + 1:right + 1])


def _format_financial_value(metric: str, match: re.Match[str], text: str) -> str:
    value = (match.group("value") or "").strip()
    unit = (match.groupdict().get("unit") or "").strip()
    matched_text = match.group(0) or ""
    if value and not value.startswith("-") and re.search(rf"\(\s*{re.escape(value)}\s*\)", matched_text):
        value = f"-{value}"
    if unit == "%":
        return f"{value}%"
    if unit:
        return f"Nu. {value} {unit}"
    local_window = text[max(0, match.start() - 220): min(len(text), match.end() + 220)].lower()
    if (
        metric in _METRICS_MILLION_FROM_CONTEXT
        and re.fullmatch(r"-?[0-9][0-9,]*(?:\.[0-9]+)?", value)
        and "." in value
        and "figures in million" in local_window
    ):
        return f"Nu. {value} million"
    if metric in _FINANCIAL_VALUE_FORMATS:
        return _FINANCIAL_VALUE_FORMATS[metric].format(value=value)
    return f"Nu. {value}"


def _is_financial_metric_match_valid(metric: str, sentence: str) -> bool:
    sentence_l = (sentence or "").lower()
    if metric == "loan_assets":
        blocked = {"pending forclosure", "transferred", "npl", "non-performing"}
        if any(term in sentence_l for term in blocked):
            return False
    return True


def _extract_financial_metric(text: str, metric: str) -> Dict[str, str] | None:
    for pat in _FINANCIAL_PATTERNS.get(metric, []):
        m = pat.search(text)
        if not m:
            continue
        sentence = _extract_financial_sentence(text, m.start(), m.end())
        if not _is_financial_metric_match_valid(metric, sentence):
            continue
        return {
            "metric": metric,
            "label": _FINANCIAL_METRIC_LABELS[metric],
            "value": _format_financial_value(metric, m, text),
            "sentence": sentence,
        }
    return None


def _render_financial_primary(year: str, item: Dict[str, str]) -> str:
    if item.get("metric") == "net_npl_ratio":
        return f"In {year}, BIL's {item['label']} was {item['value']}."
    return f"In {year}, BIL reported {item['label']} of {item['value']}."


def _display_metric_label(label: str) -> str:
    if (label or "").lower() == "net npl ratio":
        return "Net NPL ratio"
    cleaned = (label or "").strip()
    if not cleaned:
        return ""
    return cleaned[:1].upper() + cleaned[1:]


def _group_financial_highlights(
    highlights: List[Dict[str, str]],
    exclude_metrics: set[str] | None = None,
    max_items: int | None = None,
) -> List[Tuple[str, List[Dict[str, str]]]]:
    highlight_map = {
        item.get("metric", ""): item
        for item in highlights
        if item.get("metric")
    }
    excluded = set(exclude_metrics or set())
    grouped: List[Tuple[str, List[Dict[str, str]]]] = []
    count = 0

    for section_title, metric_keys in _FINANCIAL_SECTION_ORDER:
        items: List[Dict[str, str]] = []
        for key in metric_keys:
            if key in excluded or key not in highlight_map:
                continue
            items.append(highlight_map[key])
            count += 1
            if max_items is not None and count >= max_items:
                break
        if items:
            grouped.append((section_title, items))
        if max_items is not None and count >= max_items:
            break

    return grouped


def _render_highlight_sections_md(
    sections: List[Tuple[str, List[Dict[str, str]]]],
    heading: str,
) -> str:
    if not sections:
        return ""

    lines = [f"**{heading}**"]
    for section_title, items in sections:
        lines.append(f"\n**{section_title}**")
        for item in items:
            lines.append(f"- {_display_metric_label(item['label'])}: {item['value']}")
    return "\n".join(lines)


def _render_missing_annual_report(year: str, available_years: List[str]) -> tuple[str, str]:
    ordered = sorted(available_years)
    if ordered:
        earliest = ordered[0]
        latest = ordered[-1]
        if year < earliest:
            answer = f"I couldn't find a BIL annual report for {year} in the indexed documents. The earliest report I have is {earliest}."
        elif year > latest:
            answer = f"I don't have the {year} annual report in the indexed documents yet. The latest report I have is {latest}."
        else:
            answer = f"I found references to the {year} annual report, but I couldn't extract a reliable summary from the indexed copy."
    else:
        answer = f"I couldn't find a BIL annual report for {year} in the indexed documents."

    answer_md = f"**BIL {year} Annual Report**\n\n{answer}"
    return answer, answer_md


def _render_annual_report_summary(year: str, highlights: List[Dict[str, str]]) -> tuple[str, str]:
    if not highlights:
        return "", ""

    highlight_map = {item["metric"]: item for item in highlights if item.get("metric")}
    lead = [highlight_map[key] for key in _FINANCIAL_SUMMARY_LEAD_ORDER if key in highlight_map][:3]
    if not lead:
        lead = highlights[:3]
    fragments = [f"{item['label']} of {item['value']}" for item in lead]

    if len(highlights) <= 2:
        answer = f"I found the {year} annual report, but I could only extract a limited summary from the current indexed copy: {fragments[0]}."
        intro = f"I found the {year} report, but the extracted text for this year is limited. Here are the figures I could read reliably."
    elif len(fragments) == 1:
        answer = f"For {year}, BIL's annual report highlights {fragments[0]}."
        intro = f"Here is the main picture from the {year} annual report."
    elif len(fragments) == 2:
        answer = f"For {year}, BIL's annual report highlights {fragments[0]} and {fragments[1]}."
        intro = f"Here is the main picture from the {year} annual report."
    else:
        answer = f"For {year}, BIL's annual report highlights " + ", ".join(fragments[:-1]) + f", and {fragments[-1]}."
        intro = f"Here is the main picture from the {year} annual report."

    sections = _group_financial_highlights(highlights, max_items=10)
    answer_md = (
        f"**BIL {year} Annual Report**\n\n"
        f"{intro}\n\n"
        f"{_render_highlight_sections_md(sections, 'Key figures')}"
    ).strip()

    return answer, answer_md

@tool
def bil_extract_financial_fact(query: str) -> str:
    """
    Use for year-specific BIL annual-report and financial-result questions.
    Returns JSON with either a direct metric answer or a compact annual-report summary.
    """
    if not _looks_like_financial_fact_query(query):
        return json.dumps({"found": False}, ensure_ascii=False)

    years = re.findall(r"\b(20\d{2})\b", query or "")
    if not years:
        return json.dumps({"found": False}, ensure_ascii=False)
    year = years[0]

    reports = _load_annual_reports()
    report = reports.get(year)
    if not report or not report.get("text"):
        if _looks_like_annual_report_summary_query(query):
            answer, answer_md = _render_missing_annual_report(year, list(reports.keys()))
            return json.dumps(
                {
                    "found": True,
                    "year": year,
                    "mode": "missing_report",
                    "metric": "",
                    "answer": answer,
                    "answer_md": answer_md,
                    "highlights": [],
                    "source": "",
                },
                ensure_ascii=False,
            )
        return json.dumps({"found": False, "year": year}, ensure_ascii=False)

    report_text = report["text"]
    extracted: Dict[str, Dict[str, str]] = {}
    for metric in _FINANCIAL_HIGHLIGHT_ORDER:
        item = _extract_financial_metric(report_text, metric)
        if item:
            extracted[metric] = item

    if not extracted:
        if _looks_like_annual_report_summary_query(query):
            answer, answer_md = _render_missing_annual_report(year, list(reports.keys()))
            return json.dumps(
                {
                    "found": True,
                    "year": year,
                    "mode": "missing_report",
                    "metric": "",
                    "answer": answer,
                    "answer_md": answer_md,
                    "highlights": [],
                    "source": report.get("title", ""),
                },
                ensure_ascii=False,
            )
        return json.dumps({"found": False, "year": year}, ensure_ascii=False)

    metric = _resolve_financial_metric(query)
    wants_summary = _wants_annual_report_summary(query, metric)

    if wants_summary:
        summary_items = [extracted[key] for key in _FINANCIAL_HIGHLIGHT_ORDER if key in extracted]
        answer, answer_md = _render_annual_report_summary(year, summary_items)
        return json.dumps(
            {
                "found": True,
                "year": year,
                "mode": "summary",
                "metric": "",
                "answer": answer,
                "answer_md": answer_md,
                "highlights": summary_items[:10],
                "source": report.get("title", ""),
            },
            ensure_ascii=False,
        )

    primary = extracted.get(metric)
    if primary is None:
        primary = extracted.get("profit_after_tax") or extracted.get("gross_premium") or next(iter(extracted.values()))

    extra_items = []
    for key in _FINANCIAL_HIGHLIGHT_ORDER:
        if key == primary.get("metric") or key not in extracted:
            continue
        extra_items.append(extracted[key])
        if len(extra_items) >= 6:
            break

    answer = _render_financial_primary(year, primary)
    answer_md = f"**BIL {year}**\n\n{answer}"
    sections = _group_financial_highlights(
        [extracted[key] for key in _FINANCIAL_HIGHLIGHT_ORDER if key in extracted],
        exclude_metrics={primary.get("metric", "")},
        max_items=6,
    )
    sections_md = _render_highlight_sections_md(sections, "Other figures from the same report")
    if sections_md:
        answer_md += f"\n\n{sections_md}"
    return json.dumps(
        {
            "found": True,
            "year": year,
            "mode": "metric",
            "metric": primary.get("metric", ""),
            "answer": answer,
            "answer_md": answer_md,
            "highlights": extra_items,
            "source": report.get("title", ""),
        },
        ensure_ascii=False,
    )



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
