import json
import re
import random
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from app.config import settings
from app.text_utils import (
    looks_like_form_request,
    looks_like_form_download_request,
    looks_like_document_download_request,
)
from app.tools import (
    bil_get_forms,
    bil_get_documents,
    bil_retrieve_context,
    bil_unrelated_reply,
    bil_intent_hint,
)

load_dotenv()

# =========================
# Schemas
# =========================
class DownloadItem(BaseModel):
    title: str
    url: str


class AgentResponse(BaseModel):
    intent: str = Field(description="bil_query | form_request | unrelated | not_found")
    answer: str
    answer_md: str = Field(default="", description="Same answer in Markdown for UI rendering")
    sources: List[str] = Field(default_factory=list)
    downloads: List[DownloadItem] = Field(default_factory=list)
    confidence: str = Field(description="low | medium | high")

parser = PydanticOutputParser(pydantic_object=AgentResponse)

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_HELP_CLOSINGS = [
    "Let me know if you need anything else.",
    "I’m here if you need anything else.",
    "Happy to help if you need anything else.",
    "I can help with anything else you need.",
    "If you need more help, I’m here.",
    "I’m here to help with anything else you need.",
    "Glad to help anytime you need more support.",
]
_STYLE_HINTS = [
    "Start with a direct one-line answer, then add 3-6 focused bullets with concrete details.",
    "Use a compact summary paragraph, then grouped bullets by category.",
    "Use concise subheadings with 1-3 bullets each and avoid repeating heading names from previous turns.",
    "Keep it practical and specific; include what the user should do next in plain language.",
    "Vary sentence openings and list wording so it feels conversational, not templated.",
]
_PROCESS_STYLE_HINTS = [
    "For process/how-to requests, provide 4-6 steps and add a short 'why this matters' note for important steps.",
    "Explain the flow in phases (Before you start, Submit, After submission) with concise actionable bullets.",
    "Use a checklist style, but add brief context under each step so users understand the purpose.",
    "Include common mistakes or delays to avoid when relevant, using 1-2 compact bullets.",
]
_FOLLOWUP_STYLE_HINTS = [
    "Treat short follow-ups as continuation of prior context and answer directly without repeating generic intros.",
    "Keep follow-up responses concise but specific, with one short context reminder only if needed.",
]
_STYLE_TAGS = {
    "numbered": re.compile(r"(^|\n)\s*\d+\.\s", re.IGNORECASE),
    "key_points": re.compile(r"\bkey points\b", re.IGNORECASE),
    "overview": re.compile(r"(^|\n)\s*#+\s*overview\b", re.IGNORECASE),
    "subheading": re.compile(r"(^|\n)\s*#+\s+\w+", re.IGNORECASE),
}


def _detect_style_tags(text: str) -> set[str]:
    t = text or ""
    found: set[str] = set()
    for name, pat in _STYLE_TAGS.items():
        if pat.search(t):
            found.add(name)
    return found


def build_style_hint(query: str, history: List[Dict[str, str]]) -> str:
    qn = _norm(query)
    last_assistant = last_assistant_message(history)
    last_tags = _detect_style_tags(last_assistant)

    pool: List[str] = list(_STYLE_HINTS)
    if _is_actionable_query(query):
        pool.extend(_PROCESS_STYLE_HINTS)
    if len(qn.split()) <= 4 or is_affirmative_reply(query):
        pool.extend(_FOLLOWUP_STYLE_HINTS)

    filtered: List[str] = []
    for hint in pool:
        h = hint.lower()
        if "numbered" in h and "numbered" in last_tags:
            continue
        if "key points" in h and "key_points" in last_tags:
            continue
        if "subheading" in h and "subheading" in last_tags:
            continue
        if "overview" in h and "overview" in last_tags:
            continue
        filtered.append(hint)

    candidates = filtered or pool
    return random.choice(candidates)

# =========================
# Social intents
# =========================
_GREETINGS = {"hello", "hi", "hey", "good morning", "good afternoon", "good evening", "kuzuzangpo", "kuzuzangpo la", "kuzu"}
_FAREWELLS = {"bye", "goodbye", "see you", "see ya", "take care", "later", "okay", "ok"}
_THANKS = {"thanks", "thank you", "thx", "thanks a lot", "appreciate it"}
_TOPIC_PREFIX_RE = re.compile(
    r"^(tell me about|tell me more about|tell me|explain|describe|what is|what are|"
    r"give me info on|give me information on|information on|info on|details on|"
    r"i want to know about|i need to know about|i need info on|i want info on|"
    r"how to apply for|how do i apply for|how to claim for|how do i claim for)\s+",
    re.IGNORECASE,
)

def _norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _compact_topic(text: str) -> str:
    qn = _norm(text)
    if not qn:
        return ""
    qn = _TOPIC_PREFIX_RE.sub("", qn)
    qn = re.sub(r"\b(please|kindly|now|today)\b", "", qn)
    qn = re.sub(r"\s+", " ", qn).strip()
    return qn

def normalize_query_aliases(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return t
    # Common ASR typo: PIL -> BIL
    t = re.sub(r"\bPIL\b", "BIL", t, flags=re.IGNORECASE)
    # Common shorthand for BIL fund products
    t = re.sub(r"\bPF\b", "PPF", t, flags=re.IGNORECASE)
    t = re.sub(r"\bGFM\b", "GFM", t, flags=re.IGNORECASE)
    return t

def detect_social_intent(q: str) -> Optional[str]:
    qn = _norm(q)
    if qn in _GREETINGS:
        return "greeting"
    if qn in _FAREWELLS:
        return "farewell"
    if qn in _THANKS:
        return "thanks"
    return None


# =========================
# Markdown + JSON helpers
# =========================
def extract_first_json_object(text: str) -> str | None:
    if not text:
        return None
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None

def looks_like_markdown(md: str) -> bool:
    if not md:
        return False
    md = md.strip()
    return any(tok in md for tok in ["\n- ", "\n* ", "\n### ", "\n## ", "\n1. ", "**", "|"])

def repair_markdown_from_text(text: str) -> str:
    # Minimal repair: keep it readable, avoid inventing structure
    t = (text or "").strip()
    if not t:
        return ""
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"(\. )", ".\n", t)  # sentence line breaks
    return t.strip()

def strip_urls(text: str) -> str:
    if not text:
        return ""
    text = _URL_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def remove_question_sentences(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", t)
    kept = [p.strip() for p in parts if p.strip() and "?" not in p]
    return " ".join(kept).strip()

def remove_question_lines_md(md: str) -> str:
    t = (md or "").strip()
    if not t:
        return ""
    lines = t.splitlines()
    kept = [ln for ln in lines if "?" not in ln]
    return "\n".join(kept).strip()

def has_help_closing(text: str) -> bool:
    t = (text or "").lower()
    if not t:
        return False
    cues = [
        "anything else",
        "need help",
        "here to help",
        "happy to help",
        "glad to help",
        "more support",
    ]
    return any(c in t for c in cues)

def force_no_sources(data: Dict[str, Any]) -> None:
    data["sources"] = []

def normalize_form_answer(data: Dict[str, Any]) -> None:
    # If it's a form_request and downloads exist, keep text short and consistent
    if data.get("intent") == "form_request" and data.get("downloads"):
        data["answer"] = "I found the requested form(s). You can download them below."
        if not (data.get("answer_md") or "").strip():
            data["answer_md"] = "**I found the requested form(s).**\n\nYou can download them below."


# =========================
# Form intent + continuity (core fix)
# =========================
_VAGUE_FORM_PHRASES = {
    "form", "the form", "that form", "this form",
    "give me the form", "give me a form", "send me the form",
    "download the form", "give me the pdf", "send the pdf",
    "give me a form for that", "give me the form for that",
    "give me the form for this", "form for this", "form for that",
    "give me the form for it", "form for it",
}

def is_vague_form_request(q: str) -> bool:
    qn = _norm(q)
    if qn in _VAGUE_FORM_PHRASES:
        return True
    # short variants like: "need form", "form pls"
    if len(qn.split()) <= 4 and "form" in qn:
        return True
    return False

def extract_form_topic(q: str) -> str:
    qn = _norm(q)
    m = re.search(r"\bform\b\s+(for|of)\s+(.+)$", qn)
    if not m:
        return ""
    topic = m.group(2).strip()
    topic = re.sub(r"\b(this|that|it)\b", "", topic).strip()
    return _compact_topic(topic)

def last_user_topic(history: List[Dict[str, str]]) -> str:
    """
    Use last meaningful USER message.
    Avoid assistant messages because they contain many doc keywords -> wrong forms.
    """
    for m in reversed(history):
        if m.get("role") != "user":
            continue
        txt = normalize_query_aliases((m.get("content") or "").strip())
        if not txt:
            continue
        if is_affirmative_reply(txt):
            continue
        form_topic = extract_form_topic(txt)
        if form_topic:
            return form_topic
        if is_vague_form_request(txt):
            continue
        if _norm(txt) in {"types", "type", "options", "details", "more", "more info"}:
            continue
        # ignore social chatter
        if _norm(txt) in {"hi", "hello", "hey", "bye", "thanks", "thank you", "kuzuzangpo"}:
            continue
        return _compact_topic(txt) or txt
    return ""

def last_user_message(history: List[Dict[str, str]]) -> str:
    for m in reversed(history):
        if m.get("role") != "user":
            continue
        txt = normalize_query_aliases((m.get("content") or "").strip())
        if txt:
            return txt
    return ""

def last_assistant_message(history: List[Dict[str, str]]) -> str:
    for m in reversed(history):
        if m.get("role") != "assistant":
            continue
        txt = (m.get("content") or "").strip()
        if txt:
            return txt
    return ""

def build_recent_history_context(history: List[Dict[str, str]], max_items: int = 4) -> str:
    if not history:
        return "No recent chat context."
    items = []
    for m in history[-max_items:]:
        role = (m.get("role") or "").strip().lower()
        if role not in {"user", "assistant"}:
            continue
        content = (m.get("content") or "").strip()
        content = re.sub(r"\s+", " ", content)
        if len(content) > 220:
            content = content[:220].rstrip() + "..."
        items.append(f"{role}: {content}")
    return "\n".join(items) if items else "No recent chat context."

def is_affirmative_reply(q: str) -> bool:
    qn = _norm(q)
    if not qn:
        return False
    return qn in {"yes", "yeah", "yep", "sure", "ok", "okay", "please", "alright"}


_FOLLOWUP_FRAGMENT_WORDS = {
    "this", "that", "it", "those", "these", "same", "more", "details",
    "types", "type", "options", "option", "process", "procedure",
    "how", "what about", "and this", "for this", "for that",
}
_FOLLOWUP_DETAIL_TERMS = {
    "document", "documents", "requirement", "requirements", "required",
    "eligibility", "eligible", "benefit", "benefits", "coverage",
    "premium", "premiums", "rate", "rates", "interest", "tenure",
    "repayment", "repayments", "process", "procedure", "steps",
    "apply", "application", "claim", "claims", "contact", "contacts",
    "branch", "branches", "address", "location",
}
_FOLLOWUP_DETAIL_PHRASES = {
    "what about",
    "how about",
    "what are the documents",
    "what documents",
    "documents required",
    "required documents",
    "what are the requirements",
    "what is the process",
    "how to apply",
    "how do i apply",
    "what is the eligibility",
    "interest rate",
    "repayment period",
}
_BIL_CONTEXT_TERMS = {
    "bil", "insurance", "loan", "claim", "form", "forms", "policy", "premium",
    "motor", "fire", "travel", "engineering", "aviation", "marine", "liability",
    "ppf", "gf", "gfm", "provident", "gratuity", "download", "document",
}
_BIL_TOPIC_TERMS = {
    "bil", "insurance", "loan", "loans", "claim", "claims", "policy", "policies",
    "motor", "fire", "travel", "engineering", "aviation", "marine", "liability",
    "ppf", "gf", "gfm", "provident", "gratuity", "housing", "vehicle", "personal",
    "contractor", "tourism", "hotel", "agriculture", "livestock", "shares", "securities",
    "branch", "branches", "contact", "contacts",
}


def is_contextual_followup_fragment(q: str) -> bool:
    qn = _norm(q)
    if not qn:
        return False
    if qn in _FOLLOWUP_FRAGMENT_WORDS:
        return True
    if any(phrase in qn for phrase in _FOLLOWUP_DETAIL_PHRASES):
        return True
    words = qn.split()
    if len(words) <= 8 and any(w in {"this", "that", "it", "these", "those", "they", "them"} for w in words):
        return True
    if len(words) <= 3 and any(w in {"more", "details", "types", "options"} for w in words):
        return True
    if len(words) <= 8 and any(w in _FOLLOWUP_DETAIL_TERMS for w in words) and not any(w in _BIL_TOPIC_TERMS for w in words):
        return True
    return False


def has_explicit_bil_topic(q: str) -> bool:
    qn = _norm(q)
    if not qn:
        return False
    words = set(qn.split())
    return bool(words & _BIL_TOPIC_TERMS)


def should_bind_to_recent_topic(q: str, history: List[Dict[str, str]]) -> bool:
    if not has_recent_bil_context(history):
        return False
    if is_contextual_followup_fragment(q):
        return True

    qn = _norm(q)
    if not qn or has_explicit_bil_topic(q):
        return False

    words = qn.split()
    if len(words) > 8:
        return False

    if any(phrase in qn for phrase in _FOLLOWUP_DETAIL_PHRASES):
        return True
    if any(w in _FOLLOWUP_DETAIL_TERMS for w in words):
        return True
    return False


def has_recent_bil_context(history: List[Dict[str, str]], window: int = 6) -> bool:
    if not history:
        return False
    for m in history[-window:]:
        content = _norm(m.get("content") or "")
        if not content:
            continue
        if any(re.search(rf"\b{re.escape(t)}\b", content) for t in _BIL_CONTEXT_TERMS):
            return True
    return False

def build_retrieval_query(q: str, history: List[Dict[str, str]]) -> str:
    qn = _norm(q)
    expanded = [qn]
    topic = _norm(last_user_topic(history))

    # Expand short fund-related asks so vector search has enough signal.
    if qn in {"pf", "ppf", "gf", "gfm", "provident", "gratuity", "fund", "funds"}:
        expanded.append("private provident and gratuity fund ppf gf")
        expanded.append("ppf gfm department bil")

    if any(k in qn for k in ["ppf", "provident", "gratuity", "gf", "gfm", "fund"]):
        expanded.append("private provident fund and gratuity fund bil")
        expanded.append("ppf employee registration contribution nominee refund form")
        expanded.append("loan against private provident fund ppf")

    # Follow-ups like "how to fill this form", "for 2022", "details"
    if any(p in qn for p in ["this form", "that form", "fill this form", "fill the form", "details", "for 20"]):
        if topic:
            expanded.append(f"{qn} {topic}")
            expanded.append(topic)

    # Keep retrieval anchored to BIL for short/ambiguous user inputs.
    if len(qn.split()) <= 3 and "bil" not in qn and "bhutan insurance" not in qn:
        expanded.append(f"{qn} bil")

    # Use recent user context for very short follow-ups.
    if len(qn.split()) <= 2:
        if topic and topic != qn:
            expanded.append(f"{qn} {topic}")

    # Follow-up questions should inherit the prior topic.
    if should_bind_to_recent_topic(q, history):
        if topic and topic != qn:
            expanded.append(f"{topic} {qn}")
            expanded.append(f"{qn} {topic}")
            expanded.append(topic)

    # If recent context is BIL and query is short, bias retrieval toward BIL semantics
    # even without explicit keywords.
    if should_bind_to_recent_topic(q, history) and "bil" not in qn:
        expanded.append(f"{qn} bhutan insurance limited")

    seen = set()
    out = []
    for part in expanded:
        part = part.strip()
        if not part or part in seen:
            continue
        seen.add(part)
        out.append(part)

    return " ".join(out)

def should_promote_unrelated_to_bil(
    q: str,
    history: List[Dict[str, str]],
    hinted_intent: str,
) -> tuple[bool, List[Dict[str, Any]]]:
    """
    Dynamic fallback:
    - If intent hint says unrelated, probe retrieval confidence.
    - Promote to bil_query only when retrieved chunk confidence is strong enough.
    """
    if hinted_intent != "unrelated":
        return False, []
    if detect_social_intent(q):
        return False, []
    if looks_like_form_download_request(q) or is_vague_form_request(q):
        return False, []

    def _probe(query_text: str) -> List[Dict[str, Any]]:
        try:
            probe_json = bil_retrieve_context.run(query_text)
            probe_obj = json.loads(probe_json)
            return probe_obj.get("chunks", []) or []
        except Exception:
            return []

    def _best_score(chunks: List[Dict[str, Any]]) -> float:
        best_local = 0.0
        for c in chunks:
            try:
                best_local = max(best_local, float(c.get("score", 0.0)))
            except Exception:
                continue
        return best_local

    generic_noise = {
        "tell", "about", "what", "which", "where", "when", "why", "how",
        "for", "the", "and", "with", "this", "that", "these", "those",
        "more", "detail", "details", "please", "need", "want",
        "document", "documents", "requirement", "requirements", "required",
        "process", "procedure", "steps", "rate", "rates", "interest",
        "tenure", "premium", "coverage", "benefits", "eligibility",
    }
    raw_tokens = [
        t for t in re.findall(r"[a-z0-9]+", _norm(q))
        if len(t) > 2 and t not in generic_noise
    ]

    def _lexical_hits(chunks: List[Dict[str, Any]], tokens: List[str]) -> int:
        if not chunks or not tokens:
            return 0
        best_hits = 0
        for c in chunks:
            hay = (
                f"{c.get('title','')} "
                f"{c.get('source','')} "
                f"{str(c.get('content',''))[:1400]}"
            ).lower()
            hits = sum(1 for t in tokens if t in hay)
            best_hits = max(best_hits, hits)
        return best_hits

    # First probe with raw query only (avoids context contamination on truly unrelated asks).
    raw_chunks = _probe(q)
    raw_best = _best_score(raw_chunks)
    raw_hits = _lexical_hits(raw_chunks, raw_tokens)
    if has_explicit_bil_topic(q) and raw_tokens and raw_best >= max(settings.min_relevance, 0.24) and raw_hits >= 1:
        return True, raw_chunks

    # Only run context-expanded rescue for genuine follow-up questions.
    q_words = _norm(q).split()
    can_use_context_rescue = should_bind_to_recent_topic(q, history)
    if not can_use_context_rescue:
        return False, []

    probe_query = build_retrieval_query(q, history)
    ctx_chunks = _probe(probe_query)
    if not ctx_chunks:
        return False, []

    best = _best_score(ctx_chunks)
    min_strong = max(settings.min_relevance, 0.26)
    if has_recent_bil_context(history):
        min_strong -= 0.06
    if should_bind_to_recent_topic(q, history):
        min_strong -= 0.05
    if len(q_words) <= 3:
        min_strong -= 0.03
    min_strong = max(0.14, min_strong)

    if best >= min_strong:
        return True, ctx_chunks
    return False, []

def is_direct_question(q: str) -> bool:
    qn = _norm(q)
    if not qn:
        return False
    tokens = qn.split()
    question_words = {
        "what", "where", "when", "why", "how", "which", "who", "whom",
        "can", "could", "do", "does", "did", "is", "are", "am", "will", "would",
    }
    contact_cues = {"number", "phone", "contact", "email", "address", "location"}
    if any(t in question_words for t in tokens):
        return True
    if any(t in contact_cues for t in tokens):
        return True
    # short fragments like "for insurance?" are not treated as direct questions
    if qn.startswith("for ") and len(tokens) <= 4:
        return False
    if qn.endswith("?") and len(tokens) > 4:
        return True
    return False

def build_form_query_variants(user_query: str, history: List[Dict[str, str]]) -> List[str]:
    """
    General form query expansion:
    - If user says "form for X" -> try X + (application/proposal/claim)
    - If user is vague -> use last user topic
    - Includes canonical mapping (loan -> loan application form; motor -> motor proposal/claim)
    """
    qn = _norm(user_query)
    variants: List[str] = []

    # Always include raw normalized query
    variants.append(qn)

    # Explicit "form for/of X"
    m = re.search(r"\bform\b\s+(for|of)\s+(.+)$", qn)
    if m:
        topic = m.group(2).strip()
        variants += [
            f"{topic} form",
            f"{topic} application form",
            f"{topic} proposal form",
            f"{topic} claim form",
        ]

    # If vague -> use last user topic
    if is_vague_form_request(user_query):
        topic = _norm(last_user_topic(history))
        if topic:
            variants += [
                f"{topic} form",
                f"{topic} application form",
                f"{topic} proposal form",
                f"{topic} claim form",
            ]

            # Canonical mappings (general-purpose, helps “housing loan form”)
            if "loan" in topic:
                variants.append("loan application form")
            if any(w in topic for w in ["motor", "car", "vehicle"]):
                variants += ["motor proposal form", "motor claim form"]

    # If user asked for a product + "form" but not exact type
    if "loan" in qn and "application" not in qn:
        variants.append("loan application form")
    if any(w in qn for w in ["motor", "car", "vehicle"]) and "form" in qn:
        variants += ["motor proposal form", "motor claim form"]

    # Dedupe keep order
    seen = set()
    out: List[str] = []
    for v in variants:
        v = _norm(v)
        if not v or v in seen:
            continue
        seen.add(v)
        out.append(v)

    return out[:8]

def build_document_query_variants(user_query: str, history: List[Dict[str, str]]) -> List[str]:
    qn = _norm(user_query)
    variants: List[str] = [qn]

    years = re.findall(r"\b(20\d{2})\b", qn)
    topic = _norm(last_user_topic(history))

    if "annual report" in qn or "report" in qn:
        variants.append("annual report")
    if "handbook" in qn or "guide" in qn or "manual" in qn:
        variants.append("insurance handbook")

    if years:
        y = years[0]
        variants.append(f"annual report {y}")
        variants.append(f"report {y}")
        if topic:
            variants.append(f"{topic} {y}")

    # Short follow-up like "for 2022 to download"
    if (len(qn.split()) <= 5 or qn.startswith("for ")) and topic:
        variants.append(f"{topic} {qn}")
        variants.append(topic)

    seen = set()
    out: List[str] = []
    for v in variants:
        v = _norm(v)
        if not v or v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out[:8]


_ACTION_CUES = {
    "how",
    "process",
    "procedure",
    "step",
    "steps",
    "apply",
    "application",
    "claim",
    "claims",
    "required",
    "requirement",
    "requirements",
    "document",
    "documents",
    "form",
    "download",
    "register",
}

_PRODUCT_HINTS = {
    "motor": {"motor", "vehicle", "car", "auto"},
    "fire": {"fire"},
    "travel": {"travel"},
    "loan": {"loan", "housing", "transport", "personal", "contractor", "tourism", "hotel", "agriculture", "livestock", "shares", "securities"},
    "ppf": {"ppf", "provident", "gratuity", "gf", "gfm"},
}

_LINK_STOPWORDS = {
    "how", "do", "to", "the", "a", "an", "is", "are", "and", "for", "of", "in", "on",
    "with", "bil", "bhutan", "insurance", "limited", "what", "which", "tell", "about",
    "me", "please", "claim", "claims", "loan", "loans", "policy", "policies",
}

_DEFAULT_FORMS_LINK = {"title": "Download Forms", "url": "https://www.bil.bt/help-and-support/download-forms"}
_CLAIM_PAGE_LINKS = {
    "motor": {"title": "Motor Insurance Claim", "url": "https://www.bil.bt/claim/motor-insurance-claim"},
    "fire": {"title": "Fire Insurance Claim", "url": "https://www.bil.bt/claim/fire-insurance-claim"},
    "travel": {"title": "Travel Insurance Claim", "url": "https://www.bil.bt/claim/travel-insurance-claim"},
    "engineering": {"title": "Engineering Insurance Claim", "url": "https://www.bil.bt/claim/engineering-insurance-claim"},
    "aviation": {"title": "Aviation Insurance Claim", "url": "https://www.bil.bt/claim/aviation-insurance-claim"},
    "marine": {"title": "Marine Insurance Claim", "url": "https://www.bil.bt/claim/marine-insurance-claim"},
    "liability": {"title": "Liability Insurance Claim", "url": "https://www.bil.bt/claim/liability-insurance-claim"},
    "misc": {"title": "Miscellaneous Insurance Claim", "url": "https://www.bil.bt/claim/miscellaneous-insurance-claim"},
}
_PRODUCT_PAGE_LINKS = {
    "motor": {"title": "Motor Insurance", "url": "https://www.bil.bt/insurance/motor-insurance"},
    "fire": {"title": "Fire Insurance", "url": "https://www.bil.bt/insurance/fire-insurance"},
    "travel": {"title": "Travel Insurance", "url": "https://www.bil.bt/insurance/travel-insurance"},
    "engineering": {"title": "Engineering Insurance", "url": "https://www.bil.bt/insurance/engineering"},
    "aviation": {"title": "Aviation Insurance", "url": "https://www.bil.bt/insurance/aviation-insurance"},
    "marine": {"title": "Marine Insurance", "url": "https://www.bil.bt/insurance/marine-insurance"},
    "liability": {"title": "Liability Insurance", "url": "https://www.bil.bt/insurance/liability-insurance"},
    "misc": {"title": "Miscellaneous Insurance", "url": "https://www.bil.bt/insurance/miscellaneous-insurance"},
}
_LOAN_PAGE_LINKS = [
    ({"personal"}, {"title": "Personal Loan", "url": "https://www.bil.bt/loans/personal-loan"}),
    ({"housing"}, {"title": "Housing Loan", "url": "https://www.bil.bt/loans/housing-loan"}),
    ({"transport", "vehicle"}, {"title": "Transport Loan", "url": "https://www.bil.bt/loans/transport-loan"}),
    ({"agriculture", "livestock", "farm"}, {"title": "Agriculture and Livestock Loan", "url": "https://www.bil.bt/loans/agriculture-and-livestock"}),
    ({"hotel", "tourism"}, {"title": "Hotel and Tourism Loan", "url": "https://www.bil.bt/loans/hotel-tourism-loan"}),
    ({"contractor", "contractors"}, {"title": "Loans to Contractors", "url": "https://www.bil.bt/loans/loans-to-contractors"}),
    ({"shares", "securities"}, {"title": "Loan for Shares and Securities", "url": "https://www.bil.bt/loans/loan-for-shares-securities"}),
    ({"service"}, {"title": "Service Sector Loan", "url": "https://www.bil.bt/loans/service-sector-loan"}),
    ({"trade", "commerce"}, {"title": "Trade and Commerce Loan", "url": "https://www.bil.bt/loans/trade-commerce-loan"}),
    ({"production", "manufacturing", "industry", "industrial"}, {"title": "Production and Manufacturing Loan", "url": "https://www.bil.bt/loans/production-manufacturing-loan"}),
    ({"forestry", "logging"}, {"title": "Forestry and Logging Loan", "url": "https://www.bil.bt/loans/forestry-logging-loan"}),
    ({"mining", "quarrying"}, {"title": "Mining and Quarrying Loan", "url": "https://www.bil.bt/loans/mining-quarrying-loan"}),
]
_EXTRA_PRODUCT_WORDS = {
    "engineering": {"engineering", "contractor", "erection", "machinery"},
    "aviation": {"aviation", "aircraft"},
    "marine": {"marine", "cargo", "transit"},
    "liability": {"liability", "third-party", "third", "party", "workmen", "workman", "compensation"},
    "misc": {"miscellaneous", "student", "sports", "fidelity", "burglary"},
}


def _is_actionable_query(q: str) -> bool:
    qn = _norm(q)
    if not qn:
        return False
    toks = set(qn.split())
    return bool(_ACTION_CUES & toks)


def _query_products(text: str) -> set[str]:
    qn = _norm(text)
    found = set()
    for label, words in _PRODUCT_HINTS.items():
        if any(re.search(rf"\b{re.escape(w)}\b", qn) for w in words):
            found.add(label)
    return found


def _infer_products(q: str, history: List[Dict[str, str]]) -> set[str]:
    found = _query_products(q)
    if found:
        return found

    topic = last_user_topic(history)
    if topic:
        return _query_products(topic)
    return set()


def _derive_action_form_queries(q: str, history: List[Dict[str, str]]) -> List[str]:
    products = _infer_products(q, history)
    qn = _norm(q)
    is_claim = "claim" in qn
    explicit_form_ask = looks_like_form_request(q) or looks_like_form_download_request(q)

    candidates: List[str] = []
    if "motor" in products:
        if is_claim:
            candidates += ["motor claim form", "claim intimation form"]
        else:
            candidates += ["motor proposal form"]
    if "fire" in products:
        if is_claim:
            candidates += ["fire claim form", "claim intimation form"]
        else:
            candidates += ["fire insurance proposal form"]
    if "travel" in products:
        if is_claim:
            candidates += ["claim intimation form"]
        else:
            candidates += ["travel insurance proposal form"]
    if "loan" in products:
        if is_claim:
            candidates += ["loan protection claim form"]
        else:
            candidates += ["loan application form"]
    if "ppf" in products:
        candidates += ["ppf mou", "change of nominee form"]

    if is_claim and not candidates:
        candidates += ["claim intimation form"]

    if explicit_form_ask:
        candidates += build_form_query_variants(q, history)

    seen = set()
    out: List[str] = []
    for c in candidates:
        c = _norm(c)
        if not c or c in seen:
            continue
        seen.add(c)
        out.append(c)
    return out[:8]


def _dedupe_downloads(items: List[Dict[str, Any]], limit: int = 6) -> List[Dict[str, str]]:
    seen = set()
    out: List[Dict[str, str]] = []
    for d in items or []:
        if not isinstance(d, dict):
            continue
        url = (d.get("url") or "").strip()
        if not url:
            continue
        key = url.lower().rstrip("/")
        if key in seen:
            continue
        seen.add(key)
        title = (d.get("title") or "").strip() or "Download"
        out.append({"title": title, "url": url})
        if len(out) >= limit:
            break
    return out


def _gather_action_downloads(q: str, history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    candidates = _derive_action_form_queries(q, history)
    if not candidates:
        return []

    qn = _norm(q)
    is_claim = "claim" in qn
    explicit_form_ask = looks_like_form_request(q) or looks_like_form_download_request(q)

    merged: List[Dict[str, str]] = []
    for cq in candidates:
        try:
            obj = json.loads(bil_get_forms.run(cq))
        except Exception:
            continue
        for m in (obj.get("matches") or []):
            if isinstance(m, dict):
                merged.append({"title": m.get("title", ""), "url": m.get("url", "")})
        if len(merged) >= 8:
            break

    deduped = _dedupe_downloads(merged, limit=8)
    if explicit_form_ask:
        return deduped[:6]

    filtered: List[Dict[str, str]] = []
    for d in deduped:
        t = f"{d.get('title','')} {d.get('url','')}".lower()
        if is_claim and "proposal" in t:
            continue
        if not is_claim and ("claim" in t or "intimation" in t):
            continue
        filtered.append(d)
    return filtered[:6]


def _is_file_link(url: str) -> bool:
    u = (url or "").lower().split("?")[0]
    return ("/document/forms/" in u) or u.endswith((".pdf", ".doc", ".docx"))


def _extract_action_links(
    chunks: List[Dict[str, Any]],
    q: str,
    history: Optional[List[Dict[str, str]]] = None,
    limit: int = 3,
) -> List[Dict[str, str]]:
    q_tokens = [
        t for t in re.findall(r"[a-z0-9]+", _norm(q))
        if len(t) > 2 and t not in _LINK_STOPWORDS
    ]
    q_products = _infer_products(q, history or [])

    seen = set()
    scored: List[tuple[float, Dict[str, str]]] = []
    for c in chunks or []:
        url = str(c.get("source") or "").strip()
        if not url or not url.startswith("http") or _is_file_link(url):
            continue

        key = url.lower().rstrip("/")
        if key in seen:
            continue
        seen.add(key)

        title = re.sub(r"\s+", " ", str(c.get("title") or "").strip())
        hay = f"{title} {url}".lower()

        overlap = sum(1 for t in q_tokens if t in hay)
        if q_tokens and overlap == 0:
            continue

        score = float(c.get("score") or 0.0) + (0.12 * min(3, overlap))
        if q_products:
            c_products = _query_products(hay)
            if c_products & q_products:
                score += 0.20
            elif c_products:
                continue

        pretty = title or "BIL Resource"
        pretty = re.sub(r"\s*[\-|]\s*Bhutan Insurance Limited\s*$", "", pretty, flags=re.IGNORECASE).strip()
        scored.append((score, {"title": pretty or "BIL Resource", "url": url}))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:limit]]


def _dedupe_links(items: List[Dict[str, str]], limit: int = 3) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    seen = set()
    for item in items or []:
        if not isinstance(item, dict):
            continue
        url = (item.get("url") or "").strip()
        if not url:
            continue
        key = url.lower().rstrip("/")
        if key in seen:
            continue
        seen.add(key)
        title = (item.get("title") or "").strip() or "BIL Resource"
        out.append({"title": title, "url": url})
        if len(out) >= limit:
            break
    return out


def _extract_extra_products(text: str) -> set[str]:
    qn = _norm(text)
    found = set()
    for label, words in _EXTRA_PRODUCT_WORDS.items():
        if any(re.search(rf"\b{re.escape(w)}\b", qn) for w in words):
            found.add(label)
    return found


def _fallback_action_links(q: str, history: List[Dict[str, str]], limit: int = 3) -> List[Dict[str, str]]:
    qn = _norm(q)
    history_topic = _norm(last_user_topic(history))
    merged_text = f"{qn} {history_topic}".strip()
    products = _infer_products(merged_text, history) | _extract_extra_products(merged_text)

    is_claim_flow = any(k in qn for k in ["claim", "intimation", "settlement"])
    links: List[Dict[str, str]] = []

    if is_claim_flow:
        for p in ["motor", "fire", "travel", "engineering", "aviation", "marine", "liability", "misc"]:
            if p in products and p in _CLAIM_PAGE_LINKS:
                links.append(_CLAIM_PAGE_LINKS[p])
        # Claims generally need forms too.
        links.append(_DEFAULT_FORMS_LINK)
        return _dedupe_links(links, limit=limit)

    # Product info / application flow pages.
    for p in ["motor", "fire", "travel", "engineering", "aviation", "marine", "liability", "misc"]:
        if p in products and p in _PRODUCT_PAGE_LINKS:
            links.append(_PRODUCT_PAGE_LINKS[p])

    if "loan" in products or "loan" in merged_text.split():
        matched_loan_link = None
        for keys, link in _LOAN_PAGE_LINKS:
            if any(k in merged_text.split() for k in keys):
                matched_loan_link = link
                break
        if matched_loan_link:
            links.append(matched_loan_link)
        else:
            # Generic loan fallback: safest broad page among indexed loan pages.
            links.append({"title": "Personal Loan", "url": "https://www.bil.bt/loans/personal-loan"})

    if any(k in merged_text for k in ["ppf", "provident", "gratuity", "gf", "gfm"]):
        links.append(_DEFAULT_FORMS_LINK)

    # For process/apply requests, forms page is often useful.
    if any(k in qn for k in ["apply", "application", "proposal", "form", "document", "download"]):
        links.append(_DEFAULT_FORMS_LINK)

    return _dedupe_links(links, limit=limit)


def _md_has_links(md: str) -> bool:
    return bool(re.search(r"\[[^\]]+\]\(https?://", md or "", flags=re.IGNORECASE))


def _format_action_context(links: List[Dict[str, str]], downloads: List[Dict[str, str]]) -> str:
    if not links and not downloads:
        return "No additional action resources."
    lines: List[str] = []
    if links:
        lines.append("LINKS:")
        for l in links[:3]:
            lines.append(f"- {l.get('title','Resource')}: {l.get('url','')}")
    if downloads:
        lines.append("DOWNLOADS:")
        for d in downloads[:4]:
            lines.append(f"- {d.get('title','Download')}: {d.get('url','')}")
    return "\n".join(lines)


def _append_links_to_answer_md(md: str, links: List[Dict[str, str]], has_downloads: bool) -> str:
    body = (md or "").strip()
    if links and not _md_has_links(body):
        lines = ["**Helpful links**"]
        for l in links[:3]:
            t = (l.get("title") or "BIL Resource").strip()
            u = (l.get("url") or "").strip()
            if u:
                lines.append(f"- [{t}]({u})")
        body = (body + "\n\n" + "\n".join(lines)).strip() if body else "\n".join(lines)

    if has_downloads and "attached below" not in body.lower():
        body = (body + "\n\nDownloadable files are attached below.").strip() if body else "Downloadable files are attached below."

    return body


def enforce_downloads_rules(data: Dict[str, Any], user_query: str) -> None:
    """
    Keep downloads controlled:
    - explicit form/document download asks are allowed
    - actionable process/claim/application queries may include relevant attachments
    - simple informational asks should not include random files
    """
    form_download_now = looks_like_form_download_request(user_query) or is_vague_form_request(user_query)
    doc_download_now = looks_like_document_download_request(user_query)
    actionable_now = _is_actionable_query(user_query)
    if not form_download_now and not doc_download_now and not actionable_now:
        data["downloads"] = []
        # also ensure we don't label it form_request accidentally
        if data.get("intent") == "form_request":
            data["intent"] = "bil_query"
        return

    dls = data.get("downloads") or []
    if not isinstance(dls, list):
        data["downloads"] = []
        return

    cleaned = []
    for d in dls:
        if not isinstance(d, dict):
            continue
        url = (d.get("url") or "").strip()
        title = (d.get("title") or "").strip()
        if url:
            cleaned.append({"title": title or "Download form", "url": url})

    data["downloads"] = cleaned


def finalize(data: Dict[str, Any], user_query: str = "") -> Dict[str, Any]:
    force_no_sources(data)

    data.setdefault("downloads", [])
    data.setdefault("confidence", "low")
    data.setdefault("intent", "not_found")
    data.setdefault("answer", "")
    data.setdefault("answer_md", "")
    data.setdefault("client_delay_ms", None)

    data["answer"] = strip_urls(str(data.get("answer", "")))

    # Prevent hallucinated / irrelevant downloads
    enforce_downloads_rules(data, user_query)

    normalize_form_answer(data)

    md = str(data.get("answer_md", "") or "").strip()
    if not md:
        data["answer_md"] = repair_markdown_from_text(str(data.get("answer", "")))
    elif not looks_like_markdown(md):
        data["answer_md"] = repair_markdown_from_text(md)
    else:
        data["answer_md"] = md

    # Remove any questions per UX requirement
    data["answer"] = remove_question_sentences(str(data.get("answer", "")))
    data["answer_md"] = remove_question_lines_md(str(data.get("answer_md", "")))

    # Add a gentle closing help line (no question) only when missing
    if data.get("intent") != "unrelated":
        if not has_help_closing(data.get("answer", "")) and not has_help_closing(data.get("answer_md", "")):
            if _HELP_CLOSINGS and random.random() < 0.6:
                closing = random.choice(_HELP_CLOSINGS)
                if closing not in data["answer"]:
                    data["answer"] = (data["answer"] + " " + closing).strip() if data["answer"] else closing
                if closing not in data["answer_md"]:
                    data["answer_md"] = (data["answer_md"] + f"\n\n{closing}").strip() if data["answer_md"] else closing

    return data


# =========================
# Prompt (KEEP EXACTLY THE SAME)
# =========================
SYSTEM_TEMPLATE = """
You are the official website chatbot for Bhutan Insurance Limited (BIL) (bil.bt).

KNOWN CONTACT INFO (use when asked):
- Toll-free number: 2011
- Phone: +975-2-339892/93/94 (use +975, not 00975)

OUTPUT (MUST FOLLOW):
- Return ONLY valid JSON that matches the schema exactly.
- Do NOT include URLs or links in answer.
- You SHOULD include official BIL links in answer_md whenever they help complete the user task.
- Do NOT include sources in the user-facing text.
- If the user requests forms/documents, do NOT list form names in text. Provide them ONLY via downloads[].

FIELDS:
- answer: plain text (no markdown symbols).
- answer_md: the same message formatted in Markdown for a chat UI.
- sources: always return an empty list [].
- downloads: array of objects like {{ "title": "...", "url": "..." }}.

BEHAVIOR:
- Use tools to fetch BIL context/forms.
- For BIL questions: answer ONLY using retrieved context.
- If context is missing, set intent="not_found" and ask ONE clarifying question.
- For unrelated questions: set intent="unrelated" and gently redirect.
- For process/how-to/application/claim questions, provide one complete response in a single message:
  1) clear flow/steps, 2) short explanation for why each major step matters, 3) key requirements/documents (if known), 4) helpful links from ACTIONABLE RESOURCES, 5) mention downloads if attached.

DYNAMIC RESPONSE STYLE (IMPORTANT):
- Match the response style to the user’s intent and how they asked.
- First line must directly address the user’s main ask.
- Do not repeat the same response pattern every turn; vary structure naturally.
- Use RECENT CHAT CONTEXT to interpret follow-up fragments and to avoid monotonous formatting.
- Do not rely on exact keywords only; infer semantically similar asks and likely intent from context.
- If the user asks for process/steps, explain slightly more than a bare checklist (brief rationale + practical notes).
- Then choose ONE layout below for answer_md (pick the most suitable; do not use all):

LAYOUT TOOLBOX (choose 1):
1) Quick Answer + Why (best for advice/yes-no/personal questions)
   - Start with a direct answer (Yes/No/It depends) then 2–4 bullets.

2) Summary + Key Points (best for “tell me about…”)
   - Short intro then bullets under a heading.

3) Steps Checklist (best for “how to”, “process”, “what to do”)
   - Use numbered steps (3–6 max), with a short explanation line for key steps.

4) Guided Flow (best for process explanations that need context)
   - Use sections like "Before you start", "Do this", "After submission", each with concise bullets.

5) Compare (best for “difference between”, “which is better”)
   - Use a small markdown table OR two subheadings.

6) Requirements (best for “documents needed”, “eligibility”, “what do I need”)
   - Use a short list grouped by category.

7) Options List (best for lists like products/loans)
   - Use subheadings per option with 2–3 bullets each (compact).
   
8) Form Request Confirmation (best for form requests)
   - Confirm form found and available for download (no links).
   
9) IF other scenarios: use your best judgment to pick a clear, concise format.

STYLE:
- answer_md should use headings (if absolutely necessary), bullets, and line breaks for readability.
- Keep it concise but complete: prefer 1–3 short sections, typically 5–10 bullets total.
- Do NOT end with a question. If you add a closing, make it a brief helpful statement (no question) and vary the wording.
- Avoid “brochure tone” for personal questions; sound like helpful support staff.
- NO need to provide headers for all the answers; give headers for only the most relevant answers.

SELF-CHECK:
Before finalizing:
1) Did I answer the user’s exact ask in the first line?
2) Did I choose the best ONE layout from the toolbox?
3) Is answer_md clean and scannable?
4) Did I avoid sources and avoid ending with a question?

Return ONLY JSON matching this schema:
{format_instructions}
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_TEMPLATE),
        ("human", "USER QUERY:\n{query}\n\nRECENT CHAT CONTEXT:\n{history_context}\n\nSTYLE GUIDANCE:\n{style_hint}\n\nACTIONABLE RESOURCES:\n{action_context}\n\nCONTEXT:\n{context}\n"),
    ]
).partial(format_instructions=parser.get_format_instructions())


def _llm() -> ChatOpenAI:
    kwargs = {
        "model": settings.chat_model,
        "api_key": settings.openai_api_key,
        "model_kwargs": {"response_format": {"type": "json_object"}},
    }
    # Some models (e.g., gpt-5-nano) only allow default temperature
    if not str(settings.chat_model).startswith("gpt-5"):
        kwargs["temperature"] = 0.35
    return ChatOpenAI(**kwargs)


# =========================
# Main runner
# =========================
def run_agent(query: str, history: List[Dict[str, str]]) -> Dict[str, Any]:
    q = normalize_query_aliases((query or "").strip())
    if not q:
        return finalize({
            "intent": "not_found",
            "answer": "Please type your question.",
            "answer_md": "Please type your question.",
            "downloads": [],
            "sources": [],
            "confidence": "low",
        }, user_query=q)

    # 0) Social intent
    social = detect_social_intent(q)
    if social == "greeting":
        return finalize({
            "intent": "unrelated",
            "answer": "Hello! I can help with BIL insurance, claims, loans, and forms.",
            "answer_md": "**Hello!**\n\nI can help with BIL insurance, claims, loans, and forms.",
            "downloads": [],
            "sources": [],
            "confidence": "high",
            "client_delay_ms": 1500,
        }, user_query=q)

    if social == "thanks":
        return finalize({
            "intent": "unrelated",
            "answer": "You’re welcome! Let me know if you need anything else.",
            "answer_md": "You’re welcome!\n\nIf you need help with insurance, loans, or forms, just ask.",
            "downloads": [],
            "sources": [],
            "confidence": "high",
            "client_delay_ms": 1500,
        }, user_query=q)

    if social == "farewell":
        return finalize({
            "intent": "unrelated",
            "answer": "Goodbye! Have a great day.",
            "answer_md": "Goodbye!\n\nHave a great day.",
            "downloads": [],
            "sources": [],
            "confidence": "high",
            "client_delay_ms": 1500,
        }, user_query=q)

    # 1) Intent hint
    try:
        hint = bil_intent_hint.run(q)
    except Exception:
        hint = "bil_query"

    # Smart continuation: short detail follow-ups in an ongoing BIL thread should
    # stay in BIL mode even when explicit product keywords are missing.
    if hint == "unrelated" and should_bind_to_recent_topic(q, history):
        hint = "bil_query"

    # Dynamic fallback: if hint says unrelated but retrieval confidence is strong,
    # treat it as a BIL query (helps short follow-up fragments).
    prefetched_chunks: List[Dict[str, Any]] = []
    promote_to_bil, probe_chunks = should_promote_unrelated_to_bil(q, history, hint)
    if promote_to_bil:
        hint = "bil_query"
        prefetched_chunks = probe_chunks

    # 2) Forms path (ONLY when this message explicitly asks to get a form)
    formish_now = looks_like_form_download_request(q) or is_vague_form_request(q) or hint == "form_request"
    if formish_now:
        candidate_queries = _derive_action_form_queries(q, history)

        merged: List[Dict[str, Any]] = []
        seen_urls = set()

        for cq in candidate_queries:
            try:
                tool_out = bil_get_forms.run(cq)
                obj = json.loads(tool_out)
                matches = obj.get("matches", []) or []
                for m in matches:
                    url = (m.get("url") or "").strip()
                    title = (m.get("title") or "").strip()
                    if not url or url in seen_urls:
                        continue
                    seen_urls.add(url)
                    merged.append({"title": title, "url": url})
            except Exception:
                continue

        if merged:
            is_claim = "claim" in _norm(q)
            if not is_claim:
                filtered = []
                for item in merged:
                    hay = f"{item.get('title', '')} {item.get('url', '')}".lower()
                    if "claim" in hay or "intimation" in hay:
                        continue
                    filtered.append(item)
                if filtered:
                    merged = filtered

            return finalize({
                "intent": "form_request",
                "answer": "I found the requested form(s). You can download them below.",
                "answer_md": "**I found the requested form(s).**\n\nYou can download them below.",
                "downloads": merged[:4],
                "sources": [],
                "confidence": "high",
            }, user_query=q)

        # Graceful fallback when no matching form exists
        if is_vague_form_request(q):
            topic = last_user_topic(history)
            if topic:
                return finalize({
                    "intent": "not_found",
                    "answer": "I don’t have a specific form for that request. If you need a form, tell me what it’s for (example: motor claim, travel proposal, loan application).",
                    "answer_md": "I don’t have a specific form for that request.\n\nIf you need a form, tell me what it’s for (example: motor claim, travel proposal, loan application).",
                    "downloads": [],
                    "sources": [],
                    "confidence": "low",
                }, user_query=q)

        # Nicer clarification (no “I couldn’t find…”)
        return finalize({
            "intent": "not_found",
            "answer": "If you need a form, tell me what it’s for (example: motor claim, travel proposal, loan application).",
            "answer_md": "If you need a form, tell me what it’s for (example: motor claim, travel proposal, loan application).",
            "downloads": [],
            "sources": [],
            "confidence": "low",
        }, user_query=q)

    # 2b) Document download path (annual reports, handbooks, guides, etc.)
    doc_download_now = looks_like_document_download_request(q)
    if doc_download_now:
        candidate_queries = build_document_query_variants(q, history)
        merged: List[Dict[str, Any]] = []
        seen_urls = set()

        for cq in candidate_queries:
            try:
                tool_out = bil_get_documents.run(cq)
                obj = json.loads(tool_out)
                matches = obj.get("matches", []) or []
                for m in matches:
                    url = (m.get("url") or "").strip()
                    title = (m.get("title") or "").strip()
                    if not url or url in seen_urls:
                        continue
                    seen_urls.add(url)
                    merged.append({"title": title, "url": url})
            except Exception:
                continue

        if merged:
            return finalize({
                "intent": "bil_query",
                "answer": "I found the requested document(s). You can download them below.",
                "answer_md": "**I found the requested document(s).**\n\nYou can download them below.",
                "downloads": merged[:6],
                "sources": [],
                "confidence": "high",
            }, user_query=q)

    # 3) Unrelated
    if hint == "unrelated":
        msg = bil_unrelated_reply.run(q) if hasattr(bil_unrelated_reply, "run") else bil_unrelated_reply(q)
        return finalize({
            "intent": "unrelated",
            "answer": msg,
            "answer_md": repair_markdown_from_text(msg),
            "downloads": [],
            "sources": [],
            "confidence": "high",
            "client_delay_ms": 1500,
        }, user_query=q)

    # 4) BIL Query: retrieve context + ask LLM
    if prefetched_chunks:
        chunks = prefetched_chunks
    else:
        try:
            retrieval_query = build_retrieval_query(q, history)
            ctx_json = bil_retrieve_context.run(retrieval_query)
            ctx_obj = json.loads(ctx_json)
            chunks = ctx_obj.get("chunks", []) or []
        except Exception:
            chunks = []

    if not chunks:
        # Retry with prior topic for vague follow-ups.
        retry_topic = last_user_topic(history) or last_user_message(history)
        if retry_topic and _norm(retry_topic) != _norm(q):
            retry_query = retry_topic
            if should_bind_to_recent_topic(q, history):
                retry_query = f"{retry_topic} {q}".strip()
            try:
                ctx_json = bil_retrieve_context.run(retry_query)
                ctx_obj = json.loads(ctx_json)
                chunks = ctx_obj.get("chunks", []) or []
            except Exception:
                chunks = []

    if not chunks:
        return finalize({
            "intent": "not_found",
            "answer": "I don’t have that information for a specific product. If you want details, tell me which insurance or loan product.",
            "answer_md": "I don’t have that information for a specific product.\n\nIf you want details, tell me which insurance or loan product.",
            "downloads": [],
            "sources": [],
            "confidence": "low",
        }, user_query=q)

    context = "\n\n".join([f"- {c.get('content','')}" for c in chunks[: settings.top_k]])
    history_context = build_recent_history_context(history)
    style_hint = build_style_hint(q, history)

    action_links: List[Dict[str, str]] = []
    action_downloads: List[Dict[str, str]] = []
    if _is_actionable_query(q):
        action_links = _extract_action_links(chunks, q, history=history, limit=3)
        if len(action_links) < 2:
            fallback_links = _fallback_action_links(q, history, limit=3)
            action_links = _dedupe_links(action_links + fallback_links, limit=3)
        action_downloads = _gather_action_downloads(q, history)
    action_context = _format_action_context(action_links, action_downloads)

    llm = _llm()
    msgs = prompt.format_messages(
        query=q,
        context=context,
        history_context=history_context,
        style_hint=style_hint,
        action_context=action_context,
    )
    out = llm.invoke(msgs).content or ""

    # Parse
    try:
        parsed = parser.parse(out)
        parsed_obj = parsed.model_dump()
        parsed_downloads = _dedupe_downloads(parsed_obj.get("downloads") or [], limit=6)
        merged_downloads = _dedupe_downloads(parsed_downloads + action_downloads, limit=6)
        parsed_obj["downloads"] = merged_downloads
        parsed_obj["answer_md"] = _append_links_to_answer_md(
            str(parsed_obj.get("answer_md", "")),
            action_links,
            bool(merged_downloads),
        )
        return finalize(parsed_obj, user_query=q)
    except Exception:
        try:
            candidate = extract_first_json_object(out)
            if candidate:
                obj = json.loads(candidate)
                if isinstance(obj, dict):
                    parsed_downloads = _dedupe_downloads(obj.get("downloads") or [], limit=6)
                    merged_downloads = _dedupe_downloads(parsed_downloads + action_downloads, limit=6)
                    obj["downloads"] = merged_downloads
                    obj["answer_md"] = _append_links_to_answer_md(
                        str(obj.get("answer_md", "")),
                        action_links,
                        bool(merged_downloads),
                    )
                    return finalize(obj, user_query=q)
        except Exception:
            pass

        return finalize({
            "intent": "not_found",
            "answer": "I couldn’t process that. Please try again.",
            "answer_md": "I couldn’t process that. Please try again.",
            "downloads": [],
            "sources": [],
            "confidence": "low",
        }, user_query=q)
