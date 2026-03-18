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
    bil_extract_financial_fact,
    bil_get_form_context,
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
    "I can also help with other BIL questions if needed.",
    "I can help further with BIL products, claims, loans, or forms.",
    "I can continue with the next BIL detail whenever needed.",
]
_CLOSING_STYLE_HINTS = [
    "Keep the closing warm but restrained, like support staff wrapping up a useful reply.",
    "Make the closing specific to the topic when possible, instead of using a generic help phrase.",
    "Use one short sentence with no question mark and no markdown.",
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
_GREETING_STYLE_HINTS = [
    "Greet naturally and keep it concise, like a helpful assistant on the BIL website.",
    "Offer help with BIL topics without sounding scripted or overly formal.",
    "If there is recent BIL context, you may lightly mention continuing that topic.",
]
_UNRELATED_STYLE_HINTS = [
    "Sound natural and direct, like support staff acknowledging the user's topic before redirecting.",
    "Keep the wording graceful and specific; avoid canned apology phrases unless they fit naturally.",
    "Vary the phrasing so the reply does not sound repeated across unrelated questions.",
]
_OVERVIEW_STYLE_HINTS = [
    "For broad or overview requests, start with a short plain-language overview, then list the main types or key points with one-line descriptions.",
    "Do not jump into procedures, rates, forms, or document lists unless the user asked for them explicitly.",
    "If the topic is a claim type or product, explain what it is and when it applies before mentioning next steps.",
]
_OPTIONS_STYLE_HINTS = [
    "For broad lists of products or options, keep the intro to 1-2 short sentences and move the detail into bullets.",
    "Group broader option lists into a few clear categories or representative examples instead of one long dense paragraph.",
]
_LOAN_RATE_STYLE_HINTS = [
    "For broad loan rate questions, summarize multiple loan products from the retrieved context. Do not answer with only one example if several loan rates are available.",
    "If the retrieved context appears partial, say the list includes examples and mention that other loan products are also available instead of implying the list is exhaustive.",
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


def _is_general_overview_request(query: str) -> bool:
    qn = _norm(query)
    if not qn or _is_actionable_query(query):
        return False
    overview_prefixes = (
        "tell me about",
        "tell me more about",
        "more on",
        "what about",
        "explain",
        "describe",
        "information on",
        "info on",
    )
    if any(qn.startswith(prefix) for prefix in overview_prefixes):
        return True
    if len(qn.split()) <= 4 and has_explicit_bil_topic(qn):
        return True
    return False


def build_style_hint(query: str, history: List[Dict[str, str]]) -> str:
    qn = _norm(query)
    last_assistant = last_assistant_message(history)
    last_tags = _detect_style_tags(last_assistant)

    base_pool: List[str] = list(_STYLE_HINTS)
    filtered_base: List[str] = []
    for hint in base_pool:
        h = hint.lower()
        if "numbered" in h and "numbered" in last_tags:
            continue
        if "key points" in h and "key_points" in last_tags:
            continue
        if "subheading" in h and "subheading" in last_tags:
            continue
        if "overview" in h and "overview" in last_tags:
            continue
        filtered_base.append(hint)
    base_choice = random.choice(filtered_base or base_pool)

    extras: List[str] = []
    if _is_general_overview_request(query):
        extras.extend(_OVERVIEW_STYLE_HINTS)
        extras.append(random.choice(_OPTIONS_STYLE_HINTS))
    elif _is_actionable_query(query):
        extras.append(random.choice(_PROCESS_STYLE_HINTS))

    if _is_broad_loan_rate_query(query):
        extras.append(random.choice(_LOAN_RATE_STYLE_HINTS))

    if len(qn.split()) <= 4 or is_affirmative_reply(query):
        extras.append(random.choice(_FOLLOWUP_STYLE_HINTS))

    if extras:
        return " ".join(extras + [base_choice])
    return base_choice

# =========================
# Social intents
# =========================
_GREETINGS = {
    "hello", "hi", "hey", "hola", "greetings", "namaste",
    "good morning", "good afternoon", "good evening",
    "kuzuzangpo", "kuzuzangpo la", "kuzuzangpola", "kuzu",
}
_GREETING_SINGLE_WORDS = {"hello", "hi", "hey", "hola", "greetings", "namaste", "kuzuzangpo", "kuzuzangpola", "kuzu"}
_GREETING_PATTERNS = [
    re.compile(r"^(hello|hi|hey|hola|greetings|namaste|good morning|good afternoon|good evening|kuzuzangpo|kuzuzangpo la|kuzuzangpola|kuzu)(\s+norbu)?$", re.IGNORECASE),
]
_FAREWELLS = {"bye", "goodbye", "see you", "see ya", "take care", "later", "okay", "ok"}
_THANKS = {"thanks", "thank you", "thx", "thanks a lot", "appreciate it"}
_IDENTITY_PATTERNS = [
    re.compile(r"^(who are you|what are you|tell me about yourself|introduce yourself)$", re.IGNORECASE),
    re.compile(r"^(what is your name|what s your name|your name|who is norbu|are you norbu)$", re.IGNORECASE),
    re.compile(r"^(who am i chatting with|who am i talking to)$", re.IGNORECASE),
    re.compile(r"^(what can you do|how can you help|what do you do)$", re.IGNORECASE),
]
_TOPIC_PREFIX_RE = re.compile(
    r"^(tell me about|tell me more about|tell me|explain|describe|what is|what are|"
    r"more on|what about|can you tell me about|can you explain|"
    r"give me|can you give me|show me|send me|share|provide|download|download the|open|open the|"
    r"give me info on|give me information on|information on|info on|details on|"
    r"i want to know about|i need to know about|i need info on|i want info on|"
    r"how to apply for|how do i apply for|how to claim for|how do i claim for|"
    r"how to apply|how do i apply|how to claim|how do i claim|how to file|how do i file|"
    r"how to submit|how do i submit|documents required for|required documents for|requirements for)\s+",
    re.IGNORECASE,
)
_TOPIC_PRONOUNS = {"it", "this", "that", "them", "these", "those", "one"}
_GENERIC_TOPIC_FOLLOWUPS = {
    "types", "type", "options", "details", "more", "more info", "some more",
    "more details", "tell me more", "anything else", "next",
}


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
    qn = re.sub(r"^the\s+", "", qn)
    qn = re.sub(r"\b(pdf|file|files|document|documents|link|links)\b", "", qn)
    qn = re.sub(r"\s+", " ", qn).strip()
    if not qn or qn in _TOPIC_PRONOUNS:
        return ""
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
    tokens = qn.split()
    compact_greeting = " ".join(tok for tok in tokens if tok not in {"norbu", "la"}).strip()

    if (
        qn in _GREETINGS
        or compact_greeting in _GREETINGS
        or compact_greeting in _GREETING_SINGLE_WORDS
        or any(p.search(qn) for p in _GREETING_PATTERNS)
    ):
        return "greeting"
    if qn in _FAREWELLS:
        return "farewell"
    if qn in _THANKS:
        return "thanks"
    if any(p.search(qn) for p in _IDENTITY_PATTERNS):
        return "identity"
    return None


def build_identity_reply(q: str) -> tuple[str, str]:
    qn = _norm(q)
    if any(term in qn for term in ["what can you do", "how can you help", "what do you do"]):
        answer = (
            "I’m Norbu, the AI assistant for Bhutan Insurance Limited. "
            "I can help with BIL insurance products, claims, loans, forms, branches, contact details, and annual reports."
        )
    else:
        answer = (
            "I’m Norbu, the AI assistant for Bhutan Insurance Limited. "
            "I’m here to help with BIL insurance, claims, loans, forms, contact details, and annual reports."
        )
    return answer, repair_markdown_from_text(answer)


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

def markdown_to_plain_text(md: str) -> str:
    t = _URL_RE.sub("", md or "")
    if not t:
        return ""
    t = re.sub(r"(?m)^\s*#{1,6}\s*", "", t)
    t = re.sub(r"\*\*(.*?)\*\*", r"\1", t)
    t = re.sub(r"__(.*?)__", r"\1", t)
    t = re.sub(r"`([^`]*)`", r"\1", t)
    lines: List[str] = []
    for raw in t.splitlines():
        line = raw.strip()
        if not line:
            continue
        line = re.sub(r"^\s*[-*]\s+", "", line)
        line = re.sub(r"^\s*\d+\.\s+", "", line)
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines).strip()

def normalize_form_markdown(md: str) -> str:
    t = _URL_RE.sub("", md or "")
    if not t:
        return ""
    t = t.replace("\r\n", "\n")
    t = re.sub(r"([.!])\s+(\*\*)", r"\1\n\n\2", t)
    t = re.sub(r"\s+(\*\*)", r"\n\n\1", t)
    t = re.sub(r"(\*\*[^*]+\*\*)\s*-\s*", r"\1\n- ", t)
    t = re.sub(r"\s+-\s+", "\n- ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
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
    if data.get("intent") != "form_request" or not data.get("downloads"):
        return

    answer = str(data.get("answer", "") or "").strip()
    if not answer:
        data["answer"] = "I found the requested form(s). You can download them below."

    if not (data.get("answer_md") or "").strip():
        data["answer_md"] = repair_markdown_from_text(str(data.get("answer", "") or ""))


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


def _has_topic_signal(text: str) -> bool:
    qn = _norm(text)
    if not qn:
        return False
    return bool(_query_products(qn) or has_explicit_bil_topic(qn) or re.search(r"\b20\d{2}\b", qn))


def _should_skip_user_topic(text: str) -> bool:
    qn = normalize_query_aliases((text or "").strip())
    if not qn:
        return True

    qn_norm = _norm(qn)
    compact = _compact_topic(qn)
    if not compact:
        return True

    if qn_norm in {"hi", "hello", "hey", "bye", "thanks", "thank you", "kuzuzangpo"}:
        return True
    if is_affirmative_reply(qn) or is_vague_form_request(qn):
        return True
    if extract_followup_year(qn_norm) or _is_generic_context_query(qn_norm):
        return True
    if compact in _GENERIC_TOPIC_FOLLOWUPS:
        return True

    form_topic = extract_form_topic(qn)
    if form_topic or _has_topic_signal(qn_norm) or _has_topic_signal(compact):
        return False

    tokens = set(compact.split())
    if tokens and tokens <= _TOPIC_PRONOUNS:
        return True
    if len(tokens) <= 4 and (tokens & _FOLLOWUP_DETAIL_TERMS):
        return True
    if len(tokens) <= 6 and (tokens & _TOPIC_PRONOUNS) and (tokens & _FOLLOWUP_DETAIL_TERMS):
        return True
    if re.search(r"\bwhat\s+does\s+(it|this|that|them|these|those)\b", qn_norm):
        return True
    if any(phrase in qn_norm for phrase in _FOLLOWUP_DETAIL_PHRASES):
        return True
    if _is_actionable_query(qn):
        return True
    return False


def last_user_topic(history: List[Dict[str, str]]) -> str:
    """
    Use the most recent USER turn that still carries topic signal.
    Skip vague follow-ups like "how to file it", "give me the form", or "for 2020?"
    so later continuity can stay anchored to the real product/report topic.
    """
    for m in reversed(history):
        if m.get("role") != "user":
            continue
        txt = normalize_query_aliases((m.get("content") or "").strip())
        if not txt:
            continue
        form_topic = extract_form_topic(txt)
        if form_topic:
            return form_topic
        if _should_skip_user_topic(txt):
            continue
        compact = _compact_topic(txt)
        if compact:
            return compact
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


def last_assistant_topic(history: List[Dict[str, str]]) -> str:
    for m in reversed(history):
        if m.get("role") != "assistant":
            continue
        txt = normalize_query_aliases((m.get("content") or "").strip())
        if not txt:
            continue
        topic = _extract_topic_from_history_text(txt)
        if topic:
            return topic
    return ""


def last_active_topic(history: List[Dict[str, str]]) -> str:
    user_topic = last_user_topic(history)
    if user_topic:
        return user_topic

    assistant_topic = last_assistant_topic(history)
    if assistant_topic:
        return assistant_topic

    return last_user_message(history) or last_assistant_message(history)

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
    "cover", "covers", "covered", "exclusion", "exclusions",
    "premium", "premiums", "rate", "rates", "interest", "tenure",
    "repayment", "repayments", "process", "procedure", "steps",
    "apply", "application", "claim", "claims", "contact", "contacts",
    "branch", "branches", "address", "location", "income", "earnings",
    "report", "reports", "financial",
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
    "same for",
}
_BIL_CONTEXT_TERMS = {
    "bil", "insurance", "loan", "claim", "form", "forms", "policy", "premium",
    "motor", "fire", "travel", "engineering", "aviation", "marine", "liability",
    "ppf", "gf", "gfm", "provident", "gratuity", "download", "document",
    "annual", "report", "financial", "income", "earnings",
}
_BIL_TOPIC_TERMS = {
    "bil", "insurance", "loan", "loans", "claim", "claims", "policy", "policies",
    "motor", "fire", "travel", "engineering", "aviation", "marine", "liability",
    "ppf", "gf", "gfm", "provident", "gratuity", "housing", "vehicle", "personal",
    "contractor", "tourism", "hotel", "agriculture", "livestock", "shares", "securities",
    "branch", "branches", "contact", "contacts", "annual", "report", "reports",
    "financial", "income", "earnings",
}
_BROAD_INSURANCE_OVERVIEW_HINT = (
    "insurance products personal accident auto insurance money insurance fire insurance enhanced rural policy "
    "marine cargo transit fidelity guarantee aviation loan protection burglary machinery breakdown "
    "construction project liability workmen compensation student care"
)
_BROAD_LOAN_OVERVIEW_HINT = (
    "loan products personal loan housing loan transport loan agriculture and livestock loan "
    "hotel and tourism loan loans to contractors loan for shares and securities "
    "service sector loan trade and commerce loan production and manufacturing loan "
    "forestry and logging loan mining and quarrying loan"
)
_BROAD_CLAIM_OVERVIEW_HINT = (
    "motor insurance claim fire insurance claim travel insurance claim engineering insurance claim "
    "aviation insurance claim marine insurance claim liability insurance claim miscellaneous insurance claim"
)
_YEAR_FOLLOWUP_RE = re.compile(
    r"^(?:for|in|about|what about|how about|same for|and for)?\s*(20\d{2})\??$",
    re.IGNORECASE,
)
_GENERIC_CONTEXT_TERMS = {
    "report", "reports", "annual", "document", "documents", "file", "files",
    "it", "this", "that", "same", "one",
}
_CONTEXT_STOPWORDS = {
    "can", "could", "would", "please", "tell", "me", "about", "the", "a", "an",
    "give", "show", "share", "send", "provide", "need", "want", "for", "of", "to",
}


def extract_followup_year(q: str) -> str:
    qn = _norm(q)
    if not qn:
        return ""
    m = _YEAR_FOLLOWUP_RE.match(qn)
    return m.group(1) if m else ""


def _retarget_topic_year(topic: str, year: str) -> str:
    topic_n = _compact_topic(topic)
    if not topic_n or not year:
        return topic_n
    if re.search(r"\b20\d{2}\b", topic_n):
        return re.sub(r"\b20\d{2}\b", year, topic_n)
    return f"{topic_n} {year}".strip()


def _is_generic_context_query(q: str) -> bool:
    qn = _norm(q)
    if not qn:
        return False
    words = [w for w in qn.split() if w not in _CONTEXT_STOPWORDS]
    if not words:
        return False
    return all(w in _GENERIC_CONTEXT_TERMS for w in words)


def is_contextual_followup_fragment(q: str) -> bool:
    qn = _norm(q)
    if not qn:
        return False
    if extract_followup_year(qn):
        return True
    if _is_generic_context_query(qn):
        return True
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
    if _is_generic_context_query(qn):
        return False
    words = set(qn.split())
    return bool(words & _BIL_TOPIC_TERMS)


def should_bind_to_recent_topic(q: str, history: List[Dict[str, str]]) -> bool:
    if not has_recent_bil_context(history):
        return False
    if extract_followup_year(q):
        return True
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


def contextualize_query_from_history(q: str, history: List[Dict[str, str]]) -> str:
    qn = normalize_query_aliases((q or "").strip())
    if not qn or not has_recent_bil_context(history):
        return qn

    topic = last_active_topic(history) or last_user_message(history)
    if not topic:
        return qn

    year = extract_followup_year(qn)
    if year:
        retargeted = _retarget_topic_year(topic, year)
        return retargeted or qn

    topic_n = _compact_topic(topic)
    qn_compact = _compact_topic(qn) or qn

    if _is_generic_context_query(qn) and topic_n:
        return topic_n

    if "report" in _norm(qn) and "annual report" in topic_n and not re.search(r"\b20\d{2}\b", qn):
        return topic_n

    if has_explicit_bil_topic(qn):
        return qn

    if should_bind_to_recent_topic(qn, history):
        if topic_n and topic_n not in qn_compact:
            pronoun_cleaned = re.sub(r"\b(it|this|that|them|those|these)\b", "", qn_compact)
            pronoun_cleaned = re.sub(r"\s+", " ", pronoun_cleaned).strip()
            if pronoun_cleaned:
                return f"{pronoun_cleaned} {topic_n}".strip()
            return topic_n
    return qn


def build_retrieval_query(q: str, history: List[Dict[str, str]]) -> str:
    queries = build_retrieval_queries(q, history)
    return " ".join(queries)


def _is_broad_insurance_overview_query(text: str) -> bool:
    qn = _norm(text)
    if not qn or _is_actionable_query(text):
        return False
    if _query_products(qn) or _extract_extra_products(qn):
        return False
    broad_cues = (
        "insurance products",
        "types of insurance",
        "insurance options",
        "insurance policies",
        "what insurance",
        "what policies",
    )
    if any(cue in qn for cue in broad_cues):
        return True
    return qn in {"insurance", "insurances", "policies", "products", "insurance products"}


def _is_broad_loan_overview_query(text: str) -> bool:
    qn = _norm(text)
    if not qn or _is_actionable_query(text) or _is_broad_loan_rate_query(text):
        return False
    loan_topic = _extract_specific_loan_topic(qn)
    if loan_topic and loan_topic != "loan":
        return False
    broad_cues = (
        "loan products",
        "types of loans",
        "types of loan",
        "loan options",
        "what loans",
        "loans offered",
        "loans in bil",
    )
    if any(cue in qn for cue in broad_cues):
        return True
    return qn in {"loan", "loans", "bil loan", "bil loans"}


def _is_broad_claim_overview_query(text: str) -> bool:
    qn = _norm(text)
    if not qn or _is_actionable_query(text):
        return False
    if _looks_like_claim_topic_text(qn):
        return False
    broad_cues = ("claim types", "types of claims", "claim products", "claims in bil", "claims offered")
    if any(cue in qn for cue in broad_cues):
        return True
    return qn in {"claim", "claims", "insurance claim", "insurance claims"}


def _wants_diverse_retrieval(text: str) -> bool:
    return (
        _is_broad_loan_rate_query(text)
        or _is_broad_loan_overview_query(text)
        or _is_broad_insurance_overview_query(text)
        or _is_broad_claim_overview_query(text)
    )


def build_retrieval_queries(q: str, history: List[Dict[str, str]]) -> List[str]:
    contextualized = _norm(contextualize_query_from_history(q, history))
    qn = contextualized or _norm(q)
    expanded = [qn]
    topic = _norm(last_active_topic(history))
    year = extract_followup_year(q)

    # Expand short fund-related asks so vector search has enough signal.
    if qn in {"pf", "ppf", "gf", "gfm", "provident", "gratuity", "fund", "funds"}:
        expanded.append("private provident and gratuity fund ppf gf")
        expanded.append("ppf gfm department bil")

    if any(k in qn for k in ["ppf", "provident", "gratuity", "gf", "gfm", "fund"]):
        expanded.append("private provident fund and gratuity fund bil")
        expanded.append("ppf employee registration contribution nominee refund form")
        expanded.append("loan against private provident fund ppf")

    if year and topic:
        expanded.append(_retarget_topic_year(topic, year))

    if _is_broad_loan_rate_query(qn):
        expanded.append("loan interest rates bil")
        expanded.append(_BROAD_LOAN_RATE_RETRIEVAL_HINT)

    if _is_broad_loan_overview_query(qn):
        expanded.append("loan products bil")
        expanded.append(_BROAD_LOAN_OVERVIEW_HINT)

    if _is_broad_insurance_overview_query(qn):
        expanded.append("insurance products bil")
        expanded.append(_BROAD_INSURANCE_OVERVIEW_HINT)

    if _is_broad_claim_overview_query(qn):
        expanded.append("insurance claims bil")
        expanded.append(_BROAD_CLAIM_OVERVIEW_HINT)

    # Follow-ups like "how to fill this form", "for 2022", "details"
    if any(p in qn for p in ["this form", "that form", "fill this form", "fill the form", "details", "for 20"]):
        if topic and topic not in qn:
            expanded.append(f"{qn} {topic}")
            expanded.append(topic)

    # Keep retrieval anchored to BIL for short/ambiguous user inputs.
    if len(qn.split()) <= 3 and "bil" not in qn and "bhutan insurance" not in qn:
        expanded.append(f"{qn} bil")

    # Use recent user context for very short follow-ups.
    if len(qn.split()) <= 2:
        if topic and topic != qn and topic not in qn:
            expanded.append(f"{qn} {topic}")

    # Follow-up questions should inherit the prior topic.
    if should_bind_to_recent_topic(q, history):
        if topic and topic != qn and topic not in qn:
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

    return out[:8]


def _retrieve_context_once(query_text: str) -> List[Dict[str, Any]]:
    try:
        ctx_json = bil_retrieve_context.run(query_text)
        ctx_obj = json.loads(ctx_json)
        return ctx_obj.get("chunks", []) or []
    except Exception:
        return []


def _merge_retrieved_chunks(
    query_variants: List[str],
    broad: bool = False,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for q_index, query_text in enumerate(query_variants):
        chunks = _retrieve_context_once(query_text)
        for rank, chunk in enumerate(chunks):
            content = re.sub(r"\s+", " ", str(chunk.get("content") or "")).strip()
            if not content:
                continue
            source = str(chunk.get("source") or "").strip()
            title = str(chunk.get("title") or "").strip()
            key = "||".join([
                source.lower().rstrip("/"),
                title.lower(),
                content[:260].lower(),
            ])
            try:
                base_score = float(chunk.get("score", 0.0))
            except Exception:
                base_score = 0.0
            sort_score = base_score + max(0.0, 0.05 - (0.01 * q_index)) + max(0.0, 0.03 - (0.004 * rank))
            candidate = {
                "content": content,
                "source": source,
                "title": title,
                "score": base_score,
                "_sort_score": sort_score,
            }
            current = merged.get(key)
            if current is None or sort_score > float(current.get("_sort_score", 0.0)):
                merged[key] = candidate

    ordered = list(merged.values())
    ordered.sort(key=lambda item: float(item.get("_sort_score", 0.0)), reverse=True)
    if broad:
        first_pass: List[Dict[str, Any]] = []
        leftovers: List[Dict[str, Any]] = []
        seen_sources = set()
        for item in ordered:
            source_key = (item.get("source") or item.get("title") or "").strip().lower()
            if source_key and source_key not in seen_sources:
                seen_sources.add(source_key)
                first_pass.append(item)
            else:
                leftovers.append(item)
        ordered = first_pass + leftovers

    max_items = limit or (max(settings.top_k + 4, 10) if broad else max(settings.top_k, 6))
    cleaned: List[Dict[str, Any]] = []
    for item in ordered[:max_items]:
        cleaned.append({
            "content": item.get("content", ""),
            "source": item.get("source", ""),
            "title": item.get("title", ""),
            "score": item.get("score", 0.0),
        })
    return cleaned

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
        return _retrieve_context_once(query_text)

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

    probe_queries = build_retrieval_queries(q, history)
    ctx_chunks = _merge_retrieved_chunks(
        probe_queries,
        broad=_wants_diverse_retrieval(q),
        limit=max(settings.top_k + 2, 8),
    )
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
    - If user is vague -> use the most recent active topic from user/assistant history
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

    # If vague -> use the most recent active topic in the conversation
    if is_vague_form_request(user_query):
        topic = _norm(last_active_topic(history))
        if topic:
            topic_is_claim = bool(re.search(r"\b(claim|claims|intimation|settlement|settle)\b", topic))
            variants.append(f"{topic} form")
            if topic_is_claim:
                variants += [
                    f"{topic} claim form",
                    "claim intimation form",
                ]
            else:
                variants += [
                    f"{topic} application form",
                    f"{topic} proposal form",
                    f"{topic} claim form",
                ]

            # Canonical mappings (general-purpose, helps “housing loan form”)
            if "loan" in topic:
                variants.append("loan protection claim form" if topic_is_claim else "loan application form")
            if any(w in topic for w in ["motor", "car", "vehicle"]):
                variants += ["motor claim form"]
                if not topic_is_claim:
                    variants += ["motor proposal form"]

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
    topic = _norm(last_active_topic(history))

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
    "required",
    "requirement",
    "requirements",
    "document",
    "documents",
    "form",
    "download",
    "register",
    "submit",
    "settlement",
    "settle",
    "intimation",
    "file",
}
_PROCESS_ACTION_PHRASES = {
    "how to",
    "how do i",
    "what is the process",
    "what is the procedure",
    "claim process",
    "application process",
    "how to claim",
    "how do i claim",
    "how to apply",
    "how do i apply",
    "how to file",
    "how do i file",
    "file a claim",
    "make a claim",
    "claim settlement",
    "documents required",
    "required documents",
    "what documents",
    "what are the requirements",
    "where to submit",
    "what to do",
}
_CLAIM_PROCESS_TERMS = {
    "file",
    "submit",
    "settlement",
    "settle",
    "intimation",
    "intimate",
    "report",
    "documents",
    "document",
    "required",
    "requirements",
    "process",
    "procedure",
    "steps",
    "step",
    "form",
    "download",
}

_PRODUCT_HINTS = {
    "motor": {"motor", "vehicle", "car", "auto"},
    "fire": {"fire"},
    "travel": {"travel"},
    "loan": {"loan", "loans", "housing", "transport", "personal", "contractor", "tourism", "hotel", "agriculture", "livestock", "shares", "securities"},
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


def _extract_specific_loan_topic(text: str) -> str:
    qn = _norm(text)
    if not qn:
        return ""

    matches: List[str] = []
    for keywords, link in _LOAN_PAGE_LINKS:
        title = _norm(link.get("title", ""))
        if not title:
            continue
        if title in qn:
            matches.append(title)
            continue
        if "loan" not in qn:
            continue
        if any(re.search(rf"\b{re.escape(word)}\b", qn) for word in keywords):
            matches.append(title)

    ordered: List[str] = []
    seen = set()
    for item in matches:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)

    if len(ordered) == 1:
        return ordered[0]
    if len(ordered) > 1:
        return "loan"
    return ""


def _looks_like_claim_topic_text(text: str) -> bool:
    qn = _norm(text)
    if not qn:
        return False

    if re.search(
        r"\b(motor|fire|travel|engineering|aviation|marine|liability|miscellaneous)\s+claims?\b",
        qn,
    ):
        return True

    claim_process_phrases = (
        "claim form",
        "claim intimation",
        "claim process",
        "claim procedure",
        "claim settlement",
        "file a claim",
        "make a claim",
        "how to claim",
        "how do i claim",
        "how to file the claim",
        "how do i file the claim",
        "submit the claim",
    )
    return any(phrase in qn for phrase in claim_process_phrases)


def _extract_topic_from_history_text(text: str) -> str:
    qn = _norm(text)
    if not qn:
        return ""

    year_match = re.search(r"\b(20\d{2})\b", qn)
    if "annual report" in qn:
        return f"{year_match.group(1)} annual report" if year_match else "annual report"

    loan_topic = _extract_specific_loan_topic(qn)
    if loan_topic:
        return loan_topic

    products = _query_products(qn)
    claimish = _looks_like_claim_topic_text(qn)

    if len(products) == 1:
        label = next(iter(products))
        if label == "loan":
            return "loan"
        if label == "ppf":
            return "ppf"
        return f"{label} claim" if claimish else f"{label} insurance"

    if len(products) > 1:
        if "loan" in products and products <= {"loan", "ppf"}:
            return "loan"
        if claimish:
            return "claim"
        if products & {"motor", "fire", "travel", "engineering", "aviation", "marine", "liability", "misc"}:
            return "insurance"

    return ""


_BROAD_LOAN_RATE_RETRIEVAL_HINT = (
    "loan interest rates loan against pf personal mortgaged loan housing loan "
    "transport commercial loan transport non-commercial loan agriculture and livestock loan "
    "hotel and tourism loan loans to contractors loan for shares and securities "
    "service sector loan trade and commerce loan production and manufacturing loan "
    "forestry and logging loan mining and quarrying loan"
)


def _is_broad_loan_rate_query(text: str) -> bool:
    qn = _norm(text)
    if not qn:
        return False

    rate_cues = ("interest rate", "interest rates", "loan rate", "loan rates", "rates")
    if not any(cue in qn for cue in rate_cues):
        return False

    if any(term in qn for term in ["premium", "premiums", "insurance premium"]):
        return False

    loan_topic = _extract_specific_loan_topic(qn)
    if loan_topic and loan_topic != "loan":
        return False

    broad_cues = ("various", "different", "some", "their", "all", "available", "offered", "offer")
    loanish = any(
        term in qn
        for term in [
            "loan",
            "loans",
            "housing",
            "personal",
            "transport",
            "shares",
            "securities",
            "contractor",
            "tourism",
            "hotel",
            "agriculture",
            "livestock",
            "service sector",
            "trade",
            "commerce",
            "production",
            "manufacturing",
            "forestry",
            "logging",
            "mining",
            "quarrying",
            "bil",
        ]
    )
    return loanish and ("loan" in qn or "loans" in qn or any(cue in qn for cue in broad_cues))


def _is_actionable_query(q: str) -> bool:
    qn = _norm(q)
    if not qn:
        return False

    if looks_like_form_request(q) or looks_like_form_download_request(q) or looks_like_document_download_request(q):
        return True

    toks = set(qn.split())
    if any(phrase in qn for phrase in _PROCESS_ACTION_PHRASES):
        return True

    if "claim" in toks or "claims" in toks:
        return bool(toks & _CLAIM_PROCESS_TERMS)

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

    topic = last_active_topic(history)
    if topic:
        return _query_products(topic)
    return set()


def _is_claim_context(q: str, history: Optional[List[Dict[str, str]]] = None) -> bool:
    merged = _norm(q)
    if history:
        topic = _norm(last_active_topic(history) or last_user_message(history))
        if topic:
            merged = f"{merged} {topic}".strip()
    claim_terms = {"claim", "claims", "intimation", "settlement", "settle"}
    return any(re.search(rf"\b{re.escape(term)}\b", merged) for term in claim_terms)


def _derive_action_form_queries(q: str, history: List[Dict[str, str]]) -> List[str]:
    products = _infer_products(q, history)
    qn = _norm(q)
    is_claim = _is_claim_context(q, history)
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


_FORM_RESULT_TERMS = {"form", "proposal", "application", "claim", "intimation", "registration", "nomination", "authorization", "refund", "mou"}
_NONCLAIM_FORM_TERMS = {"proposal", "application", "registration", "nomination", "authorization", "refund", "mou"}
_CLAIM_FORM_TERMS = {"claim", "intimation", "settlement"}


def _is_form_like_download(item: Dict[str, str]) -> bool:
    text = f"{item.get('title', '')} {item.get('url', '')}".lower()
    return any(term in text for term in _FORM_RESULT_TERMS)


def _filter_form_downloads_by_context(items: List[Dict[str, str]], is_claim: bool) -> List[Dict[str, str]]:
    if not items:
        return []

    filtered = [item for item in items if _is_form_like_download(item)]
    ordered = filtered or list(items)

    claim_matches = [
        item for item in ordered
        if any(term in f"{item.get('title', '')} {item.get('url', '')}".lower() for term in _CLAIM_FORM_TERMS)
    ]
    nonclaim_matches = [
        item for item in ordered
        if any(term in f"{item.get('title', '')} {item.get('url', '')}".lower() for term in _NONCLAIM_FORM_TERMS)
    ]

    if is_claim and claim_matches:
        return claim_matches
    if not is_claim and nonclaim_matches:
        return nonclaim_matches
    return ordered


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
    is_claim = _is_claim_context(q, history)
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
    filtered = _filter_form_downloads_by_context(deduped, is_claim=is_claim)
    if explicit_form_ask and filtered:
        return filtered[:6]
    if filtered:
        return filtered[:6]
    return deduped[:6]


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
    history_topic = _norm(last_active_topic(history))
    merged_text = f"{qn} {history_topic}".strip()
    products = _infer_products(merged_text, history) | _extract_extra_products(merged_text)

    is_claim_flow = _is_claim_context(q, history)
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



def _describe_request_scope(q: str, history: List[Dict[str, str]]) -> str:
    qn = _norm(q)
    topic = _compact_topic(last_active_topic(history))
    year_match = re.search(r"\b(20\d{2})\b", qn)
    year = extract_followup_year(q) or (year_match.group(1) if year_match else "")

    if year and "annual report" in topic:
        return f"the {year} annual report"

    specific_loan = _extract_specific_loan_topic(qn or topic)
    if specific_loan and specific_loan != "loan":
        return specific_loan

    products = _infer_products(qn, history) | _extract_extra_products(qn) | _extract_extra_products(topic)
    if "travel" in products:
        return "travel insurance"
    if "motor" in products:
        return "motor claim" if _is_claim_context(q, history) else "motor insurance"
    if "fire" in products:
        return "fire claim" if _is_claim_context(q, history) else "fire insurance"
    if "loan" in products:
        if _is_broad_loan_rate_query(q) or _is_broad_loan_overview_query(q):
            return "BIL loan products"
        return "loan products"
    if not products and ("insurance" in qn or "policies" in qn):
        return "BIL insurance products"
    if not products and ("loan" in qn or "loans" in qn):
        return "BIL loan products"
    if not products and ("claim" in qn or "claims" in qn):
        return "BIL claims information"
    if "ppf" in products:
        return "PPF/GF services"
    if year:
        return f"the {year} report"
    if topic:
        return topic
    return ""


def _build_not_found_response(q: str, history: List[Dict[str, str]]) -> Dict[str, Any]:
    scope = _describe_request_scope(q, history)
    if _wants_diverse_retrieval(q):
        subject = scope or "that BIL topic"
        answer = (
            f"I couldn’t pull a reliable full overview of {subject} from the indexed BIL content yet. "
            "If you narrow it to one product, coverage area, rate, or report year, I can answer more precisely."
        )
    elif should_bind_to_recent_topic(q, history):
        subject = scope or "that follow-up"
        answer = (
            f"I couldn’t pull enough indexed detail for {subject} yet. "
            "If you repeat the product or report name in the request, I can narrow it better."
        )
    elif scope:
        answer = (
            f"I couldn’t pull enough indexed detail for {scope} yet. "
            "If you ask for a specific part like coverage, documents, process, rates, or the form, I can narrow it better."
        )
    else:
        answer = (
            "I couldn’t pull enough indexed detail for that request yet. "
            "If you mention the product, form, or report year, I can narrow it better."
        )
    return {
        "intent": "not_found",
        "answer": answer,
        "answer_md": answer,
        "downloads": [],
        "sources": [],
        "confidence": "low",
    }

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
    data.setdefault("suppress_help_closing", False)

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

    # Add a dynamic closing only to shorter informational replies; long answers and download replies already feel complete.
    answer_text = str(data.get("answer", "") or "")
    has_downloads = bool(data.get("downloads"))
    if data.get("intent") != "unrelated" and not data.get("suppress_help_closing") and not has_downloads:
        if not has_help_closing(answer_text) and not has_help_closing(data.get("answer_md", "")):
            if len(answer_text) <= 220 and random.random() < 0.35:
                closing = _build_dynamic_closing(user_query, answer_text, str(data.get("intent", "")))
                if closing and closing not in data["answer"]:
                    data["answer"] = (data["answer"] + " " + closing).strip() if data["answer"] else closing
                if closing and closing not in data["answer_md"]:
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
- For process/how-to/application questions, and for claim questions that explicitly ask about filing, documents, requirements, settlement, forms, or steps, provide one complete response in a single message:
  1) clear flow/steps, 2) short explanation for why each major step matters, 3) key requirements/documents (if known), 4) helpful links from ACTIONABLE RESOURCES, 5) mention downloads if attached.
- If the user asks generally about a product or claim type (for example "tell me about motor claim"), start with a short overview of what it is, when it applies, and the key things to know. Only add a brief next-step summary if useful; do not jump straight into numbered steps unless the user asked for process details.

DYNAMIC RESPONSE STYLE (IMPORTANT):
- Match the response style to the user’s intent and how they asked.
- First line must directly address the user’s main ask.
- Do not repeat the same response pattern every turn; vary structure naturally.
- Use RECENT CHAT CONTEXT to interpret follow-up fragments and to avoid monotonous formatting.
- Do not rely on exact keywords only; infer semantically similar asks and likely intent from context.
- If the user asks for process/steps, explain slightly more than a bare checklist (brief rationale + practical notes).
- If the user asks "tell me about", "more on", or a similar overview request, prefer a concise overview first rather than a procedure.
- For broad product/rate questions, do not imply the list is exhaustive unless the retrieved context clearly supports that. If you show a subset, say it includes some key products and note that other products are also available.
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
- Keep the opening paragraph short. If there are several facts or options, move them into bullets instead of packing them into the intro.
- Keep it concise but complete: prefer 1–3 short sections, typically 5–10 bullets total. For broad loan-rate/product-list questions, it is fine to exceed this slightly so the list does not become misleadingly incomplete.
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


CLOSING_SYSTEM_TEMPLATE = """
You write one short closing sentence for the Bhutan Insurance Limited (BIL) website chatbot.

Write exactly one sentence that:
- sounds natural and varied, not canned
- fits the user's topic and the answer that was just given
- stays within 8-18 words
- offers further help only for relevant BIL topics
- does not repeat the wording already used in the answer
- uses no markdown, no bullets, no links, and no question mark
- does not mention anything outside BIL services
"""

closing_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", CLOSING_SYSTEM_TEMPLATE),
        (
            "human",
            """USER QUERY:
{query}

INTENT:
{intent}

ANSWER SO FAR:
{answer}

STYLE GUIDANCE:
{style_hint}
""",
        ),
    ]
)


FORM_GUIDE_SYSTEM_TEMPLATE = """
You write the chat reply that accompanies downloadable Bhutan Insurance Limited (BIL) forms.

Write concise markdown for a chat UI.
Choose the most relevant format based on the attached form or forms and the supplied context instead of using one fixed template.

Guidelines:
- start with a short confirmation that the form or forms are attached below
- use a compact structure that fits the form, such as short sections, bullets, or a brief checklist
- explain what the form is for when that is clear from the context
- include the most useful things the user should know before filling it, but only when supported by the context
- if multiple forms are attached, explain how they differ or work together when that is clear
- labels like `**What it's for**`, `**Before you fill it**`, or `**How they work together**` are optional, not required
- keep it concise and easy to scan
- do not use markdown headings with `#`
- do not use links and do not end with a question
- do not invent fields, documents, or steps that are not supported by the context
"""

form_guide_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", FORM_GUIDE_SYSTEM_TEMPLATE),
        (
            "human",
            """USER QUERY:
{query}

RECENT CHAT CONTEXT:
{history_context}

ACTIVE BIL TOPIC:
{topic}

FORMS ATTACHED:
{form_titles}

FORM CONTEXT:
{context}
""",
        ),
    ]
)


GREETING_SYSTEM_TEMPLATE = """
You write the greeting reply for Norbu, the AI assistant for Bhutan Insurance Limited (BIL).

Write a short reply that:
- greets the user naturally
- makes it clear you can help with BIL topics like insurance, claims, loans, forms, contact details, branches, and annual reports
- may lightly mention continuing a recent BIL topic if one is provided
- sounds warm but not overly enthusiastic or canned
- stays within 1-2 short sentences
- uses no bullets, no markdown headings, no links, and no question at the end
"""

greeting_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", GREETING_SYSTEM_TEMPLATE),
        (
            "human",
            """USER QUERY:
{query}

RECENT CHAT CONTEXT:
{history_context}

RECENT BIL TOPIC:
{recent_topic}

STYLE GUIDANCE:
{style_hint}
""",
        ),
    ]
)


UNRELATED_SYSTEM_TEMPLATE = """
You are the official website chatbot for Bhutan Insurance Limited (BIL).

The user's latest message is outside the scope of BIL support.

Write a short reply that:
- acknowledges what the user is asking about in the first sentence
- clearly says you cannot help with that specific topic here
- does NOT answer the unrelated question itself
- naturally redirects to the BIL topics you can help with: insurance products, claims, loans, forms, contact details, branches, and annual reports
- if a recent BIL topic is provided, you may mention it as something you can continue helping with
- sounds natural and varied, not canned
- stays within 2-3 sentences
- uses no bullets, no markdown headings, no links, and no question at the end
"""

unrelated_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", UNRELATED_SYSTEM_TEMPLATE),
        (
            "human",
            """USER QUERY:
{query}

RECENT CHAT CONTEXT:
{history_context}

RECENT BIL TOPIC:
{recent_topic}

STYLE GUIDANCE:
{style_hint}
""",
        ),
    ]
)



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


def _llm_plain() -> ChatOpenAI:
    kwargs = {
        "model": settings.chat_model,
        "api_key": settings.openai_api_key,
    }
    if not str(settings.chat_model).startswith("gpt-5"):
        kwargs["temperature"] = 0.55
    return ChatOpenAI(**kwargs)


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                if item.strip():
                    parts.append(item.strip())
                continue
            if isinstance(item, dict):
                piece = item.get("text") or item.get("content") or ""
                if isinstance(piece, str) and piece.strip():
                    parts.append(piece.strip())
                continue
            piece = getattr(item, "text", None) or getattr(item, "content", None)
            if isinstance(piece, str) and piece.strip():
                parts.append(piece.strip())
        return " ".join(parts).strip()
    return str(content or "").strip()


def _build_unrelated_llm_reply(query: str, history: List[Dict[str, str]]) -> str:
    recent_topic = last_active_topic(history) if has_recent_bil_context(history) else ""
    style_hint = random.choice(_UNRELATED_STYLE_HINTS)
    try:
        llm = _llm_plain()
        msgs = unrelated_prompt.format_messages(
            query=query,
            history_context=build_recent_history_context(history),
            recent_topic=recent_topic or "None",
            style_hint=style_hint,
        )
        out = llm.invoke(msgs)
        text = _message_content_to_text(getattr(out, "content", ""))
        text = strip_urls(text)
        text = re.sub(r"\s+", " ", text).strip()
        if text and not text.startswith("{"):
            return text
    except Exception:
        pass
    return bil_unrelated_reply.run(query) if hasattr(bil_unrelated_reply, "run") else bil_unrelated_reply(query)



def _build_greeting_llm_reply(query: str, history: List[Dict[str, str]]) -> str:
    recent_topic = last_active_topic(history) if has_recent_bil_context(history) else ""
    style_hint = random.choice(_GREETING_STYLE_HINTS)
    try:
        llm = _llm_plain()
        msgs = greeting_prompt.format_messages(
            query=query,
            history_context=build_recent_history_context(history),
            recent_topic=recent_topic or "None",
            style_hint=style_hint,
        )
        out = llm.invoke(msgs)
        text = _message_content_to_text(getattr(out, "content", ""))
        text = strip_urls(text)
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"[?]+$", "", text).strip()
        if text and not text.startswith("{"):
            return text
    except Exception:
        pass
    if recent_topic:
        return f"Hello. I can help with BIL topics, including continuing with {recent_topic}."
    return "Hello. I can help with BIL insurance, claims, loans, forms, contact details, branches, and annual reports."



def build_start_greeting(history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    history = list(history or [])
    greeting = _build_greeting_llm_reply("hello", history)
    return finalize(
        {
            "intent": "unrelated",
            "answer": greeting,
            "answer_md": repair_markdown_from_text(greeting),
            "downloads": [],
            "sources": [],
            "confidence": "high",
            "client_delay_ms": 1200,
        },
        user_query="hello",
    )



def _build_dynamic_closing(query: str, answer: str, intent: str) -> str:
    style_hint = random.choice(_CLOSING_STYLE_HINTS)
    try:
        llm = _llm_plain()
        msgs = closing_prompt.format_messages(
            query=query,
            intent=intent or "bil_query",
            answer=answer,
            style_hint=style_hint,
        )
        out = llm.invoke(msgs)
        text = _message_content_to_text(getattr(out, "content", ""))
        text = strip_urls(text)
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"[?!.]+$", "", text).strip()
        if text and len(text.split()) >= 4:
            return f"{text}."
    except Exception:
        pass
    return random.choice(_HELP_CLOSINGS)


def _fallback_form_download_reply(downloads: List[Dict[str, str]]) -> Dict[str, str]:
    titles = [
        (d.get("title") or "Form").strip()
        for d in downloads[:4]
        if (d.get("title") or "").strip()
    ]
    if not titles:
        md = "I've attached the relevant BIL form below.\n\n**Before you fill it**\n- Review the requested details in the form before submission."
        return {"answer": markdown_to_plain_text(md), "answer_md": md}

    if len(titles) == 1:
        title = titles[0]
        md = (
            f"I've attached the relevant form below.\n\n"
            f"**What it's for**\n"
            f"- {title}\n\n"
            f"**Before you fill it**\n"
            f"- Review the required fields carefully and complete them accurately.\n"
            f"- Keep the supporting details for your BIL request ready before submission."
        )
        return {"answer": markdown_to_plain_text(md), "answer_md": md}

    md = (
        "I've attached the relevant forms below.\n\n"
        "**What they're for**\n"
        + "\n".join(f"- {title}" for title in titles[:4])
        + "\n\n**Before you fill them**\n"
        "- Review each form carefully and complete the sections that apply to your request.\n"
        "- Keep the related policy, identity, or product details ready before submission."
    )
    if any("intimation" in title.lower() for title in titles) and any("claim" in title.lower() for title in titles):
        md += "\n\n**How they work together**\n- Use the intimation form to notify BIL first, then complete the detailed claim form."
    return {"answer": markdown_to_plain_text(md), "answer_md": md}


def _build_form_download_reply(query: str, history: List[Dict[str, str]], downloads: List[Dict[str, str]]) -> Dict[str, str]:
    topic = last_active_topic(history)
    form_titles = ", ".join(
        (d.get("title") or "Form").strip()
        for d in downloads[:4]
        if (d.get("title") or "").strip()
    )
    try:
        context_obj = bil_get_form_context(query, downloads[:4], topic)
    except Exception:
        context_obj = {"contexts": [], "found": False}

    contexts = context_obj.get("contexts", []) or []
    if contexts:
        context_text = "\n\n".join(
            f"TITLE: {c.get('title', '')}\nSOURCE: {c.get('source', '')}\nCONTENT: {c.get('content', '')}"
            for c in contexts[:4]
        )
    else:
        context_text = "Only the form titles are available. Stay general and do not invent details."

    try:
        llm = _llm_plain()
        msgs = form_guide_prompt.format_messages(
            query=query,
            history_context=build_recent_history_context(history),
            topic=topic or "None",
            form_titles=form_titles or "BIL form",
            context=context_text,
        )
        out = llm.invoke(msgs)
        md = _message_content_to_text(getattr(out, "content", ""))
        md = normalize_form_markdown(md)
        if md and not md.startswith("{") and looks_like_markdown(md):
            answer = markdown_to_plain_text(md)
            if answer:
                return {"answer": answer, "answer_md": md}
    except Exception:
        pass

    return _fallback_form_download_reply(downloads)


# =========================
# Main runner
# =========================
def run_agent(query: str, history: List[Dict[str, str]]) -> Dict[str, Any]:
    raw_q = normalize_query_aliases((query or "").strip())
    if not raw_q:
        return finalize({
            "intent": "not_found",
            "answer": "Please type your question.",
            "answer_md": "Please type your question.",
            "downloads": [],
            "sources": [],
            "confidence": "low",
        }, user_query=raw_q)

    # 0) Social intent
    social = detect_social_intent(raw_q)
    if social == "greeting":
        greeting = _build_greeting_llm_reply(raw_q, history)
        return finalize({
            "intent": "unrelated",
            "answer": greeting,
            "answer_md": repair_markdown_from_text(greeting),
            "downloads": [],
            "sources": [],
            "confidence": "high",
            "client_delay_ms": 1500,
        }, user_query=raw_q)

    if social == "thanks":
        return finalize({
            "intent": "unrelated",
            "answer": "You’re welcome! Let me know if you need anything else.",
            "answer_md": "You’re welcome!\n\nIf you need help with insurance, loans, or forms, just ask.",
            "downloads": [],
            "sources": [],
            "confidence": "high",
            "client_delay_ms": 1500,
        }, user_query=raw_q)

    if social == "farewell":
        return finalize({
            "intent": "unrelated",
            "answer": "Goodbye! Have a great day.",
            "answer_md": "Goodbye!\n\nHave a great day.",
            "downloads": [],
            "sources": [],
            "confidence": "high",
            "client_delay_ms": 1500,
        }, user_query=raw_q)

    if social == "identity":
        answer, answer_md = build_identity_reply(raw_q)
        return finalize({
            "intent": "unrelated",
            "answer": answer,
            "answer_md": answer_md,
            "downloads": [],
            "sources": [],
            "confidence": "high",
            "client_delay_ms": 1500,
        }, user_query=raw_q)

    q = contextualize_query_from_history(raw_q, history)

    # 0b) Direct annual-report financial extraction for year-specific queries.
    if not looks_like_document_download_request(q):
        try:
            fin_obj = json.loads(bil_extract_financial_fact.run(q))
        except Exception:
            fin_obj = {}
        if isinstance(fin_obj, dict) and fin_obj.get("found"):
            return finalize({
                "intent": "bil_query",
                "answer": str(fin_obj.get("answer", "") or ""),
                "answer_md": str(fin_obj.get("answer_md", "") or fin_obj.get("answer", "")),
                "downloads": [],
                "sources": [],
                "confidence": "high",
                "suppress_help_closing": True,
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
            is_claim = _is_claim_context(q, history)
            filtered = _filter_form_downloads_by_context(merged, is_claim=is_claim)
            if filtered:
                merged = filtered

            downloads = merged[:4]
            form_reply = _build_form_download_reply(q, history, downloads)
            return finalize({
                "intent": "form_request",
                "answer": str(form_reply.get("answer", "") or ""),
                "answer_md": str(form_reply.get("answer_md", "") or ""),
                "downloads": downloads,
                "sources": [],
                "confidence": "high",
            }, user_query=q)

        # Graceful fallback when no matching form exists
        if is_vague_form_request(q):
            topic = last_active_topic(history)
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
        msg = _build_unrelated_llm_reply(q, history)
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
    broad_retrieval = _wants_diverse_retrieval(q)
    if prefetched_chunks:
        chunks = prefetched_chunks
    else:
        retrieval_queries = build_retrieval_queries(q, history)
        chunks = _merge_retrieved_chunks(
            retrieval_queries,
            broad=broad_retrieval,
            limit=max(settings.top_k + 4, 10) if broad_retrieval else max(settings.top_k, 6),
        )

    if not chunks:
        # Retry with prior topic for vague follow-ups.
        retry_topic = last_active_topic(history) or last_user_message(history)
        if retry_topic and _norm(retry_topic) != _norm(q):
            retry_query = retry_topic
            if should_bind_to_recent_topic(q, history):
                retry_query = f"{retry_topic} {q}".strip()
            retry_queries = build_retrieval_queries(retry_query, history)
            retry_broad = broad_retrieval or _wants_diverse_retrieval(retry_query)
            chunks = _merge_retrieved_chunks(
                retry_queries,
                broad=retry_broad,
                limit=max(settings.top_k + 4, 10) if retry_broad else max(settings.top_k, 6),
            )

    if not chunks:
        return finalize(_build_not_found_response(q, history), user_query=q)

    context_limit = max(settings.top_k + 4, 10) if broad_retrieval else settings.top_k

    context = "\n\n".join([f"- {c.get('content','')}" for c in chunks[:context_limit]])
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

        fallback_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(out or "").strip(), flags=re.IGNORECASE | re.MULTILINE).strip()
        if fallback_text and not fallback_text.startswith("{"):
            return finalize({
                "intent": "bil_query",
                "answer": fallback_text,
                "answer_md": fallback_text,
                "downloads": action_downloads,
                "sources": [],
                "confidence": "medium",
            }, user_query=q)

        return finalize({
            "intent": "not_found",
            "answer": "I couldn’t process that. Please try again.",
            "answer_md": "I couldn’t process that. Please try again.",
            "downloads": [],
            "sources": [],
            "confidence": "low",
        }, user_query=q)
