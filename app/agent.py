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
from app.text_utils import looks_like_form_request
from app.tools import bil_get_forms, bil_retrieve_context, bil_unrelated_reply, bil_intent_hint

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

# =========================
# Social intents
# =========================
_GREETINGS = {"hello", "hi", "hey", "good morning", "good afternoon", "good evening", "kuzuzangpo", "kuzuzangpo la", "kuzu"}
_FAREWELLS = {"bye", "goodbye", "see you", "see ya", "take care", "later", "okay", "ok"}
_THANKS = {"thanks", "thank you", "thx", "thanks a lot", "appreciate it"}

def _norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def normalize_query_aliases(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return t
    # Common ASR typo: PIL -> BIL
    t = re.sub(r"\bPIL\b", "BIL", t, flags=re.IGNORECASE)
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
        if is_vague_form_request(txt):
            continue
        if _norm(txt) in {"types", "type", "options", "details", "more", "more info"}:
            continue
        # ignore social chatter
        if _norm(txt) in {"hi", "hello", "hey", "bye", "thanks", "thank you", "kuzuzangpo"}:
            continue
        return txt
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

def is_affirmative_reply(q: str) -> bool:
    qn = _norm(q)
    if not qn:
        return False
    return qn in {"yes", "yeah", "yep", "sure", "ok", "okay", "please", "alright"}

def build_retrieval_query(q: str, history: List[Dict[str, str]]) -> str:
    return q

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


def enforce_downloads_rules(data: Dict[str, Any], user_query: str) -> None:
    """
    HARD RULE: only include downloads if user is actually requesting forms in THIS turn.
    Prevents random forms appearing for normal product Qs.
    """
    formish_now = looks_like_form_request(user_query) or is_vague_form_request(user_query)
    if not formish_now:
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
- You MAY include official BIL or app store URLs in answer_md when the user explicitly asks for resources, claims, contact info, or links.
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

DYNAMIC RESPONSE STYLE (IMPORTANT):
- Match the response style to the user’s intent and how they asked.
- First line must directly address the user’s main ask.
- Then choose ONE layout below for answer_md (pick the most suitable; do not use all):

LAYOUT TOOLBOX (choose 1):
1) Quick Answer + Why (best for advice/yes-no/personal questions)
   - Start with a direct answer (Yes/No/It depends) then 2–4 bullets.

2) Summary + Key Points (best for “tell me about…”)
   - Short intro then bullets under a heading.

3) Steps Checklist (best for “how to”, “process”, “what to do”)
   - Use numbered steps (1–5 max).

4) Compare (best for “difference between”, “which is better”)
   - Use a small markdown table OR two subheadings.

5) Requirements (best for “documents needed”, “eligibility”, “what do I need”)
   - Use a short list grouped by category.

6) Options List (best for lists like products/loans)
   - Use subheadings per option with 2–3 bullets each (compact).
   
7) Form Request Confirmation (best for form requests)
   - Confirm form found and available for download (no links).
   
8) IF other scenarios: use your best judgment to pick a clear, concise format.

STYLE:
- answer_md should use headings (if absolutely necessary), bullets, and line breaks for readability.
- Keep it concise: prefer 1–2 short sections, max ~8 bullets total.
- Do NOT end with a question. If you add a closing, make it a brief helpful statement (no question) and vary the wording.
- Avoid “brochure tone” for personal questions; sound like helpful support staff.
- NO need to provide headers for all the answers; give headers for only the most relevant answers.

SELF-CHECK:
Before finalizing:
1) Did I answer the user’s exact ask in the first line?
2) Did I choose the best ONE layout from the toolbox?
3) Is answer_md clean and scannable?
4) Did I avoid URLs/sources and avoid ending with a question?

Return ONLY JSON matching this schema:
{format_instructions}
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_TEMPLATE),
        ("human", "USER QUERY:\n{query}\n\nCONTEXT:\n{context}\n"),
    ]
).partial(format_instructions=parser.get_format_instructions())


def _llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.chat_model,
        api_key=settings.openai_api_key,
        temperature=0.2,
        model_kwargs={"response_format": {"type": "json_object"}},
    )


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
        }, user_query=q)

    if social == "thanks":
        return finalize({
            "intent": "unrelated",
            "answer": "You’re welcome! Let me know if you need anything else.",
            "answer_md": "You’re welcome!\n\nIf you need help with insurance, loans, or forms, just ask.",
            "downloads": [],
            "sources": [],
            "confidence": "high",
        }, user_query=q)

    if social == "farewell":
        return finalize({
            "intent": "unrelated",
            "answer": "Goodbye! Have a great day.",
            "answer_md": "Goodbye!\n\nHave a great day.",
            "downloads": [],
            "sources": [],
            "confidence": "high",
        }, user_query=q)

    # 1) Intent hint
    try:
        hint = bil_intent_hint.run(q)
    except Exception:
        hint = "bil_query"

    # 2) Forms path (ONLY when this message is form-ish)
    formish_now = looks_like_form_request(q) or is_vague_form_request(q) or hint == "form_request"
    if formish_now:
        candidate_queries = build_form_query_variants(q, history)

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
        }, user_query=q)

    # 4) BIL Query: retrieve context + ask LLM
    try:
        retrieval_query = build_retrieval_query(q, history)
        ctx_json = bil_retrieve_context.run(retrieval_query)
        ctx_obj = json.loads(ctx_json)
        chunks = ctx_obj.get("chunks", []) or []
    except Exception:
        chunks = []

    if not chunks:
        # Retry with prior topic for vague follow-ups
        retry_topic = last_user_topic(history) or last_user_message(history)
        if retry_topic and _norm(retry_topic) != _norm(q):
            try:
                ctx_json = bil_retrieve_context.run(retry_topic)
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

    llm = _llm()
    msgs = prompt.format_messages(query=q, context=context)
    out = llm.invoke(msgs).content or ""

    # Parse
    try:
        parsed = parser.parse(out)
        return finalize(parsed.model_dump(), user_query=q)
    except Exception:
        try:
            candidate = extract_first_json_object(out)
            if candidate:
                obj = json.loads(candidate)
                if isinstance(obj, dict):
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
