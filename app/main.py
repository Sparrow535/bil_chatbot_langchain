from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from collections import defaultdict, deque
from pathlib import Path

from app.schemas import ChatRequest, GreetingRequest, BotResponse, DownloadItem, GreetingResponse, TranscribeResponse
from app.agent import build_start_greeting, run_agent
from app.incremental import start_incremental_loop
from openai import OpenAI
import tempfile
import os
from typing import Optional, Tuple

from dotenv import load_dotenv
load_dotenv()


app = FastAPI(title="BIL LangChain Chatbot API", version="1.0.0")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SESSION_HISTORY = defaultdict(lambda: deque(maxlen=12))
_incremental_started = False


def _is_non_english_text(text: str) -> bool:
    s = (text or "").strip()
    if not s:
        return False

    total_letters = 0
    latin_letters = 0
    for ch in s:
        if ch.isalpha():
            total_letters += 1
            if "a" <= ch.lower() <= "z":
                latin_letters += 1

    if total_letters == 0:
        return False
    return (latin_letters / total_letters) < 0.6


def _transcribe_with_fallback(tmp_path: str) -> Tuple[str, str]:
    preferred = (os.getenv("OPENAI_STT_MODEL", "gpt-4o-transcribe") or "").strip()
    fallback_raw = os.getenv("OPENAI_STT_FALLBACK_MODELS", "")
    fallback_models = [m.strip() for m in fallback_raw.split(",") if m.strip()]
    language: Optional[str] = (os.getenv("OPENAI_STT_LANGUAGE", "en") or "").strip() or None
    prompt: Optional[str] = (os.getenv("OPENAI_STT_PROMPT", "") or "").strip() or None

    candidates = [preferred] + fallback_models
    if "gpt-4o-transcribe" not in candidates:
        candidates.append("gpt-4o-transcribe")
    if "whisper-1" not in candidates:
        candidates.append("whisper-1")

    deduped = []
    seen = set()
    for m in candidates:
        if not m or m in seen:
            continue
        seen.add(m)
        deduped.append(m)

    last_error = None
    for model_name in deduped:
        kwargs = {"model": model_name}
        if language:
            kwargs["language"] = language
        # Prompt tends to help Whisper with domain names.
        if prompt and model_name.startswith("whisper"):
            kwargs["prompt"] = prompt

        attempts = [kwargs]
        if "prompt" in kwargs:
            retry_no_prompt = dict(kwargs)
            retry_no_prompt.pop("prompt", None)
            attempts.append(retry_no_prompt)

        for attempt_kwargs in attempts:
            try:
                with open(tmp_path, "rb") as f:
                    resp = client.audio.transcriptions.create(file=f, **attempt_kwargs)
                text = (getattr(resp, "text", None) or "").strip()
                if not text:
                    last_error = ValueError("Empty transcription text")
                    continue

                if language and language.lower().startswith("en") and _is_non_english_text(text):
                    last_error = ValueError("Non-English transcription candidate")
                    continue

                return text, model_name
            except Exception as e:
                last_error = e
                continue

    if last_error:
        raise last_error
    raise RuntimeError("Transcription failed")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def _startup_tasks():
    global _incremental_started
    if not _incremental_started:
        start_incremental_loop()
        _incremental_started = True

@app.get("/health")
def health():
    return {"ok": True}


def _merge_session_history(session_id: str, req_history, persist: bool = True):
    sid = session_id or "anon"
    history = list(SESSION_HISTORY[sid])

    if req_history:
        client_history = []
        for m in req_history[-8:]:
            item = {"role": m.role, "content": m.content}
            followup_query = (m.followup_query or "").strip() if hasattr(m, "followup_query") else ""
            if followup_query:
                item["followup_query"] = followup_query
            client_history.append(item)
        if not history:
            history = client_history
            if persist:
                SESSION_HISTORY[sid].extend(client_history)
        else:
            merged = list(history)
            seen = {
                (
                    str(m.get("role", "")),
                    str(m.get("content", "")),
                    str(m.get("followup_query", "")),
                )
                for m in merged
            }
            for item in client_history:
                key = (
                    item["role"],
                    item["content"],
                    str(item.get("followup_query", "")),
                )
                if key in seen:
                    continue
                merged.append(item)
                seen.add(key)
            history = merged[-12:]
            if persist:
                SESSION_HISTORY[sid].clear()
                SESSION_HISTORY[sid].extend(history)

    return sid, history


@app.post("/greeting", response_model=GreetingResponse)
def greeting(req: GreetingRequest):
    _, history = _merge_session_history(req.session_id or "anon", req.history, persist=False)
    result = build_start_greeting(history=history)
    return GreetingResponse(
        answer=result.get("answer", ""),
        answer_md=result.get("answer_md", ""),
        client_delay_ms=result.get("client_delay_ms"),
    )


@app.post("/chat", response_model=BotResponse)
def chat(req: ChatRequest):

    sid, history = _merge_session_history(req.session_id or "anon", req.history, persist=True)

    result = run_agent(query=req.message, history=history)

    SESSION_HISTORY[sid].append({"role": "user", "content": req.message})
    assistant_hist = str(result.get("answer", "") or "")
    assistant_followup_query = str(result.get("followup_query", "") or "").strip()
    raw_downloads = result.get("downloads") or []
    if isinstance(raw_downloads, list) and raw_downloads:
        titles = []
        for d in raw_downloads:
            if isinstance(d, dict):
                t = str(d.get("title", "") or "").strip()
                if t:
                    titles.append(t)
        if titles:
            assistant_hist = f"{assistant_hist} Downloads: {', '.join(titles[:4])}".strip()
    assistant_item = {"role": "assistant", "content": assistant_hist}
    if assistant_followup_query:
        assistant_item["followup_query"] = assistant_followup_query
    SESSION_HISTORY[sid].append(assistant_item)

    downloads = []
    for d in (result.get("downloads") or []):
        downloads.append(
            DownloadItem(
                title=str(d.get("title", "Download")),
                url=str(d.get("url", "")),
            )
        )
    downloads = [x for x in downloads if x.url]

    debug_payload = None
    if os.getenv("DEBUG_CHAT", "0") == "1":
        debug_payload = result.get("debug")

    return BotResponse(
        intent=result.get("intent", "not_found"),
        answer=result.get("answer", ""),
        answer_md=result.get("answer_md", ""),
        sources=result.get("sources", []),
        downloads=downloads,
        confidence=result.get("confidence", "low"),
        debug=debug_payload,
        followup_query=result.get("followup_query") or None,
    )


@app.post("/stt", response_model=TranscribeResponse)
async def stt(file: UploadFile = File(...)):
    # Basic validation
    if not file:
        raise HTTPException(status_code=400, detail="No audio file provided")

    # Save to temp file
    suffix = ".webm"
    if file.filename and "." in file.filename:
        suffix = "." + file.filename.split(".")[-1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        content = await file.read()
        tmp.write(content)

    try:
        text, used_model = _transcribe_with_fallback(tmp_path)
        if os.getenv("DEBUG_CHAT", "0") == "1":
            print(f"[stt] model={used_model} bytes={len(content)} text_len={len(text)}")
        return TranscribeResponse(text=text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

# Serve frontend (works locally and on Render)
FRONTEND_DIR = (Path(__file__).resolve().parent.parent / "frontend").resolve()
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

