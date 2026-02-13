from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from collections import defaultdict, deque
from pathlib import Path

from app.schemas import ChatRequest, BotResponse, DownloadItem, TranscribeResponse
from app.agent import run_agent
from app.incremental import start_incremental_loop
from openai import OpenAI
import tempfile
import os

from dotenv import load_dotenv
load_dotenv()


app = FastAPI(title="BIL LangChain Chatbot API", version="1.0.0")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SESSION_HISTORY = defaultdict(lambda: deque(maxlen=12))
_incremental_started = False


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

@app.post("/chat", response_model=BotResponse)
def chat(req: ChatRequest):

    sid = req.session_id or "anon"
    history = list(SESSION_HISTORY[sid])
    result = run_agent(query=req.message, history=history)

    SESSION_HISTORY[sid].append({"role": "user", "content": req.message})
    assistant_hist = str(result.get("answer", "") or "")
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
    SESSION_HISTORY[sid].append({"role": "assistant", "content": assistant_hist})

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
        debug=debug_payload
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
        # OpenAI Audio Transcriptions API :contentReference[oaicite:1]{index=1}
        with open(tmp_path, "rb") as f:
            transcription = client.audio.transcriptions.create(
                model=os.getenv("OPENAI_STT_MODEL", "gpt-4o-transcribe"),
                file=f,
                # language="en",  # optional
            )

        text = (getattr(transcription, "text", None) or "").strip()
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

