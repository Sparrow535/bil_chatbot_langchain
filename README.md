# BIL Chatbot (LangChain + FastAPI)

A production-ready chatbot backend and frontend that powers the conversational assistant used on bil.bt. The project combines lightweight FastAPI endpoints, LangChain orchestration, OpenAI models for chat and speech-to-text, and a small frontend to interact with the bot.

What this is (short)
- A conversational assistant that answers BIL-related queries using a mix of live scraping/tools and retrieval over indexed content. It exposes endpoints for greeting, chat, and speech transcription and serves a bundled frontend.

Contributions (Sparrow535)
- Implemented the FastAPI backend and API surface (app/main.py) including endpoints: `/greeting`, `/chat`, `/stt` and a health check.
- Built the agent orchestration and domain-specific logic (app/agent.py, app/tools.py) that routes user queries to the right tools and constructs assistant responses.
- Added incremental/background processing for data refreshes and periodic tasks (app/incremental.py).
- Implemented session/history handling and response shaping in the API (session merging in app/main.py and Pydantic schemas in app/schemas.py).
- Created configuration and environment-driven settings (app/config.py) and packaged requirements for easy setup (requirements.txt).
- Provided a small frontend (frontend/) and static file serving so the full demo can be run locally or deployed.

Technical summary (concise)
- Languages & runtime: Python 3.11+ (FastAPI), JavaScript/HTML for the frontend.
- Core libraries: FastAPI, Uvicorn, OpenAI official client, LangChain, FAISS (faiss-cpu), Playwright (for scraping), pypdf/python-docx for document ingestion.
- Key files:
  - app/main.py — HTTP API, session handling, STT fallback logic and static frontend mount
  - app/agent.py — agent logic and response formatting (LangChain + tool orchestration)
  - app/tools.py — site scraping, helper tools and download generation used by the agent
  - app/incremental.py — background data fetch and processing loop
  - app/schemas.py — Pydantic request/response models
  - frontend/ — web UI served by FastAPI static files
  - requirements.txt — pinned Python dependencies

How to run locally (short)
1. Clone and install dependencies (use a virtualenv):

```bash
git clone https://github.com/Sparrow535/bil_chatbot_langchain.git
cd bil_chatbot_langchain
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

2. Install Playwright browsers (required if using Playwright-based tools):

```bash
python -m playwright install
```

3. Set required environment variables (example .env):

```ini
OPENAI_API_KEY=sk-...
BIL_BASE_URL=https://www.bil.bt
# optional tuning
OPENAI_CHAT_MODEL=gpt-4.1-mini
OPENAI_EMBED_MODEL=text-embedding-3-small
PORT=8000
```

4. Start the app with Uvicorn:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

5. Open the frontend in your browser (when running locally): http://localhost:8000/ — the production demo is available at https://bil.bt

Notes and operational tips
- STT: the `/stt` endpoint uses OpenAI audio transcription models with a fallback chain; set OPENAI_STT_MODEL and OPENAI_STT_FALLBACK_MODELS as needed.
- Vector search: the project references faiss-cpu for embeddings-based retrieval. On some platforms faiss installation can be non-trivial; use the provided faiss-cpu wheel or switch to a hosted vector DB if needed.
- Playwright: scraping tools rely on Playwright — ensure browsers are installed via `playwright install` and run in an environment where headless browsing is allowed.
- Secrets: store API keys in environment variables or a secrets store. Do not commit keys to the repository.
- Debugging: set `DEBUG_CHAT=1` to receive additional debug payloads from the agent (careful: may expose internal data).

Project layout (top-level)

```
app/           — FastAPI app, agent logic, tools, and schemas (app/main.py, app/agent.py, app/tools.py...)
frontend/      — Web UI served statically by FastAPI (mounted at /)
data/          — raw or processed data (ingestion targets)
scripts/       — utility scripts (ingestion, housekeeping)
requirements.txt — pinned Python dependencies
```

Demo
- The chatbot is live on the public site: https://bil.bt — try asking BIL-related questions there.

If you want next
- I can add a short Troubleshooting or Deploy section (Render, Heroku, or a containerized Dockerfile).
- I can draft a minimal .env.example with all recommended settings.
- I can add a CONTRIBUTING.md and a simple GitHub Action to run linters or tests.

---
Made by Sparrow535 — backend, agent orchestration, and deployment wiring