import sys
from pathlib import Path

# --- ensure project root in PYTHONPATH ---
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

# --- load .env ---
from dotenv import load_dotenv
load_dotenv()

import os
import json
from typing import List, Dict, Any

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from tqdm import tqdm

from app.text_utils import chunk_text, clean_text


RAW_JSONL = Path("data/raw.jsonl")
VS_DIR = Path("data/faiss_index")
FORMS_JSON = Path("data/forms.json")
INDEX_REPORT = Path("data/index_report.json")

# --- TUNING ---
BATCH_SIZE = 64          # safe size (32–128 recommended)
MAX_CHUNKS = None        # set to int for testing, e.g. 5000


def load_raw() -> List[Dict[str, Any]]:
    items = []
    with RAW_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def classify_section(source: str, title: str, text: str) -> str:
    s = (source or "").lower()
    t = (title or "").lower()
    x = (text or "").lower()

    if "/loans" in s or "loan" in t:
        return "Loan"
    if "claim" in s or "claims" in s or "claim" in t or "claims" in t or "claim" in x[:800]:
        return "Claims"
    if "contact" in s or "contact" in t or "address" in x[:800] or "phone" in x[:800]:
        return "Contact"

    insurance_terms = ["insurance", "policy", "premium", "coverage", "insured", "insurer"]
    if any(w in t for w in insurance_terms) or any(w in x[:900] for w in insurance_terms):
        return "Insurance"

    return "Other"


def build():
    if not RAW_JSONL.exists():
        raise RuntimeError("data/raw.jsonl missing. Run scripts/crawl_bil.py first.")

    raw = load_raw()
    documents: List[Document] = []

    # --- Build documents + chunks ---
    for item in raw:
        text = clean_text(item.get("text", ""))
        if not text:
            continue

        source = item.get("source", "")
        title = item.get("title", "")
        doc_type = item.get("type", "page")

        section = classify_section(source, title, text)
        chunks = chunk_text(text)

        for idx, ck in enumerate(chunks):
            ck = clean_text(ck)
            if not ck:
                continue

            documents.append(
                Document(
                    page_content=ck,
                    metadata={
                        "source": source,
                        "title": title,
                        "type": doc_type,
                        "section_type": section,
                        "chunk_index": idx,
                    },
                )
            )

            if MAX_CHUNKS and len(documents) >= MAX_CHUNKS:
                break
        if MAX_CHUNKS and len(documents) >= MAX_CHUNKS:
            break

    if not documents:
        raise RuntimeError("No chunks created. Crawl may be empty.")

    print(f"🔹 Total chunks to embed: {len(documents)}")

    embeddings = OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
        api_key=os.getenv("OPENAI_API_KEY", ""),
    )

    # --- Build FAISS incrementally ---
    vectorstore = None

    for i in tqdm(range(0, len(documents), BATCH_SIZE), desc="Embedding batches"):
        batch_docs = documents[i:i + BATCH_SIZE]

        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch_docs, embeddings)
        else:
            vectorstore.add_documents(batch_docs)

    VS_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VS_DIR))

    # --- Index report ---
    forms_count = 0
    if FORMS_JSON.exists():
        try:
            forms_count = len(json.loads(FORMS_JSON.read_text(encoding="utf-8")))
        except Exception:
            pass

    report = {
        "indexed_pages_and_docs": len(raw),
        "indexed_chunks": len(documents),
        "batch_size": BATCH_SIZE,
        "vectorstore_dir": str(VS_DIR),
        "forms_catalog_count": forms_count,
        "notes": [
            "Embeddings generated in batches to avoid OpenAI token limits.",
            "Each chunk is tagged with section_type metadata.",
        ],
    }

    INDEX_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("✅ Index build complete")
    print(" - FAISS directory:", VS_DIR)
    print(" - Index report:", INDEX_REPORT)


if __name__ == "__main__":
    build()
