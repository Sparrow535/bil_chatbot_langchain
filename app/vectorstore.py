from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from app.config import settings

VS_DIR = Path("data/faiss_index")

def load_vectorstore() -> FAISS:
    if not VS_DIR.exists():
        raise RuntimeError("Vector store not found. Run scripts/build_index.py first.")
    embeddings = OpenAIEmbeddings(model=settings.embed_model, api_key=settings.openai_api_key)
    return FAISS.load_local(
        folder_path=str(VS_DIR),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )
