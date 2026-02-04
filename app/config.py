import os
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

class Settings(BaseModel):
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    bil_base_url: str = os.getenv("BIL_BASE_URL", "https://www.bil.bt")

    chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
    embed_model: str = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

    top_k: int = int(os.getenv("TOP_K", "5"))
    min_relevance: float = float(os.getenv("MIN_RELEVANCE", "0.20"))

    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))

settings = Settings()
