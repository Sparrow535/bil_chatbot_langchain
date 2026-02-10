from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict, Any

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: Optional[str] = None
    history: List[ChatMessage] = Field(default_factory=list)

class DownloadItem(BaseModel):
    title: str
    url: str

class BotResponse(BaseModel):
    intent: Literal["bil_query", "form_request", "unrelated", "not_found"]
    answer: str
    answer_md: str = ""
    sources: List[str] = Field(default_factory=list)
    downloads: List[DownloadItem] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"] = "low"
    debug: Optional[Dict[str, Any]] = None
    client_delay_ms: Optional[int] = None

class TranscribeResponse(BaseModel):
    text: str
