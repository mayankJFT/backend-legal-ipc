from pydantic import BaseModel, Field
from typing import Dict, List, Optional

# === API Models ===
class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[float] = None

class QueryRequest(BaseModel):
    query: str
    model_name: str = "gpt-4o-mini"  # Fastest model as default
    conversation_id: Optional[str] = None
    strategy: str = "simple"  # Default to faster strategy
    max_tokens: int = 1500  # Reduced default for speed
    temperature: float = 0.1  # Lower for faster processing
    stream: bool = True  # Enable streaming by default for faster perceived response
    include_history: bool = False  # Disabled by default for speed

class ResponseMetadata(BaseModel):
    model: str
    strategy: str
    chunks_retrieved: int
    tokens_used: int
    processing_time: float
    conversation_id: str

class QueryResponse(BaseModel):
    response: str
    metadata: ResponseMetadata
    context_sources: List[Dict[str, str]] = []

class HealthResponse(BaseModel):
    status: str
    version: str
    available_models: List[str]