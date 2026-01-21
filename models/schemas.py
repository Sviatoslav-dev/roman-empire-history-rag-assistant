"""Pydantic schemas for API requests and responses."""
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from enum import Enum


class MessageRole(str, Enum):
    """Message role types."""
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    """Chat message model."""
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)


class RetrievedImage(BaseModel):
    """Retrieved image metadata."""
    id: str
    url: Optional[str] = None
    local_path: Optional[str] = None
    caption: Optional[str] = None
    page_title: Optional[str] = None
    score: float


class RetrievedContext(BaseModel):
    """Retrieved context for RAG."""
    text_chunks: List[str] = Field(default_factory=list)
    images: List[RetrievedImage] = Field(default_factory=list)
