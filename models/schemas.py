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


class Chat(BaseModel):
    """Chat model."""
    id: str
    title: Optional[str] = None
    messages: List[ChatMessage] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


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


class CreateChatRequest(BaseModel):
    """Request to create a new chat."""
    title: Optional[str] = None


class MessageRequest(BaseModel):
    """Request to send a message."""
    content: str
    image_url: Optional[str] = None  # For future image query support


class MessageResponse(BaseModel):
    """Response to a message."""
    answer: str
    context: RetrievedContext
    chat: Chat


class ChatListResponse(BaseModel):
    """Response with list of chats."""
    chats: List[Chat]
