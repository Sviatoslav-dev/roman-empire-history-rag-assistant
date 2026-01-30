import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Iterable, List, Literal, Optional


# In-memory single-chat store
_IN_MEMORY_CHAT: Optional[dict] = None

Role = Literal["user", "assistant"]


@dataclass
class ChatMessage:
    role: Role
    content: str
    images: Optional[list[dict]] = None  # e.g., [{"url": str, "local_path": str, "caption": str}]
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class Chat:
    id: str
    title: str
    messages: List[ChatMessage]
    updated_at: str


# --- File backend helpers ---


def _serialize_chat(chat: Chat) -> dict:
    return {
        "id": chat.id,
        "title": chat.title,
        "updated_at": chat.updated_at,
        "messages": [asdict(m) for m in chat.messages],
    }


def _deserialize_chat(raw: dict) -> Chat:
    messages = [ChatMessage(**m) for m in raw.get("messages", [])]
    return Chat(id=raw.get("id", ""), title=raw.get("title", "Chat"), messages=messages, updated_at=raw.get("updated_at", ""))


# --- Public API (single-chat semantics) ---

def create_chat(title: str) -> Chat:
    """Create a new single chat (overwrites any existing single chat)."""
    chat_id = uuid.uuid4().hex
    now = datetime.utcnow().isoformat()
    chat = Chat(id=chat_id, title=title, messages=[], updated_at=now)
    save_chat(chat)
    return chat


def save_chat(chat: Chat) -> None:
    """Persist the single Chat according to the selected backend."""
    global _IN_MEMORY_CHAT

    # Keep chat only in memory for the current process; do not write to disk or session.
    _IN_MEMORY_CHAT = _serialize_chat(chat)

def load_chat() -> Optional[Chat]:
    """Load the single chat. chat_id is ignored for single-chat mode."""

    if _IN_MEMORY_CHAT is None:
        return None
    return _deserialize_chat(_IN_MEMORY_CHAT)


def add_message(chat: Chat, role: Role, content: str, images: Optional[Iterable[dict]] = None) -> Chat:
    msg = ChatMessage(role=role, content=content, images=list(images) if images else None)
    chat.messages.append(msg)
    chat.updated_at = datetime.utcnow().isoformat()
    save_chat(chat)
    return chat
