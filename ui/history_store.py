"""Lightweight local chat history store for the Streamlit UI."""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Literal, Optional

STORE_DIR = Path.home() / ".cache" / "roman_rag_ui" / "chats"
STORE_DIR.mkdir(parents=True, exist_ok=True)

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


def _chat_path(chat_id: str) -> Path:
    return STORE_DIR / f"{chat_id}.json"


def create_chat(title: str) -> Chat:
    chat_id = uuid.uuid4().hex
    now = datetime.utcnow().isoformat()
    chat = Chat(id=chat_id, title=title, messages=[], updated_at=now)
    save_chat(chat)
    return chat


def save_chat(chat: Chat) -> None:
    path = _chat_path(chat.id)
    data = {
        "id": chat.id,
        "title": chat.title,
        "updated_at": chat.updated_at,
        "messages": [asdict(m) for m in chat.messages],
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def load_chat(chat_id: str) -> Optional[Chat]:
    path = _chat_path(chat_id)
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    messages = [ChatMessage(**m) for m in raw.get("messages", [])]
    return Chat(id=raw["id"], title=raw.get("title", "Untitled"), messages=messages, updated_at=raw.get("updated_at", ""))


def list_chats() -> list[Chat]:
    chats: list[Chat] = []
    for file in sorted(STORE_DIR.glob("*.json")):
        loaded = load_chat(file.stem)
        if loaded:
            chats.append(loaded)
    # Sort by updated_at desc
    chats.sort(key=lambda c: c.updated_at, reverse=True)
    return chats


def delete_chat(chat_id: str) -> None:
    path = _chat_path(chat_id)
    if path.exists():
        path.unlink()


def add_message(chat: Chat, role: Role, content: str, images: Optional[Iterable[dict]] = None) -> Chat:
    msg = ChatMessage(role=role, content=content, images=list(images) if images else None)
    chat.messages.append(msg)
    chat.updated_at = datetime.utcnow().isoformat()
    save_chat(chat)
    return chat
