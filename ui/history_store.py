"""Lightweight chat history store with selectable backends.

Supports two backends:
- file (default): stores chats as JSON files under ~/.cache/roman_rag_ui/chats (original behaviour)
- session: stores chats in the Streamlit `st.session_state` (per-user session in a deployed Streamlit app)

Set the backend with the environment variable `HISTORY_STORE_BACKEND=session` to enable session storage.
"""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Literal, Optional

# Backend selection via env var. Default is 'file' (original behaviour).
_BACKEND = os.getenv("HISTORY_STORE_BACKEND", "file").lower()

# File-store directory (only used when backend == 'file'). Created lazily.
_STORE_DIR = Path.home() / ".cache" / "roman_rag_ui" / "chats"

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

def _ensure_store_dir() -> None:
    if _BACKEND != "file":
        return
    _STORE_DIR.mkdir(parents=True, exist_ok=True)


def _chat_path(chat_id: str) -> Path:
    return _STORE_DIR / f"{chat_id}.json"


def _serialize_chat(chat: Chat) -> dict:
    return {
        "id": chat.id,
        "title": chat.title,
        "updated_at": chat.updated_at,
        "messages": [asdict(m) for m in chat.messages],
    }


def _deserialize_chat(raw: dict) -> Chat:
    messages = [ChatMessage(**m) for m in raw.get("messages", [])]
    return Chat(id=raw["id"], title=raw.get("title", "Untitled"), messages=messages, updated_at=raw.get("updated_at", ""))


# --- Session backend helpers (Streamlit) ---

def _get_session_store() -> dict:
    """Return the in-memory dict used to store chats in the Streamlit session.

    Lazily imports Streamlit and raises a helpful error if not available.
    """
    try:
        import streamlit as st
    except Exception as e:  # pragma: no cover - environment dependent
        raise RuntimeError("Streamlit is not available: cannot use 'session' history backend") from e

    # Use a dedicated key in session_state to avoid collisions
    if "_history_store_chats" not in st.session_state:
        st.session_state["_history_store_chats"] = {}
    return st.session_state["_history_store_chats"]


# --- Public API (preserve original function signatures) ---

def create_chat(title: str) -> Chat:
    chat_id = uuid.uuid4().hex
    now = datetime.utcnow().isoformat()
    chat = Chat(id=chat_id, title=title, messages=[], updated_at=now)

    save_chat(chat)
    return chat


def save_chat(chat: Chat) -> None:
    """Persist a Chat according to the selected backend."""
    if _BACKEND == "session":
        store = _get_session_store()
        store[chat.id] = _serialize_chat(chat)
        return

    # file backend
    _ensure_store_dir()
    path = _chat_path(chat.id)
    path.write_text(json.dumps(_serialize_chat(chat), ensure_ascii=False, indent=2))


def load_chat(chat_id: str) -> Optional[Chat]:
    if _BACKEND == "session":
        store = _get_session_store()
        raw = store.get(chat_id)
        if not raw:
            return None
        return _deserialize_chat(raw)

    # file backend
    _ensure_store_dir()
    path = _chat_path(chat_id)
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    return _deserialize_chat(raw)


def list_chats() -> list[Chat]:
    """Return all chats sorted by updated_at descending."""
    chats: list[Chat] = []
    if _BACKEND == "session":
        store = _get_session_store()
        for raw in store.values():
            chats.append(_deserialize_chat(raw))
        chats.sort(key=lambda c: c.updated_at, reverse=True)
        return chats

    # file backend
    _ensure_store_dir()
    for file in sorted(_STORE_DIR.glob("*.json")):
        loaded = load_chat(file.stem)
        if loaded:
            chats.append(loaded)
    chats.sort(key=lambda c: c.updated_at, reverse=True)
    return chats


def delete_chat(chat_id: str) -> None:
    if _BACKEND == "session":
        store = _get_session_store()
        if chat_id in store:
            del store[chat_id]
        return

    # file backend
    _ensure_store_dir()
    path = _chat_path(chat_id)
    if path.exists():
        path.unlink()


def add_message(chat: Chat, role: Role, content: str, images: Optional[Iterable[dict]] = None) -> Chat:
    msg = ChatMessage(role=role, content=content, images=list(images) if images else None)
    chat.messages.append(msg)
    chat.updated_at = datetime.utcnow().isoformat()
    save_chat(chat)
    return chat
