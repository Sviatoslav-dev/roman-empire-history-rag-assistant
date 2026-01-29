"""Streamlit UI for the Roman Empire RAG assistant.

Features:
- Chat messages with optional image uploads
- Assistant replies can render retrieved images with pinned source links
- Sidebar chat list with load/delete/new chat controls
- Local JSON-backed chat history
"""
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import streamlit as st

from rag.pipeline import RAGPipeline
from rag.rag_model import RetrievedImage
from ui import history_store

# Project root used to resolve relative local image paths for display
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@st.cache_resource(show_spinner=False)
def get_pipeline() -> RAGPipeline:
    """Cache pipeline so we do not re-initialize heavy models between reruns."""
    return RAGPipeline()


def _save_uploaded_image(file) -> Optional[str]:
    if file is None:
        return None
    suffix = Path(file.name).suffix or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.getbuffer())
        return tmp.name


def _image_src(img: dict) -> Optional[str]:
    """Return a usable image src. Prefer existing local_path (resolved), else url."""
    local = img.get("local_path")
    if local:
        base = Path(local)
        candidates = []
        if base.is_absolute():
            candidates.append(base)
        else:
            candidates.extend([
                PROJECT_ROOT / base,
                PROJECT_ROOT / "data_ingestion" / base,
                Path.cwd() / base,
            ])
        for path in candidates:
            if path.exists():
                return str(path)
    url = img.get("url")
    return url if url else None


def _render_message(msg: history_store.ChatMessage, msg_index: int | None = None):
    with st.chat_message(msg.role):
        if msg.images:
            # Show images above the text: hero + smaller thumbnails
            images = [img for img in msg.images if img]
            if msg.role == "assistant" and len(images) > 1:
                # Show the first image as the hero (no visible slider)
                hero = images[0]
                hero_src = _image_src(hero)
                if hero_src:
                    hero_caption = hero.get("caption") or hero.get("page_title")
                    source = hero.get("url") or hero.get("local_path")
                    st.image(hero_src, caption=hero_caption or "Image", width=480)
                    if source:
                        st.caption(f"Source: {source}")

                cols = st.columns(min(5, len(images)))
                for col, img in zip(cols, images):
                    thumb_src = _image_src(img)
                    if thumb_src:
                        with col:
                            thumb_cap = img.get("caption") or img.get("page_title") or ""
                            st.image(thumb_src, caption=thumb_cap, width=140)
            else:
                for img in images:
                    src = _image_src(img)
                    caption_parts = []
                    if img.get("caption"):
                        caption_parts.append(img["caption"])
                    if src:
                        caption_parts.append(f"Source: {src}")
                    caption = " | ".join(caption_parts) if caption_parts else None
                    if src:
                        st.image(src, caption=caption, width=480)

        st.markdown(msg.content)


def _render_history(messages: list[history_store.ChatMessage]):
    for idx, msg in enumerate(messages):
        # Attach index for unique widget keys where needed
        _render_message(msg, msg_index=idx)


def _prepare_image_payload(images: list[RetrievedImage]) -> list[dict]:
    payload = []
    for img in images:
        payload.append({
            "url": img.url,
            "local_path": img.local_path,
            "caption": img.caption,
            "page_title": img.page_title,
            "score": img.score,
        })
    return payload


def _ensure_active_chat(title: str | None = None) -> history_store.Chat:
    if chat := st.session_state.get("active_chat"):
        return chat
    chat = history_store.create_chat(title or f"Chat {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
    st.session_state["active_chat_id"] = chat.id
    st.session_state["active_chat"] = chat
    return chat


def _load_chat(chat_id: str):
    chat = history_store.load_chat(chat_id)
    st.session_state["active_chat_id"] = chat_id
    st.session_state["active_chat"] = chat


def _build_chat_history_for_pipeline(messages: list[history_store.ChatMessage]) -> list[tuple[str, str]]:
    """Convert UI ChatMessages to (role, content) tuples for query rewriting."""
    history: list[tuple[str, str]] = []
    for m in messages:
        if m.content:
            history.append((m.role, m.content))
    return history


# ---- Page layout ----
st.set_page_config(page_title="Roman Empire RAG Assistant", page_icon="🏛️")
st.title("🏛️ Roman Empire RAG Assistant")
st.caption("Ask questions about the Roman Empire. Attach an image if helpful; answers may include images with source links.")

pipeline = get_pipeline()

# Sidebar: chats management
with st.sidebar:
    st.header("Chats")

    # keep track of whether the "New chat" form is visible
    if "show_new_chat" not in st.session_state:
        st.session_state["show_new_chat"] = False

    # Top-level "New chat" button (like ChatGPT)
    if st.button("New chat", use_container_width=True, key="new_chat_button"):
        # Show the new-chat form and prefill the name with a timestamped default
        st.session_state["show_new_chat"] = True
        st.session_state["new_chat_name"] = f"Chat {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}"

    # If user clicked New chat, show a compact form to enter the name + Create/Cancel
    if st.session_state.get("show_new_chat"):
        new_title = st.text_input("Chat name", key="new_chat_name")
        cols = st.columns([3, 1])
        with cols[0]:
            if st.button("Create", use_container_width=True, key="create_chat_button"):
                title = new_title.strip() or f"Chat {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}"
                chat = history_store.create_chat(title)
                st.session_state["active_chat_id"] = chat.id
                st.session_state["active_chat"] = chat
                # Clear the temporary name and hide form
                st.session_state.pop("new_chat_name", None)
                st.session_state["show_new_chat"] = False
                st.rerun()
        with cols[1]:
            if st.button("Cancel", use_container_width=True, key="cancel_new_chat_button"):
                # Hide form and clear the temporary name
                st.session_state["show_new_chat"] = False
                st.session_state.pop("new_chat_name", None)

    # List chats as a vertical list of buttons (current chat is prefixed with an arrow)
    chats = history_store.list_chats()
    if chats:
        for c in chats:
            is_active = st.session_state.get("active_chat_id") == c.id
            label = f"{'➤ ' if is_active else ''}{c.title}"
            if st.button(label, key=f"open_chat_{c.id}", use_container_width=True):
                _load_chat(c.id)
                st.rerun()
    else:
        st.info("No chats yet — create one below.")

    # Delete current chat (kept below the list)
    if st.session_state.get("active_chat_id"):
        if st.button("Delete current chat", use_container_width=True):
            history_store.delete_chat(st.session_state["active_chat_id"])
            st.session_state.pop("active_chat_id", None)
            st.session_state.pop("active_chat", None)
            st.rerun()

# Main chat area
active_chat: Optional[history_store.Chat] = st.session_state.get("active_chat")
if active_chat:
    _render_history(active_chat.messages)
else:
    st.info("Start a new chat or select one from the sidebar.")

# Input controls
uploaded_image = st.file_uploader("Optional: add an image", type=["png", "jpg", "jpeg", "webp"], label_visibility="collapsed")
user_text = st.chat_input("Ask about the Roman Empire…")

if user_text:
    chat = _ensure_active_chat()

    # Persist user message
    chat = history_store.add_message(chat, role="user", content=user_text)
    st.session_state["active_chat"] = chat

    query_image_path = _save_uploaded_image(uploaded_image)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            chat_history = _build_chat_history_for_pipeline(chat.messages)
            answer, retrieved_ctx = pipeline.generate_answer(
                question=user_text,
                query_image_path=query_image_path,
                chat_history=chat_history,
            )

    # Save assistant message with images
    images_payload = _prepare_image_payload(retrieved_ctx.images)
    chat = history_store.add_message(chat, role="assistant", content=answer, images=images_payload)
    st.session_state["active_chat"] = chat

    # Render freshly added assistant message (user message already visible on rerun)
    _render_message(history_store.ChatMessage(role="assistant", content=answer, images=images_payload))

    # Force rerun so state reflects new messages and uploader resets
    st.rerun()
