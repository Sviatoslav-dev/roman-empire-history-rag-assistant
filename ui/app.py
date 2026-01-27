"""Streamlit UI for the Roman Empire RAG assistant.

Features:
- Chat messages with optional image uploads
- Assistant replies can render retrieved images with pinned source links
- Sidebar chat list with load/delete/new chat controls
- Local JSON-backed chat history
"""
import tempfile
from datetime import datetime
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
                slider_suffix = msg.created_at.replace(":", "-") if msg.created_at else ""
                slider_key = f"hero_idx_{msg_index}_{slider_suffix}"
                hero_idx = st.slider("Preview image", 0, len(images) - 1, 0, key=slider_key)
                hero = images[hero_idx] if 0 <= hero_idx < len(images) else None
                if hero:
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
    chat = history_store.create_chat(title or f"Chat {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
    st.session_state["active_chat_id"] = chat.id
    st.session_state["active_chat"] = chat
    return chat


def _load_chat(chat_id: str):
    chat = history_store.load_chat(chat_id)
    st.session_state["active_chat_id"] = chat_id
    st.session_state["active_chat"] = chat


# ---- Page layout ----
st.set_page_config(page_title="Roman Empire RAG Assistant", page_icon="🏛️")
st.title("🏛️ Roman Empire RAG Assistant")
st.caption("Ask questions about the Roman Empire. Attach an image if helpful; answers may include images with source links.")

pipeline = get_pipeline()

# Sidebar: chats management
with st.sidebar:
    st.header("Chats")
    chats = history_store.list_chats()
    chat_options = {c.title: c.id for c in chats}
    selected_title = None
    if chats:
        selected_title = st.selectbox("Open chat", options=list(chat_options.keys()), index=0)
    else:
        st.caption("No chats yet — create one below.")

    if selected_title:
        _load_chat(chat_options[selected_title])

    with st.expander("New chat"):
        new_title = st.text_input("Title", value="New chat")
        if st.button("Create", use_container_width=True, type="primary"):
            chat = history_store.create_chat(new_title or "Untitled chat")
            st.session_state["active_chat_id"] = chat.id
            st.session_state["active_chat"] = chat
            st.rerun()

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
            answer, retrieved_ctx = pipeline.generate_answer(question=user_text, query_image_path=query_image_path)

    # Save assistant message with images
    images_payload = _prepare_image_payload(retrieved_ctx.images)
    chat = history_store.add_message(chat, role="assistant", content=answer, images=images_payload)
    st.session_state["active_chat"] = chat

    # Render freshly added assistant message (user message already visible on rerun)
    _render_message(history_store.ChatMessage(role="assistant", content=answer, images=images_payload))

    # Force rerun so state reflects new messages and uploader resets
    st.rerun()
