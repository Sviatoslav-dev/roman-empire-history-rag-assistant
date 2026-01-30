"""Streamlit UI for the Roman Empire RAG assistant.

Features:
- Chat messages with optional image uploads
- Assistant replies can render retrieved images with pinned source links
- Single persistent chat (no multiple chats list)
- Local JSON-backed chat history
"""
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import io
import urllib.request

import streamlit as st

from rag.langchain_pipeline import RAGPipeline
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


def _local_image_src(img: dict) -> Optional[str]:
    """Return a resolved local image path if it exists, otherwise None.

    This helper only considers local_path candidates and never falls back to remote URL.
    """
    local = img.get("local_path")
    if not local:
        return None
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
    return None


def _render_message(msg: history_store.ChatMessage, msg_index: int | None = None):
    with st.chat_message(msg.role):
        if msg.images:
            # Show images above the text: render up to 5 images in a single horizontal row,
            # all the same size to avoid duplication and visual hierarchy.
            images = [img for img in msg.images if img]
            if images:
                # Limit to 5 images horizontally
                display_images = images[:5]
                cols = st.columns(min(5, len(display_images)))
                for col, img in zip(cols, display_images):
                    # Prefer a local image path when available
                    local_src = _local_image_src(img)
                    remote_src = None
                    if local_src:
                        resolved_src = local_src
                    else:
                        # fall back to remote url if no local file exists
                        remote_src = _image_src(img)
                        resolved_src = str(remote_src) if remote_src else None

                    if not resolved_src:
                        continue

                    with col:
                        caption = img.get("caption") or None

                        # If resolved_src is a local file, render it directly and skip remote fetch.
                        if local_src:
                            if caption:
                                def open_caption(key):
                                    st.session_state[key] = True

                                def close_caption(key):
                                    st.session_state[key] = False

                                key = f"caption_open_{hash(resolved_src)}"
                                st.session_state.setdefault(key, False)

                                st.image(resolved_src, width=200)

                                if len(caption) > 120:
                                    if st.session_state[key]:
                                        st.caption(caption)
                                        st.button(
                                            "▴ Collapse",
                                            key=f"btn_close_{key}",
                                            on_click=close_caption,
                                            args=(key,),
                                            type="tertiary",
                                        )
                                    else:
                                        st.caption(caption[:120] + "…")
                                        st.button(
                                            "▾ Show more",
                                            key=f"btn_open_{key}",
                                            on_click=open_caption,
                                            args=(key,),
                                            type="tertiary",
                                        )
                                else:
                                    st.caption(caption)
                            else:
                                st.image(resolved_src, width=200)
                        else:
                            # Try to download remote images server-side and pass bytes to st.image. This
                            # avoids client-side hotlinking/cors issues that often produce broken images.
                            rendered = False
                            try:
                                if resolved_src.startswith("http://") or resolved_src.startswith("https://"):
                                    with urllib.request.urlopen(resolved_src, timeout=6) as resp:
                                        img_bytes = resp.read()
                                    if caption:
                                        st.image(io.BytesIO(img_bytes), caption=caption, width=200)
                                    else:
                                        st.image(io.BytesIO(img_bytes), width=200)
                                    rendered = True
                            except Exception:
                                rendered = False

                            if not rendered:
                                # Fallback to letting Streamlit load the src directly (may work for some hosts).
                                if caption:
                                    st.image(resolved_src, caption=caption, width=200)
                                else:
                                    st.image(resolved_src, width=200)

                        # Add a small 'Open source' link under the image when a page URL is available.
                        page_url = img.get("url")
                        if page_url and (str(page_url).startswith("http://") or str(page_url).startswith("https://")):
                            link_html = f'<div style="text-align:center"><a href="{page_url}" target="_blank" rel="noopener noreferrer">Open source</a></div>'
                            st.markdown(link_html, unsafe_allow_html=True)

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
    """Return the single persistent chat, creating it if missing."""
    if chat := st.session_state.get("active_chat"):
        return chat
    # Attempt to load existing chat from store
    chat = history_store.load_chat()
    if not chat:
        chat = history_store.create_chat(title or f"Chat {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
    st.session_state["active_chat"] = chat
    return chat


def _build_chat_history_for_pipeline(messages: list[history_store.ChatMessage]) -> list[tuple[str, str]]:
    """Convert UI ChatMessages to (role, content) tuples for query rewriting."""
    history: list[tuple[str, str]] = []
    for m in messages:
        if m.content:
            history.append((m.role, m.content))
    return history


# ---- Page layout ----n
st.set_page_config(page_title="Roman Empire Assistant", page_icon="🏛️")
st.title("🏛️ Roman Empire Assistant")
st.caption("Ask questions about the Roman Empire. Attach an image if helpful; answers may include images with source links.")

pipeline = get_pipeline()

# # Sidebar: simple control for clearing the chat
# with st.sidebar:
#     st.header("Chat")
#     if st.button("Clear chat", use_container_width=True):
#         history_store.delete_chat()
#         st.session_state.pop("active_chat", None)
#         st.rerun()

# Main chat area
active_chat: Optional[history_store.Chat] = st.session_state.get("active_chat")
if active_chat:
    _render_history(active_chat.messages)

# Input controls
# Use a placeholder so we can remove/hide the uploader while the assistant is thinking.
if "uploader_visible" not in st.session_state:
    st.session_state["uploader_visible"] = True

uploader_placeholder = st.empty()
uploaded_image = None
if st.session_state.get("uploader_visible", True):
    uploaded_image = uploader_placeholder.file_uploader(
        "Optional: add an image",
        type=["png", "jpg", "jpeg", "webp"],
        label_visibility="collapsed",
        key="attached_image",
    )

user_text = st.chat_input("Ask about the Roman Empire…")

if user_text:
    chat = _ensure_active_chat()

    # Persist the uploaded image to a temp file (if any) before saving the user message
    query_image_path = _save_uploaded_image(uploaded_image)

    # Build image payload for the user's message so the uploaded image shows immediately.
    user_images_payload = None
    if query_image_path:
        user_images_payload = [{
            "url": None,
            "local_path": query_image_path,
            "caption": None,
            "page_title": None,
            "score": None,
        }]

    # Persist user message (include uploaded image payload so it renders inline)
    chat = history_store.add_message(chat, role="user", content=user_text, images=user_images_payload)
    st.session_state["active_chat"] = chat
    # Immediately render the saved user message (with image if present) so it's visible while processing.
    _render_history([chat.messages[-1]])

    # Hide the uploader while we process the request to prevent user interactions
    # and keep the UI focused. We remove the widget from the UI immediately by
    # clearing the placeholder; also mark session state so the uploader remains
    # hidden after rerun until we explicitly show it again.
    try:
        uploader_placeholder.empty()
    except Exception:
        # placeholder may not exist in some runs; ignore
        pass
    st.session_state["uploader_visible"] = False

    # Ensure uploader is restored even if generation fails by using try/finally
    retrieved_ctx = None
    answer = None
    try:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chat_history = _build_chat_history_for_pipeline(chat.messages)
                answer, retrieved_ctx = pipeline.generate_answer(
                    question=user_text,
                    query_image_path=query_image_path,
                    chat_history=chat_history,
                    top_k_text=5,  # можна змінити
                    top_k_images=3,  # можна змінити
                )

        # Save assistant message with images
        images_payload = _prepare_image_payload(retrieved_ctx.images if retrieved_ctx else [])
        chat = history_store.add_message(chat, role="assistant", content=answer or "(No answer)", images=images_payload)
        st.session_state["active_chat"] = chat

        # Render freshly added assistant message (user message already visible on rerun)
        _render_message(history_store.ChatMessage(role="assistant", content=answer or "(No answer)", images=images_payload))

    finally:
        # After processing (successful or not), make the uploader visible again for next input.
        st.session_state["uploader_visible"] = True
        # Optionally clear the attached file from session state so uploader resets.
        try:
            # Remove any lingering attached_image_* keys from session_state.
            for k in list(st.session_state.keys()):
                if str(k).startswith("attached_image_"):
                    try:
                        st.session_state.pop(k, None)
                    except Exception:
                        pass
            old_idx = st.session_state.get("uploader_key_index", 0)
            new_idx = old_idx + 1
            st.session_state["uploader_key_index"] = new_idx
        except Exception:
            st.session_state["uploader_key_index"] = 0
        # After processing (successful or not), make the uploader visible again for next input.
        st.session_state["uploader_visible"] = True
        # Force rerun so state reflects new messages and uploader visibility
        st.rerun()
