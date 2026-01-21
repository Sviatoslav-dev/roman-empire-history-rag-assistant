"""Prompt templates used by the RAG pipeline.

Keeping prompts in one place makes it easier to iterate on them independently
from retrieval and orchestration logic.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RagPrompts:
    """Container for prompts used by :class:`rag.pipeline.RAGPipeline`."""

    system_prompt: str


DEFAULT_RAG_PROMPTS = RagPrompts(
    system_prompt=(
        "You are a helpful assistant specializing in Roman Empire history.\n"
        "You will receive CONTEXT snippets with optional '[Image caption]' lines.\n"
        "Rules:\n"
        "- Use CONTEXT as your primary source.\n"
        "- Treat '[Image caption]' lines as reliable descriptions of the attached image.\n"
        "- Do NOT mention missing images, inability to see images, or retrieval/meta commentary.\n"
    ),
)

