"""RAG pipeline for question answering."""
import argparse
import os
from typing import Optional, Tuple

from dotenv import load_dotenv

from rag.langchain_adapter import build_langchain_rag_chain
from rag.rag_model import RetrievedImage, RetrievedContext
from rag.retriever import QdrantRetriever
from rag.prompts import DEFAULT_RAG_PROMPTS, RagPrompts
from rag.llm_client import LLMClient
from logger import get_logger

load_dotenv()

logger = get_logger(__name__)


class RAGPipeline:
    """High-level RAG pipeline that retrieves context and generates an answer."""

    def __init__(self, *, prompts: RagPrompts = DEFAULT_RAG_PROMPTS):
        self.retriever = QdrantRetriever()
        self.llm = LLMClient()
        self.prompts = prompts

    # ---- Small helpers ----
    @staticmethod
    def _build_user_prompt(context_text: str, question: str) -> str:
        return f"CONTEXT:\n{context_text}\n\nQUESTION: {question}\nANSWER:"

    def _retrieve_text(self, question: str, top_k_text: int) -> tuple[list[str], list[int]]:
        results = self.retriever.search_text(question, top_k=top_k_text)
        texts = [t for (t, _s, _m, _cid) in results]
        ids = [cid for (_t, _s, _m, cid) in results]
        return texts, ids

    def _retrieve_images_with_links(
        self,
        query_image_path: str,
        top_k_images: int,
        *,
        question: str,
        top_k_links: int = 3,
    ) -> tuple[list[RetrievedImage], list[int], dict[int, list[str]]]:
        """Retrieve images, then top linkages (caption-embedded) closest to the question."""
        image_results = self.retriever.search_images(query_image_path, top_k=top_k_images)
        image_ids = [payload.get("image_id") for payload, _s in image_results if payload.get("image_id") is not None]

        # Rank links by caption similarity to the question, limited to the retrieved images
        link_results = self.retriever.search_links_by_text(question, image_ids=image_ids, top_k=top_k_links)

        image_linked_chunk_ids: list[int] = []
        chunk_captions: dict[int, list[str]] = {}
        links_by_image: dict[int, list[dict]] = {}
        for link_payload, _score in link_results:
            cid = link_payload.get("text_chunk_id")
            img_id = link_payload.get("image_id")
            if cid is None or img_id is None:
                continue
            image_linked_chunk_ids.append(int(cid))
            cap = link_payload.get("caption")
            cap = cap.strip() if isinstance(cap, str) else ""
            if cap:
                chunk_captions.setdefault(int(cid), []).append(cap)
            links_by_image.setdefault(img_id, []).append(link_payload)

        retrieved_images: list[RetrievedImage] = []
        for i, (image_metadata, score) in enumerate(image_results):
            image_id = image_metadata.get("image_id")
            per_image_links = links_by_image.get(image_id) if image_id is not None else None
            per_image_captions = []
            if per_image_links:
                for l in per_image_links:
                    cap = l.get("caption")
                    cap = cap.strip() if isinstance(cap, str) else ""
                    if cap:
                        per_image_captions.append(cap)

            retrieved_images.append(
                RetrievedImage(
                    id=str(i),
                    url=image_metadata.get("image_url"),
                    local_path=image_metadata.get("image_path") or image_metadata.get("local_path"),
                    caption=(per_image_captions[0] if per_image_captions else None),
                    page_title=None,
                    score=float(score),
                )
            )

        return retrieved_images, image_linked_chunk_ids, chunk_captions

    def _collect_images_from_text_chunks(self, chunk_ids: list[int]) -> list[RetrievedImage]:
        """Fetch and deduplicate images linked to the given text chunk ids."""
        images_by_id: dict[str, RetrievedImage] = {}
        for cid in chunk_ids:
            for payload in self.retriever.get_images_by_text_chunk_id(cid):
                image_id = payload.get("image_id") or payload.get("id") or payload.get("image_path") or payload.get("image_url")
                if not image_id:
                    continue
                sid = str(image_id)
                if sid in images_by_id:
                    continue
                images_by_id[sid] = RetrievedImage(
                    id=sid,
                    url=payload.get("image_url"),
                    local_path=payload.get("image_path") or payload.get("local_path"),
                    caption=payload.get("caption"),
                    page_title=payload.get("page_title"),
                    score=0.0,
                )
        return list(images_by_id.values())

    @staticmethod
    def _merge_ranked_ids(primary: list[int], secondary: list[int], limit: int) -> list[int]:
        merged: list[int] = []
        seen: set[int] = set()
        for cid in primary:
            if cid not in seen:
                merged.append(cid)
                seen.add(cid)
        for cid in secondary:
            if cid not in seen:
                merged.append(cid)
                seen.add(cid)
        return merged[:limit]

    def _load_chunk_text(self, chunk_id: int, text_by_id: dict[int, str]) -> Optional[str]:
        if chunk_id in text_by_id:
            return text_by_id[chunk_id]
        res = self.retriever.get_text_chunk_by_id(chunk_id)
        if not res:
            return None
        text, _m = res
        return text

    @staticmethod
    def _attach_captions(chunk_text: str, captions: list[str]) -> str:
        if not captions:
            return chunk_text

        seen: set[str] = set()
        uniq_caps: list[str] = []
        for c in captions:
            if c not in seen:
                uniq_caps.append(c)
                seen.add(c)

        caps_text = "\n".join([f"[Image caption] {c}" for c in uniq_caps])
        return f"{caps_text}\n{chunk_text}"

    @staticmethod
    def _format_history(messages: list[tuple[str, str]], max_turns: int = 6) -> str:
        """Format recent chat history as plain text lines for query rewriting."""
        if not messages:
            return ""
        recent = messages[-max_turns:]
        lines = []
        for role, content in recent:
            role_label = "User" if role == "user" else "Assistant"
            lines.append(f"{role_label}: {content}")
        return "\n".join(lines)

    def _rewrite_question(self, question: str, history: list[tuple[str, str]] | None) -> str:
        """Rewrite the question using chat history to make it standalone."""
        if not history:
            return question
        history_text = self._format_history(history)
        user_prompt = self.prompts.query_rewrite_user_template.format(
            history=history_text,
            question=question,
        )
        rewritten = self.llm.generate(
            user_prompt,
            system_prompt=self.prompts.query_rewrite_system_prompt,
            max_new_tokens=128,
            temperature=0.2,
            top_p=0.9,
        )
        return rewritten.strip() or question

    def generate_answer(
        self,
        question: str,
        top_k_text: int = 5,
        top_k_images: int = 3,
        query_image_path: Optional[str] = None,
        chat_history: Optional[list[tuple[str, str]]] = None,
    ) -> Tuple[str, RetrievedContext]:
        """Question -> answer (+ retrieved context).

        Note: conversational history support is intentionally omitted for now.
        """

        rewritten_question = self._rewrite_question(question, chat_history)

        # --- A) Text retrieval always happens ---
        text_chunks_by_text, text_chunk_ids_by_text = self._retrieve_text(rewritten_question, top_k_text)

        # Text-only
        if not query_image_path:
            context_text = "\n\n".join(text_chunks_by_text)
            user_prompt = self._build_user_prompt(context_text, rewritten_question)
            logger.info("CONTEXT: ", context_text)
            logger.info("SYSTEM_PROMPT: ", self.prompts.system_prompt)
            logger.info("USER_PROMPT: ", user_prompt)
            answer = self.llm.generate(user_prompt, system_prompt=self.prompts.system_prompt)
            linked_images = self._collect_images_from_text_chunks(text_chunk_ids_by_text)
            return answer, RetrievedContext(text_chunks=text_chunks_by_text, images=linked_images)

        # --- B) Image retrieval + links ---
        retrieved_images, image_linked_chunk_ids, chunk_captions = self._retrieve_images_with_links(
            query_image_path=query_image_path,
            top_k_images=top_k_images,
            question=rewritten_question,
            top_k_links=3,
        )

        # --- C) Merge chunk ids ---
        merged_chunk_ids = self._merge_ranked_ids(text_chunk_ids_by_text, image_linked_chunk_ids, limit=top_k_text)

        # Reuse text chunks we already have
        text_by_id = {cid: txt for cid, txt in zip(text_chunk_ids_by_text, text_chunks_by_text)}

        merged_text_chunks: list[str] = []
        for cid in merged_chunk_ids:
            chunk_text = self._load_chunk_text(cid, text_by_id)
            if not chunk_text:
                continue
            merged_text_chunks.append(self._attach_captions(chunk_text, chunk_captions.get(cid, [])))

        context_text = "\n\n".join(merged_text_chunks)
        user_prompt = self._build_user_prompt(context_text, rewritten_question)

        logger.debug(
            "Built prompt (len=%s). text_chunks=%s; images=%s",
            len(user_prompt),
            len(merged_text_chunks),
            len(retrieved_images),
        )

        answer = self.llm.generate(user_prompt, system_prompt=self.prompts.system_prompt)
        return answer, RetrievedContext(text_chunks=merged_text_chunks, images=retrieved_images)


# ---- Simple runner to quickly try the pipeline locally ----
if __name__ == "__main__":
    # Minimal smoke run so you can execute: `python -m rag.pipeline`
    # or `python rag/pipeline.py` from the project root.
    parser = argparse.ArgumentParser(description="Run RAG pipeline demo")
    parser.add_argument(
        "--question", "-q",
        default=os.environ.get(
            "RAG_DEMO_QUESTION",
            "What can you say about this picture?"
        ),
        help="Question to ask the RAG pipeline",
    )
    parser.add_argument(
        "--image-path", "-i",
        default="../tests/data/images/Colosseum_in_Rome,_Italy_-_April_2007.jpg",
        help="Path to the query image (use --no-image to disable image retrieval)",
    )
    parser.add_argument("--no-image", action="store_true", help="Run without image retrieval")
    args = parser.parse_args()

    sample_question = args.question
    image_path = None if args.no_image else args.image_path
    logger.info("Running RAG pipeline demo")

    pipeline = RAGPipeline()
    answer, retrieved = pipeline.generate_answer(sample_question, query_image_path=image_path)

    # Optional: LangChain demo if desired
    if os.environ.get("RAG_USE_LANGCHAIN", "0") == "1":
        logger.info("Running LangChain RetrievalQA demo")
        lc_chain = build_langchain_rag_chain(top_k_text=5)
        lc_result = lc_chain.invoke({"query": sample_question})
        logger.info("LangChain answer: %s", lc_result.get("result"))

    logger.info("Question: %s", sample_question)
    logger.info("Answer: %s", answer)

    # Briefly summarize retrieved context
    logger.info("Retrieved context summary: text_chunks=%s images=%s", len(retrieved.text_chunks), len(retrieved.images))
    if retrieved.text_chunks:
        preview = (retrieved.text_chunks[0] or "").strip().replace("\n", " ")
        logger.debug("First chunk preview: %s", preview[:200])

    if retrieved.images:
        for i, img in enumerate(retrieved.images, 1):
            logger.debug(
                "Retrieved image[%s]: title=%s caption=%s score=%.3f path=%s",
                i,
                img.page_title or "n/a",
                (img.caption or "No caption")[:100],
                img.score,
                img.local_path or img.url or "n/a",
            )
