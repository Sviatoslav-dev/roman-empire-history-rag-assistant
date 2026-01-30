from typing import Optional, List, Tuple
from dotenv import load_dotenv

from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.chat_models.base import BaseChatModel
from pydantic import PrivateAttr

from rag.prompts import RagPrompts, DEFAULT_RAG_PROMPTS
from rag.rag_model import RetrievedImage, RetrievedContext
from rag.retriever import QdrantRetriever
from rag.llm_client import LLMClient
from logger import get_logger
from langchain.schema import ChatResult, ChatGeneration, AIMessage

load_dotenv()

logger = get_logger(__name__)


class LangChainLLMWrapper(BaseChatModel):
    """Wrapper for your existing LLMClient to work with LangChain"""
    _llm_client: LLMClient = PrivateAttr()

    def __init__(self, llm_client: LLMClient):
        super().__init__()
        self._llm_client = llm_client

    def _generate(self, messages: List[BaseMessage], **kwargs):
        # Extract system prompt if present
        system_prompt = ""
        user_prompt = ""

        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_prompt = msg.content
            elif isinstance(msg, HumanMessage):
                user_prompt = msg.content

        # Generate using your existing LLMClient
        answer = self._llm_client.generate(
            user_prompt,
            system_prompt=system_prompt,
            max_new_tokens=kwargs.get('max_new_tokens', 512),
            temperature=kwargs.get('temperature', 0.1),
            top_p=kwargs.get('top_p', 0.9),
        )


        # Return LangChain compatible response
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=answer))]
        )

    @property
    def _llm_type(self) -> str:
        return "llm_client_wrapper"


class RAGPipeline:
    """High-level RAG pipeline that retrieves context and generates an answer."""

    def __init__(self, *, prompts: RagPrompts = DEFAULT_RAG_PROMPTS):
        self.retriever = QdrantRetriever()
        self.llm = LLMClient()
        self.prompts = prompts

        # Create LangChain components
        self.langchain_llm = LangChainLLMWrapper(self.llm)
        self.output_parser = StrOutputParser()

        # Build the answer generation chain
        self._build_answer_chain()

    def _build_answer_chain(self):
        """Build LangChain chain for answer generation"""

        # Define the prompt template
        self.answer_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.prompts.system_prompt),
            HumanMessagePromptTemplate.from_template(
                "CONTEXT:\n{context_text}\n\nQUESTION: {question}\nANSWER:"
            )
        ])

        # Create the chain
        self.answer_chain = (
                RunnablePassthrough()
                | self.answer_prompt
                | self.langchain_llm
                | self.output_parser
        )

        # Build query rewrite chain
        self.query_rewrite_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.prompts.query_rewrite_system_prompt),
            HumanMessagePromptTemplate.from_template(self.prompts.query_rewrite_user_template)
        ])

        self.query_rewrite_chain = (
                self.query_rewrite_prompt
                | self.langchain_llm
                | self.output_parser
        )

    # ---- Small helpers (unchanged from original) ----
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
        allowed_image_ids: set[str] = set()
        for link_payload, _score in link_results:
            cid = link_payload.get("text_chunk_id")
            img_id = link_payload.get("image_id")
            if cid is None or img_id is None:
                continue
            image_linked_chunk_ids.append(int(cid))
            allowed_image_ids.add(img_id)
            cap = link_payload.get("caption")
            cap = cap.strip() if isinstance(cap, str) else ""
            if cap:
                chunk_captions.setdefault(int(cid), []).append(cap)
            links_by_image.setdefault(img_id, []).append(link_payload)

        if not allowed_image_ids:
            return [], image_linked_chunk_ids, chunk_captions

        retrieved_images: list[RetrievedImage] = []
        for (image_metadata, score) in image_results:
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
                    id=image_id,
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
                image_id = payload["image_id"]
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

    def _filter_images_by_caption_similarity(
            self,
            question: str,
            images: list[RetrievedImage],
            top_k: int | None = None,
    ) -> list[RetrievedImage]:
        """Keep only images whose caption embeddings are similar to the question."""
        image_ids: list[str] = []
        id_map: dict[str, RetrievedImage] = {}
        for img in images:
            image_ids.append(img.id)
            id_map[img.id] = img

        if not image_ids:
            return []

        limit = top_k if top_k is not None else len(image_ids)
        link_results = self.retriever.search_links_by_text(
            question, image_ids=image_ids, top_k=limit, score_threshold=0.5
        )
        allowed: set[str] = set()
        for payload, _score in link_results:
            img_id = payload["image_id"]
            allowed.add(img_id)

        if not allowed:
            return []

        return [id_map[iid] for iid in image_ids if iid in allowed and iid in id_map]

    def _dedupe_images(self, images: list[RetrievedImage]) -> list[RetrievedImage]:
        """Remove duplicate images by image id (fallback to url/local_path when id is missing).

        Keeps the first-seen ordering but, if a later duplicate has a higher score, it replaces the earlier
        entry with the higher-scored one.
        """
        if not images:
            return []

        best_map: dict[str | None, RetrievedImage] = {}
        order: list[str | None] = []

        def _key(img: RetrievedImage):
            if img.id is not None:
                return str(img.id)
            # fallback to url/local_path so we still dedupe obvious duplicates without an id
            return img.url or img.local_path or None

        for img in images:
            k = _key(img)
            if k not in best_map:
                best_map[k] = img
                order.append(k)
            else:
                # prefer higher score if available
                existing = best_map[k]
                try:
                    existing_score = float(existing.score or 0.0)
                except Exception:
                    existing_score = 0.0
                try:
                    new_score = float(img.score or 0.0)
                except Exception:
                    new_score = 0.0
                if new_score > existing_score:
                    best_map[k] = img

        return [best_map[k] for k in order]

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
        return "\n\n".join(lines)

    def _rewrite_question(self, question: str, history: list[tuple[str, str]] | None) -> str:
        """Rewrite the question using chat history to make it standalone."""
        if not history:
            return question

        history_text = self._format_history(history)

        # Use LangChain chain for query rewriting
        rewritten = self.query_rewrite_chain.invoke({
            "history": history_text,
            "question": question
        })

        return rewritten.strip() or question

    def generate_answer(
            self,
            question: str,
            top_k_text: int = 5,
            top_k_images: int = 3,
            query_image_path: Optional[str] = None,
            chat_history: Optional[list[tuple[str, str]]] = None,
    ) -> Tuple[str, RetrievedContext]:
        """Question -> answer (+ retrieved context)."""

        rewritten_question = self._rewrite_question(question, chat_history)

        # --- A) Text retrieval always happens ---
        text_chunks_by_text, text_chunk_ids_by_text = self._retrieve_text(rewritten_question, top_k_text)

        # Text-only
        if not query_image_path:
            context_text = "\n\n".join(text_chunks_by_text)

            # Use LangChain chain for answer generation
            answer = self.answer_chain.invoke({
                "context_text": context_text,
                "question": rewritten_question
            })

            linked_images = self._collect_images_from_text_chunks(text_chunk_ids_by_text)
            linked_images = self._filter_images_by_caption_similarity(
                rewritten_question, linked_images, top_k=len(linked_images) or None
            )
            linked_images = self._dedupe_images(linked_images)
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

        logger.debug(
            "Built prompt (len=%s). text_chunks=%s; images=%s",
            len(context_text),
            len(merged_text_chunks),
            len(retrieved_images),
        )

        # Use LangChain chain for answer generation
        answer = self.answer_chain.invoke({
            "context_text": context_text,
            "question": rewritten_question
        })

        retrieved_images = self._dedupe_images(retrieved_images)
        return answer, RetrievedContext(text_chunks=merged_text_chunks, images=retrieved_images)

if __name__ == "__main__":
    # pipeline = RAGPipeline()
    #
    # # Просте питання без історії
    # question = "Who was the first Roman Emperor?"
    #
    # # Викликаємо генерацію відповіді
    # answer, context = pipeline.generate_answer(
    #     question=question,
    #     top_k_text=3,
    #     chat_history=None  # Без історії
    # )
    #
    # print(f"Питання: {question}")
    # print(f"Відповідь: {answer}")
    # print(f"Знайдено текстових фрагментів: {len(context.text_chunks)}")
    # print(f"Знайдено зображень: {len(context.images)}")

    # Minimal smoke run so you can execute: `python -m rag.pipeline`
    # or `python rag/pipeline.py` from the project root.
    # parser = argparse.ArgumentParser(description="Run RAG pipeline demo")
    # parser.add_argument(
    #     "--question", "-q",
    #     default=os.environ.get(
    #         "RAG_DEMO_QUESTION",
    #         "What can you say about this picture?"
    #     ),
    #     help="Question to ask the RAG pipeline",
    # )
    # parser.add_argument(
    #     "--image-path", "-i",
    #     default="../tests/data/images/Colosseum_in_Rome,_Italy_-_April_2007.jpg",
    #     help="Path to the query image (use --no-image to disable image retrieval)",
    # )
    # parser.add_argument("--no-image", action="store_true", help="Run without image retrieval")
    # args = parser.parse_args()

    # sample_question = "Explain the key reforms of Augustus that transformed the Roman Republic into the Principate. Provide 3–5 bullets and cite specific offices/institutions."#args.question
    # sample_question = "Explain how the Principate differed from the Roman Republic in terms of political institutions"#args.question
    # sample_question = "What was the fertility rate in Roman Egypt for ages 32?"#args.question
    # sample_question = "How much did the Decius Trajanus's antoninianus weight and what was it's diameter?"#args.question
    # sample_question = "How much did the sestertius with Emperor Maximinus Thrax standing between two legionary banners on Reverse weight during the Military campaigns in the north?"#args.question
    sample_question = "When did Augustus reign?"  # args.question
    # sample_question = "Who is the person that stand on the left?"#args.question
    # sample_question = "What were the fertility rates in Roman Egypt?"#args.question
    # image_path = "../tests/data/images/Venice_–_The_Tetrarchs_03.jpg" #if args.no_image else args.image_path
    image_path = None
    logger.info("Running RAG pipeline demo")

    pipeline = RAGPipeline()
    answer, retrieved = pipeline.generate_answer(sample_question, query_image_path=image_path, top_k_text=5)

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

