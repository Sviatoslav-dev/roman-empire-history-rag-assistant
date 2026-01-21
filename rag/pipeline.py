"""RAG pipeline for question answering."""
import os
from pathlib import Path
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from llama_cpp import Llama

from models.schemas import ChatMessage, RetrievedContext, RetrievedImage
from rag.retriever import QdrantRetriever
from logger import get_logger

# --- LangChain additions ---
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models.llms import LLM

load_dotenv()

logger = get_logger(__name__)

# GGUF model path - Qwen 2.5 3B Instruct quantized
PROJECT_ROOT = Path(__file__).parent.parent
LLM_MODEL_PATH = str(PROJECT_ROOT / "models" / "qwen2.5-3b-instruct-q4_k_m.gguf")#os.getenv("LLM_MODEL_PATH", "models/Qwen2.5-3B-Instruct.Q4_K_M.gguf")
# Number of layers to offload to GPU (0 = CPU only, -1 = all layers)
N_GPU_LAYERS = int(os.getenv("N_GPU_LAYERS", "-1"))
# Context window size
N_CTX = int(os.getenv("N_CTX", "4096"))

class LLMClient:
    """LLM client using llama-cpp-python for GGUF models."""

    def __init__(self, model_path: str = None, n_gpu_layers: int = None, n_ctx: int = None):
        """Initialize LLM client.

        Args:
            model_path: Path to GGUF model file (default: LLM_MODEL_PATH env var)
            n_gpu_layers: Number of layers to offload to GPU (default: N_GPU_LAYERS env var)
            n_ctx: Context window size (default: N_CTX env var)
        """
        self.model_path = model_path or LLM_MODEL_PATH
        self.n_gpu_layers = n_gpu_layers if n_gpu_layers is not None else N_GPU_LAYERS
        self.n_ctx = n_ctx if n_ctx is not None else N_CTX

        logger.info("Loading LLM model: %s", self.model_path)
        logger.info("GPU layers: %s", self.n_gpu_layers)
        logger.info("Context size: %s", self.n_ctx)

        self.model = Llama(
            model_path=self.model_path,
            n_gpu_layers=self.n_gpu_layers,
            n_ctx=self.n_ctx,
            verbose=False,
        )
        logger.info("Model loaded successfully")


    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        *,
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate text from prompt.

        If `system_prompt` is provided, uses the chat API with system/user roles to
        reduce instruction leakage.
        """
        if self.model is None:
            return "\u26a0\ufe0f LLM not loaded. Please configure your model. See error messages above."

        stop = [
            "User:",
            "Question:",
            "\n\n\n",
            "CONTEXT:",
            "QUESTION:",
            # common prompt-leak patterns
            "You may receive",
            "SYSTEM INSTRUCTIONS",
        ]

        if system_prompt:
            resp = self.model.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_new_tokens,
                stop=stop,
            )
            return (resp["choices"][0]["message"]["content"] or "").strip()

        response = self.model(
            prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            echo=False,
            stop=stop,
        )

        return response["choices"][0]["text"].strip()


# ---- LangChain wrappers ----
class LangChainLLM(LLM):
    """LangChain LLM wrapper that delegates to our LLMClient."""
    def __init__(self, client: LLMClient, max_new_tokens: int = 256, temperature: float = 0.7):
        super().__init__()
        self._client = client
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature

    @property
    def _llm_type(self) -> str:
        return "custom-llm-client"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        text = self._client.generate(prompt, max_new_tokens=self._max_new_tokens, temperature=self._temperature)
        if stop:
            for s in stop:
                if s in text:
                    text = text.split(s)[0]
        return text


class QdrantLangChainRetriever(BaseRetriever):
    """Adapter that exposes `QdrantRetriever.search_text()` as a LangChain retriever.

    This allows using the same underlying Qdrant collections with LangChain chains.
    """
    def __init__(self, retriever: QdrantRetriever, top_k_text: int = 5):
        super().__init__()
        self._retriever = retriever
        self._top_k = top_k_text

    def _get_relevant_documents(self, query: str) -> List[Document]:
        results = self._retriever.search_text(query, top_k=self._top_k)
        docs: List[Document] = []
        for text, score, metadata, chunk_id in results:
            docs.append(Document(page_content=text, metadata={**metadata, "score": score, "chunk_id": chunk_id}))
        return docs

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        # For simplicity, use sync path
        return self._get_relevant_documents(query)


def build_langchain_rag_chain(
    llm_client: Optional[LLMClient] = None,
    retriever: Optional[QdrantRetriever] = None,
    top_k_text: int = 5
) -> RetrievalQA:
    """Build a LangChain `RetrievalQA` chain using project components.

    Args:
        llm_client: Optional pre-configured LLMClient for generation.
        retriever: Optional QdrantRetriever instance.
        top_k_text: Number of text chunks to retrieve per query.

    Returns:
        A `RetrievalQA` chain that pulls context from Qdrant and generates answers.
    """
    llm_client = llm_client or LLMClient()
    lc_llm = LangChainLLM(llm_client)
    retriever = retriever or QdrantRetriever()
    lc_retriever = QdrantLangChainRetriever(retriever, top_k_text=top_k_text)

    template = (
        "You are a helpful assistant specializing in Roman Empire history. "
        "Use the provided context to answer succinctly and accurately.\n\n"
        "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    chain = RetrievalQA.from_chain_type(
        llm=lc_llm,
        chain_type="stuff",
        retriever=lc_retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return chain


class RAGPipeline:
    """High-level RAG pipeline that retrieves context and generates an answer."""

    def __init__(self):
        """Initialize retriever and LLM.

        Note:
            LLM loading can be expensive; instantiate once and reuse if possible.
        """
        self.retriever = QdrantRetriever()
        self.llm = LLMClient()
    
    def generate_answer(
        self,
        question: str,
        chat_history: Optional[List[ChatMessage]] = None,
        top_k_text: int = 5,
        top_k_images: int = 3,
        query_image_path: Optional[str] = None
    ) -> Tuple[str, RetrievedContext]:
        """Generate an answer using retrieval-augmented generation.

        Behavior:
        - Text-only query: retrieve `top_k_text` text chunks and answer based on them.
        - Image-assisted query: retrieve similar images, then fetch linked text chunks
          via LINK_COLLECTION, merge with text retrieval results, prefix relevant
          captions as '[Image caption] ...' lines.

        Args:
            question: User question text.
            chat_history: Optional chat history (currently not injected into the prompt).
            top_k_text: Max number of text chunks to include in context.
            top_k_images: Max number of similar images to retrieve.
            query_image_path: Optional local path to a query image.

        Returns:
            (answer, RetrievedContext)

        Logging:
            Prompt contents are not logged; only metadata may be logged at DEBUG level.
        """

        system_prompt = (
            "You are a helpful assistant specializing in Roman Empire history.\n"
            "You will receive CONTEXT snippets with optional '[Image caption]' lines.\n"
            "Rules:\n"
            "- Use CONTEXT as your primary source.\n"
            "- Treat '[Image caption]' lines as reliable descriptions of the attached image.\n"
            "- Do NOT mention missing images, inability to see images, or retrieval/meta commentary.\n"
            # "- Be concise and direct.\n"
            # "- If the user asks 'What is depicted on this image?', answer in ONE short sentence naming the depicted subject.\n"
        )

        # --- A) Text retrieval always happens (textual part of prompt) ---
        text_results = self.retriever.search_text(question, top_k=top_k_text)
        text_chunks_by_text: List[str] = [t for (t, _s, _m, _cid) in text_results]
        text_chunk_ids_by_text: List[int] = [cid for (_t, _s, _m, cid) in text_results]

        # Text-only prompt => only return text chunks, no images.
        if not query_image_path:
            context_text = "\n\n".join(text_chunks_by_text)

            if chat_history:
                history_text = "\n".join([
                    f"{msg.role}: {msg.content}"
                    for msg in chat_history[-5:]
                ])
                user_prompt = f"CONTEXT:\n{context_text}\n\nQUESTION: {question}\nANSWER:"
                answer = self.llm.generate(user_prompt, system_prompt=system_prompt)
            else:
                user_prompt = f"CONTEXT:\n{context_text}\n\nQUESTION: {question}\nANSWER:"
                answer = self.llm.generate(user_prompt, system_prompt=system_prompt)

            return answer, RetrievedContext(text_chunks=text_chunks_by_text, images=[])

        # --- B) Image retrieval: nearest images -> linked chunks + captions ---
        image_results = self.retriever.search_images(query_image_path, top_k=top_k_images)

        image_linked_chunk_ids: List[int] = []
        retrieved_images: List[RetrievedImage] = []

        # Map chunk_id -> list of captions that appear in that chunk (relationship-level)
        chunk_captions: dict[int, List[str]] = {}

        # NOTE: captions are relationship-level, so we read them from LINK_COLLECTION.
        for i, (image_metadata, score) in enumerate(image_results):
            image_id = image_metadata.get("image_id")

            # Pull the link records for this image to get real stored captions
            links = []
            if image_id is not None:
                links = self.retriever.get_links_by_image_id(int(image_id), limit=200)

            per_image_captions: List[str] = []
            for link in links:
                cid = link.get("text_chunk_id")
                if cid is None:
                    continue

                image_linked_chunk_ids.append(cid)

                cap = link.get("caption")
                if isinstance(cap, str):
                    cap = cap.strip()
                else:
                    cap = ""

                if cap:
                    per_image_captions.append(cap)
                    if cid not in chunk_captions:
                        chunk_captions[cid] = []
                    chunk_captions[cid].append(cap)

            retrieved_images.append(
                RetrievedImage(
                    id=str(i),
                    url=image_metadata.get("image_url"),
                    local_path=image_metadata.get("image_path") or image_metadata.get("local_path"),
                    caption=(per_image_captions[0] if per_image_captions else None),
                    page_title=None,
                    score=float(score)
                )
            )

        # --- C) Merge + dedupe chunk ids ---
        merged_chunk_ids: List[int] = []
        seen_chunk_ids = set()

        # Preserve ranking from text retrieval first
        for cid in text_chunk_ids_by_text:
            if cid not in seen_chunk_ids:
                merged_chunk_ids.append(cid)
                seen_chunk_ids.add(cid)

        # Then add chunks from image links
        for cid in image_linked_chunk_ids:
            if cid not in seen_chunk_ids:
                merged_chunk_ids.append(cid)
                seen_chunk_ids.add(cid)

        # Limit overall context size
        merged_chunk_ids = merged_chunk_ids[:top_k_text]

        merged_text_chunks: List[str] = []
        # Reuse text chunks we already have
        text_by_id = {cid: text_chunks_by_text[idx] for idx, cid in enumerate(text_chunk_ids_by_text)}

        for cid in merged_chunk_ids:
            if cid in text_by_id:
                chunk_text = text_by_id[cid]
            else:
                res = self.retriever.get_text_chunk_by_id(cid)
                if not res:
                    continue
                chunk_text, _m = res

            # Attach captions near the chunk
            caps = chunk_captions.get(cid, [])
            if caps:
                # Deduplicate captions while preserving order
                seen_caps = set()
                uniq_caps = []
                for c in caps:
                    if c not in seen_caps:
                        uniq_caps.append(c)
                        seen_caps.add(c)

                caps_text = "\n".join([f"[Image caption] {c}" for c in uniq_caps])
                merged_text_chunks.append(f"{caps_text}\n{chunk_text}")
            else:
                merged_text_chunks.append(chunk_text)

        context_text = "\n\n".join(merged_text_chunks)

        # Remove the old separate captions_block and build prompt with just the context_text.
        user_prompt = f"CONTEXT:\n{context_text}\n\nQUESTION: {question}\nANSWER:"
        logger.debug(
            "Built prompt (len=%s). chat_history=%s; text_chunks=%s; images=%s",
            len(user_prompt),
            bool(chat_history),
            len(merged_text_chunks),
            len(retrieved_images),
        )
        answer = self.llm.generate(user_prompt, system_prompt=system_prompt)

        return answer, RetrievedContext(text_chunks=merged_text_chunks, images=retrieved_images)


# ---- Simple runner to quickly try the pipeline locally ----
if __name__ == "__main__":
    # Minimal smoke run so you can execute: `python -m rag.pipeline`
    # or `python rag/pipeline.py` from the project root.
    sample_question = os.environ.get(
        "RAG_DEMO_QUESTION",
        # "Who was Augustus?"
        # "What is Byzantine Empire?"
        # "What was the fertility rate in Roman Egypt for ages 25\u201329?"
        "What can you say about this picture?"
    )

    # image_path = "../tests/data/images/Tunisia-3363_-_Amphitheatre_Spectacle.jpg"
    image_path = "../tests/data/images/Colosseum_in_Rome,_Italy_-_April_2007.jpg"
    # image_path = "../tests/data/images/colosseum.png"
    # image_path = None

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
