"""RAG pipeline for question answering."""
import os
from pathlib import Path
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from llama_cpp import Llama

from models.schemas import ChatMessage, RetrievedContext, RetrievedImage
from rag.retriever import QdrantRetriever

# --- LangChain additions ---
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models.llms import LLM

load_dotenv()

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

        print(f"Loading LLM model: {self.model_path}")
        print(f"GPU layers: {self.n_gpu_layers}")
        print(f"Context size: {self.n_ctx}")

        self.model = Llama(
            model_path=self.model_path,
            n_gpu_layers=self.n_gpu_layers,
            n_ctx=self.n_ctx,
            verbose=False,
        )
        print(f"✓ Model loaded successfully")


    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """Generate text from prompt."""
        if self.model is None:
            return "⚠️ LLM not loaded. Please configure your model. See error messages above."

        response = self.model(
            prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            echo=False,
            stop=["User:", "Question:", "\n\n\n"],
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
    """Adapter to use our existing QdrantRetriever inside LangChain."""
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


def build_langchain_rag_chain(llm_client: Optional[LLMClient] = None, retriever: Optional[QdrantRetriever] = None, top_k_text: int = 5) -> RetrievalQA:
    """Create a LangChain RetrievalQA chain using our components."""
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

    # We construct a RetrievalQA that uses our adapter retriever.
    chain = RetrievalQA.from_chain_type(
        llm=lc_llm,
        chain_type="stuff",
        retriever=lc_retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return chain


class RAGPipeline:
    """RAG pipeline for generating answers."""
    
    def __init__(self):
        """Initialize RAG pipeline."""
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
        """Generate answer using RAG.

        Args:
            question: User's text question
            chat_history: Previous chat messages for context
            top_k_text: Number of text chunks to retrieve
            top_k_images: Number of images to retrieve
            query_image_path: Optional path to query image for image-based search

        Returns:
            Tuple of (answer, retrieved_context)
        """

        if query_image_path:
            # Image-based retrieval: find similar images, then get their associated text chunks
            image_results = self.retriever.search_images(query_image_path, top_k=top_k_images)

            # Collect unique text chunk IDs from retrieved images
            text_chunk_ids = set()
            retrieved_images = []

            for i, (image_metadata, score) in enumerate(image_results):
                # Add image to results
                retrieved_images.append(
                    RetrievedImage(
                        id=str(i),
                        url=image_metadata.get("image_url"),
                        local_path=image_metadata.get("local_path"),
                        caption=image_metadata.get("caption", ""),
                        page_title=image_metadata.get("page_title"),
                        score=float(score)
                    )
                )

                # Collect text chunk ID for later retrieval
                chunk_id = image_metadata.get("text_chunk_id")
                if chunk_id is not None:
                    text_chunk_ids.add(chunk_id)

            # Retrieve text chunks associated with found images
            text_chunks = []
            for chunk_id in list(text_chunk_ids)[:top_k_text]:
                result = self.retriever.get_text_chunk_by_id(chunk_id)
                if result:
                    text, metadata = result
                    text_chunks.append(text)
        else:
            # Text-based retrieval: find relevant text chunks, then get their associated images
            text_results = self.retriever.search_text(question, top_k=top_k_text)
            text_chunks = [text for text, score, metadata, chunk_id in text_results]

            # Retrieve images associated with the retrieved text chunks
            retrieved_images = []
            seen_image_paths = set()  # To avoid duplicates

            for text, score, metadata, chunk_id in text_results:
                # Get all images for this text chunk
                chunk_images = self.retriever.get_images_by_text_chunk_id(chunk_id)

                for img_metadata in chunk_images:
                    # Check if we've already added this image
                    img_path = img_metadata.get("local_path") or img_metadata.get("image_url")
                    if img_path and img_path not in seen_image_paths:
                        seen_image_paths.add(img_path)
                        retrieved_images.append(
                            RetrievedImage(
                                id=str(len(retrieved_images)),
                                url=img_metadata.get("image_url"),
                                local_path=img_metadata.get("local_path"),
                                caption=img_metadata.get("caption", ""),
                                page_title=img_metadata.get("page_title"),
                                score=float(score)  # Use the text chunk's relevance score
                            )
                        )

                        # Stop if we have enough images
                        if len(retrieved_images) >= top_k_images:
                            break

                if len(retrieved_images) >= top_k_images:
                    break

        # Build context from retrieved chunks
        context_text = "\n\n".join(text_chunks)
        
        # Build prompt with context
        system_prompt = """You are a helpful assistant specializing in Roman Empire history. 
Answer questions based on the provided context. Be concise and accurate."""
        
        if chat_history:
            history_text = "\n".join([
                f"{msg.role}: {msg.content}"
                for msg in chat_history[-5:]  # Last 5 messages for context
            ])
            prompt = f"""{system_prompt}

Previous conversation:
{history_text}

Context:
{context_text}

User: {question}
Assistant:"""
        else:
            prompt = f"""{system_prompt}

Context:
{context_text}

User: {question}
Assistant:"""
        
        # Generate answer
        print(prompt)
        answer = self.llm.generate(prompt)
        
        # Build retrieved context
        context = RetrievedContext(
            text_chunks=text_chunks,
            images=retrieved_images
        )
        
        return answer, context


# ---- Simple runner to quickly try the pipeline locally ----
if __name__ == "__main__":
    # Minimal smoke run so you can execute: `python -m rag.pipeline`
    # or `python rag/pipeline.py` from the project root.
    sample_question = os.environ.get(
        "RAG_DEMO_QUESTION",
        # "Who was Augustus?"
        # "What is Byzantine Empire?"
        "What can you say about this picture?"
    )

    image_path = "../tests/data/images/Tunisia-3363_-_Amphitheatre_Spectacle.jpg"

    print("Running RAG pipeline demo...\n")

    pipeline = RAGPipeline()
    answer, retrieved = pipeline.generate_answer(sample_question, query_image_path=image_path)

    # Optional: LangChain demo if desired
    if os.environ.get("RAG_USE_LANGCHAIN", "0") == "1":
        print("\nRunning LangChain RetrievalQA demo...\n")
        lc_chain = build_langchain_rag_chain(top_k_text=5)
        lc_result = lc_chain.invoke({"query": sample_question})
        print("LangChain answer:")
        print(lc_result.get("result"))

    print("Question:")
    print(sample_question)
    print("\nAnswer:")
    print(answer)

    # Briefly summarize retrieved context
    print("\nRetrieved context summary:")
    print(f"- Text chunks: {len(retrieved.text_chunks)}")
    if retrieved.text_chunks:
        preview = (retrieved.text_chunks[0] or "").strip().replace("\n", " ")
        print(f"  First chunk preview: {preview[:200]}{'...' if len(preview) > 200 else ''}")

    print(f"\n- Images retrieved: {len(retrieved.images)}")
    if retrieved.images:
        for i, img in enumerate(retrieved.images, 1):
            print(f"  [{i}] Title: {img.page_title or 'n/a'}")
            print(f"      Caption: {(img.caption or 'No caption')[:100]}")
            print(f"      Score: {img.score:.3f}")
            print(f"      Path: {img.local_path or img.url or 'n/a'}")
