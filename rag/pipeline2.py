"""RAG pipeline for question answering."""
import os
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from models.schemas import ChatMessage, RetrievedContext, RetrievedImage
from rag.retriever import QdrantRetriever

# --- LangChain additions ---
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models.llms import LLM

load_dotenv()

# LLaMA 3 8B is the recommended model, but falls back to lightweight for compatibility
LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH", "meta-llama/Llama-3.1-8B-Instruct")
# Set to True to use 4-bit quantization (reduces memory from 16GB to ~5GB)
USE_QUANTIZATION = os.getenv("USE_QUANTIZATION", "true").lower() == "true"

class LLMClient:
    """LLM client supporting LLaMA 3 8B, seq2seq (Flan-T5), and causal LM models."""

    def __init__(self, model_name: str = None, use_quantization: bool = None):
        """Initialize LLM client.

        Args:
            model_name: HuggingFace model name (default: LLM_MODEL_PATH env var)
            use_quantization: Use 4-bit quantization for LLaMA models (default: USE_QUANTIZATION env var)
        """
        self.model_name = model_name or LLM_MODEL_PATH
        self.use_quantization = use_quantization if use_quantization is not None else USE_QUANTIZATION

        # Determine device: MPS for Apple Silicon, CUDA for NVIDIA, CPU otherwise
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        print(f"Loading LLM model: {self.model_name}")
        print(f"Device: {self.device}")
        print(f"Quantization: {self.use_quantization}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Set pad token if not set (needed for LLaMA)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Prepare quantization config for LLaMA models
        quantization_config = None
        if self.use_quantization and ("llama" in self.model_name.lower() or "meta-llama" in self.model_name.lower()):
            # quantization_config = BitsAndBytesConfig(
            #     load_in_4bit=True,
            #     bnb_4bit_compute_dtype=torch.float16,
            #     bnb_4bit_quant_type="nf4",
            #     bnb_4bit_use_double_quant=True,
            # )
            quantization_config = None
            print("Using 4-bit quantization (reduces memory from ~16GB to ~5GB)")

        # Try seq2seq model first (Flan-T5, T5, BART)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32 if self.device == "cpu" else torch.float16,
            # quantization_config=quantization_config,
            device_map="auto" if quantization_config else None,
        )
        self.model_type = "seq2seq"
        if not quantization_config:
            self.model.to(self.device)

        self.model.eval()  # Set to evaluation mode
        print(f"✓ Model loaded successfully as {self.model_type} model")


    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """Generate text from prompt."""
        if self.model is None or self.tokenizer is None:
            # Placeholder response
            return "⚠️ LLM not loaded. Please configure your model. See error messages above."

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,  # LLaMA supports longer contexts
            padding=True
        )

        # Move to device if not using quantization (quantization uses device_map)
        if not self.use_quantization or self.model_type == "seq2seq":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            if self.model_type == "seq2seq":
                # For seq2seq models (Flan-T5), we generate directly
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True if temperature > 0 else False,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
                )
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:
                # For causal LM models (GPT-2, LLaMA, etc.)
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True if temperature > 0 else False,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,  # Reduce repetition for LLaMA
                )
                full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Remove the prompt from the response
                response = full_text[len(prompt):].strip()

        return response


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
        for text, score, metadata in results:
            docs.append(Document(page_content=text, metadata={**metadata, "score": score}))
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
        top_k_images: int = 3
    ) -> Tuple[str, RetrievedContext]:
        """Generate answer using RAG."""
        # Retrieve relevant text chunks
        text_results = self.retriever.search_text(question, top_k=top_k_text)
        text_chunks = [text for text, score, metadata in text_results]

        # Retrieve images based on text query
        # We'll search for images related to the question
        # In the future, you can add image query support for "What is this?" queries
        image_results = self.retriever.search_images_by_text(question, top_k=top_k_images)

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
        
        # Build retrieved context with actual image data
        retrieved_images = []
        for i, (image_metadata, score) in enumerate(image_results):
            retrieved_images.append(
                RetrievedImage(
                    id=str(i),
                    url=image_metadata.get("image_url"),
                    local_path=image_metadata.get("image_path"),
                    caption=image_metadata.get("caption", ""),
                    page_title=image_metadata.get("page_title"),
                    score=float(score)
                )
            )
        
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
        "Who was Augustus?"
    )

    print("Running RAG pipeline demo...\n")

    pipeline = RAGPipeline()
    answer, retrieved = pipeline.generate_answer(sample_question)

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
    print(f"- Images: {len(retrieved.images)}")
    if retrieved.images:
        img = retrieved.images[0]
        print(f"  First image: title={img.page_title or 'n/a'}, caption={(img.caption or '')[:80]}")
