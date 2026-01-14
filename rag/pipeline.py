"""RAG pipeline for question answering."""
import os
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from models.schemas import ChatMessage, RetrievedContext, RetrievedImage
from rag.retriever import QdrantRetriever

load_dotenv()

LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH")

class LLMClient:
    """LLM client for LLaMA 3 8B."""
    
    def __init__(self):
        """Initialize LLM client."""
        # Note: This is a placeholder. You'll need to configure based on your setup:
        # - Local inference with llama-cpp-python
        # - HuggingFace transformers
        # - API endpoint (OpenAI-compatible)
        self.model_name = LLM_MODEL_PATH
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # For now, we'll use a simple placeholder
        # Replace this with your actual LLaMA 3 8B setup
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
        except Exception as e:
            print(f"Warning: Could not load model {self.model_name}: {e}")
            print("Using placeholder model. Please configure your LLM setup.")
            self.tokenizer = None
            self.model = None
    
    def generate(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7
    ) -> str:
        """Generate text from prompt."""
        if self.model is None or self.tokenizer is None:
            # Placeholder response
            return "This is a placeholder response. Please configure your LLaMA 3 8B model."
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from the response
        response = response[len(prompt):].strip()
        return response


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
