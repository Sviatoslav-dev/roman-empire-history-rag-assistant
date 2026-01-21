"""LangChain adapters for project components.

This module is optional: it exists to let you reuse the project's Qdrant
collections with LangChain chains, without coupling LangChain to the core
pipeline.
"""

from __future__ import annotations

from typing import List, Optional

from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.documents import Document
from langchain_core.language_models.llms import LLM
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever

from rag.llm_client import LLMClient
from rag.retriever import QdrantRetriever


class LangChainLLM(LLM):
    """LangChain LLM wrapper that delegates to :class:`rag.llm_client.LLMClient`."""

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
    """Expose :meth:`rag.retriever.QdrantRetriever.search_text` as a LangChain retriever."""

    def __init__(self, retriever: QdrantRetriever, top_k_text: int = 5):
        super().__init__()
        self._retriever = retriever
        self._top_k = top_k_text

    def _get_relevant_documents(self, query: str) -> List[Document]:
        results = self._retriever.search_text(query, top_k=self._top_k)
        return [
            Document(page_content=text, metadata={**metadata, "score": score, "chunk_id": chunk_id})
            for text, score, metadata, chunk_id in results
        ]

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)


def build_langchain_rag_chain(
    llm_client: Optional[LLMClient] = None,
    retriever: Optional[QdrantRetriever] = None,
    top_k_text: int = 5,
) -> RetrievalQA:
    """Build a LangChain RetrievalQA chain using project components."""

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

    return RetrievalQA.from_chain_type(
        llm=lc_llm,
        chain_type="stuff",
        retriever=lc_retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

