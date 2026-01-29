import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from llama_cpp import Llama

from logger import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent

load_dotenv()

# Defaults, with env overrides
DEFAULT_MODEL_PATH = str(PROJECT_ROOT / os.getenv("LLM_MODEL_PATH"))
DEFAULT_N_GPU_LAYERS = int(os.getenv("N_GPU_LAYERS", "-1"))
DEFAULT_N_CTX = int(os.getenv("N_CTX", "4096"))


class LLMClient:
    """LLM client using llama-cpp-python for GGUF models."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        n_gpu_layers: Optional[int] = None,
        n_ctx: Optional[int] = None,
    ) -> None:
        """Initialize LLM client.

        Args:
            model_path: Optional path to a GGUF model file. Defaults to the resolved
                `LLM_MODEL_PATH` (see :func:`resolve_model_path`).
            n_gpu_layers: Number of layers to offload to GPU (defaults to env
                var `N_GPU_LAYERS`, falling back to -1).
            n_ctx: Context window size (defaults to env var `N_CTX`, falling back
                to 4096).
        """
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.n_gpu_layers = n_gpu_layers if n_gpu_layers is not None else DEFAULT_N_GPU_LAYERS
        self.n_ctx = n_ctx if n_ctx is not None else DEFAULT_N_CTX

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
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        *,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate text from prompt.

        If `system_prompt` is provided, uses the chat API with system/user roles.
        """
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
            # llama-cpp-python may return dict or an iterator depending on streaming.
            if isinstance(resp, dict):
                message = resp.get("choices", [{}])[0].get("message", {})
                return (message.get("content") or "").strip()

            # If streaming was enabled somehow, collect chunks.
            content_parts: list[str] = []
            for event in resp:  # type: ignore[assignment]
                if isinstance(event, dict):
                    delta = event.get("choices", [{}])[0].get("delta", {})
                    part = delta.get("content")
                    if isinstance(part, str):
                        content_parts.append(part)
            return "".join(content_parts).strip()

        response = self.model(
            prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            echo=False,
            stop=stop,
        )

        if isinstance(response, dict):
            return str(response.get("choices", [{}])[0].get("text", "")).strip()

        text_parts: list[str] = []
        for event in response:  # type: ignore[assignment]
            if isinstance(event, dict):
                txt = event.get("choices", [{}])[0].get("text")
                if isinstance(txt, str):
                    text_parts.append(txt)
        return "".join(text_parts).strip()
