"""Application settings and official PSB environment aliases."""
import math
import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    llm_base_url: str = ""
    llm_api_key: str = field(default="", repr=False)
    llm_model: str = ""
    request_timeout: float = 50.0
    llm_timeout: float = 15.0
    max_context_chars: int = 40000

    temperature: float = 0.0
    max_tokens: int = 3000
    max_workers: int = 2
    retry_attempts: int = 3
    retry_delay: float = 0.25
    top_k: int = 10
    chunker: str = ""
    emb_model: str = ""
    reranker_model: str = ""
    reranker_top_n: int = 5

    def validate_llm(self):
        missing = [name for name, value in (
            ("LLM_BASE_URL/OPENAI_BASE_URL", self.llm_base_url),
            ("LLM_API_KEY/OPENAI_API_KEY", self.llm_api_key),
            ("LLM_MODEL/OPENAI_MODEL/GEN_MODEL", self.llm_model)) if not value]
        if missing:
            from dependencies import BackendUnavailable
            raise BackendUnavailable("Missing LLM settings: " + ", ".join(missing))

    def __post_init__(self):
        if not math.isfinite(self.request_timeout) or not 0 < self.request_timeout < 60:
            raise ValueError("Request timeout must be between 0 and 60 seconds")
        if not math.isfinite(self.llm_timeout) or not 0 < self.llm_timeout <= self.request_timeout:
            raise ValueError("Invalid LLM timeout")
        if any(value < 1 for value in (self.max_tokens, self.max_workers, self.top_k, self.reranker_top_n)):
            raise ValueError("Worker, token and retrieval limits must be positive")
        if not 1 <= self.retry_attempts <= 5 or not math.isfinite(self.retry_delay) or not 0 <= self.retry_delay <= 5:
            raise ValueError("Invalid retry policy")
        if not math.isfinite(self.temperature) or not 0 <= self.temperature <= 2:
            raise ValueError("Invalid temperature")
        if self.max_context_chars < 1:
            raise ValueError("Context budget must be positive")

    @classmethod
    def from_env(cls):
        return cls(
            llm_base_url=first_env("LLM_BASE_URL", "OPENAI_BASE_URL", "ASSISTANT_LLM_BASE_URL"),
            llm_api_key=first_env("LLM_API_KEY", "OPENAI_API_KEY", "ASSISTANT_LLM_API_KEY"),
            llm_model=first_env("LLM_MODEL", "OPENAI_MODEL", "GEN_MODEL", "ASSISTANT_LLM_MODEL"),
            request_timeout=float(os.getenv("ASSISTANT_REQUEST_TIMEOUT", "50")),
            llm_timeout=float(os.getenv("ASSISTANT_LLM_TIMEOUT", "15")),
            temperature=float(os.getenv("TEMPERATURE", "0")),
            max_tokens=int(os.getenv("MAX_TOKENS", "3000")),
            max_workers=int(os.getenv("MAX_WORKERS", "2")),
            top_k=int(os.getenv("TOP_K", "10")),
            chunker=os.getenv("CHUNKER", ""),
            emb_model=os.getenv("EMB_MODEL", ""),
            reranker_model=os.getenv("RERANKER_MODEL", ""),
            reranker_top_n=int(os.getenv("RERANKER_TOP_N", "5")),
            max_context_chars=int(os.getenv("ASSISTANT_CONTEXT_CHARS", "40000")),
        )


def first_env(*names):
    return next((os.environ[name] for name in names if os.environ.get(name)), "")
