"""Official OpenAI-compatible wire protocol, without SDK URL rewriting."""
import asyncio
import httpx
from config import Settings
from dependencies import BackendUnavailable, UpstreamError, UpstreamTimeout


class OpenAICompatibleClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._slots = asyncio.Semaphore(settings.max_workers)

    async def complete(self, messages: list[dict[str, str]]) -> str:
        settings = self.settings
        settings.validate_llm()
        url = settings.llm_base_url.rstrip("/") + "/chat/completions"
        async with self._slots:
            for attempt in range(settings.retry_attempts):
                try:
                    async with httpx.AsyncClient(timeout=settings.llm_timeout) as client:
                        response = await client.post(url, headers={
                            "Authorization": "Bearer " + settings.llm_api_key,
                        }, json={"model": settings.llm_model, "messages": messages,
                                 "temperature": settings.temperature, "max_tokens": settings.max_tokens})
                        if response.status_code in (401, 403):
                            raise BackendUnavailable("LLM authorization failed (401/403); contact organizers")
                        response.raise_for_status()
                        content = response.json()["choices"][0]["message"]["content"]
                        if not isinstance(content, str) or not content.strip():
                            raise ValueError("Empty completion")
                        return content
                except httpx.HTTPStatusError as exc:
                    code = exc.response.status_code
                    if code != 429 and not 500 <= code < 600:
                        raise UpstreamError(f"LLM HTTP {code}") from None
                    failure = UpstreamError(f"LLM HTTP {code}; retry limit exhausted")
                except httpx.TimeoutException:
                    failure = UpstreamTimeout("LLM timeout; retry limit exhausted")
                except httpx.TransportError:
                    failure = UpstreamError("LLM network error; retry limit exhausted")
                except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError):
                    raise UpstreamError("Malformed LLM response or invalid request") from None
                if attempt + 1 < settings.retry_attempts:
                    await asyncio.sleep(settings.retry_delay * (2 ** attempt))
            raise failure from None
