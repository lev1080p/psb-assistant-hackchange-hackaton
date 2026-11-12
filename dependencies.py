"""Backend-owned injection boundaries, not the participant #1 module API.

A future bridge translates these calls to the actual search/document interfaces.
No implementation or fake knowledge base is installed in production.

Integration requirements:
- SearchFactory(archive_path: Path, settings: Settings) initializes once and
  returns SearchComponents(evidence, documents). solution.run already does this;
  API bootstrap must do so once before create_app, never once per question.
- Assistant prepares SearchQuery before calling EvidenceProvider. The future
  bridge translates this payload to participant #1's confirmed API; it never
  receives ChatRequest or assistant dialogue.
- Query fields: current query, bounded prior USER context, confirmed filters.
  No automatic date/document filters are inferred yet. Context is not evidence.
- Evidence returns list[Fragment], ranked most useful first; [] is a successful
  empty search. Fragment IDs must be unique, stable and refer to real sources.
- Documents returns Document for a stable exact revision ID, None for unknown ID.
- Raise BackendUnavailable for absent setup, UpstreamError for service/data
  failures, UpstreamTimeout for deadlines. Do not turn failures into [] or None.
- Providers share a reusable index, not user/session mutable state. Protect
  mutable internals with locks or serialize non-thread-safe engines in the
  bridge; blocking work must not block the async event loop. MAX_WORKERS limits
  batch concurrency but does not make an engine safe by itself.
- These are our adapter requirements, not claims about existing participant code.
"""
from __future__ import annotations

from typing import Protocol
from schemas import ChatRequest, Document, Fragment


class EvidenceProvider(Protocol):
    async def __call__(self, request: SearchQuery) -> list[Fragment]: ...


class DocumentProvider(Protocol):
    async def __call__(self, document_id: str) -> Document | None: ...


class LanguageModel(Protocol):
    async def complete(self, messages: list[dict[str, str]]) -> str: ...


class BackendUnavailable(Exception):
    """Missing configuration or dependency."""


class UpstreamError(Exception):
    """Failed or malformed upstream response."""


class UpstreamTimeout(Exception):
    """Upstream deadline exceeded."""


# Backend integration contract; does not prescribe participant #1's module API.
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from config import Settings


@dataclass(frozen=True)
class SearchComponents:
    evidence: EvidenceProvider
    documents: DocumentProvider


class SearchFactory(Protocol):
    async def __call__(self, archive_path: Path, settings: "Settings") -> SearchComponents:
        """Initialize once; return reusable, concurrency-safe provider callbacks."""
        ...


async def initialize_search(archive_path: Path, settings: "Settings") -> SearchComponents:
    """Replace only this bridge once the real module contract is confirmed."""
    raise BackendUnavailable("Search integration is unfinished: connect participant #1's module")


@dataclass(frozen=True)
class SearchQuery:
    """Proposed bridge payload, NOT participant #1's existing API.

    Context is untrusted user input, never legal evidence. Filters must contain
    only confirmed constraints; the default bridge makes no guesses about dates
    or document identities.
    """
    query: str
    user_context: tuple[str, ...] = ()
    filters: tuple[tuple[str, str], ...] = ()


def prepare_search_query(request: ChatRequest) -> SearchQuery:
    # Revalidate to apply the same budget even to mutated injected requests.
    request = ChatRequest.model_validate(request.model_dump())
    context = tuple(m.content for m in request.history if m.role == "user")
    return SearchQuery(query=request.question, user_context=context)
