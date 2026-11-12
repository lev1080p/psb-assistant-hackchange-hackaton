"""One answer workflow for HTTP and future CSV submission."""
import asyncio
import json
from pydantic import ValidationError
from config import Settings
from dependencies import EvidenceProvider, LanguageModel, UpstreamError, prepare_search_query
from prompts import answer_messages, verification_messages
from schemas import ChatRequest, Draft, SupportCheck
from verification import finalize_answer, insufficient_result, validate_citations


class Assistant:
    def __init__(self, evidence: EvidenceProvider, llm: LanguageModel, settings: Settings):
        self.evidence = evidence
        self.llm = llm
        self.settings = settings

    async def answer(self, request: ChatRequest):
        request = ChatRequest.model_validate(request.model_dump())
        async with asyncio.timeout(self.settings.request_timeout):
            search_query = prepare_search_query(request)
            fragments = await self.evidence(search_query)
            # Reject ambiguous IDs rather than silently choosing an arbitrary source.
            if len({f.fragment_id for f in fragments}) != len(fragments):
                raise UpstreamError("Duplicate evidence IDs")
            selected = []
            budget = self.settings.max_context_chars
            for fragment in fragments:
                size = len(json.dumps(fragment.model_dump(), ensure_ascii=False))
                if size <= budget:
                    selected.append(fragment)
                    budget -= size
            try:
                draft = Draft.model_validate_json(await self.llm.complete(answer_messages(request, selected)))
            except ValidationError as exc:
                raise UpstreamError("Invalid LLM answer format") from exc
            if draft.status != "answer":
                from schemas import AssistantResult
                return AssistantResult(status=draft.status, text=draft.text)
            if not validate_citations(draft, selected):
                return insufficient_result()
            try:
                check = SupportCheck.model_validate_json(await self.llm.complete(
                    verification_messages(request, draft, selected)))
            except ValidationError as exc:
                raise UpstreamError("Invalid LLM verification format") from exc
            if not check.supported:
                return insufficient_result()
            return finalize_answer(draft, selected)
