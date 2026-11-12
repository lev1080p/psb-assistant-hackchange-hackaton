"""Shared backend contracts; independent of the future search implementation."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class HistoryMessage(StrictModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class ChatRequest(StrictModel):
    question: str = Field(min_length=1, max_length=10000)
    history: list[HistoryMessage] = Field(default_factory=list)

    @field_validator("history", mode="before")
    @classmethod
    def bound_history(cls, value):
        import json
        if not isinstance(value, list):
            return value  # Let Pydantic report a malformed request.
        selected = []
        for raw in reversed(value[-24:]):
            message = HistoryMessage.model_validate(raw).model_copy(deep=True)
            if len(message.content) > 4000:
                message.content = "[Начало сообщения опущено]\n" + message.content[-3900:]
            candidate = [message] + selected
            if len(json.dumps([m.model_dump() for m in candidate], ensure_ascii=False)) > 12000:
                break
            selected = candidate
        # Avoid orphaned assistant replies after dropping earlier turns.
        while selected and selected[0].role == "assistant":
            selected.pop(0)
        return selected


class SourceModel(StrictModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)

    @field_validator("*", mode="after")
    @classmethod
    def nonblank_required(cls, value, info):
        if info.field_name in {"fragment_id", "document_id", "document_title", "text"}:
            if isinstance(value, str) and not value.strip():
                raise ValueError("Required source field must not be blank")
        return value


class Fragment(SourceModel):
    fragment_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    document_title: str = Field(min_length=1)
    section: str = ""
    text: str = Field(min_length=1)
    revision: str = ""
    valid_from: str = ""
    valid_to: str = ""
    start: None = None
    end: None = None


class Document(SourceModel):
    document_id: str = Field(min_length=1)
    text: str = Field(min_length=1)


class Claim(StrictModel):
    text: str = Field(min_length=1)
    fragment_ids: list[str] = Field(min_length=1)


class DraftClaim(Claim):
    # Each cited source must have an exact supporting quote.
    quotes: dict[str, str]


class Draft(StrictModel):
    status: Literal["answer", "clarification", "insufficient"]
    text: str = Field(min_length=1)
    claims: list[DraftClaim] = Field(default_factory=list)


class SupportCheck(StrictModel):
    supported: bool


class AssistantResult(StrictModel):
    status: Literal["answer", "clarification", "insufficient"]
    text: str = Field(min_length=1)
    claims: list[Claim] = Field(default_factory=list)
    fragments: list[Fragment] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_sources(self):
        ids = [fragment.fragment_id for fragment in self.fragments]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate fragment IDs")
        if any(not set(claim.fragment_ids) <= set(ids) for claim in self.claims):
            raise ValueError("Unknown fragment ID")
        if self.status == "answer" and not self.claims:
            raise ValueError("An answer requires supported claims")
        if self.status != "answer" and (self.claims or self.fragments):
            raise ValueError("Non-answer results must not assert supported claims")
        return self
