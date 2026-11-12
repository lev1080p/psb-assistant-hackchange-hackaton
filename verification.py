"""Deterministic citation checks, followed by a separate model support check.

Future source-location integration (not enabled; Fragment.start/end stay None):
1. Obtain Document.text for the exact document/revision of the fragment.
2. Check full_text[python_start:python_end] == fragment.text for real offsets,
   with 0 <= start < end <= len(full_text). Never normalize either text.
3. If offsets are unavailable, enumerate exact matches, including overlapping
   occurrences. Infer a position only when exactly one occurrence exists.
4. Multiple occurrences require the real position from the document component;
   do not choose the first match as an authoritative location.
5. For a verified Python code-point offset p, the JavaScript UTF-16 offset is
   len(full_text[:p].encode("utf-16-le")) // 2. Convert BOTH boundaries and verify
   against the same immutable text version sent to the browser.
6. No exact match: report a source consistency error, never invent coordinates
   or modify the quote. The current frontend falls back to the first match when
   offsets are null, so ambiguity cannot be eliminated in that mode. Repeated
   fragments must remain an explicitly documented integration limitation until
   verified real coordinates are enabled.
"""
from schemas import AssistantResult, Claim, Draft, Fragment


def insufficient_result():
    return AssistantResult(status="insufficient", text=(
        "Не удалось подтвердить юридический вывод предоставленными фрагментами. "
        "Для ответа нужны дополнительные или более подходящие положения документов."))


def validate_citations(draft: Draft, fragments: list[Fragment]) -> bool:
    by_id = {item.fragment_id: item for item in fragments}
    if len(by_id) != len(fragments) or not draft.claims:
        return False
    for claim in draft.claims:
        ids = set(claim.fragment_ids)
        if len(ids) != len(claim.fragment_ids) or ids != set(claim.quotes):
            return False
        for source_id in ids:
            quote = claim.quotes[source_id]
            if source_id not in by_id or not quote.strip() or quote not in by_id[source_id].text:
                return False
    return True


def finalize_answer(draft: Draft, fragments: list[Fragment]) -> AssistantResult:
    used = {source_id for claim in draft.claims for source_id in claim.fragment_ids}
    claims = [Claim(text=item.text, fragment_ids=item.fragment_ids) for item in draft.claims]
    return AssistantResult(status="answer", text="\n\n".join(item.text for item in claims),
                           claims=claims, fragments=[f for f in fragments if f.fragment_id in used])
