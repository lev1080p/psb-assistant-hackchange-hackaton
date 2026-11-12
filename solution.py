"""PSB CSV entry point. No API server, fake retrieval, or package installation.

CSV dialogue labels are confirmed by official demo_questions.csv. Only complete
alternating user/assistant turns are parsed; ambiguous input stays opaque.
A different confirmed decoder can be injected without changing the answer workflow. Real search must be wired in dependencies.initialize_search.
"""
import asyncio
import csv
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Callable
from pydantic import ValidationError
from assistant import Assistant
from config import Settings
from dependencies import (BackendUnavailable, LanguageModel, SearchFactory,
                          UpstreamError, UpstreamTimeout, initialize_search)
from llm_client import OpenAICompatibleClient
from schemas import AssistantResult, ChatRequest

LOGGER = logging.getLogger(__name__)
MAX_ANSWER_CHARS = 10000
# Leave a minute for CSV output and shutdown within the 20-minute platform limit.
RUN_TIMEOUT_SECONDS = 19 * 60
TECHNICAL_ERROR = (
    "Техническая ошибка обработки: ответ не получен или не прошёл проверку. "
    "Это не означает отсутствия юридических оснований в базе знаний.")


def input_paths(data_dir: Path) -> tuple[Path, Path]:
    paths = data_dir / "questions.csv", data_dir / "knowledge-base.zip"
    for path in paths:
        if not path.is_file():
            raise ValueError(f"Required input file is missing: {path.name}")
        try:
            with path.open("rb") as stream:
                stream.read(1)
        except OSError:
            raise ValueError(f"Required input file is unreadable: {path.name}") from None
    return paths


def read_questions(path: Path) -> list[tuple[str, str]]:
    rows = []
    seen = set()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)) or not {"id", "question"} <= set(reader.fieldnames):
            raise ValueError("questions.csv must contain unique id and question columns")
        for number, row in enumerate(reader, 2):
            if None in row or row.get("id") is None or row.get("question") is None:
                raise ValueError(f"Malformed input row {number}")
            case_id, question = row["id"], row["question"]
            if not case_id.strip() or case_id in seen:
                raise ValueError(f"Empty or duplicate ID at row {number}")
            seen.add(case_id)
            rows.append((case_id, question))
    if not rows:
        raise ValueError("questions.csv contains no questions")
    return rows


def parse_question(raw: str) -> ChatRequest:
    """Parse only the complete dialogue format observed in official demo CSV."""
    matches = list(re.finditer(r"^(Пользователь|Ассистент):[ \t]*", raw, re.MULTILINE))
    if len(matches) < 3 or raw[:matches[0].start()].strip():
        return ChatRequest(question=raw)
    turns = []
    for index, match in enumerate(matches):
        role = "user" if match.group(1) == "Пользователь" else "assistant"
        expected = "user" if index % 2 == 0 else "assistant"
        end = matches[index + 1].start() if index + 1 < len(matches) else len(raw)
        content = raw[match.end():end].strip()
        if role != expected or not content:
            return ChatRequest(question=raw)
        turns.append({"role": role, "content": content})
    if turns[-1]["role"] != "user":
        return ChatRequest(question=raw)
    return ChatRequest(question=turns[-1]["content"], history=turns[:-1])


def format_submission_answer(result: AssistantResult) -> str:
    # Revalidate even if an injected dependency used model_construct or mutated a model.
    result = AssistantResult.model_validate(result.model_dump())
    if result.status == "answer":
        by_id = {f.fragment_id: f for f in result.fragments}
        parts = []
        for claim in result.claims:
            labels = list(dict.fromkeys(
                " · ".join(filter(None, (by_id[i].document_title, by_id[i].section)))
                for i in claim.fragment_ids))
            parts.append(claim.text + "\nОснования: " + "; ".join(labels) + ".")
        text = "\n\n".join(parts)
    else:
        text = result.text
    text = text.strip()
    if not text or len(text) > MAX_ANSWER_CHARS:
        # Never cut off a legal condition, exception or source label silently.
        raise ValueError("Answer is empty or exceeds 10000 characters")
    return text


def write_submission(path: Path, rows: list[tuple[str, str]]) -> None:
    ids = [case_id for case_id, _ in rows]
    if len(ids) != len(set(ids)) or any(not case_id.strip() or not answer.strip() or
                                      len(answer) > MAX_ANSWER_CHARS for case_id, answer in rows):
        raise ValueError("Invalid submission rows")
    # Publish only a complete CSV, never leave a partially written final file.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                         dir=path.parent, prefix=".submission-", delete=False) as stream:
            temporary = Path(stream.name)
            writer = csv.writer(stream)
            writer.writerow(("id", "answer"))
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


async def run(*, data_dir: Path | None = None, output_path: Path | None = None,
              settings: Settings | None = None, search_factory: SearchFactory | None = None,
              llm: LanguageModel | None = None,
              question_parser: Callable[[str], ChatRequest] = parse_question) -> Path:
    data_dir = data_dir if data_dir is not None else Path(os.getenv("DATA_DIR") or "data")
    output_path = output_path if output_path is not None else Path.cwd() / "submission.csv"
    settings = settings if settings is not None else Settings.from_env()
    questions_path, archive_path = input_paths(data_dir)
    questions = read_questions(questions_path)
    if llm is None:
        settings.validate_llm()
        llm = OpenAICompatibleClient(settings)
    answers: list[str | None] = [None] * len(questions)
    initialized = False
    try:
        async with asyncio.timeout(RUN_TIMEOUT_SECONDS):
            factory = search_factory if search_factory is not None else initialize_search
            components = await factory(archive_path, settings)
            initialized = True
            assistant = Assistant(components.evidence, llm, settings)
            # Fixed-size worker pool, so neither tasks nor requests grow with row count.
            remaining = iter(enumerate(questions))

            async def worker():
                for index, (_, raw_question) in remaining:
                    if not raw_question.strip():
                        answers[index] = "Уточните, пожалуйста, ваш вопрос и обстоятельства ситуации."
                        continue
                    try:
                        request = question_parser(raw_question)
                        result = await assistant.answer(request)
                        answers[index] = format_submission_answer(result)
                    except BackendUnavailable:
                        # Missing integration/authentication is a fatal deployment error.
                        raise
                    except (UpstreamError, UpstreamTimeout, TimeoutError, ValidationError, ValueError) as exc:
                        LOGGER.error("Case row %d failed (%s)", index + 2, type(exc).__name__)
                        answers[index] = TECHNICAL_ERROR

            async with asyncio.TaskGroup() as tasks:
                for _ in range(min(settings.max_workers, len(questions))):
                    tasks.create_task(worker())
    except TimeoutError:
        if not initialized:
            raise BackendUnavailable("Search initialization exceeded the run time budget") from None
        LOGGER.error("Batch time budget exhausted; unfinished rows marked as technical errors")
    write_submission(output_path, [(case_id, answers[index] or TECHNICAL_ERROR)
                                   for index, (case_id, _) in enumerate(questions)])
    return output_path


def main() -> int:
    try:
        asyncio.run(run())
    except (BackendUnavailable, ValueError, OSError, csv.Error) as exc:
        # Only controlled configuration/input diagnostics; never log proxy bodies or keys.
        LOGGER.error("Submission not created: %s", exc)
        return 1
    except ExceptionGroup:
        # TaskGroup wraps failures; report deployment class without upstream text/secrets.
        LOGGER.error("Submission not created: dependency failure; check search integration "
                     "and LLM authorization (401/403)")
        return 1
    except Exception:
        LOGGER.error("Submission not created: unexpected processing failure")
        return 1
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
