"""Run: uvicorn api:app. Inject real evidence/document providers via create_app."""
import asyncio
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from starlette.staticfiles import StaticFiles
from assistant import Assistant
from config import Settings
from dependencies import (BackendUnavailable, DocumentProvider, EvidenceProvider,
                          LanguageModel, UpstreamError, UpstreamTimeout)
from llm_client import OpenAICompatibleClient
from schemas import AssistantResult, ChatRequest, Document


async def _missing_evidence(request):
    raise BackendUnavailable("Search integration is not configured")


async def _missing_document(document_id):
    raise BackendUnavailable("Document integration is not configured")


def create_app(*, evidence: EvidenceProvider | None = None,
               documents: DocumentProvider | None = None,
               llm: LanguageModel | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings if settings is not None else Settings.from_env()
    assistant = Assistant(evidence if evidence is not None else _missing_evidence,
                          llm if llm is not None else OpenAICompatibleClient(settings), settings)
    document_provider = documents if documents is not None else _missing_document
    app = FastAPI(title="Юридический ассистент")
    static_dir = Path(__file__).parent / "web" / "static"
    static_files = StaticFiles(directory=static_dir, check_dir=False)

    @app.get("/style.css", include_in_schema=False)
    @app.get("/app.js", include_in_schema=False)
    async def static_asset(request: Request):
        # Only these two fixed routes are exposed. StaticFiles also rejects
        # resolved paths (including symlinks) outside its root.
        return await static_files.get_response(request.url.path.lstrip("/"), request.scope)

    @app.get("/", include_in_schema=False)
    async def index():
        path = static_dir / "index.html"
        if not path.is_file():
            raise HTTPException(404, "Страница интерфейса не найдена")
        return FileResponse(path)

    @app.post("/api/chat", response_model=AssistantResult)
    async def chat(request: ChatRequest):
        try:
            return await assistant.answer(request)
        except BackendUnavailable:
            raise HTTPException(503, "Интеграция поиска или LLM не настроена") from None
        except (TimeoutError, UpstreamTimeout):
            raise HTTPException(504, "Превышено время обработки запроса") from None
        except UpstreamError:
            raise HTTPException(502, "Не удалось получить корректный ответ зависимого сервиса") from None

    @app.get("/api/documents/{document_id}", response_model=Document)
    async def get_document(document_id: str):
        try:
            async with asyncio.timeout(settings.request_timeout):
                document = await document_provider(document_id)
            if document is None:
                raise HTTPException(404, "Документ не найден")
            if document.document_id != document_id:
                raise UpstreamError("Document ID mismatch")
            return document
        except BackendUnavailable:
            raise HTTPException(503, "Интеграция документов не настроена") from None
        except (TimeoutError, UpstreamTimeout):
            raise HTTPException(504, "Превышено время загрузки документа") from None
        except UpstreamError:
            raise HTTPException(502, "Не удалось загрузить документ") from None

    return app


app = create_app()
