"""Synthetic fixtures only; no real knowledge-base content or proxy requests."""
import json
import unittest
from unittest.mock import patch
import httpx
from pydantic import ValidationError
from assistant import Assistant
from config import Settings
from dependencies import BackendUnavailable, UpstreamError, UpstreamTimeout, SearchQuery, prepare_search_query
from llm_client import OpenAICompatibleClient
from schemas import AssistantResult, ChatRequest, Fragment


SOURCE = Fragment(fragment_id="test-f1", document_id="test-doc", document_title="ТЕСТОВЫЙ документ",
                  section="п. 1", text="ТЕСТ: требуется согласование.")


def draft(source_id="test-f1", quote="требуется согласование."):
    return {"status": "answer", "text": "Не использовать этот непроверенный текст",
            "claims": [{"text": "ТЕСТ: нужно согласование (ТЕСТОВЫЙ документ, п. 1).",
                        "fragment_ids": [source_id], "quotes": {source_id: quote}}]}


class FakeEvidence:
    def __init__(self, fragments=None):
        self.fragments = [SOURCE] if fragments is None else fragments
        self.requests = []

    async def __call__(self, request):
        self.requests.append(request)
        return self.fragments


class FakeLLM:
    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.messages = []

    async def complete(self, messages):
        self.messages.append(messages)
        value = self.outputs.pop(0)
        if isinstance(value, Exception):
            raise value
        return json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value


class SchemaTests(unittest.TestCase):
    def test_invalid_requests(self):
        for data in ({"question": " "}, {"question": "x", "history": [{"role": "system", "content": "x"}]},
                     {"question": "x", "unexpected": True}):
            with self.subTest(data=data), self.assertRaises(ValidationError):
                ChatRequest.model_validate(data)

    def test_sources_and_offsets(self):
        with self.assertRaises(ValidationError):
            Fragment(**{**SOURCE.model_dump(), "start": 0})
        with self.assertRaises(ValidationError):
            AssistantResult(status="answer", text="x", claims=[{"text": "x", "fragment_ids": ["fake"]}])
        for timeout in (0, 60, float("nan")):
            with self.assertRaises(ValueError):
                Settings(request_timeout=timeout)


class AssistantTests(unittest.IsolatedAsyncioTestCase):
    async def test_history_and_verified_answer(self):
        evidence, llm = FakeEvidence(), FakeLLM(draft(), {"supported": True})
        request = ChatRequest(question="Текущий вопрос", history=[{"role": "user", "content": "Предыдущий вопрос"},
                                                               {"role": "assistant", "content": "Старая гипотеза"}])
        result = await Assistant(evidence, llm, Settings()).answer(request)
        self.assertEqual(result.status, "answer")
        self.assertEqual(evidence.requests[0], prepare_search_query(request))
        payload = json.loads(llm.messages[0][1]["content"])
        self.assertEqual(payload["history"], [m.model_dump() for m in request.history])
        self.assertEqual(llm.messages[0][1]["content"].count("Текущий вопрос"), 1)
        self.assertNotIn("непроверенный", result.text)
        self.assertEqual(result.fragments, [SOURCE])
        self.assertEqual(result.claims[0].fragment_ids, [SOURCE.fragment_id])

    async def test_non_answers(self):
        for status in ("clarification", "insufficient"):
            with self.subTest(status=status):
                llm = FakeLLM({"status": status, "text": "Укажите дату"})
                result = await Assistant(FakeEvidence([]), llm, Settings()).answer(ChatRequest(question="Вопрос"))
                self.assertEqual(result.status, status)
                self.assertEqual(result.fragments, [])

    async def test_invalid_citations_and_missing_sources(self):
        for fragments, value in (([SOURCE], draft("invented")), ([SOURCE], draft(quote="Выдуманная цитата")),
                                 ([], draft())):
            llm = FakeLLM(value)
            result = await Assistant(FakeEvidence(fragments), llm, Settings()).answer(ChatRequest(question="x"))
            self.assertEqual(result.status, "insufficient")
            self.assertEqual(result.claims, [])
            self.assertEqual(len(llm.messages), 1)

    async def test_semantic_rejection(self):
        result = await Assistant(FakeEvidence(), FakeLLM(draft(), {"supported": False}), Settings()).answer(ChatRequest(question="x"))
        self.assertEqual(result.status, "insufficient")

    async def test_malformed_output(self):
        with self.assertRaises(UpstreamError):
            await Assistant(FakeEvidence(), FakeLLM("not JSON"), Settings()).answer(ChatRequest(question="x"))

    async def test_duplicate_sources(self):
        with self.assertRaises(UpstreamError):
            await Assistant(FakeEvidence([SOURCE, SOURCE]), FakeLLM(), Settings()).answer(ChatRequest(question="x"))

    async def test_context_budget(self):
        llm = FakeLLM({"status": "insufficient", "text": "Нет оснований"})
        await Assistant(FakeEvidence(), llm, Settings(max_context_chars=1)).answer(ChatRequest(question="x"))
        self.assertEqual(json.loads(llm.messages[0][1]["content"])["evidence"], [])


class APITests(unittest.IsolatedAsyncioTestCase):
    async def call(self, app, method, path, **kwargs):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            return await client.request(method, path, **kwargs)

    async def test_chat_contract(self):
        from api import create_app
        for status in ("answer", "clarification", "insufficient"):
            llm = FakeLLM(draft(), {"supported": True}) if status == "answer" else FakeLLM({"status": status, "text": "Тест"})
            response = await self.call(create_app(evidence=FakeEvidence(), llm=llm), "POST", "/api/chat",
                                       json={"question": "Тест", "history": []})
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(set(data), {"status", "text", "claims", "fragments"})
            self.assertEqual(data["status"], status)
            for fragment in data["fragments"]:
                self.assertIsNone(fragment["start"])
                self.assertIsNone(fragment["end"])

    async def test_validation_and_errors(self):
        from api import create_app
        app = create_app()
        self.assertEqual((await self.call(app, "POST", "/api/chat", json={"question": " "})).status_code, 422)
        self.assertEqual((await self.call(app, "POST", "/api/chat", json={"question": "x"})).status_code, 503)
        for error, expected in ((UpstreamError("secret"), 502), (UpstreamTimeout("secret"), 504),
                                (BackendUnavailable("secret"), 503), (TimeoutError("secret"), 504)):
            app = create_app(evidence=FakeEvidence(), llm=FakeLLM(error))
            response = await self.call(app, "POST", "/api/chat", json={"question": "x"})
            self.assertEqual(response.status_code, expected)
            self.assertNotIn("secret", response.text)
        app = create_app(evidence=FakeEvidence(), llm=FakeLLM("bad JSON"))
        self.assertEqual((await self.call(app, "POST", "/api/chat", json={"question": "x"})).status_code, 502)

    async def test_document_and_html(self):
        from api import create_app
        from pathlib import Path
        from schemas import Document
        async def documents(document_id):
            if document_id == SOURCE.document_id:
                return Document(document_id=document_id, text=SOURCE.text)
            return None
        app = create_app(documents=documents)
        response = await self.call(app, "GET", "/api/documents/test-doc")
        self.assertEqual(response.json(), {"document_id": "test-doc", "text": SOURCE.text})
        self.assertEqual(response.status_code, 200)
        self.assertEqual((await self.call(app, "GET", "/api/documents/unknown")).status_code, 404)
        response = await self.call(app, "GET", "/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, (Path(__file__).parent / "web/static/index.html").read_bytes())

    async def test_document_errors(self):
        from api import create_app
        from schemas import Document
        for error, expected in ((BackendUnavailable(), 503), (UpstreamError(), 502), (UpstreamTimeout(), 504)):
            async def documents(document_id):
                raise error
            response = await self.call(create_app(documents=documents), "GET", "/api/documents/x")
            self.assertEqual(response.status_code, expected)
        async def wrong_document(document_id):
            return Document(document_id="other", text="Тест")
        self.assertEqual((await self.call(create_app(documents=wrong_document), "GET", "/api/documents/x")).status_code, 502)

    async def test_whole_request_deadline(self):
        import asyncio
        from api import create_app
        async def blocked(request):
            await asyncio.Event().wait()
        settings = Settings(request_timeout=0.02, llm_timeout=0.01)
        response = await self.call(create_app(evidence=blocked, settings=settings), "POST", "/api/chat", json={"question": "x"})
        self.assertEqual(response.status_code, 504)


class ClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_settings(self):
        with self.assertRaises(BackendUnavailable):
            await OpenAICompatibleClient(Settings()).complete([])

    async def test_wire_contract_and_errors(self):
        settings = Settings(llm_base_url="https://synthetic.invalid/v1", llm_api_key="test-only", llm_model="test-model")
        real_client = httpx.AsyncClient
        for mode in ("ok", "http_error", "timeout", "malformed"):
            def handler(request):
                self.assertEqual(str(request.url), "https://synthetic.invalid/v1/chat/completions")
                self.assertEqual(request.headers["Authorization"], "Bearer test-only")
                self.assertEqual(json.loads(request.content)["model"], "test-model")
                if mode == "timeout":
                    raise httpx.ReadTimeout("synthetic", request=request)
                if mode == "http_error":
                    return httpx.Response(429, json={"error": "test"})
                if mode == "malformed":
                    return httpx.Response(200, json={"choices": []})
                return httpx.Response(200, json={"choices": [{"message": {"content": "test-result"}}]})
            with patch("llm_client.httpx.AsyncClient", side_effect=lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw)):
                client = OpenAICompatibleClient(settings)
                if mode == "ok":
                    self.assertEqual(await client.complete([]), "test-result")
                else:
                    with self.assertRaises(UpstreamTimeout if mode == "timeout" else UpstreamError):
                        await client.complete([])


class HistoryAndSourceRegressionTests(unittest.IsolatedAsyncioTestCase):
    async def test_long_history_no_422(self):
        from api import create_app
        history = [{"role": role, "content": f"message-{i}-{role}"}
                   for i in range(70) for role in ("user", "assistant")]
        evidence = FakeEvidence()
        app = create_app(evidence=evidence, llm=FakeLLM({"status": "clarification", "text": "Уточнение"}))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/api/chat", json={"question": "current", "history": history})
        self.assertEqual(response.status_code, 200)
        actual = evidence.requests[0].user_context
        self.assertLessEqual(len(actual), 24)
        self.assertEqual(actual[-1], "message-69-user")
        self.assertTrue(all(item.endswith("-user") for item in actual))

    async def test_large_messages_budget_and_question_once(self):
        from prompts import answer_messages
        history = [{"role": role, "content": "z" * 30000 + "TAIL"}
                   for _ in range(80) for role in ("user", "assistant")]
        request = ChatRequest(question="UNIQUE_CURRENT", history=history)
        self.assertLessEqual(len(json.dumps([m.model_dump() for m in request.history], ensure_ascii=False)), 12000)
        self.assertTrue(request.history[-1].content.endswith("TAIL"))
        self.assertIn("опущено", request.history[-1].content)
        messages = answer_messages(request, [])
        self.assertEqual(messages[1]["content"].count("UNIQUE_CURRENT"), 1)
        self.assertEqual(ChatRequest.model_validate(request.model_dump()), request)
        self.assertEqual(ChatRequest(question="another").history, [])

    async def test_exact_source_roundtrip_and_nonblank(self):
        from schemas import Document
        text = " \t\r\nМаркер 😀\nвнутренние  пробелы\r\n "
        fragment = Fragment(**{**SOURCE.model_dump(), "text": text})
        document = Document(document_id="d", text=text)
        self.assertEqual(fragment.text, text)
        self.assertEqual(document.text, text)
        self.assertEqual(Fragment.model_validate_json(fragment.model_dump_json()).text, text)
        self.assertIsNone(fragment.start)
        self.assertIsNone(fragment.end)
        for field in ("fragment_id", "document_id", "document_title", "text"):
            with self.subTest(field=field), self.assertRaises(ValidationError):
                Fragment(**{**SOURCE.model_dump(), field: " \t\n"})
        with self.assertRaises(ValidationError):
            Document(document_id="d", text=" \n")

    async def test_search_bridge_has_no_assistant_history(self):
        from dependencies import prepare_search_query
        request = ChatRequest(question="current", history=[
            {"role": "user", "content": "context"},
            {"role": "assistant", "content": "untrusted assertion"}])
        query = prepare_search_query(request)
        self.assertEqual(query.query, "current")
        self.assertEqual(query.user_context, ("context",))
        self.assertEqual(query.filters, ())
        self.assertFalse(hasattr(query, "history"))


class SearchQueryIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_assistant_passes_search_query_not_chat_request(self):
        evidence = FakeEvidence([])
        llm = FakeLLM({"status": "insufficient", "text": "Нет оснований"})
        request = ChatRequest(question="current-marker", history=[
            {"role": "user", "content": "previous-user-marker"},
            {"role": "assistant", "content": "assistant-marker"}])
        await Assistant(evidence, llm, Settings()).answer(request)
        self.assertEqual(len(evidence.requests), 1)
        query = evidence.requests[0]
        self.assertIsInstance(query, SearchQuery)
        self.assertNotIsInstance(query, ChatRequest)
        self.assertEqual(query.query, "current-marker")
        self.assertEqual(query.user_context, ("previous-user-marker",))
        self.assertEqual(query.filters, ())
        self.assertNotIn("assistant-marker", repr(query))
        self.assertEqual(repr(query).count("current-marker"), 1)
        # Full bounded history still reaches the LLM for conversation understanding.
        payload = json.loads(llm.messages[0][1]["content"])
        self.assertEqual(payload["history"][1]["content"], "assistant-marker")

    async def test_separate_calls_do_not_share_context(self):
        evidence = FakeEvidence([])
        llm = FakeLLM(*[{"status": "insufficient", "text": "Нет оснований"}] * 2)
        assistant = Assistant(evidence, llm, Settings())
        await assistant.answer(ChatRequest(question="first", history=[{"role": "user", "content": "first-context"}]))
        await assistant.answer(ChatRequest(question="second"))
        self.assertEqual(evidence.requests[0].user_context, ("first-context",))
        self.assertEqual(evidence.requests[1].user_context, ())


if __name__ == "__main__":
    unittest.main()
