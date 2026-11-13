"""Offline CSV/configuration integration tests with neutral synthetic markers."""
import asyncio
import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch
import httpx
from config import Settings
from dependencies import BackendUnavailable, SearchComponents, UpstreamError, UpstreamTimeout
from llm_client import OpenAICompatibleClient
from schemas import AssistantResult, ChatRequest, Claim, Fragment
from solution import (format_submission_answer, input_paths, parse_question, read_questions,
                      run, write_submission, TECHNICAL_ERROR)
from verification import validate_citations
from schemas import Draft


FRAGMENT = Fragment(fragment_id="fixture-f", document_id="fixture-d",
                    document_title='Тестовая запись, "A"', section="метка A", text="Маркер A")


def verified_result(text="Маркер A"):
    return AssistantResult(status="answer", text=text,
                           claims=[Claim(text=text, fragment_ids=[FRAGMENT.fragment_id])], fragments=[FRAGMENT])


class NeutralLLM:
    async def complete(self, messages):
        payload = json.loads(messages[1]["content"])
        if "draft" in payload:
            return '{"supported": true}'
        return json.dumps({"status": "answer", "text": "Маркер A", "claims": [{
            "text": "Маркер A", "fragment_ids": [FRAGMENT.fragment_id],
            "quotes": {FRAGMENT.fragment_id: "Маркер A"}}]})


class CSVTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def make_csv(self, rows, header=("id", "question")):
        path = self.root / "questions.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(rows)
        return path

    def test_ids_and_order(self):
        rows = [("0001", "строка 1"), (" 02 ", "строка 2"), ("1e3", "строка 3")]
        self.assertEqual(read_questions(self.make_csv(rows)), rows)

    def test_bad_ids_and_headers(self):
        for rows in ([('a', 'x'), ('a', 'y')], [('', 'x')], []):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                read_questions(self.make_csv(rows))
        with self.assertRaises(ValueError):
            read_questions(self.make_csv([('x', 'y')], ("id", "answer")))

    def test_missing_files(self):
        with self.assertRaisesRegex(ValueError, 'questions.csv'):
            input_paths(self.root)
        self.make_csv([('x', 'y')])
        with self.assertRaisesRegex(ValueError, 'knowledge-base.zip'):
            input_paths(self.root)

    def test_csv_escaping_and_answer_limits(self):
        output = self.root / 'submission.csv'
        rows = [('0001', 'Текст, "кавычки"\nновая строка'), ('b', 'я' * 10000)]
        write_submission(output, rows)
        with output.open(encoding="utf-8", newline="") as stream:
            data = list(csv.reader(stream))
        self.assertEqual(data, [['id', 'answer']] + [list(row) for row in rows])
        original = output.read_bytes()
        for invalid in ([('x', '')], [('x', 'a' * 10001)], [('x', 'a'), ('x', 'b')]):
            with self.assertRaises(ValueError):
                write_submission(output, invalid)
            self.assertEqual(output.read_bytes(), original)

    def test_source_labels_from_metadata(self):
        answer = format_submission_answer(verified_result())
        self.assertIn(FRAGMENT.document_title, answer)
        self.assertIn(FRAGMENT.section, answer)
        self.assertNotIn(FRAGMENT.fragment_id, answer)
        with self.assertRaises(ValueError):
            format_submission_answer(verified_result('x' * 10000))
        for status in ('clarification', 'insufficient'):
            self.assertEqual(format_submission_answer(AssistantResult(status=status, text='Пояснение')), 'Пояснение')

    def test_official_dialogue_labels_are_parsed(self):
        raw = 'Пользователь: строка\nАссистент: строка\nПользователь: последняя строка'
        request = parse_question(raw)
        self.assertEqual(request.question, 'последняя строка')
        self.assertEqual([m.role for m in request.history], ['user', 'assistant'])
        self.assertEqual(request.history[0].content, 'строка')

    def test_citation_integrity(self):
        for source_id, quote in ((FRAGMENT.fragment_id, 'Маркер A'), ('invented', 'Маркер A'),
                                 (FRAGMENT.fragment_id, 'Маркер B')):
            value = Draft(status='answer', text='Маркер A', claims=[{
                'text': 'Маркер A', 'fragment_ids': [source_id], 'quotes': {source_id: quote}}])
            self.assertEqual(validate_citations(value, [FRAGMENT]), source_id == FRAGMENT.fragment_id and quote == 'Маркер A')


    def test_utf8_bom_input(self):
        path = self.make_csv([('001', 'Тестовый вопрос')])
        path.write_bytes(b'\xef\xbb\xbf' + path.read_bytes())
        self.assertEqual(read_questions(path), [('001', 'Тестовый вопрос')])

    def test_ambiguous_dialogue_stays_opaque(self):
        for raw in ('Обычная строка', 'Пользователь: один\nПользователь: два\nПользователь: три',
                    'Введение\nПользователь: один\nАссистент: два\nПользователь: три'):
            self.assertEqual(parse_question(raw).question, raw)
            self.assertEqual(parse_question(raw).history, [])


class BatchTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # Opaque placeholder: production code must not parse/index the archive itself.
        (self.root / 'knowledge-base.zip').write_bytes(b'test-only-placeholder')
        self.output = self.root / 'submission.csv'
        self.calls = []
        self.factory_calls = []

    def questions(self, rows):
        with (self.root / 'questions.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['id', 'question'])
            writer.writerows(rows)

    async def factory(self, path, settings):
        self.factory_calls.append((path, settings))
        async def evidence(request):
            self.calls.append(request)
            if request.query == 'slow':
                await asyncio.sleep(0.01)
            return [FRAGMENT]
        async def documents(document_id):
            return None
        return SearchComponents(evidence, documents)

    async def execute(self, **kwargs):
        return await run(data_dir=self.root, output_path=self.output, search_factory=self.factory,
                         llm=NeutralLLM(), settings=Settings(max_workers=2), **kwargs)

    async def test_independence_single_init_order_empty_question(self):
        self.questions([('001', 'slow'), ('002', 'fast'), ('003', ' ')])
        await self.execute()
        with self.output.open(encoding='utf-8', newline='') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([r['id'] for r in rows], ['001', '002', '003'])
        self.assertTrue(all(r['answer'].strip() for r in rows))
        self.assertEqual(len(self.factory_calls), 1)
        self.assertEqual(self.factory_calls[0][0], self.root / 'knowledge-base.zip')
        self.assertEqual(len(self.calls), 2)
        self.assertTrue(all(not r.user_context for r in self.calls))

    async def test_confirmed_decoder_injection(self):
        self.questions([('a', 'encoded-a'), ('b', 'encoded-b')])
        def decoder(raw):
            return ChatRequest(question=raw, history=[{'role': 'user', 'content': 'context-' + raw}])
        await self.execute(question_parser=decoder)
        self.assertEqual({r.query: r.user_context[0] for r in self.calls},
                         {'encoded-a': 'context-encoded-a', 'encoded-b': 'context-encoded-b'})

    async def test_official_dialogue_through_solution_to_search(self):
        self.questions([('a', 'Пользователь: context-a\nАссистент: not-evidence\nПользователь: current-a'),
                        ('b', 'current-b')])
        await self.execute()
        by_query = {q.query: q for q in self.calls}
        self.assertEqual(by_query['current-a'].user_context, ('context-a',))
        self.assertEqual(by_query['current-b'].user_context, ())

    async def test_missing_real_search_does_not_create_output(self):
        self.questions([('a', 'x')])
        with self.assertRaisesRegex(BackendUnavailable, 'integration is unfinished'):
            await run(data_dir=self.root, output_path=self.output, llm=NeutralLLM())
        self.assertFalse(self.output.exists())

    async def test_missing_llm_configuration(self):
        self.questions([('a', 'x')])
        with self.assertRaisesRegex(BackendUnavailable, 'Missing LLM settings'):
            await run(data_dir=self.root, output_path=self.output, settings=Settings(), search_factory=self.factory)
        self.assertEqual(self.factory_calls, [])
        self.assertFalse(self.output.exists())

    async def test_technical_failure_has_nonlegal_diagnostic(self):
        self.questions([('a', 'x')])
        failing = AsyncMock()
        failing.complete.side_effect = UpstreamTimeout('not logged')
        await run(data_dir=self.root, output_path=self.output, search_factory=self.factory, llm=failing)
        with self.output.open(encoding='utf-8', newline='') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(rows[0]['answer'], TECHNICAL_ERROR)

    async def test_batch_timeout_keeps_every_id(self):
        self.questions([('a', 'x'), ('b', 'y')])
        blocked = AsyncMock()
        async def wait(messages):
            await asyncio.Event().wait()
        blocked.complete.side_effect = wait
        with patch('solution.RUN_TIMEOUT_SECONDS', 0.02):
            await run(data_dir=self.root, output_path=self.output, search_factory=self.factory, llm=blocked)
        with self.output.open(encoding='utf-8', newline='') as stream:
            self.assertEqual([r['id'] for r in csv.DictReader(stream)], ['a', 'b'])

    async def test_fatal_auth_does_not_publish_output(self):
        self.questions([('a', 'x')])
        failing = AsyncMock()
        failing.complete.side_effect = BackendUnavailable('LLM authorization failed')
        with self.assertRaises(ExceptionGroup):
            await run(data_dir=self.root, output_path=self.output, search_factory=self.factory, llm=failing)
        self.assertFalse(self.output.exists())

    async def test_data_dir_env_and_default_output(self):
        self.questions([('a', 'x')])
        with patch.dict(os.environ, {'DATA_DIR': str(self.root)}), patch('solution.Path.cwd', return_value=self.root):
            path = await run(search_factory=self.factory, llm=NeutralLLM())
        self.assertEqual(path, self.output)


class ConfigAndRetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_env_alias_priority_and_generation(self):
        env = {'LLM_MODEL': 'first', 'OPENAI_MODEL': 'second', 'GEN_MODEL': 'third',
               'LLM_BASE_URL': 'https://example.invalid/v1', 'OPENAI_BASE_URL': 'ignored',
               'LLM_API_KEY': 'synthetic-secret', 'OPENAI_API_KEY': 'ignored',
               'TEMPERATURE': '0.3', 'MAX_TOKENS': '900', 'MAX_WORKERS': '3', 'TOP_K': '7',
               'CHUNKER': 'supplied', 'EMB_MODEL': 'embedding', 'RERANKER_MODEL': 'reranker', 'RERANKER_TOP_N': '4'}
        with patch.dict(os.environ, env, clear=True):
            settings = Settings.from_env()
            self.assertEqual(settings.llm_model, 'first')
            self.assertEqual(settings.max_workers, 3)
            self.assertEqual(settings.top_k, 7)
            self.assertEqual(settings.chunker, 'supplied')
            self.assertNotIn('synthetic-secret', repr(settings))
            del os.environ['LLM_MODEL']
            self.assertEqual(Settings.from_env().llm_model, 'second')
            del os.environ['OPENAI_MODEL']
            self.assertEqual(Settings.from_env().llm_model, 'third')

    async def test_bounded_retry_and_correct_endpoint(self):
        real_client = httpx.AsyncClient
        for mode, expected_count, expected_error in (('recovered', 3, None), ('401', 1, BackendUnavailable),
                ('429', 3, UpstreamError), ('503', 3, UpstreamError), ('network', 3, UpstreamError),
                ('timeout', 3, UpstreamTimeout), ('400', 1, UpstreamError)):
            calls = []
            def handler(request):
                calls.append(request)
                self.assertEqual(str(request.url), 'https://example.invalid/proxy/chat/completions')
                payload = json.loads(request.content)
                self.assertEqual((payload['temperature'], payload['max_tokens']), (0.4, 800))
                if mode == 'network':
                    raise httpx.ConnectError('test', request=request)
                if mode == 'timeout':
                    raise httpx.ReadTimeout('test', request=request)
                if mode == 'recovered' and len(calls) == 3:
                    return httpx.Response(200, json={'choices': [{'message': {'content': 'ok'}}]})
                return httpx.Response(429 if mode == 'recovered' else int(mode), json={})
            settings = Settings(llm_base_url='https://example.invalid/proxy/', llm_api_key='test-only',
                                llm_model='env-model', temperature=0.4, max_tokens=800)
            with self.subTest(mode=mode), patch('llm_client.httpx.AsyncClient',
                    side_effect=lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw)), \
                    patch('llm_client.asyncio.sleep', new_callable=AsyncMock) as sleep:
                if expected_error:
                    with self.assertRaises(expected_error):
                        await OpenAICompatibleClient(settings).complete([])
                else:
                    self.assertEqual(await OpenAICompatibleClient(settings).complete([]), 'ok')
                self.assertEqual(len(calls), expected_count)
                self.assertEqual(sleep.await_count, expected_count - 1)


if __name__ == '__main__':
    unittest.main()
