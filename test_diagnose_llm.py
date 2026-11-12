"""Offline diagnostics tests; all LLM completions are synthetic stubs."""
import importlib
import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch
import diagnose_llm
from config import Settings
from dependencies import BackendUnavailable
from test_backend import FakeLLM


DRAFT = {"status": "answer", "text": "Marker ALPHA", "claims": [{
    "text": "Marker ALPHA", "fragment_ids": ["diagnostic-f"],
    "quotes": {"diagnostic-f": "Marker ALPHA"}}]}


class DiagnosticsTests(unittest.IsolatedAsyncioTestCase):
    async def test_three_stages_and_no_content_in_reports(self):
        llm = FakeLLM("PRIVATE_RESPONSE", DRAFT, {"supported": True})
        reports = await diagnose_llm.diagnose(llm, Settings())
        self.assertEqual([r['status'] for r in reports], ['ok'] * 3)
        self.assertEqual(len(llm.messages), 3)
        self.assertTrue(all(r['seconds'] >= 0 for r in reports))
        self.assertNotIn('PRIVATE_RESPONSE', repr(reports))

    async def test_safe_failure_and_skip(self):
        reports = await diagnose_llm.diagnose(FakeLLM("ok", "PRIVATE_INVALID_JSON"), Settings())
        self.assertEqual([r['status'] for r in reports], ['ok', 'invalid_model_output', 'skipped'])
        self.assertNotIn('PRIVATE_INVALID_JSON', repr(reports))
        reports = await diagnose_llm.diagnose(FakeLLM(BackendUnavailable('PRIVATE_SECRET')), Settings())
        self.assertEqual(reports[0]['status'], 'configuration_or_authorization_error')
        self.assertNotIn('PRIVATE_SECRET', repr(reports))

    async def test_import_and_no_run_flag_do_not_create_client(self):
        with patch('llm_client.OpenAICompatibleClient') as client, redirect_stdout(io.StringIO()):
            importlib.reload(diagnose_llm)
            self.assertEqual(diagnose_llm.main([]), 0)
            client.assert_not_called()
        # Restore real binding after testing module import.
        importlib.reload(diagnose_llm)


if __name__ == '__main__':
    unittest.main()
