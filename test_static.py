"""Local ASGI tests only; no external services and no changes to frontend assets."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import httpx
from api import create_app


class StaticTests(unittest.IsolatedAsyncioTestCase):
    async def get(self, app, path):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            return await client.get(path)

    async def test_frontend_resources_and_content_types(self):
        app = create_app()
        root = Path(__file__).parent / 'web/static'
        for path, name, types in (
            ('/', 'index.html', {'text/html'}),
            ('/style.css', 'style.css', {'text/css'}),
            ('/app.js', 'app.js', {'text/javascript', 'application/javascript'})):
            with self.subTest(path=path):
                response = await self.get(app, path)
                self.assertEqual(response.status_code, 200)
                self.assertIn(response.headers['content-type'].split(';')[0], types)
                self.assertEqual(response.content, (root / name).read_bytes())

    async def test_unknown_assets_and_traversal(self):
        app = create_app()
        for path in ('/missing.css', '/config.py', '/.git/config',
                     '/%2e%2e/config.py', '/%2e%2e/%2e%2e/requirements.txt',
                     '/style.css/../../config.py', '/app.js%2f..%2f..%2fconfig.py'):
            with self.subTest(path=path):
                self.assertEqual((await self.get(app, path)).status_code, 404)

    async def test_missing_frontend_files_and_directory(self):
        # Redirect only the application root; do not remove or modify real files.
        with tempfile.TemporaryDirectory() as directory:
            fake_module = Path(directory) / 'api.py'
            with patch('api.__file__', str(fake_module)):
                app_without_directory = create_app()
                for path in ('/', '/style.css', '/app.js'):
                    self.assertEqual((await self.get(app_without_directory, path)).status_code, 404)
                (Path(directory) / 'web/static').mkdir(parents=True)
                app_without_files = create_app()
                for path in ('/', '/style.css', '/app.js'):
                    self.assertEqual((await self.get(app_without_files, path)).status_code, 404)

    async def test_symlink_outside_static_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            static = root / 'web/static'
            static.mkdir(parents=True)
            outside = root / 'private.txt'
            outside.write_text('outside static root')
            (static / 'app.js').symlink_to(outside)
            with patch('api.__file__', str(root / 'api.py')):
                app = create_app()
                self.assertEqual((await self.get(app, '/app.js')).status_code, 404)


if __name__ == '__main__':
    unittest.main()
