"""Portable-release checks without HTTP binding, model inference, or services."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from c1_webui import AppServer, RequestHandler


class PortableWebUITest(unittest.TestCase):
    def test_disabled_gallery_does_not_load_local_dataset(self):
        with patch("c1_webui.ThreadingHTTPServer.__init__", return_value=None), \
                patch("c1_webui.Gallery", side_effect=AssertionError("local dataset read")):
            server = AppServer(("127.0.0.1", 7860), object(), enable_gallery=False)
        self.assertIsNone(server.gallery)

    def test_disabled_gallery_routes_return_404(self):
        for method, paths in [("do_GET", ["/api/gallery", "/api/gallery/image/1", "/gallery.html"]),
                              ("do_POST", ["/api/gallery/selection", "/api/gallery/finalize"])]:
            for path in paths:
                with self.subTest(method=method, path=path):
                    handler = RequestHandler.__new__(RequestHandler)
                    handler.path = path
                    handler.server = SimpleNamespace(gallery=None)
                    errors = []
                    handler.send_error = lambda code, message: errors.append(code)
                    getattr(handler, method)()
                    self.assertEqual(errors, [404])


if __name__ == "__main__":
    unittest.main()
