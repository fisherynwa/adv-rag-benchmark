import json
import pathlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

BENCHMARK = pathlib.Path(__file__).resolve().parents[1] / "benchmark"


@pytest.fixture(scope="session")
def answer_key():
    return json.loads((BENCHMARK / "answer_key.json").read_text())["questions"]


@pytest.fixture(scope="session")
def report_pages():
    """Page number -> whitespace-normalised text of the rendered PDF."""
    from pypdf import PdfReader
    from ragbench.metrics import norm
    reader = PdfReader(BENCHMARK / "kerrowan_annual_report_2025.pdf")
    return {i + 1: norm(p.extract_text()) for i, p in enumerate(reader.pages)}


class _FakeOllama(BaseHTTPRequestHandler):
    """Answers every chat request with a reply all LlamaIndex judges can parse. Keep-alive, like the real server."""
    protocol_version = "HTTP/1.1"
    reply = "4.0\nYES, supported. [RESULT] 3.0"

    def log_message(self, *args):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        body = json.dumps({"model": "fake", "created_at": "2026-01-01T00:00:00Z",
                           "message": {"role": "assistant", "content": self.reply},
                           "done": True, "done_reason": "stop"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture(scope="session")
def fake_ollama_url():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeOllama)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
