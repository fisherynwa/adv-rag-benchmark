import json
import os
import pathlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

BENCHMARK = pathlib.Path(__file__).resolve().parents[1] / "benchmark"


def _private_nltk_data():
    """Give NLTK a private copy of LlamaIndex's bundled sentence-splitter data.

    NLTK refuses to read data files that are hard-linked (uv's default on Windows) or group/world-writable
    (common on CI runners). A fresh copy, readable and writable only by its owner, passes both checks.
    """
    import importlib.util
    import shutil
    import stat
    import tempfile
    source = pathlib.Path(importlib.util.find_spec("llama_index.core").origin).parent / "_static" / "nltk_cache"
    target = pathlib.Path(tempfile.mkdtemp(prefix="nltk_data_"))          # created private (0700)
    if source.is_dir():
        shutil.copytree(source, target, dirs_exist_ok=True, copy_function=shutil.copyfile)
    for path in [target, *target.rglob("*")]:
        path.chmod(path.stat().st_mode & ~(stat.S_IWGRP | stat.S_IWOTH))
    os.environ["NLTK_DATA"] = str(target)


_private_nltk_data()          # before any test imports LlamaIndex


@pytest.fixture(scope="session")
def answer_key():
    return json.loads((BENCHMARK / "answer_key.json").read_text(encoding="utf-8"))["questions"]


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
    requests = 0

    def log_message(self, *args):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        type(self).requests += 1
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
    # Keep local requests away from any proxy. Setting NO_PROXY also stops Python reading the Windows proxy settings.
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = "127.0.0.1,localhost"
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeOllama)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()