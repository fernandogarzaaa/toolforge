"""OpenAI-compatible generation backend: request shape and proxy bypass for local servers."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from toolforge.formatting import IM_END
from toolforge.generate import OpenAICompletionsGenerator, opener_for


class _Handler(BaseHTTPRequestHandler):
    bodies: list = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Handler.bodies.append((self.path, body))
        out = json.dumps({"choices": [{"text": "echo:" + body["prompt"]}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    _Handler.bodies = []
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/v1"
    srv.shutdown()
    srv.server_close()


def test_generate_posts_raw_prompts_and_bypasses_proxy(server, monkeypatch):
    # An unreachable proxy in the environment must not be used for a loopback endpoint.
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    gen = OpenAICompletionsGenerator(server + "/", model_id="m1")
    assert gen.url == server + "/completions"
    outs = gen.generate(["a", "b"], 7, temperature=0.5, seed=3)
    assert outs == ["echo:a", "echo:b"]
    path, body = _Handler.bodies[0]
    assert path == "/v1/completions"
    assert body == {"model": "m1", "prompt": "a", "max_tokens": 7, "temperature": 0.5, "seed": 3, "stop": [IM_END]}
    assert [b["prompt"] for _, b in _Handler.bodies] == ["a", "b"]


@pytest.mark.parametrize("url", ["http://127.0.0.1:1/v1", "http://localhost:8097/v1", "http://[::1]:80/x"])
def test_loopback_urls_get_no_proxy_handler(url, monkeypatch):
    monkeypatch.setenv("http_proxy", "http://proxy.example:3128")
    handlers = opener_for(url).handlers
    proxies = [h for h in handlers if type(h).__name__ == "ProxyHandler"]
    assert all(h.proxies == {} for h in proxies)  # no handler routes through the env proxy


def test_remote_urls_keep_environment_proxies(monkeypatch):
    monkeypatch.setenv("http_proxy", "http://proxy.example:3128")
    handlers = opener_for("http://api.example.com/v1").handlers
    proxies = [h for h in handlers if type(h).__name__ == "ProxyHandler"]
    assert proxies and proxies[0].proxies.get("http") == "http://proxy.example:3128"
