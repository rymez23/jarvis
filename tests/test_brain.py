"""
Tests for the brain, using a tiny fake Ollama server (no real model needed).
Run with:  python -m pytest
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from jarvis.brain import Brain


class FakeOllama(BaseHTTPRequestHandler):
    received = []          # every /api/chat request body, so tests can inspect them

    def log_message(self, *args):   # keep test output quiet
        pass

    def do_GET(self):
        self._send({"version": "fake"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOllama.received.append(body)
        last = body["messages"][-1]["content"]
        self._send({"message": {"role": "assistant", "content": f" echo: {last} "},
                    "eval_count": 20, "eval_duration": 4_000_000_000, "load_duration": 1_500_000_000})

    def _send(self, data):
        raw = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@pytest.fixture
def server():
    FakeOllama.received = []
    srv = HTTPServer(("127.0.0.1", 0), FakeOllama)       # port 0 = pick any free port
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_ask_returns_text_and_speed(server):
    r = Brain("m", "be Jarvis", server).ask("hello")
    assert r.text == "echo: hello"
    assert r.tokens_per_second == 5.0 and r.load_seconds == 1.5


def test_personality_sent_first_and_history_remembered(server):
    b = Brain("m", "be Jarvis", server)
    b.ask("first")
    b.ask("second")
    msgs = FakeOllama.received[-1]["messages"]
    assert msgs[0] == {"role": "system", "content": "be Jarvis"}
    assert [m["content"] for m in msgs[1:]] == ["first", "echo: first", "second"]


def test_history_is_capped(server):
    b = Brain("m", "p", server, max_history=4)
    for i in range(5):
        b.ask(f"msg {i}")
    assert len(b.history) <= 5      # 4 kept + the newest reply
    assert b.history[-2]["content"] == "msg 4"


def test_forget_and_is_running(server):
    b = Brain("m", "p", server)
    b.ask("x")
    b.forget()
    assert b.history == [] and b.is_running()
    assert not Brain("m", "p", "http://127.0.0.1:9").is_running()   # nothing listens on port 9


class FakeOllamaWithTools(FakeOllama):
    """First call: asks for the get_time tool. Second call: answers using the result."""
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOllama.received.append(body)
        if body.get("tools"):
            self._send({"message": {"role": "assistant", "content": "",
                                    "tool_calls": [{"function": {"name": "get_time", "arguments": {}}}]},
                        "eval_count": 5, "eval_duration": 1_000_000_000})
        else:
            tool_result = body["messages"][-1]["content"]
            self._send({"message": {"role": "assistant", "content": f"Sir, {tool_result}"},
                        "eval_count": 5, "eval_duration": 1_000_000_000})


def test_tool_call_round_trip():
    FakeOllama.received = []
    srv = HTTPServer(("127.0.0.1", 0), FakeOllamaWithTools)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        ran = []
        b = Brain("m", "p", f"http://127.0.0.1:{srv.server_port}")
        r = b.ask("clock please", tools=[{"type": "function"}],
                  run_tool=lambda name, args: ran.append(name) or "It's 9:00 PM.")
        assert ran == ["get_time"]
        assert r.text == "Sir, It's 9:00 PM." and r.tokens == 10
        assert "tools" not in FakeOllama.received[-1]          # second request offers no tools
        roles = [m["role"] for m in b.history]
        assert roles == ["user", "assistant", "tool", "assistant"]
    finally:
        srv.shutdown()


def test_remember_adds_quick_answers_to_history(server):
    b = Brain("m", "p", server)
    b.remember("what time is it", "It's 9:00 PM.")
    assert [m["content"] for m in b.history] == ["what time is it", "It's 9:00 PM."]


def test_context_is_added_to_every_request(server):
    clock = iter(["It is Monday.", "It is Tuesday."])
    b = Brain("m", "be Jarvis", server, context=lambda: next(clock))
    b.ask("one")
    assert FakeOllama.received[-1]["messages"][0]["content"] == "be Jarvis\n\nIt is Monday."
    b.ask("two")
    assert FakeOllama.received[-1]["messages"][0]["content"] == "be Jarvis\n\nIt is Tuesday."   # refreshed each time


def test_now_context_has_real_date():
    from datetime import datetime
    from jarvis.main import now_context
    text = now_context()
    assert datetime.now().strftime("%A") in text and str(datetime.now().year) in text
