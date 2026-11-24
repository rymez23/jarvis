"""
THE BRAIN: sends your words to the local AI model and gets its reply.

How it works:
  Ollama runs in the background on this PC and listens at http://127.0.0.1:11434
  (127.0.0.1 means "this computer"; nothing goes to the internet). We send it
  JSON over HTTP and it sends JSON back. That's the same way programs talk to
  web services, just pointed at your own machine.

  The model has no memory between requests. To make it remember the
  conversation, we resend the recent messages every time. That list is
  `self.history`. A longer history means more text to read each turn, which is
  slower on your CPU, so it's capped by "max_history" in settings.json.

Uses only Python's standard library.
"""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass


@dataclass
class Reply:
    text: str
    tokens: int            # how many tokens (word pieces) the model generated
    seconds: float         # how long generating them took
    load_seconds: float    # time spent loading the model into memory (first request only)

    @property
    def tokens_per_second(self) -> float:
        return self.tokens / self.seconds if self.seconds else 0.0


class Brain:
    def __init__(self, model: str, personality: str, url: str = "http://127.0.0.1:11434",
                 max_history: int = 10, think: bool = False, timeout: float = 300, context=None):
        self.model = model
        self.personality = personality
        self.url = url.rstrip("/")
        self.max_history = max_history
        self.think = think        # "thinking" models reason before answering: smarter but much slower on CPU
        self.timeout = timeout
        self.history: list[dict] = []
        # context: an optional function returning up-to-date facts (like the current
        # date and time) that are added to the instructions on EVERY request. The model
        # has no clock; without this it guesses, and guesses wrong.
        self.context = context

    # ------------------------------------------------------------------ talking to Ollama

    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            self.url + path, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.load(resp)

    def is_running(self) -> bool:
        """True if Ollama is up. (GET /api/version is the cheapest request it accepts.)"""
        try:
            with urllib.request.urlopen(self.url + "/api/version", timeout=3):
                return True
        except OSError:
            return False

    # ------------------------------------------------------------------ the main job

    def ask(self, text: str, tools: list | None = None, run_tool=None) -> Reply:
        """Send one message (plus the recent conversation) and return the reply.

        If `tools` is given, the model may answer with a request to run one
        (a "tool call") instead of text. We run it with `run_tool(name, args)`,
        add the result to the conversation, and ask again for the final wording.
        Only one round of tool calls is allowed, so a confused model can't loop.
        """
        self.history.append({"role": "user", "content": text})
        self.history = self.history[-self.max_history:]          # forget the oldest messages

        data = self._chat(tools)
        tokens, seconds, load = self._timings(data)
        calls = data["message"].get("tool_calls") or []
        if calls and run_tool:
            self.history.append({"role": "assistant", "content": data["message"].get("content", ""),
                                 "tool_calls": calls})
            for call in calls:
                fn = call.get("function", {})
                result = run_tool(fn.get("name", ""), fn.get("arguments") or {})
                self.history.append({"role": "tool", "tool_name": fn.get("name", ""), "content": result})
            data = self._chat(None)                          # no tools this time: just answer
            t2, s2, l2 = self._timings(data)
            tokens, seconds, load = tokens + t2, seconds + s2, load + l2

        answer = data["message"].get("content", "").strip()
        self.history.append({"role": "assistant", "content": answer})
        return Reply(text=answer, tokens=tokens, seconds=seconds, load_seconds=load)

    def _chat(self, tools: list | None) -> dict:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": self.system_prompt()}] + self.history,
            "stream": False,       # wait for the whole answer
            "think": self.think,
            "keep_alive": "30m",   # keep the model loaded in RAM for 30 min so the next answer is quicker
        }
        if tools:
            payload["tools"] = tools
        return self._post("/api/chat", payload)

    def system_prompt(self) -> str:
        extra = self.context() if self.context else ""
        return f"{self.personality}\n\n{extra}".strip()

    @staticmethod
    def _timings(data: dict) -> tuple[int, float, float]:
        # Ollama reports its timings in nanoseconds (billionths of a second)
        return (data.get("eval_count", 0), data.get("eval_duration", 0) / 1e9,
                data.get("load_duration", 0) / 1e9)

    def remember(self, user_text: str, answer: str) -> None:
        """Add an exchange that was handled without the model (a quick-match action),
        so the model still knows about it later in the conversation."""
        self.history += [{"role": "user", "content": user_text}, {"role": "assistant", "content": answer}]
        self.history = self.history[-self.max_history:]

    def forget(self) -> None:
        """Start a fresh conversation."""
        self.history.clear()
