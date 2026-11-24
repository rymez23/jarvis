"""
ACTIONS (stage 4): things Jarvis can DO, not just talk about.

Each action is an ordinary Python function that returns a sentence for Jarvis
to say. There are two ways a request reaches an action:

  1. QUICK MATCH (instant, reliable)
     Common phrasings are recognised with regular expressions (text patterns),
     e.g. "what time is it". No AI involved, so no waiting and no mistakes.

  2. TOOL CALLING (flexible, slower, sometimes wrong)
     Anything else goes to the AI model along with a description of each
     action ("tools"). If the model decides an action fits, Ollama returns a
     structured request like {"name": "open_app", "arguments": {"app": "vs code"}}
     instead of text. We run that function, give the model the result, and it
     words the final answer. Small models sometimes pick the wrong tool, which
     is why the quick matches exist.

SAFETY: open_app only opens apps on the APPS list below. The model can't make
Jarvis run any other program or command, whatever it says.

To add an action:
  1. write a function that returns a sentence,
  2. add it to TOOLS (name, description, parameters),
  3. optionally add a quick-match pattern in Actions.quick_match().
"""
from __future__ import annotations

import os
import re
import sys
from datetime import datetime
from pathlib import Path

from .brief_reader import find, latest_brief_path, read_brief

# ---------------------------------------------------------------- apps Jarvis may open
# name -> (what to launch, other names people use)
APPS: dict[str, tuple[str, list[str]]] = {
    "chrome":       ("chrome", ["google chrome", "browser", "the browser", "internet"]),
    "edge":         ("msedge", ["microsoft edge"]),
    "vs code":      (str(Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Microsoft VS Code" / "Code.exe"),
                     ["vscode", "visual studio code", "code", "code editor", "my editor"]),
    "notepad":      ("notepad.exe", ["note pad", "notes"]),
    "calculator":   ("calc.exe", ["calc"]),
    "file explorer": ("explorer.exe", ["explorer", "files", "my files", "file manager"]),
    "settings":     ("ms-settings:", ["windows settings", "pc settings"]),
    "terminal":     ("wt.exe", ["windows terminal", "command prompt", "powershell", "command line"]),
    "task manager": ("taskmgr.exe", []),
}


def resolve_app(spoken: str) -> str | None:
    """'Visual Studio Code' -> 'vs code'. Returns None if it isn't on the list."""
    spoken = re.sub(r"[^a-z ]", "", spoken.lower()).strip()
    spoken = re.sub(r"^(the|my|up)\s+", "", spoken).strip()
    for name, (_target, aliases) in APPS.items():
        if spoken == name or spoken in aliases:
            return name
    return None


def launch(target: str) -> None:
    """Ask Windows to open a program, like double-clicking it. Doesn't wait for it."""
    if sys.platform == "win32":
        os.startfile(target)


# ---------------------------------------------------------------- small helpers

def ordinal(n: int) -> str:
    """1 -> '1st', 22 -> '22nd', 13 -> '13th' (11-13 are the exceptions)."""
    suffix = "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def speakable(text: str) -> str:
    """Make page text sound right: arrows and menu chevrons become words, and long
    version numbers are shortened ('158.0.8127.37' -> '158'; nobody wants to hear the rest)."""
    text = re.sub(r"\b(\d+)(?:\.\d+){2,}\b", r"\1", text)
    return re.sub(r"\s+", " ", text.replace("→", " to ").replace("›", ",")).strip()


def first_sentence(text: str) -> str:
    text = re.sub(r"\s*\(.*?\)", "", text)           # drop "(Settings › Windows Update)"
    return text.split(". ")[0].rstrip(".") + "."


# ---------------------------------------------------------------- the actions

class Actions:
    def __init__(self, brief_path: Path | None = None, launcher=launch, clock=datetime.now):
        self.brief_path = brief_path or latest_brief_path()
        self.launcher = launcher        # swappable so tests don't really open apps
        self.clock = clock              # swappable so tests can pick the time

    def get_time(self) -> str:
        return f"It's {self.clock().strftime('%I:%M %p').lstrip('0')}."

    def get_date(self) -> str:
        now = self.clock()
        return f"Today is {now.strftime('%A, %B')} {ordinal(now.day)}."

    def open_app(self, app: str) -> str:
        name = resolve_app(app)
        if not name:
            return f"I'm not allowed to open {app}. I can open: {', '.join(APPS)}."
        self.launcher(APPS[name][0])
        return f"Opening {name}."

    def _load_brief(self):
        if not self.brief_path.exists():
            return None, None
        made_at, sections = read_brief(self.brief_path)
        today = self.clock().strftime("%B %d, %Y").replace(" 0", " ")
        return (made_at if today not in made_at else ""), sections

    def read_morning_brief(self) -> str:
        stale, sections = self._load_brief()
        if sections is None:
            return "I can't find a morning brief. Run the brief command first."
        parts = []
        if stale:
            parts.append(f"This is an older brief, from {stale.split(' · ')[0]}.")

        cal = find(sections, "Today's calendar")
        if cal:
            events = [f"{i.time} {i.link}" for i in cal.items]
            if events:
                parts.append("Today you have " + ", and ".join(events) + ".")
            elif "Run:" in cal.note or "Couldn't reach" in cal.note:
                parts.append("Your calendar isn't connected right now.")   # don't read setup commands aloud
            else:
                parts.append(cal.note or "Your calendar is clear.")

        mail = find(sections, "Unread email")
        if mail:
            if mail.items:
                firsts = [f"{i.bold} about {i.link}" for i in mail.items[:3]]
                more = len(mail.items) - len(firsts)
                parts.append(f"You have {len(mail.items)} unread emails, including "
                             + "; ".join(firsts) + (f"; and {more} more." if more > 0 else "."))
            elif "Run:" in mail.note or "Couldn't reach" in mail.note:
                parts.append("Email isn't connected right now.")
            else:
                parts.append("No new email.")

        pc = self.pc_updates(sections)
        if pc:
            parts.append(pc)

        heads = []
        for title, sec in sections.items():
            if title.endswith(" news") and sec.items:
                heads.append(f"In {title.removesuffix(' news').lower()}: {sec.items[0].link}.")
        if heads:
            parts.append("Top headlines. " + " ".join(heads))
        return " ".join(speakable(p) for p in parts)

    def pc_updates(self, sections=None) -> str:
        if sections is None:
            _stale, sections = self._load_brief()
            if sections is None:
                return "I can't find a morning brief, which is where I get update news."
        pc = find(sections, "Your PC")
        if not pc:
            return ""
        needs = [speakable(first_sentence(i.text)) for i in pc.items if "up to date" not in i.text]
        return " ".join(needs) if needs else "Your PC is fully up to date."

    # ---------------------------------------------------------------- quick match

    def quick_match(self, text: str) -> str | None:
        """Handle common phrasings directly. Returns the answer, or None to ask the model."""
        t = text.lower().strip(" ?.!")
        if re.search(r"\bwhat(?:'s| is)? the time\b|\bwhat time is it\b|\bgot the time\b", t):
            return self.get_time()
        if re.search(r"\bwhat(?:'s| is)? (?:the |today'?s )?date\b|\bwhat day is (?:it|today)\b", t):
            return self.get_date()
        if re.search(r"\bbrief(?:ing)?\b", t):
            return self.read_morning_brief()
        if re.search(r"\bupdates?\b", t) and re.search(r"\b(?:pc|computer|laptop|any)\b", t):
            return self.pc_updates()
        m = re.search(r"\b(?:open|launch|start|bring up)\s+(.+)$", t)
        if m and resolve_app(m.group(1)):
            return self.open_app(m.group(1))
        return None

    # ---------------------------------------------------------------- tool calling

    def run_tool(self, name: str, args: dict) -> str:
        """Run a tool the model asked for. Unknown names are refused, not guessed."""
        funcs = {"get_time": self.get_time, "get_date": self.get_date,
                 "open_app": self.open_app, "read_morning_brief": self.read_morning_brief,
                 "pc_updates": self.pc_updates}
        if name not in funcs:
            return f"There is no tool called {name}."
        try:
            return funcs[name](**args) if name == "open_app" else funcs[name]()
        except TypeError:
            return f"The {name} tool was called with the wrong details."


# Descriptions the model reads to decide which tool fits (Ollama's "tools" format).
TOOLS = [
    {"type": "function", "function": {
        "name": "get_time", "description": "Get the current local time.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "get_date", "description": "Get today's date and day of the week.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "open_app", "description": "Open an application on PJ's Windows PC.",
        "parameters": {"type": "object", "required": ["app"], "properties": {
            "app": {"type": "string", "enum": list(APPS), "description": "Which app to open"}}}}},
    {"type": "function", "function": {
        "name": "read_morning_brief",
        "description": "Summarise PJ's morning brief: today's calendar, unread email, PC updates, top news.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "pc_updates", "description": "Say which software updates PJ's PC needs (Chrome, Windows, apps).",
        "parameters": {"type": "object", "properties": {}}}},
]
