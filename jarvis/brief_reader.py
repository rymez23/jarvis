"""
Reads the morning brief's saved web page, so Jarvis can say it out loud.

The morning brief project saves today's page to
    C:\\Users\\<you>\\AppData\\Local\\morning-brief\\pages\\latest.html
Rather than importing that project's code (it uses a different Python), we read
the HTML file it already produces. Each part of the page looks like:

    <section><h2>Unread email (3)</h2><ul><li><b>Amazon</b>: <a>Your package shipped</a>
        <div class="muted small">preview text</div></li> ...</ul></section>

Python's built-in HTMLParser calls our methods for every opening tag, closing
tag, and piece of text, in order. We use that to collect, per section, the
title and each list item's parts.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path


def latest_brief_path() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".morning-brief"))
    return base / "morning-brief" / "pages" / "latest.html"


@dataclass
class Item:
    text: str = ""                # all the item's own text (not its preview or sub-list)
    link: str = ""                # text of its first link (a headline, an email subject)
    bold: str = ""                # bold text (an email's sender)
    time: str = ""                # calendar time, e.g. "6:30 AM"
    children: list[str] = field(default_factory=list)   # nested list items (e.g. update names)


@dataclass
class Section:
    title: str
    items: list[Item] = field(default_factory=list)
    note: str = ""                # a message such as "Nothing scheduled today."


class _Parser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.sections: list[Section] = []
        self.made_at = ""                         # the line under the greeting: date and time
        self._stack: list[tuple[str, str]] = []   # open tags as (tag, class), innermost last
        self._li_depth = 0

    # which tags are we currently inside?
    def _inside(self, tag, cls=None):
        return any(t == tag and (cls is None or cls in c.split()) for t, c in self._stack)

    def handle_starttag(self, tag, attrs):
        cls = dict(attrs).get("class") or ""
        self._stack.append((tag, cls))
        if tag == "li":
            self._li_depth += 1
            if self._li_depth == 1 and self.sections:
                self.sections[-1].items.append(Item())
            elif self._li_depth == 2 and self.sections and self.sections[-1].items:
                self.sections[-1].items[-1].children.append("")

    def handle_endtag(self, tag):
        if tag == "li":
            self._li_depth -= 1
        # pop back to the matching tag (HTML is forgiving, so be forgiving too)
        for i in range(len(self._stack) - 1, -1, -1):
            if self._stack[i][0] == tag:
                del self._stack[i:]
                break

    def handle_data(self, data):
        text = data.strip()
        if not text:
            return
        if self._inside("p", "sub"):
            self.made_at += text
        elif self._inside("h2"):
            self.sections.append(Section(title=text))
        elif not self.sections:
            return
        elif self._li_depth >= 2:
            item = self.sections[-1].items[-1]
            item.children[-1] = (item.children[-1] + " " + text).strip()
        elif self._li_depth == 1:
            item = self.sections[-1].items[-1]
            if self._inside("div", "small"):
                return                              # skip previews/summaries: too long to read aloud
            if self._inside("span", "time"):
                item.time = text
            elif self._inside("a") and not item.link:
                item.link = text
            elif self._inside("b") and not item.bold:
                item.bold = text
            item.text = (item.text + " " + text).strip()
        elif self._inside("p"):
            self.sections[-1].note = (self.sections[-1].note + " " + text).strip()


def read_brief(path: Path | None = None) -> tuple[str, dict[str, Section]]:
    """Return (when the brief was made, {section title: Section})."""
    parser = _Parser()
    parser.feed((path or latest_brief_path()).read_text(encoding="utf-8"))
    return parser.made_at, {s.title: s for s in parser.sections}


def find(sections: dict[str, Section], starts_with: str) -> Section | None:
    """Section titles can change ('Unread email (3)'), so match on how they start."""
    return next((s for title, s in sections.items() if title.startswith(starts_with)), None)
