"""
Tests for actions (stage 4). Apps aren't really opened (a fake launcher records
them), the clock is fixed, and sample brief pages stand in for the real one.
Run with:  python -m pytest
"""
from datetime import datetime
from pathlib import Path

import pytest

from jarvis.brief_reader import find, read_brief
from jarvis.tools import TOOLS, Actions, ordinal, resolve_app, speakable

DATA = Path(__file__).parent / "data"
MORNING = lambda: datetime(2026, 11, 24, 5, 30)          # same day the sample brief was made


@pytest.fixture
def actions():
    opened = []
    a = Actions(brief_path=DATA / "sample_brief.html", launcher=opened.append, clock=MORNING)
    a.opened = opened
    return a


# ---------------------------------------------------------------- reading the brief page

def test_brief_reader_finds_sections_and_items():
    made_at, sections = read_brief(DATA / "sample_brief.html")
    assert made_at.startswith("Tuesday, November 24, 2026")
    cal = find(sections, "Today's calendar")
    assert [(i.time, i.link) for i in cal.items] == [("6:30 AM", "Shift"), ("All day", "Payday")]
    mail = find(sections, "Unread email")
    assert [(i.bold, i.link) for i in mail.items][0] == ("Amazon", "Your package shipped")
    pc = find(sections, "Your PC")
    assert pc.items[1].children == ["Defender update KB2267602"]
    assert find(sections, "Science news").note == "No new stories since your last brief."


# ---------------------------------------------------------------- the actions

def test_time_and_date(actions):
    assert actions.get_time() == "It's 5:30 AM."
    assert actions.get_date() == "Today is Tuesday, November 24th."
    assert [ordinal(n) for n in (1, 2, 3, 4, 11, 12, 13, 21, 22, 23)] == \
        ["1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd", "23rd"]


def test_open_only_allowed_apps(actions):
    assert actions.open_app("Visual Studio Code") == "Opening vs code."
    assert actions.opened[-1].endswith("Code.exe")
    assert actions.open_app("the browser") == "Opening chrome."
    assert actions.open_app("cmd.exe /c del everything").startswith("I'm not allowed to open")
    assert len(actions.opened) == 2                       # the bad one launched nothing
    assert resolve_app("my files") == "file explorer"


def test_read_morning_brief(actions):
    said = actions.read_morning_brief()
    assert "Today you have 6:30 AM Shift, and All day Payday." in said
    assert "You have 3 unread emails, including Amazon about Your package shipped" in said
    assert "Chrome update available: 158.0 to 159.0." in said
    assert "Windows Update: 1 pending." in said
    assert "In tech: Tech headline 1." in said and "older brief" not in said


def test_brief_from_another_day_says_so(actions):
    actions.clock = lambda: datetime(2026, 11, 26, 7, 0)
    assert actions.read_morning_brief().startswith("This is an older brief, from Tuesday, November 24, 2026.")


def test_quiet_brief_and_missing_brief(tmp_path):
    a = Actions(brief_path=DATA / "sample_brief_quiet.html", launcher=lambda t: None, clock=MORNING)
    said = a.read_morning_brief()
    assert "Your calendar isn't connected right now." in said and "Email isn't connected" in said
    assert "Run:" not in said
    assert "Your PC is fully up to date." in said
    assert Actions(brief_path=tmp_path / "none.html").read_morning_brief().startswith("I can't find")


# ---------------------------------------------------------------- quick match vs. the model

@pytest.mark.parametrize("said, expected", [
    ("What time is it?", "It's 5:30 AM."),
    ("Jarvis, what's the date", "Today is Tuesday, November 24th."),
    ("Can you tell me what day of the week it is today?", "Today is Tuesday, November 24th."),
    ("Which day is it", "Today is Tuesday, November 24th."),
    ("What's today?", "Today is Tuesday, November 24th."),
    ("Open VS Code", "Opening vs code."),
    ("Could you launch the browser", "Opening chrome."),
    ("Any updates on my computer?", "Chrome update available"),
    ("Read me my morning brief", "Today you have"),
])
def test_quick_match(actions, said, expected):
    assert actions.quick_match(said).startswith(expected)


@pytest.mark.parametrize("said", ["Tell me a joke", "Open your heart to me", "Start my day right",
                                  "Which day is my shift this week?"])
def test_everything_else_goes_to_the_model(actions, said):
    assert actions.quick_match(said) is None
    assert actions.opened == []


def test_run_tool_refuses_unknown_tools(actions):
    assert actions.run_tool("delete_files", {}) == "There is no tool called delete_files."
    assert actions.run_tool("open_app", {"app": "calculator"}) == "Opening calculator."
    assert actions.run_tool("open_app", {}) == "The open_app tool was called with the wrong details."
    assert {t["function"]["name"] for t in TOOLS} == {"get_time", "get_date", "open_app",
                                                      "read_morning_brief", "pc_updates"}


def test_speakable_shortens_long_versions():
    assert speakable("Chrome update available: 158.0.8127.37 → 159.0.8154.13.") == \
        "Chrome update available: 158 to 159."
    assert speakable("Python 3.15 is out") == "Python 3.15 is out"     # short versions stay
