# Jarvis

An offline, voice-driven assistant for a CPU-only laptop, built one part at a time.
Everything runs on this PC through Ollama; nothing goes to the internet.

## Commands

    jarvis                     chat by typing (stage 1)
    jarvis --model gemma4:e2b  use a different downloaded model this time
    jarvis --bench             speed-test every downloaded model
    jarvis --voices            hear every downloaded voice
    jarvis --say "Hello"       speak a line with the current voice
    jarvis --hear              mic test: record once and show what was heard

In chat: press **Enter on an empty line and speak** (it stops when you go quiet), or type.
Say "goodbye Jarvis" to exit. `/mute` toggles speech, `/stats` shows speed, `/reset` forgets the conversation, `/quit` exits.

## Things Jarvis can do

| Say something like | What happens |
|---|---|
| "What time is it?" / "What's the date?" | Tells you |
| "Open VS Code" / "Launch the browser" | Opens it (only apps on the list in `jarvis/tools.py`) |
| "Read me my morning brief" | Reads today's calendar, email, PC updates and top headlines |
| "Any updates on my computer?" | Says what needs updating, from the morning brief |

Common phrasings are matched instantly in code; anything else goes to the model,
which can choose an action itself ("tool calling"). To add an action, see the top
of `jarvis/tools.py`.

## Settings (settings.json)

- `model`: which model to use (see `ollama list`)
- `personality`: the instructions Jarvis always follows
- `max_history`: how many recent messages it remembers (fewer = faster)
- `think`: let "thinking" models reason first (smarter, much slower on CPU)
- `voice`: `name` (a downloaded Piper voice), `speed` (1.0 normal, 0.9 faster), `enabled`
- `ears`: `model` (Whisper model; `base.en` is downloaded), `enabled`
- `actions`: `true` lets Jarvis use the actions above

## Where things live

| What | Where |
|---|---|
| Code (this folder) | `OneDrive\Documents\jarvis` |
| Python 3.13 environment (Piper etc.) | `C:\Users\<you>\.venvs\jarvis` (not synced) |
| Voices | `C:\Users\<you>\.jarvis\voices` |
| Speech-to-text model | `C:\Users\<you>\.jarvis\whisper` |
| AI models | `C:\Users\<you>\.ollama\models` (managed by Ollama) |

The `jarvis` command uses the Python 3.13 environment. To add a package to it:

    uv pip install --python %USERPROFILE%\.venvs\jarvis\Scripts\python.exe <package>

## Build plan

1. Brain: Ollama + text chat. **Done.**
2. Voice out: Piper text-to-speech. **Done.**
3. Voice in: push-to-talk with faster-whisper. **Done.**
4. Actions: time, open apps, read the morning brief aloud. **Done.**
5. Wake word: "Jarvis..."

Stages 2+ need Python 3.13 in a virtual environment (voice libraries lag behind 3.15).
