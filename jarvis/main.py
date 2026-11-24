"""
Talk to Jarvis. Press Enter and speak, or type; he answers out loud.

    jarvis                    chat with the model set in settings.json
    jarvis --model gemma4:e2b chat with a different model this time
    jarvis --bench            speed-test every downloaded model, to choose one
    jarvis --voices           hear every downloaded voice, to choose one
    jarvis --say "Hello"      just say something (handy for testing a voice)
    jarvis --hear             record once and show what Jarvis heard (mic test)
    jarvis --wake             hands-free: say "Hey Jarvis", wait for the beep, then speak

In the chat:  press Enter on an empty line to speak; say "goodbye Jarvis" to exit.
              /mute toggles speech,  /reset forgets the conversation,
              /stats shows speed,  /quit exits.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

from .brain import Brain, Reply
from .tools import TOOLS, Actions
from . import voice as voice_mod

ROOT = Path(__file__).resolve().parent.parent


def load_settings() -> dict:
    return json.loads((ROOT / "settings.json").read_text(encoding="utf-8"))


def installed_models(url: str) -> list[str]:
    with urllib.request.urlopen(url.rstrip("/") + "/api/tags", timeout=5) as resp:
        return sorted(m["name"] for m in json.load(resp)["models"])


def load_voice(settings: dict):
    """Return a Voice, or None if speech is off or not set up (text-only chat still works)."""
    v = settings.get("voice", {})
    if not v.get("enabled", False):
        return None
    try:
        return voice_mod.Voice(v["name"], v.get("speed", 1.0))
    except ImportError:
        print("(Voice off: Piper isn't installed in this Python. Use the 'jarvis' command.)\n")
    except FileNotFoundError as e:
        print(f"(Voice off: {e})\n")
    return None


def load_ears(settings: dict):
    """Return Ears, or None if speech input is off or not set up (typing still works)."""
    e = settings.get("ears", {})
    if not e.get("enabled", False):
        return None
    try:
        from .ears import Ears
        return Ears(e.get("model", "base.en"))
    except ImportError:
        print("(Mic off: faster-whisper isn't installed in this Python. Use the 'jarvis' command.)\n")
    except Exception as err:      # e.g. the model hasn't been downloaded yet
        print(f"(Mic off: {err})\n")
    return None


GOODBYES = ("goodbye jarvis", "goodbye, jarvis", "bye jarvis", "that's all jarvis")


def respond(brain: Brain, actions: Actions | None, text: str) -> Reply:
    """Quick-match actions first (instant); otherwise ask the model, offering the tools."""
    if actions:
        answer = actions.quick_match(text)
        if answer:
            brain.remember(text, answer)
            return Reply(text=answer, tokens=0, seconds=0.0, load_seconds=0.0)
        return brain.ask(text, tools=TOOLS, run_tool=actions.run_tool)
    return brain.ask(text)


def chat(brain: Brain, voice=None, ears=None, actions: Actions | None = None) -> None:
    print(f"Jarvis ({brain.model}{', voice ' + voice.name if voice else ''}) is ready. "
          + ("Press Enter and speak, or type. Say 'goodbye Jarvis' to exit.\n" if ears
             else "Type /quit to exit.\n"))
    show_stats = False
    muted = False
    while True:
        try:
            text = input("You: " if not ears else "You (Enter = speak): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text and ears:
            text = ears.listen(on_start=lambda: print("  Listening... (stop talking when done)", flush=True))
            if not text:
                print("  (didn't catch that)\n")
                continue
            print(f"  You said: {text}")
            if text.lower().strip(" .!") in GOODBYES:
                if voice and not muted:
                    voice.say("Very good. Goodbye, PJ.")
                break
        if not text:
            continue
        if text in ("/quit", "/exit"):
            break
        if text == "/reset":
            brain.forget()
            print("(conversation forgotten)\n")
            continue
        if text == "/mute":
            muted = not muted
            print(f"(voice {'muted' if muted else 'on'})\n")
            continue
        if text == "/stats":
            show_stats = not show_stats
            print(f"(speed stats {'on' if show_stats else 'off'})\n")
            continue

        print("Jarvis is thinking...", end="\r", flush=True)
        reply = respond(brain, actions, text)
        print(" " * 24, end="\r")                      # clear the "thinking" line
        print(f"Jarvis: {reply.text}")
        if show_stats:
            print(f"  [{reply.tokens} tokens in {reply.seconds:.1f}s = "
                  f"{reply.tokens_per_second:.1f} tokens/s; load {reply.load_seconds:.1f}s]")
        print()
        if voice and not muted:
            voice.say(reply.text)       # waits until he's finished speaking


# Three typical requests: a fact, a short task, and a follow-up that tests memory.
BENCH_PROMPTS = [
    "In one sentence, what is the capital of Arizona?",
    "Give me a two-sentence pep talk for a 6:30 AM shift.",
    "What did I just ask you for?",
]


def bench(settings: dict) -> None:
    """Run the same prompts on every installed model and print a comparison."""
    models = installed_models(settings["ollama_url"])
    print(f"Testing {len(models)} models with {len(BENCH_PROMPTS)} prompts each. "
          f"This can take a few minutes on a laptop CPU.\n")
    results = []
    for model in models:
        brain = Brain(model, settings["personality"], settings["ollama_url"],
                      think=settings.get("think", False))
        print(f"== {model} ==")
        start = time.monotonic()
        speeds, load = [], 0.0
        for prompt in BENCH_PROMPTS:
            r = brain.ask(prompt)
            speeds.append(r.tokens_per_second)
            load = max(load, r.load_seconds)
            print(f"  Q: {prompt}\n  A: {r.text}\n     ({r.tokens_per_second:.1f} tokens/s)")
        total = time.monotonic() - start
        avg = sum(speeds) / len(speeds)
        results.append((model, avg, load, total))
        print()
        # unload it so the next model gets the RAM (keep_alive 0 = unload now)
        brain._post("/api/generate", {"model": model, "keep_alive": 0})

    print("== Summary (higher tokens/s = faster replies; ~5+ feels OK for speech) ==")
    for model, avg, load, total in sorted(results, key=lambda r: -r[1]):
        print(f"  {model:20} {avg:5.1f} tokens/s   first load {load:5.1f}s   all 3 answers {total:5.1f}s")


def hands_free(brain: Brain, voice, ears, actions: Actions | None, wake) -> None:
    """Loop forever: wait for "Hey Jarvis", beep, listen, answer out loud.
    Say "goodbye Jarvis" (after waking him) or press Ctrl+C to stop."""
    from .wake import beep
    print('Hands-free mode. Say "Hey Jarvis", wait for the beep, then speak. Ctrl+C to stop.\n')
    try:
        while True:
            print("  (waiting for 'Hey Jarvis')", flush=True)
            wake.wait_for_mic()
            beep()
            text = ears.listen(on_start=lambda: print("  Listening...", flush=True))
            if not text:
                print("  (didn't catch that)\n")
                continue
            print(f"  You said: {text}")
            if text.lower().strip(" .!") in GOODBYES:
                if voice:
                    voice.say("Very good. Goodbye, PJ.")
                break
            reply = respond(brain, actions, text)
            print(f"Jarvis: {reply.text}\n")
            if voice:
                voice.say(reply.text)
    except KeyboardInterrupt:
        print("\nStopped.")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Jarvis, your offline assistant (text mode).")
    p.add_argument("--model", help="use this model instead of the one in settings.json")
    p.add_argument("--bench", action="store_true", help="speed-test all downloaded models")
    p.add_argument("--voices", action="store_true", help="hear each downloaded voice")
    p.add_argument("--say", metavar="TEXT", help="speak TEXT with the current voice and exit")
    p.add_argument("--hear", action="store_true", help="record once and print what was heard")
    p.add_argument("--wake", action="store_true", help='hands-free mode: say "Hey Jarvis" to talk')
    args = p.parse_args(argv)

    settings = load_settings()
    if args.voices:
        speed = settings.get("voice", {}).get("speed", 1.0)
        for name in voice_mod.installed_voices():
            print(f"Playing {name} ...")
            voice_mod.Voice(name, speed).say(
                f"Good morning, PJ. This is the {name.split('-')[1].replace('_', ' ')} voice. "
                "Your calendar is clear and the coffee won't make itself.")
        print('\nTo choose one, put its name in settings.json under "voice" -> "name".')
        return 0
    if args.hear:
        ears = load_ears(settings)
        if ears:
            heard = ears.listen(on_start=lambda: print("Listening... say something, then stop talking.", flush=True))
            print(f"Heard: {heard!r}" if heard else "Heard nothing.")
        return 0
    if args.say:
        v = load_voice(settings)
        if v:
            v.say(args.say)
        return 0
    brain = Brain(args.model or settings["model"], settings["personality"], settings["ollama_url"],
                  settings.get("max_history", 10), settings.get("think", False))
    if not brain.is_running():
        print("Ollama isn't running. Open the Ollama app from the Start menu, then try again.")
        return 1
    if args.bench:
        bench(settings)
    else:
        actions = Actions() if settings.get("actions", True) else None
        voice, ears = load_voice(settings), load_ears(settings)
        if args.wake:
            if not ears:
                print("Hands-free mode needs the microphone (ears) working.")
                return 1
            try:
                from .wake import WakeWord
                wake = WakeWord(settings.get("wake", {}).get("threshold", 0.5))
            except ImportError:
                print("Hands-free mode needs openwakeword. Use the 'jarvis' command.")
                return 1
            hands_free(brain, voice, ears, actions, wake)
        else:
            chat(brain, voice, ears, actions)
    return 0


if __name__ == "__main__":
    sys.exit(main())
