"""Jarvis: an offline voice assistant, built one part at a time.

Parts (each is one file, so any part can be swapped without touching the others):
    brain.py   thinking   - talks to the local AI model through Ollama   (stage 1, done)
    voice.py   speaking   - Piper text-to-speech                         (stage 2, done)
    ears.py    hearing    - faster-whisper speech-to-text                (stage 3, done)
    tools.py   actions    - time, open apps, read the morning brief      (stage 4, done)
    brief_reader.py            reads the morning brief's saved page for tools.py
    wake.py    wake word  - "Hey Jarvis" with openWakeWord             (stage 5, done)
"""
