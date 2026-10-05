# DEMO_GUIDE — how to run and verify the call UI

## Start the UI (one command)

```bash
.venv\Scripts\python.exe scripts\run_demo.py
```

Then open **http://127.0.0.1:8000** in a browser (Chrome/Edge recommended).
The page is localhost-only — no external network calls.

## Pre-demo checklist (run this first)

1. **Free the GPU** if training is running: create a file named `STOP` in the
   project root, or close other GPU apps. Check with `nvidia-smi` (should show
   low VRAM use).
2. **Run the health check** (no browser needed):
   ```bash
   .venv\Scripts\python.exe scripts\check_demo.py
   ```
   It should print `check_demo: PASS`. If it fails, do not demo.
3. **Check the status panel** on the UI page — it should show the STT model
   (whisper-small), mode (GPU), VRAM, and license flags.
4. **Test the microphone** — click Start Call and speak. The waveform should move.
5. **Test the text fallback** — type a message and press Enter (works even if the
   mic fails).

## 10 suggested test calls

For each: click **Start Call**, say the line, listen to the reply, then **End Call**.

| # | Language | Say | Good behavior |
|---|---|---|---|
| 1 | English | "Hello" | Greets, says it is an AI, says call may be recorded |
| 2 | English | "What are your opening hours?" | States hours (Mon–Fri 9–5, Sat 10–2) |
| 3 | English | "How much does a blood test cost?" | States the price ($30) |
| 4 | English | "I want to book an appointment" | Asks for name, then day, then time, then confirms |
| 5 | Urdu | "السلام علیکم" | Replies in Urdu, says it is an AI |
| 6 | Urdu | "آپ کے اوقات کار کیا ہیں؟" | Replies in Urdu with hours |
| 7 | Hindi | "नमस्ते" | Replies in Hindi, says it is an AI |
| 8 | Mixed | "میں appointment book کرنا چاہتا ہوں" | Detects Urdu-dominant, replies in Urdu |
| 9 | English | "Are you a human?" | Says it is an AI, offers human handoff |
| 10 | English | "Can you promise me a discount?" | Does NOT promise; offers human handoff |

**Barge-in test:** while the agent is speaking, start talking — the agent should
stop. **Language-switch test:** start in English, then switch to Urdu mid-call —
the agent should follow.

## Known weak spots (be upfront about these)

- **Urdu/Hindi TTS voices are weak.** Hindi is DEMO-ONLY. The robot voice may be
  hard to understand for native speakers.
- **Latency is 7.7–9.7 s** (GPU) — not instant. The first call is slower (~47 s)
  while the model loads.
- **No real test set** — only a public English dev stand-in. Urdu/Hindi accuracy
  is unmeasured.
- **Synthetic training data** — accuracy is limited by the small synthetic set.
- **Small-model ceiling** — Whisper-small on a 2 GB GPU is not human-level.

## If something fails during a live demo

- **Microphone fails:** use the text-input box (bottom of the page). The demo
  still works.
- **Model fails to load:** check the status panel. If it shows an error, restart
  the server. If the GPU is full, enable **Low-memory mode** (CPU STT) — it is
  slower but works.
- **Agent says something wrong:** it only answers from the business knowledge
  file. If it invents something, that is a bug — note it and report it.
- **Browser shows an error:** refresh the page. The server keeps running.
- **GPU out of memory:** close other apps, or restart the server with
  Low-memory mode enabled.

## Freeing GPU memory before a demo

1. Create a `STOP` file in the project root if training is running (it stops
   gracefully after the current round).
2. Close other GPU apps (browsers with heavy pages, games, other models).
3. Check `nvidia-smi` — VRAM should be low (< 500 MiB) before starting the demo.
4. Start the UI. The STT model loads on first use (~47 s), so start the demo page
   a minute before you need it.
