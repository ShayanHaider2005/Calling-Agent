"""Call UI server (localhost only): a phone-call screen for testing the agent.

One command:  py scripts/run_demo.py
Then open http://127.0.0.1:8000

The browser captures the microphone (Web Audio API), streams audio over a
WebSocket, and the server runs VAD -> STT -> brain -> TTS, streaming the reply
back. Barge-in: the agent stops when the caller starts speaking.

A text-input fallback works if the microphone fails. A "low-memory mode" runs
the STT on CPU when the GPU is busy or too full.
"""
import argparse
import asyncio
import base64
import io
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from src.pipeline import Pipeline, EnergyVAD
from src.brain import Brain, detect_language

app = FastAPI()

# Serve the UI page from the same directory
HERE = os.path.dirname(__file__)


class CallSession:
    """One active call: VAD -> STT -> brain -> TTS, with barge-in."""

    def __init__(self, scenario="clinic", lang_hint="auto", low_memory=False):
        self.scenario = scenario
        self.lang_hint = lang_hint  # auto | english | urdu | hindi
        self.low_memory = low_memory
        self.pipeline = Pipeline(use_llm=False)
        # force CPU if low_memory
        if low_memory:
            self.pipeline.stt.device = "cpu"
        self.vad = EnergyVAD()
        self.listening = False
        self.frames = []
        self.silence_frames = 0
        self.speaking = False  # agent is speaking (for barge-in)
        self.call_log = []
        self.language = "english"

    def set_language_hint(self, hint):
        self.lang_hint = hint

    def process_audio(self, pcm):
        """Process one audio chunk. Returns a reply dict when speech ends."""
        if self.vad.is_speech(pcm):
            if not self.listening:
                self.listening = True
                self.frames = []
            self.frames.append(pcm)
            self.silence_frames = 0
            return None
        else:
            if self.listening:
                self.silence_frames += 1
                if self.silence_frames >= self.vad.silence_frames:
                    self.listening = False
                    audio = np_concat(self.frames)
                    self.frames = []
                    return self._handle_turn(audio)
            return None

    def _handle_turn(self, audio):
        """STT -> brain -> TTS for one turn. Returns a reply dict."""
        t0 = time.time()
        result = self.pipeline.process_array(audio)
        # apply language hint
        if self.lang_hint != "auto":
            result.detected_language = self.lang_hint
            result.reply_text = self.pipeline.brain._answer(
                result.intent, self.lang_hint)
            result.language = self.lang_hint
        # re-synthesize in the hinted language if needed
        if self.lang_hint != "auto" and result.reply_text:
            tts = self.pipeline._get_tts(self.lang_hint)
            result.reply_audio = tts.synthesize(result.reply_text)
        turn = {
            "transcript": result.transcript,
            "reply_text": result.reply_text,
            "intent": result.intent,
            "language": result.detected_language,
            "latency": result.total_latency,
            "action": result.action,
            "reply_audio_b64": base64.b64encode(result.reply_audio).decode() if result.reply_audio else "",
        }
        self.call_log.append(turn)
        return turn

    def text_turn(self, text):
        """Text-input fallback. Returns a reply dict."""
        result = self.pipeline.process_text(text)
        turn = {
            "transcript": text,
            "reply_text": result.reply_text,
            "intent": result.intent,
            "language": result.detected_language,
            "latency": result.total_latency,
            "action": result.action,
            "reply_audio_b64": "",
        }
        self.call_log.append(turn)
        return turn

    def interrupt(self):
        """Barge-in: stop the agent speaking."""
        self.speaking = False
        self.listening = False
        self.frames = []


def np_concat(frames):
    import numpy as np
    return np.concatenate(frames, axis=0)


# Global session (one call at a time)
session = None


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(HERE, "static", "index.html"), encoding="utf-8") as f:
        return f.read()


@app.get("/api/status")
def status():
    """Status panel data: models, sizes, mode, VRAM, license flags."""
    import torch
    mode = "gpu" if torch.cuda.is_available() else "cpu"
    vram = f"{torch.cuda.memory_allocated()/1024**2:.0f}/{torch.cuda.get_device_properties(0).total_memory/1024**2:.0f} MiB" if torch.cuda.is_available() else "n/a"
    return {
        "stt_model": "openai/whisper-small (8-bit + LoRA)",
        "mode": mode,
        "vram": vram,
        "flags": "Hindi TTS DEMO-ONLY; Urdu TTS weak",
    }


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    global session
    await ws.accept()
    try:
        while True:
            msg = await ws.receive()
            if isinstance(msg, bytes):
                # audio chunk (float32 PCM, 16 kHz mono)
                import numpy as np
                pcm = np.frombuffer(msg, dtype=np.float32)
                if session is None:
                    continue
                # barge-in: if agent speaking and user speaks, interrupt
                if session.speaking and session.vad.is_speech(pcm):
                    session.interrupt()
                    await ws.send_json({"type": "interrupted"})
                reply = session.process_audio(pcm)
                if reply:
                    session.speaking = True
                    await ws.send_json({"type": "reply", **reply})
                    session.speaking = False
            elif isinstance(msg, str):
                data = json.loads(msg)
                if data.get("type") == "start":
                    session = CallSession(
                        scenario=data.get("scenario", "clinic"),
                        lang_hint=data.get("lang_hint", "auto"),
                        low_memory=data.get("low_memory", False),
                    )
                    await ws.send_json({"type": "started"})
                elif data.get("type") == "text" and session:
                    reply = session.text_turn(data.get("text", ""))
                    await ws.send_json({"type": "reply", **reply})
                elif data.get("type") == "end":
                    if session:
                        await ws.send_json({"type": "ended", "log": session.call_log})
                    session = None
    except WebSocketDisconnect:
        session = None
    except Exception as e:
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    print(f"Call UI on http://{args.host}:{args.port} (localhost only)")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
