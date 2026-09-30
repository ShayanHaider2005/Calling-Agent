"""Local web demo (localhost only): click to talk, hear the reply, see transcript + latency.

Usage:
  py scripts/web_demo.py
  # then open http://localhost:8000

The browser captures the microphone as WAV (Web Audio API) and POSTs it to /chat.
The server runs the pipeline (STT -> brain -> TTS) and returns the reply audio,
transcript, intent and per-stage latency. One global pipeline keeps brain state
across turns (single-user demo).
"""
import argparse
import base64
import io
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

from src.pipeline import Pipeline

PIPELINE = None


def get_pipeline():
    global PIPELINE
    if PIPELINE is None:
        PIPELINE = Pipeline()
    return PIPELINE


app = FastAPI()

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Calling-Agent demo</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 720px; margin: 2rem auto; padding: 0 1rem; }
  button { font-size: 1.2rem; padding: 0.8rem 1.6rem; cursor: pointer; }
  button.recording { background: #d33; color: #fff; }
  #log { background: #f4f4f4; border-radius: 8px; padding: 1rem; min-height: 120px; margin-top: 1rem; }
  .turn { margin: 0.5rem 0; }
  .caller { color: #333; }
  .agent { color: #06c; }
  .meta { color: #888; font-size: 0.85rem; }
  audio { vertical-align: middle; margin-left: 0.5rem; }
</style>
</head>
<body>
<h1>Calling-Agent demo</h1>
<p>Click the button and speak. The agent replies out loud in your language.</p>
<button id="btn">Click to talk</button>
<div id="log"></div>
<script>
let mediaRecorder, chunks = [], recording = false;
let audioCtx;

function wavFromBlob(blob) {
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.readAsArrayBuffer(blob);
  });
}

async function startRecording() {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const source = audioCtx.createMediaStreamSource(stream);
  const dest = audioCtx.createMediaStreamDestination();
  source.connect(dest);
  chunks = [];
  mediaRecorder = new MediaRecorder(dest.stream);
  mediaRecorder.ondataavailable = (e) => chunks.push(e.data);
  mediaRecorder.start();
  recording = true;
  document.getElementById('btn').textContent = 'Recording... click to stop';
  document.getElementById('btn').classList.add('recording');
}

function stopRecording() {
  return new Promise((resolve) => {
    mediaRecorder.onstop = () => {
      recording = false;
      const blob = new Blob(chunks, { type: mediaRecorder.mimeType });
      document.getElementById('btn').textContent = 'Click to talk';
      document.getElementById('btn').classList.remove('recording');
      resolve(blob);
    };
    mediaRecorder.stop();
  });
}

async function sendAudio(blob) {
  // decode to PCM and build a WAV in-browser (avoids server-side ffmpeg)
  const buf = await wavFromBlob(blob);
  const audioBuffer = await audioCtx.decodeAudioData(buf);
  const pcm = audioBuffer.getChannelData(0); // float32 mono
  const wav = encodeWAV(pcm, audioBuffer.sampleRate);
  const form = new FormData();
  form.append('file', new Blob([wav], { type: 'audio/wav' }), 'audio.wav');
  const res = await fetch('/chat', { method: 'POST', body: form });
  return await res.json();
}

function encodeWAV(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const writeStr = (off, s) => { for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i)); };
  writeStr(0, 'RIFF'); view.setUint32(4, 36 + samples.length * 2, true); writeStr(8, 'WAVE');
  writeStr(12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true);
  view.setUint16(22, 1, true); view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  writeStr(36, 'data'); view.setUint32(40, samples.length * 2, true);
  let off = 44;
  for (let i = 0; i < samples.length; i++, off += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }
  return view.buffer;
}

function addTurn(html) {
  const log = document.getElementById('log');
  const div = document.createElement('div');
  div.className = 'turn';
  div.innerHTML = html;
  log.prepend(div);
}

document.getElementById('btn').addEventListener('click', async () => {
  if (!recording) { await startRecording(); return; }
  const blob = await stopRecording();
  addTurn('<div class="caller">you: <i>sending...</i></div>');
  try {
    const data = await sendAudio(blob);
    const audioB64 = data.reply_audio_b64 || '';
    const audioUrl = audioB64 ? 'data:audio/wav;base64,' + audioB64 : '';
    addTurn(
      '<div class="caller">you: ' + (data.transcript || '') + '</div>' +
      '<div class="agent">agent (' + (data.intent || '') + ', ' + (data.detected_language || '') + '): ' +
        (data.reply_text || '') +
        (audioUrl ? '<audio controls src="' + audioUrl + '"></audio>' : '') + '</div>' +
      '<div class="meta">latency: ' + (data.total_latency || 0).toFixed(2) + 's ' +
        JSON.stringify(data.latencies || {}) + '</div>'
    );
  } catch (e) {
    addTurn('<div class="caller">error: ' + e + '</div>');
  }
});
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


@app.post("/chat")
async def chat(file: UploadFile = File(...)):
    t0 = time.time()
    data = await file.read()
    import soundfile as sf
    import numpy as np
    pcm, sr = sf.read(io.BytesIO(data), dtype="float32")
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=1)
    # write to a temp wav for the pipeline
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = f.name
    try:
        sf.write(tmp, pcm, sr)
        result = get_pipeline().process_file(tmp)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
    out = result.to_dict()
    out["reply_audio_b64"] = base64.b64encode(result.reply_audio).decode() if result.reply_audio else ""
    out["server_total"] = time.time() - t0
    return JSONResponse(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    print(f"web demo on http://{args.host}:{args.port} (localhost only)")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
