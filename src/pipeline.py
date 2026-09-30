"""End-to-end pipeline: audio in -> VAD -> STT -> lang detect -> brain -> TTS -> audio out.

Supports three input modes:
  - text:   type/pipe text directly (fully testable, no audio needed)
  - file:   read a wav file
  - mic:    record from the microphone (with barge-in and end-of-speech)

Per-stage latency is logged to results/latency_log.jsonl on every turn.
"""
import json
import os
import time

import numpy as np

from src.brain import Brain, detect_language

SAMPLE_RATE = 16000
LATENCY_LOG = os.path.join("results", "latency_log.jsonl")


# ------------------------------------------------------------------ VAD

class EnergyVAD:
    """Simple energy-based voice-activity detection on 16 kHz mono PCM.

    Not as good as webrtcvad but dependency-light and works on any platform.
    """

    def __init__(self, sample_rate=SAMPLE_RATE, frame_ms=30,
                 silence_ms=600, speech_ms=250, threshold=0.01):
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.silence_ms = silence_ms
        self.speech_ms = speech_ms
        self.threshold = threshold
        self.frame_len = int(sample_rate * frame_ms / 1000)
        self.silence_frames = int(silence_ms / frame_ms)
        self.speech_frames = int(speech_ms / frame_ms)

    def is_speech(self, pcm):
        """pcm: 1-D float array in [-1, 1]. True if RMS above threshold."""
        if len(pcm) == 0:
            return False
        rms = float(np.sqrt(np.mean(np.square(pcm))))
        return rms > self.threshold

    def find_speech(self, pcm):
        """Return (start, end) sample indices of the speech region, or (0, 0)."""
        n = len(pcm)
        if n == 0:
            return 0, 0
        n_frames = n // self.frame_len
        if n_frames == 0:
            return (0, n) if self.is_speech(pcm) else (0, 0)
        energy = np.array([
            np.sqrt(np.mean(np.square(pcm[i * self.frame_len:(i + 1) * self.frame_len])))
            for i in range(n_frames)
        ])
        speech = energy > self.threshold
        # find first run of >= speech_frames, then extend until silence_frames
        start = None
        run = 0
        for i, s in enumerate(speech):
            if s:
                run += 1
                if run >= self.speech_frames and start is None:
                    start = (i - run + 1) * self.frame_len
            else:
                if start is not None:
                    # count trailing silence
                    j = i
                    while j < n_frames and not speech[j]:
                        j += 1
                    if (j - i) >= self.silence_frames or j >= n_frames:
                        # speech ends where the silence begins (frame i)
                        end = min(i * self.frame_len, n)
                        return int(start), int(end)
                run = 0
        if start is not None:
            return int(start), n
        return 0, 0


# ------------------------------------------------------------------ STT

class WhisperSTT:
    """Speech-to-text via a Hugging Face Whisper model (lazy-loaded)."""

    def __init__(self, model="openai/whisper-tiny", device=None):
        self.model = model
        self.device = device
        self._pipe = None

    def _load(self):
        if self._pipe is None:
            import torch
            from transformers import pipeline
            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._pipe = pipeline(
                "automatic-speech-recognition", model=self.model,
                device=0 if str(device).startswith("cuda") else -1,
            )
        return self._pipe

    def transcribe(self, audio_path):
        pipe = self._load()
        out = pipe(audio_path)
        return out["text"].strip()

    def transcribe_array(self, pcm, sample_rate=16000):
        """Transcribe a numpy float32 array directly (no ffmpeg needed)."""
        pipe = self._load()
        out = pipe({"raw": pcm, "sampling_rate": sample_rate})
        return out["text"].strip()


# ------------------------------------------------------------------ pipeline

class PipelineResult:
    def __init__(self):
        self.transcript = ""
        self.detected_language = "english"
        self.intent = ""
        self.reply_text = ""
        self.reply_audio = b""
        self.action = "speak"
        self.latencies = {}
        self.total_latency = 0.0

    def to_dict(self):
        return {
            "transcript": self.transcript,
            "detected_language": self.detected_language,
            "intent": self.intent,
            "reply_text": self.reply_text,
            "action": self.action,
            "latencies": self.latencies,
            "total_latency": self.total_latency,
        }


class Pipeline:
    def __init__(self, stt_model="openai/whisper-tiny", use_llm=False,
                 tts_engine="auto", scenario="scenarios/clinic.yaml"):
        self.vad = EnergyVAD()
        self.stt = WhisperSTT(model=stt_model)
        self.brain = Brain(scenario_path=scenario, use_llm=use_llm)
        self.tts_engine = tts_engine
        self._tts = {}

    def _get_tts(self, lang):
        if lang not in self._tts:
            from src.tts import TTS
            self._tts[lang] = TTS(lang, engine=self.tts_engine)
        return self._tts[lang]

    def _log_latency(self, result):
        os.makedirs(os.path.dirname(LATENCY_LOG), exist_ok=True)
        with open(LATENCY_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")

    # ------------------------------------------------------------- modes
    def process_text(self, text):
        """Text in -> reply text out. No audio. Fully testable."""
        t0 = time.time()
        result = PipelineResult()
        result.detected_language = detect_language(text)
        resp = self.brain.process(text)
        result.transcript = text
        result.intent = resp.intent
        result.reply_text = resp.text
        result.action = resp.action
        result.latencies = {"brain": time.time() - t0}
        result.total_latency = sum(result.latencies.values())
        self._log_latency(result)
        return result

    def process_file(self, audio_path):
        """Wav file in -> reply audio out."""
        t0 = time.time()
        result = PipelineResult()

        import soundfile as sf
        pcm, sr = sf.read(audio_path, dtype="float32")
        if pcm.ndim > 1:
            pcm = pcm.mean(axis=1)
        if sr != SAMPLE_RATE:
            pcm = _resample(pcm, sr, SAMPLE_RATE)

        # VAD
        tv = time.time()
        start, end = self.vad.find_speech(pcm)
        result.latencies["vad"] = time.time() - tv
        speech = pcm[start:end]
        if len(speech) == 0:
            result.reply_text = ""
            result.action = "speak"
            result.latencies["total"] = time.time() - t0
            result.total_latency = result.latencies["total"]
            self._log_latency(result)
            return result

        # STT (pass array directly — no ffmpeg needed)
        ts = time.time()
        transcript = self.stt.transcribe_array(speech, SAMPLE_RATE)
        result.latencies["stt"] = time.time() - ts
        result.transcript = transcript
        result.detected_language = detect_language(transcript)

        # Brain
        tb = time.time()
        resp = self.brain.process(transcript)
        result.latencies["brain"] = time.time() - tb
        result.intent = resp.intent
        result.reply_text = resp.text
        result.action = resp.action

        # TTS
        if resp.text and resp.action != "stop":
            tt = time.time()
            tts = self._get_tts(resp.language)
            result.reply_audio = tts.synthesize(resp.text)
            result.latencies["tts"] = time.time() - tt

        result.latencies["total"] = time.time() - t0
        result.total_latency = result.latencies["total"]
        self._log_latency(result)
        return result

    def process_mic(self, max_turns=100):
        """Live mic loop with barge-in and end-of-speech. Prints to console."""
        import sounddevice as sd

        print("Mic mode: speak to the agent. Say 'goodbye' or press Ctrl+C to end.")
        vad = self.vad
        silence_frames = 0
        recording = False
        frames = []

        def callback(indata, frames_n, time_info, status):
            nonlocal recording, frames, silence_frames
            chunk = indata[:, 0].copy()
            if vad.is_speech(chunk):
                if not recording:
                    recording = True
                    frames = []
                    print("  [caller speaking]")
                frames.append(chunk)
                silence_frames = 0
            else:
                if recording:
                    silence_frames += 1
                    if silence_frames >= vad.silence_frames:
                        # end of speech
                        recording = False
                        silence_frames = 0
                        audio = np.concatenate(frames, axis=0)
                        self._handle_mic_turn(audio)

        try:
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                                dtype="float32", callback=callback):
                while True:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nended.")

    def _handle_mic_turn(self, audio):
        """Process one mic turn (barge-in handled by stopping playback)."""
        import sounddevice as sd
        # stop any playing reply (barge-in)
        try:
            sd.stop()
        except Exception:
            pass
        result = self.process_array(audio)
        print(f"  caller: {result.transcript}")
        print(f"  agent ({result.intent}): {result.reply_text}")
        print(f"  latency: {result.total_latency:.2f}s")
        if result.reply_audio:
            import soundfile as sf
            import io
            pcm, sr = _wav_bytes_to_pcm(result.reply_audio)
            sd.play(pcm, sr)
            sd.wait()

    def process_array(self, audio):
        """numpy array in -> result (used by mic mode)."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        try:
            import soundfile as sf
            sf.write(tmp, audio, SAMPLE_RATE)
            return self.process_file(tmp)
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def _resample(pcm, orig_sr, target_sr):
    if orig_sr == target_sr:
        return pcm
    import math
    gcd = math.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    # simple linear interpolation resampling (good enough for speech)
    n_out = int(len(pcm) * target_sr / orig_sr)
    x_old = np.linspace(0, 1, len(pcm))
    x_new = np.linspace(0, 1, n_out)
    return np.interp(x_new, x_old, pcm).astype(np.float32)


def _wav_bytes_to_pcm(wav_bytes):
    import io
    import soundfile as sf
    data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    return data, sr


# ------------------------------------------------------------------ CLI

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Calling-Agent pipeline")
    ap.add_argument("mode", choices=["text", "file", "mic"], nargs="?", default="text")
    ap.add_argument("--input", "-i", default=None, help="text or audio file path")
    ap.add_argument("--stt-model", default="openai/whisper-tiny")
    ap.add_argument("--use-llm", action="store_true")
    ap.add_argument("--tts-engine", default="auto")
    args = ap.parse_args()

    pipe = Pipeline(stt_model=args.stt_model, use_llm=args.use_llm,
                    tts_engine=args.tts_engine)

    if args.mode == "text":
        if args.input:
            # single-shot: process one turn and exit (good for testing)
            result = pipe.process_text(args.input)
            print(f"caller: {result.transcript}")
            print(f"agent ({result.intent}, {result.detected_language}): {result.reply_text}")
            print(f"latency: {result.total_latency:.2f}s {result.latencies}")
        else:
            # Interactive REPL: maintains brain state across turns.
            print("Text mode: type and press Enter. 'quit' to end.")
            while True:
                try:
                    text = input("You: ")
                except EOFError:
                    break
                if text.lower() in ("quit", "exit", "q"):
                    break
                result = pipe.process_text(text)
                print(f"caller: {result.transcript}")
                print(f"agent ({result.intent}, {result.detected_language}): {result.reply_text}")
                print(f"latency: {result.total_latency:.2f}s {result.latencies}")
                if result.action == "end":
                    break
    elif args.mode == "file":
        if not args.input:
            print("file mode needs --input <wav>")
            return
        result = pipe.process_file(args.input)
        print(f"caller: {result.transcript}")
        print(f"agent ({result.intent}, {result.detected_language}): {result.reply_text}")
        print(f"latency: {result.total_latency:.2f}s {result.latencies}")
        if result.reply_audio:
            out = "results/reply.wav"
            os.makedirs("results", exist_ok=True)
            with open(out, "wb") as f:
                f.write(result.reply_audio)
            print(f"reply audio: {out}")
    else:
        pipe.process_mic()


if __name__ == "__main__":
    main()
