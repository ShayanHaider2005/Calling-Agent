"""Mouth: text-to-speech with a common interface and per-language voice config.

Primary engine: Piper (local, CPU, MIT license) — see LICENSES.md.
Fallback: pyttsx3 (SAPI5 system voices) for any language the system supports.

Voice files are downloaded under models/ (git-ignored) from
https://huggingface.co/rhasspy/piper-voices (permissive licenses).
"""
import os
import subprocess
import tempfile

VOICE_DIR = os.path.join("models", "piper_voices")

# Per-language Piper voice config: (repo path in piper-voices, quality)
# Filled in after surveying available voices; see TTS_NOTES.md.
PIPER_VOICES = {
    "english": ("en_US-amy-medium", "en/en_US/amy/medium/en_US-amy-medium.onnx"),
    "hindi": ("hi_IN-rohan-medium", "hi/hi_IN/rohan/medium/hi_IN-rohan-medium.onnx"),
    "urdu": ("ur_PK-fasih-medium", "ur/ur_PK/fasih/medium/ur_PK-fasih-medium.onnx"),
}


def pcm_to_wav_bytes(pcm_bytes, sample_rate=22050, channels=1, sample_width=2):
    """Wrap raw PCM bytes in a WAV container (returned as bytes)."""
    import io
    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sample_width)
        w.setframerate(sample_rate)
        w.writeframes(pcm_bytes)
    return buf.getvalue()


def voice_onnx_path(lang):
    if lang not in PIPER_VOICES:
        return None
    return os.path.join(VOICE_DIR, PIPER_VOICES[lang][1])


def download_voice(lang, force=False):
    """Download a Piper voice (.onnx + .onnx.json) from rhasspy/piper-voices."""
    import shutil
    from huggingface_hub import hf_hub_download

    if lang not in PIPER_VOICES:
        raise ValueError(f"no piper voice configured for {lang}")
    rel = PIPER_VOICES[lang][1]
    dest = os.path.join(VOICE_DIR, rel)
    json_dest = dest + ".json"
    if os.path.exists(dest) and os.path.exists(json_dest) and not force:
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    for suffix in ("", ".json"):
        cached = hf_hub_download(repo_id="rhasspy/piper-voices",
                                 filename=rel + suffix, repo_type="model")
        shutil.copyfile(cached, dest + suffix)
    return dest


class PiperTTS:
    def __init__(self, lang="english"):
        self.lang = lang
        self.voice = None

    def _load(self):
        if self.voice is not None:
            return self.voice
        from piper import PiperVoice
        path = voice_onnx_path(self.lang)
        if not path or not os.path.exists(path):
            path = download_voice(self.lang)
        self.voice = PiperVoice.load(path)
        return self.voice

    def synthesize(self, text):
        """Return WAV bytes (16 kHz mono) for the given text."""
        voice = self._load()
        out = bytearray()
        sample_rate = 22050  # piper default
        for chunk in voice.synthesize(text):
            # piper yields AudioChunk objects with audio_int16_bytes
            if hasattr(chunk, "audio_int16_bytes"):
                out += chunk.audio_int16_bytes
                sample_rate = getattr(chunk, "sample_rate", sample_rate)
            elif isinstance(chunk, (bytes, bytearray)):
                out += bytes(chunk)
        return pcm_to_wav_bytes(bytes(out), sample_rate)


class SapiTTS:
    """Fallback TTS via pyttsx3 / SAPI5 (system voices)."""

    def __init__(self, lang="english"):
        self.lang = lang
        self.engine = None

    def _load(self):
        if self.engine is None:
            import pyttsx3
            self.engine = pyttsx3.init()
        return self.engine

    def synthesize(self, text):
        """Return WAV bytes. Uses a temp file because SAPI5 writes to file."""
        engine = self._load()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        try:
            engine.save_to_file(text, path)
            engine.runAndWait()
            with open(path, "rb") as f:
                return f.read()
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass


class TTS:
    """Common interface: TTS(lang).synthesize(text) -> wav bytes."""

    def __init__(self, lang="english", engine="auto"):
        self.lang = lang
        self.engine = engine  # auto | piper | sapi
        self._piper = None
        self._sapi = None

    def synthesize(self, text):
        if self.engine in ("auto", "piper"):
            try:
                if self._piper is None:
                    self._piper = PiperTTS(self.lang)
                return self._piper.synthesize(text)
            except Exception as e:
                if self.engine == "piper":
                    raise
                print(f"  [tts] piper failed ({e}); falling back to sapi")
        if self._sapi is None:
            self._sapi = SapiTTS(self.lang)
        return self._sapi.synthesize(text)


def text_to_wav(text, lang="english", out_path=None, engine="auto"):
    """Convenience: synthesize text and optionally write a wav file."""
    tts = TTS(lang, engine=engine)
    data = tts.synthesize(text)
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(data)
    return data
