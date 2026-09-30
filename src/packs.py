"""Language packs: per-language adapter + TTS voice + detection config.

Each language has a pack (packs/<lang>/pack.yaml) that declares:
  - language:   english | urdu | hindi
  - adapter:    optional path to a LoRA adapter for the STT (fine-tuned per language)
  - tts_voice:  the Piper voice to use for this language
  - detection:  script regex + stopword hints for language identification

The pipeline detects the caller's language, loads the matching pack, and uses
its adapter (if any) for STT and its tts_voice for TTS. A pack with no adapter
falls back to the base multilingual STT model.

release_gate.py checks that a new pack improves its own language on the dev set
and does not regress the others beyond a tolerance.
"""
import os

import yaml

PACK_DIR = os.path.join("packs")

# Default Piper voices per language (mirror src/tts.py PIPER_VOICES).
DEFAULT_VOICES = {
    "english": "en/en_US/amy/medium/en_US-amy-medium.onnx",
    "urdu": "ur/ur_PK/fasih/medium/ur_PK-fasih-medium.onnx",
    "hindi": "hi/hi_IN/rohan/medium/hi_IN-rohan-medium.onnx",
}


class LanguagePack:
    def __init__(self, language, adapter=None, tts_voice=None, detection=None):
        self.language = language
        self.adapter = adapter          # path to LoRA adapter, or None
        self.tts_voice = tts_voice      # piper voice repo path
        self.detection = detection or {}

    def __repr__(self):
        return f"LanguagePack(lang={self.language}, adapter={self.adapter})"


def load_pack(language, pack_dir=PACK_DIR):
    """Load a language pack. Falls back to a default pack if no YAML exists."""
    path = os.path.join(pack_dir, language, "pack.yaml")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return LanguagePack(
            language=cfg.get("language", language),
            adapter=cfg.get("adapter"),
            tts_voice=cfg.get("tts_voice", DEFAULT_VOICES.get(language)),
            detection=cfg.get("detection", {}),
        )
    # default pack: no adapter, default voice
    return LanguagePack(language=language,
                        tts_voice=DEFAULT_VOICES.get(language))


def available_languages(pack_dir=PACK_DIR):
    """List languages that have a pack directory."""
    if not os.path.isdir(pack_dir):
        return []
    return sorted(d for d in os.listdir(pack_dir)
                  if os.path.isdir(os.path.join(pack_dir, d)))
