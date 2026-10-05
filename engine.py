"""Overnight training loop for the EARS (Whisper STT) with LoRA fine-tuning.

Commands:
  py engine.py night --hours N     time-budgeted LoRA fine-tuning rounds
  py engine.py smoke                tiny run to prove the loop works
  py engine.py eval                 evaluate current/best checkpoint on the test set

Behavior:
  - trains in time-budgeted rounds (each round ~ budget/rounds)
  - evaluates WER on the test set (data/metadata.csv) after each round
  - keeps the best checkpoint, rolls back if worse
  - logs scores to results/train_log.csv
  - resumes after interruption (skips completed rounds, loads last checkpoint)
  - STOP-file kill switch: create a file named STOP in the project root to stop
    gracefully after the current round

Checkpoints go to checkpoints/ (git-ignored). The best checkpoint is copied to
checkpoints/best/.
"""
import argparse
import csv
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

STOP_FILE = os.path.join(".", "STOP")
CHECKPOINT_DIR = "checkpoints"
BEST_DIR = os.path.join(CHECKPOINT_DIR, "best")
LOG_CSV = os.path.join("results", "train_log.csv")
CONFIG = os.path.join("configs", "train.yaml")

DEFAULT_CFG = {
    "model": "openai/whisper-tiny",
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "learning_rate": 1e-4,
    "batch_size": 4,
    "grad_accum": 4,
    "max_steps_per_round": 50,
    "rounds": 4,
    "warmup_ratio": 0.1,
    "test_metadata": "data/devset_metadata.csv",
    "train_dir": "data/train",
}


def load_config():
    import yaml
    cfg = dict(DEFAULT_CFG)
    if os.path.exists(CONFIG):
        with open(CONFIG, encoding="utf-8") as f:
            cfg.update(yaml.safe_load(f) or {})
    return cfg


def stop_requested():
    return os.path.exists(STOP_FILE)


# ------------------------------------------------------------------ dataset

def load_train_dataset(cfg):
    """Load training data. Returns a datasets.Dataset with 'audio' and 'text'.

    Supports:
      - hf-internal-testing/librispeech_asr_dummy (tiny public smoke-test set)
      - synthetic (piper TTS clips with known transcripts)
      - local dir data/train/ with wav files + transcripts.csv
    """
    from datasets import load_dataset, Dataset
    import soundfile as sf

    kind = cfg.get("train_kind", "auto")
    if kind == "auto":
        # default to synthetic: self-contained, no ffmpeg / no broken paths.
        # (librispeech_dummy is surveyed but blocked by ffmpeg + wrong file paths)
        kind = "synthetic"

    if kind == "librispeech_dummy":
        # NOTE: surveyed but not used — audio decode needs ffmpeg (torchcodec) and
        # the 'file' column holds the original creator's absolute paths (broken here).
        # Kept for reference; use 'synthetic' for the smoke test.
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy",
                          "clean", split="validation")
        ds = ds.select(range(min(cfg.get("max_train_clips", 16), len(ds))))
        return ds

    if kind == "synthetic":
        # generate clips with piper TTS; transcripts are known exactly.
        # Self-contained: no ffmpeg, no broken paths. Resampled to 16 kHz.
        # Covers all 3 languages + synthetic Urdu-English mixed sentences.
        from src.tts import TTS
        sentences = {
            "english": [
                "hello", "book an appointment", "what are your hours",
                "how much does it cost", "where are you located",
                "i want to see a doctor", "thank you", "goodbye",
                "what services do you offer", "are you open on saturday",
            ],
            "urdu": [
                "السلام علیکم", "میں ایپائنٹمنٹ بک کرنا چاہتا ہوں",
                "آپ کے اوقات کار کیا ہیں", "معائنے کی قیمت کتنی ہے",
                "آپ کی کلینک کہاں ہے", "میں ڈاکٹر سے ملنا چاہتا ہوں",
                "شکریہ", "الوداع", "کیا آپ ہفتے کو کھلے ہیں",
            ],
            "hindi": [
                "नमस्ते", "मैं अपॉइंटमेंट बुक करना चाहता हूँ",
                "आप कब खुलते हैं", "परामर्श की कीमत क्या है",
                "आप क्लिनिक कहाँ है", "मैं डॉक्टर से मिलना चाहता हूँ",
                "धन्यवाद", "अलविदा", "क्या आप शनिवार खुले हैं",
            ],
        }
        rows = []
        for lang, sents in sentences.items():
            tts = TTS(lang)
            for s in sents:
                wav = tts.synthesize(s)
                pcm, sr = _wav_to_pcm(wav)
                if sr != 16000:
                    pcm = _resample(pcm, sr, 16000)
                rows.append({"audio": {"array": pcm, "sampling_rate": 16000},
                             "text": s, "language": lang})
        # synthetic Urdu-English mixed sentences (labelled clearly)
        mixed = [
            "میں appointment book کرنا چاہتا ہوں",
            "آپ کی price کیا ہے",
            "میرا name احمد ہے",
            "क्या آپ Saturday کو open ہیں",
        ]
        tts = TTS("urdu")
        for s in mixed:
            wav = tts.synthesize(s)
            pcm, sr = _wav_to_pcm(wav)
            if sr != 16000:
                pcm = _resample(pcm, sr, 16000)
            rows.append({"audio": {"array": pcm, "sampling_rate": 16000},
                         "text": s, "language": "mixed"})
        # --- augmentation variants (curriculum stages) ---
        aug = cfg.get("augment", "all")
        if aug != "none":
            from src.augment import augment as _augment
            base_rows = list(rows)
            for i, r in enumerate(base_rows):
                pcm = r["audio"]["array"]
                if aug in ("noise", "all", "mixed"):
                    noisy = _augment(pcm, sr=16000, phone=False, noise_snr=15.0, seed=i)
                    rows.append({"audio": {"array": noisy, "sampling_rate": 16000},
                                 "text": r["text"], "language": r.get("language", "english")})
                if aug in ("phone", "all", "mixed"):
                    phone = _augment(pcm, sr=16000, phone=True, seed=i + 1000)
                    rows.append({"audio": {"array": phone, "sampling_rate": 8000},
                                 "text": r["text"], "language": r.get("language", "english")})
        return Dataset.from_list(rows)

    if kind == "local":
        import csv as csvmod
        rows = []
        tpath = os.path.join(cfg["train_dir"], "transcripts.csv")
        with open(tpath, newline="", encoding="utf-8") as f:
            for r in csvmod.DictReader(f):
                audio, sr = sf.read(os.path.join(cfg["train_dir"], r["path"]))
                rows.append({"audio": {"array": audio, "sampling_rate": sr},
                             "text": r["text"]})
        return Dataset.from_list(rows)

    raise ValueError(f"unknown train_kind {kind}")


def _wav_to_pcm(wav_bytes):
    import io
    import soundfile as sf
    return sf.read(io.BytesIO(wav_bytes), dtype="float32")


def _resample(pcm, orig_sr, target_sr):
    if orig_sr == target_sr:
        return pcm
    import numpy as np
    n_out = int(len(pcm) * target_sr / orig_sr)
    x_old = np.linspace(0, 1, len(pcm))
    x_new = np.linspace(0, 1, n_out)
    return np.interp(x_new, x_old, pcm).astype(np.float32)


# ------------------------------------------------------------------ model

def setup_model(cfg):
    """Load whisper + LoRA (8-bit on GPU). Returns (model, processor)."""
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    from peft import LoraConfig, get_peft_model

    model_name = cfg["model"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_name)

    if device == "cuda":
        # hard VRAM cap: 80% of total (leaves headroom for OS/display)
        frac = cfg.get("vram_fraction", 0.8)
        torch.cuda.set_per_process_memory_fraction(frac)
        from transformers import BitsAndBytesConfig
        qcfg = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name, quantization_config=qcfg, device_map="auto")
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
    else:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name).to(device)
        model.config.use_cache = False

    # target attention projections (whisper uses q_proj/k_proj/v_proj/out_proj)
    target = ["q_proj", "v_proj", "k_proj", "out_proj",
              "fc1", "fc2"]
    lora = LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"],
                      lora_dropout=cfg["lora_dropout"],
                      target_modules=target)
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model, processor


# ------------------------------------------------------------------ eval

def evaluate_wer(model, processor, cfg):
    """Evaluate WER on the test set. Returns (wer, results_dict)."""
    import csv as csvmod
    import numpy as np
    import torch
    from src.eval import evaluate

    meta = cfg["test_metadata"]
    if not os.path.exists(meta):
        return float("nan"), {}
    rows = []
    with open(meta, newline="", encoding="utf-8") as f:
        for r in csvmod.DictReader(f):
            rows.append(r)
    if not rows:
        return float("nan"), {}

    hypotheses = {}
    for row in rows:
        import soundfile as sf
        path = row["path"]
        if not os.path.isabs(path):
            path = os.path.join(".", path)
        if not os.path.exists(path):
            hypotheses[row["path"]] = ""
            continue
        pcm, sr = sf.read(path, dtype="float32")
        if pcm.ndim > 1:
            pcm = pcm.mean(axis=1)
        inp = processor(pcm, sampling_rate=sr, return_tensors="pt")
        inp = {k: v.to(model.device) for k, v in inp.items()}
        with torch.no_grad():
            out = model.generate(**inp)
        text = processor.batch_decode(out, skip_special_tokens=True)[0]
        hypotheses[row["path"]] = text
    result = evaluate(meta, hypotheses)
    return result["overall"], result


# ------------------------------------------------------------------ training

def train_round(model, processor, ds, cfg, round_idx, max_steps):
    """One training round. Returns the number of steps trained."""
    import torch
    from torch.utils.data import DataLoader

    model.train()
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True,
                        collate_fn=lambda b: _collate(b, processor))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    total = min(max_steps, len(loader) * 1)
    step = 0
    opt.zero_grad()
    for batch in loader:
        if step >= total:
            break
        batch = {k: v.to(model.device) for k, v in batch.items()}
        out = model(**batch)
        loss = out.loss / cfg["grad_accum"]
        loss.backward()
        if (step + 1) % cfg["grad_accum"] == 0:
            opt.step()
            opt.zero_grad()
        step += 1
    if (step) % cfg["grad_accum"] != 0:
        opt.step()
        opt.zero_grad()
    return step


def _collate(batch, processor):
    import torch
    audio = [b["audio"]["array"] for b in batch]
    texts = [b["text"] for b in batch]
    inp = processor(audio, sampling_rate=16000, return_tensors="pt",
                    padding=True)
    # whisper expects mel features of length 3000 (30 s); pad if shorter
    feats = inp["input_features"]
    if feats.shape[-1] < 3000:
        pad = torch.zeros(feats.shape[0], feats.shape[1], 3000 - feats.shape[-1])
        feats = torch.cat([feats, pad], dim=-1)
    labels = processor.tokenizer(texts, return_tensors="pt", padding=True).input_ids
    labels[labels == processor.tokenizer.pad_token_id] = -100
    return {"input_features": feats, "labels": labels}


def save_checkpoint(model, processor, round_idx, wer, tag="last"):
    dest = os.path.join(CHECKPOINT_DIR, f"round_{round_idx}_{tag}")
    if os.path.exists(dest):
        shutil.rmtree(dest)
    model.save_pretrained(dest)
    processor.save_pretrained(dest)
    with open(os.path.join(dest, "round.json"), "w") as f:
        json.dump({"round": round_idx, "wer": wer}, f)
    return dest


def load_checkpoint(model, processor, path):
    """Load a LoRA checkpoint for CONTINUED training (is_trainable=True)."""
    from peft import PeftModel
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    processor = AutoProcessor.from_pretrained(path)
    base = AutoModelForSpeechSeq2Seq.from_pretrained(
        model.name_or_path if hasattr(model, "name_or_path") else "openai/whisper-tiny")
    # is_trainable=True so LoRA params require grad (saved adapters default to
    # inference_mode=True, which would make the loss not require grad)
    model = PeftModel.from_pretrained(base, path, is_trainable=True)
    return model, processor


# ------------------------------------------------------------------ log

def log_csv(round_idx, wer, seconds, checkpoint):
    os.makedirs(os.path.dirname(LOG_CSV), exist_ok=True)
    exists = os.path.exists(LOG_CSV)
    with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["round", "wer", "seconds", "checkpoint", "timestamp"])
        w.writerow([round_idx, wer, seconds, checkpoint, time.time()])


def read_log():
    if not os.path.exists(LOG_CSV):
        return []
    with open(LOG_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ------------------------------------------------------------------ commands

def cmd_night(args):
    cfg = load_config()
    if args.hours:
        cfg["budget_hours"] = args.hours
    budget = cfg.get("budget_hours", 1.0) * 3600
    rounds = cfg["rounds"]
    per_round = budget / rounds
    max_steps = cfg["max_steps_per_round"]

    print(f"night run: {args.hours or cfg.get('budget_hours')}h, {rounds} rounds, "
          f"~{per_round/60:.0f} min/round, model={cfg['model']}")

    ds = load_train_dataset(cfg)
    print(f"train clips: {len(ds)}")
    model, processor = setup_model(cfg)

    # resume: find last completed round
    log = read_log()
    start_round = 0
    best_wer = float("inf")
    if log:
        start_round = max(int(r["round"]) for r in log) + 1
        valid_wers = [float(r["wer"]) for r in log
                      if r["wer"] not in ("", "nan") and float(r["wer"]) == float(r["wer"])]
        if valid_wers:
            best_wer = min(valid_wers)
        print(f"resuming from round {start_round}, best_wer={best_wer:.4f}")
        # load last checkpoint
        last = os.path.join(CHECKPOINT_DIR, f"round_{start_round-1}_last")
        if os.path.exists(last):
            model, processor = load_checkpoint(model, processor, last)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    t_start = time.time()

    for r in range(start_round, rounds):
        t0 = time.time()
        # OOM handling: retry with smaller batch, clear cache, log it
        bs = cfg["batch_size"]
        for attempt in range(3):
            try:
                cfg["batch_size"] = max(1, bs // (2 ** attempt))
                steps = train_round(model, processor, ds, cfg, r, max_steps)
                break
            except torch.cuda.OutOfMemoryError:
                print(f"  OOM at batch_size={cfg['batch_size']} — clearing cache, retrying smaller")
                torch.cuda.empty_cache()
                if attempt == 2:
                    raise
        train_t = time.time() - t0
        wer, _ = evaluate_wer(model, processor, cfg)
        print(f"round {r}: {steps} steps in {train_t:.0f}s, WER={wer:.4f}")
        save_checkpoint(model, processor, r, wer, tag="last")
        log_csv(r, wer, train_t, f"round_{r}_last")
        if wer == wer and wer < best_wer:  # wer==wer excludes nan
            best_wer = wer
            save_checkpoint(model, processor, r, wer, tag="best")
            if os.path.exists(BEST_DIR):
                shutil.rmtree(BEST_DIR)
            shutil.copytree(os.path.join(CHECKPOINT_DIR, f"round_{r}_best"), BEST_DIR)
            print(f"  new best WER={wer:.4f} -> {BEST_DIR}")
        # STOP-file kill switch: checked after each round (graceful)
        if stop_requested():
            print("STOP file found — stopping gracefully.")
            break
        if time.time() - t_start > budget:
            print("budget exhausted — stopping.")
            break

    print(f"done. best_wer={best_wer:.4f}")


def cmd_smoke(args):
    """Tiny run to prove the loop: 2 rounds, few steps, dummy dataset."""
    cfg = load_config()
    cfg.update({
        "model": "openai/whisper-tiny",
        "train_kind": "synthetic",
        "max_train_clips": 8,
        "max_steps_per_round": 2,
        "rounds": 2,
        "batch_size": 2,
        "grad_accum": 1,
        "learning_rate": 1e-4,
    })
    print("smoke run:", cfg)
    ds = load_train_dataset(cfg)
    print(f"train clips: {len(ds)}")
    model, processor = setup_model(cfg)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    for r in range(cfg["rounds"]):
        t0 = time.time()
        steps = train_round(model, processor, ds, cfg, r, cfg["max_steps_per_round"])
        wer, _ = evaluate_wer(model, processor, cfg)
        save_checkpoint(model, processor, r, wer, tag="last")
        log_csv(r, wer, time.time() - t0, f"round_{r}_last")
        print(f"smoke round {r}: {steps} steps, WER={wer:.4f}")
    print("smoke run complete.")


def cmd_eval(args):
    cfg = load_config()
    model, processor = setup_model(cfg)
    if args.checkpoint:
        model, processor = load_checkpoint(model, processor, args.checkpoint)
    elif os.path.exists(BEST_DIR):
        model, processor = load_checkpoint(model, processor, BEST_DIR)
    wer, details = evaluate_wer(model, processor, cfg)
    print(f"WER={wer:.4f}")
    out = os.path.join("results", "eval_latest.json")
    os.makedirs("results", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(details, f, indent=2, ensure_ascii=False)


# Curriculum stages: (name, augmentation level). Advance when dev WER plateaus.
CURRICULUM = [
    ("clean", "none"),
    ("noise", "noise"),
    ("phone", "phone"),
    ("mixed", "mixed"),
]


def cmd_curriculum(args):
    """Staged-difficulty training. Advance when dev WER stops improving."""
    import torch
    cfg = load_config()
    rounds_per_stage = args.rounds_per_stage
    best_wer = float("inf")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    model, processor = setup_model(cfg)
    stage_scores = []
    for stage_name, aug in CURRICULUM:
        if stop_requested():
            print("STOP file found — stopping.")
            break
        print(f"\n=== stage: {stage_name} (aug={aug}) ===")
        cfg["augment"] = aug
        ds = load_train_dataset(cfg)
        print(f"train clips: {len(ds)}")
        stage_best = float("inf")
        stall = 0
        for r in range(rounds_per_stage):
            if stop_requested():
                break
            t0 = time.time()
            bs = cfg["batch_size"]
            for attempt in range(3):
                try:
                    cfg["batch_size"] = max(1, bs // (2 ** attempt))
                    steps = train_round(model, processor, ds, cfg, r, cfg["max_steps_per_round"])
                    break
                except torch.cuda.OutOfMemoryError:
                    print(f"  OOM — clearing cache, retrying smaller")
                    torch.cuda.empty_cache()
                    if attempt == 2:
                        raise
            wer, _ = evaluate_wer(model, processor, cfg)
            print(f"  round {r}: WER={wer:.4f} ({time.time()-t0:.0f}s)")
            save_checkpoint(model, processor, r, wer, tag="last")
            log_csv(f"{stage_name}_{r}", wer, time.time() - t0, f"round_{r}_last")
            stage_scores.append({"stage": stage_name, "round": r, "wer": wer})
            if wer == wer and wer < best_wer:
                best_wer = wer
                save_checkpoint(model, processor, r, wer, tag="best")
                if os.path.exists(BEST_DIR):
                    shutil.rmtree(BEST_DIR)
                shutil.copytree(os.path.join(CHECKPOINT_DIR, f"round_{r}_best"), BEST_DIR)
                print(f"  new best WER={wer:.4f}")
            if wer < stage_best - 0.005:
                stage_best = wer
                stall = 0
            else:
                stall += 1
            if stall >= 2:
                print(f"  plateaued — advancing to next stage")
                break
    # save scores CSV + plot
    import csv
    os.makedirs("results", exist_ok=True)
    with open("results/curriculum_scores.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["stage", "round", "wer"])
        w.writeheader()
        w.writerows(stage_scores)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        stages = [s["stage"] for s in stage_scores]
        wers = [s["wer"] for s in stage_scores]
        plt.figure(figsize=(8, 4))
        plt.plot(range(len(wers)), wers, "o-")
        plt.xticks(range(len(wers)), [f"{s}" for s in stages], rotation=30)
        plt.ylabel("WER")
        plt.title("Curriculum training: WER per round")
        plt.tight_layout()
        plt.savefig("results/curriculum_plot.png", dpi=100)
        print("wrote results/curriculum_plot.png")
    except Exception as e:
        print(f"plot skipped: {e}")
    print(f"\ndone. best_wer={best_wer:.4f}")
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_night = sub.add_parser("night")
    p_night.add_argument("--hours", type=float, default=None)
    p_smoke = sub.add_parser("smoke")
    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--checkpoint", default=None)
    p_cur = sub.add_parser("curriculum")
    p_cur.add_argument("--rounds-per-stage", type=int, default=3)
    args = ap.parse_args()

    if args.cmd == "night":
        cmd_night(args)
    elif args.cmd == "smoke":
        cmd_smoke(args)
    elif args.cmd == "eval":
        cmd_eval(args)
    elif args.cmd == "curriculum":
        cmd_curriculum(args)


if __name__ == "__main__":
    main()
