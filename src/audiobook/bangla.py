"""
Bengali (Bangla) TTS via the BosonLab/chatterbox-bangla fine-tune.

The stock multilingual model covers 23 languages but NOT Bengali. This module
adds Bangla as a 24th capability using BosonLab's fine-tuned T3 checkpoint
(vocab extended 704 -> 2530 for Bengali script).

Architecture fit: the fine-tune targets the ENGLISH ChatterboxTTS class
(precomputed-conds generate API, same S3Gen/voice-encoder layout), so
BanglaTTS subclasses it and only widens the generate() signature with an
ignored language_id — making it a drop-in for every generation path
(TTS tab, single/multi audiobooks, batch, regen).
"""

import os
import threading
import time
from pathlib import Path

try:
    # Prefer the src.* tree (same module objects as the rest of the app).
    # Falling back to the editable-installed chatterbox.* tree works but
    # creates duplicate module objects (separate T3_PROGRESS etc.).
    from src.chatterbox.tts import ChatterboxTTS
except ImportError:  # installed package layout
    from chatterbox.tts import ChatterboxTTS

try:
    from huggingface_hub import hf_hub_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

try:
    # Shared locale helper (bn/bn-BD/bn_BD -> bn). No cycle: langtext
    # imports nothing from this module.
    from src.audiobook.langtext import normalize_locale
except ImportError:
    try:
        from .langtext import normalize_locale
    except ImportError:
        normalize_locale = None


BANGLA_LANG = "bn"
BANGLA_REPO_ID = "BosonLab/chatterbox-bangla"
BANGLA_DIR = Path(__file__).resolve().parent.parent.parent / "models-bangla"
BANGLA_FILES = [
    "ve.safetensors",
    "t3_cfg.safetensors",
    "s3gen.safetensors",
    "tokenizer.json",
    "conds.pt",
]

_bangla_cache = {"model": None}
_bangla_lock = threading.Lock()


def evict_bangla_model() -> bool:
    """Drop cached Bangla weights to free RAM. Returns True if anything was freed."""
    if _bangla_cache.get("model") is not None:
        _bangla_cache["model"] = None
        import gc as _gc
        _gc.collect()
        try:
            import ctypes as _ct
            _ct.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
        print("[MEM] Evicted Bangla model from RAM", flush=True)
        return True
    return False


def is_bangla(language_id) -> bool:
    """True when the requested language should route to the Bangla model.

    Accepts bn, bn-BD, bn_BD, bn-IN (etc.) — anything else is not Bangla.
    """
    if normalize_locale is not None:
        try:
            return normalize_locale(language_id) == BANGLA_LANG
        except Exception:
            pass
    return (language_id or "").strip().lower() == BANGLA_LANG


class BanglaTTS(ChatterboxTTS):
    """English-class Chatterbox with a multilingual-compatible generate().

    All params identical to ChatterboxTTS.generate(); language_id is accepted
    and ignored (this model only speaks Bengali).
    """

    def generate(
        self,
        text,
        conds,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
        repetition_penalty=1.2,
        min_p=0.05,
        top_p=1.0,
        language_id=BANGLA_LANG,
        **kwargs,
    ):
        return super().generate(
            text,
            conds,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            min_p=min_p,
            top_p=top_p,
        )


def load_bangla_model() -> BanglaTTS:
    """Download (first use) + load the Bangla model. Singleton, CPU-only."""
    if _bangla_cache["model"] is not None:
        print("♻️ [BN] Reusing already-loaded Bangla model.")
        return _bangla_cache["model"]
    with _bangla_lock:
        if _bangla_cache["model"] is not None:
            print("♻️ [BN] Reusing already-loaded Bangla model.")
            return _bangla_cache["model"]
        if not HF_HUB_AVAILABLE:
            raise RuntimeError("huggingface_hub is required to download the Bangla model")
        _t0 = time.time()
        BANGLA_DIR.mkdir(parents=True, exist_ok=True)
        missing = [f for f in BANGLA_FILES if not (BANGLA_DIR / f).exists()]
        print(f"[BN] Bengali model dir: {BANGLA_DIR} | missing={missing or 'none'}", flush=True)
        for fpath in BANGLA_FILES:
            dest = BANGLA_DIR / fpath
            if not dest.exists():
                print(f"[BN] Downloading {fpath} from {BANGLA_REPO_ID} (large file, terminal quiet until done)...", flush=True)
                hf_hub_download(repo_id=BANGLA_REPO_ID, filename=fpath, local_dir=str(BANGLA_DIR))
                print(f"[BN] Downloaded: {fpath} ({dest.stat().st_size/1e6:.1f}MB)", flush=True)
            else:
                print(f"[BN] Already exists: {fpath}", flush=True)
        print("[BN] Loading Bangla weights to CPU...", flush=True)
        model = BanglaTTS.from_local(BANGLA_DIR, "cpu")
        print(f"[BN] Bangla model ready in {time.time()-_t0:.1f}s", flush=True)
        _bangla_cache["model"] = model
        return model


def resolve_model_for_language(model, language_id):
    """Route Bengali to the Bangla singleton; everything else passes through.

    Never raises for missing Bangla deps — falls back to the given model so
    callers degrade gracefully (multilingual tokenizer will show its limits
    on Bengali text, but nothing crashes).
    """
    if not is_bangla(language_id):
        return model
    try:
        return load_bangla_model()
    except Exception as e:
        print(f"[BN] Bangla model unavailable ({e}) — falling back to multilingual model", flush=True)
        return model
