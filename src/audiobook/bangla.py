"""
Bengali (Bangla) TTS via two interchangeable fine-tunes of the ResembleAI
Chatterbox ENGLISH base:

  * BosonLab/chatterbox-bangla      — vocab 704 -> 2530, full 5-file set
                                      (models-bangla/)
  * EMTIAZZ/chatterbox-bangla-tts   — vocab 704 -> 4240, T3-only fine-tune
                                      (models-bangla2/, base parts shared)

Both target the ENGLISH ChatterboxTTS class (precomputed-conds generate API,
same S3Gen/voice-encoder layout), so they only need to subclass it and widen
their generate() signature with an ignored language_id — making them drop-in
for every generation path (TTS tab, single/multi audiobooks, batch, regen).

The app exposes them as two language entries in the dropdown:
    bn   -> BosonLab
    bn2  -> EMTIAZZ
Memory stays one-model-at-a-time under a 3-way rule (multilingual | bn | bn2).
"""

import shutil
import threading
import time
from pathlib import Path

try:
    # Prefer the src.* tree (same module objects as the rest of the app).
    # Falling back to the editable-installed chatterbox.* tree works but
    # creates duplicate module objects (separate T3_PROGRESS etc.).
    from src.chatterbox.tts import ChatterboxTTS, Conditionals
except ImportError:  # installed package layout
    from chatterbox.tts import ChatterboxTTS, Conditionals

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None


def normalize_locale(language_id):
    """bn, bn-BD, bn_BD, bn-IN -> 'bn'; bn2, bn2-* -> 'bn2'; else as-is."""
    s = (language_id or "").strip().lower().replace("_", "-")
    if s == "bn" or s.startswith("bn-"):
        return "bn"
    if s == "bn2" or s.startswith("bn2-"):
        return "bn2"
    return s


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

BANGLA2_LANG = "bn2"
BANGLA2_REPO_ID = "EMTIAZZ/chatterbox-bangla-tts"
BANGLA2_BASE_REPO_ID = "ResembleAI/chatterbox"
BANGLA2_DIR = Path(__file__).resolve().parent.parent.parent / "models-bangla2"
BANGLA2_TUNE = "t3_bangla_888k.safetensors"
BANGLA2_TOKENIZER = "tokenizer.json"
BANGLA2_VOCAB = 4240
BANGLA2_BASE_FILES = [
    "ve.safetensors",
    "s3gen.safetensors",
    "conds.pt",
]

_bangla_caches = {"bosonlab": {"model": None}, "emtiazz": {"model": None}}
_bangla_lock = threading.Lock()


def evict_bangla_variant(variant) -> bool:
    """Drop one cached Bangla model to free RAM. True if anything freed."""
    cache = _bangla_caches.get(variant)
    if cache is None or cache.get("model") is None:
        return False
    cache["model"] = None
    import gc as _gc
    _gc.collect()
    try:
        import ctypes as _ct
        _ct.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
    print(f"[MEM] Evicted Bangla model ({variant}) from RAM", flush=True)
    return True


def evict_bangla_model() -> bool:
    """Drop ALL cached Bangla weights. True if anything freed."""
    freed = False
    for variant in list(_bangla_caches):
        if evict_bangla_variant(variant):
            freed = True
    return freed


def is_bangla(language_id) -> bool:
    """True when this language routes to a Bangla model.

    Accepts bn, bn-BD, bn_BD, bn-IN (BosonLab) and bn2, bn2-* (EMTIAZZ).
    """
    try:
        return normalize_locale(language_id) in (BANGLA_LANG, BANGLA2_LANG)
    except Exception:
        s = (language_id or "").strip().lower()
        return s in (BANGLA_LANG, BANGLA2_LANG)


def variant_for(language_id) -> str:
    """bosonlab for bn*, emtiazz for bn2*, bosonlab as the default fallback."""
    try:
        return "emtiazz" if normalize_locale(language_id) == BANGLA2_LANG else "bosonlab"
    except Exception:
        return "bosonlab"


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


class BanglaTTS2(BanglaTTS):
    """BosonLab-compatible generate() wrapper for the EMTIAZZ T3 fine-tune."""

    pass


def _ensure_emtiazz_files():
    """models-bangla2/: base parts reused from models-bangla, fine-tune from HF."""
    BANGLA2_DIR.mkdir(parents=True, exist_ok=True)

    for fpath in BANGLA2_BASE_FILES:
        dest = BANGLA2_DIR / fpath
        if dest.exists():
            continue
        src = BANGLA_DIR / fpath
        if src.exists():
            print(f"[BN2] Reusing base {fpath} from models-bangla/", flush=True)
            shutil.copy2(src, dest)
            print(f"[BN2] Copied {fpath} ({dest.stat().st_size/1e6:.1f}MB)", flush=True)
        else:
            print(f"[BN2] Downloading base {fpath} from {BANGLA2_BASE_REPO_ID} ...", flush=True)
            hf_hub_download(repo_id=BANGLA2_BASE_REPO_ID, filename=fpath, local_dir=str(BANGLA2_DIR))

    for fpath in [BANGLA2_TUNE, BANGLA2_TOKENIZER]:
        dest = BANGLA2_DIR / fpath
        if not dest.exists():
            print(f"[BN2] Downloading {fpath} from {BANGLA2_REPO_ID} (large file, please wait)...", flush=True)
            hf_hub_download(repo_id=BANGLA2_REPO_ID, filename=fpath, local_dir=str(BANGLA2_DIR))
            print(f"[BN2] Downloaded: {fpath} ({dest.stat().st_size/1e6:.1f}MB)", flush=True)


def _load_emtiazz_engine(bangla2_dir) -> BanglaTTS2:
    """Build a ChatterboxTTS-compatible engine on the EMTIAZZ T3 fine-tune.

    Load path mirrors the model card: base engine, rebuild T3 at vocab 4240,
    load tuned weights (stripping the HF `t3.` prefix), swap the 4240 tokenizer.
    """
    import torch as _torch
    from safetensors.torch import load_file as _load_file

    try:
        from src.chatterbox.models.voice_encoder import VoiceEncoder
        from src.chatterbox.models.s3gen import S3Gen
        from src.chatterbox.models.t3 import T3
        from src.chatterbox.models.t3.modules.t3_config import T3Config
        from src.chatterbox.models.tokenizers import EnTokenizer
    except ImportError:  # installed package layout
        from chatterbox.models.voice_encoder import VoiceEncoder
        from chatterbox.models.s3gen import S3Gen
        from chatterbox.models.t3 import T3
        from chatterbox.models.t3.modules.t3_config import T3Config
        from chatterbox.models.tokenizers import EnTokenizer

    ve = VoiceEncoder()
    ve.load_state_dict(_load_file(bangla2_dir / "ve.safetensors"))
    ve.to("cpu").eval()

    t3 = T3(hp=T3Config(text_tokens_dict_size=BANGLA2_VOCAB))
    finetune = _load_file(bangla2_dir / BANGLA2_TUNE)
    if any(k.startswith("t3.") for k in finetune):
        finetune = {k[len("t3."):]: v for k, v in finetune.items() if k.startswith("t3.")}
    t3.load_state_dict(finetune, strict=True)
    t3.to("cpu").eval()

    s3gen = S3Gen()
    s3gen.load_state_dict(_load_file(bangla2_dir / "s3gen.safetensors"), strict=False)
    s3gen.to("cpu").eval()

    tokenizer = EnTokenizer(str(bangla2_dir / BANGLA2_TOKENIZER))

    conds = None
    conds_path = bangla2_dir / "conds.pt"
    if conds_path.exists():
        conds = Conditionals.load(conds_path, map_location=_torch.device("cpu")).to("cpu")

    engine = ChatterboxTTS(t3, s3gen, ve, tokenizer, "cpu", conds=conds)
    model = BanglaTTS2()
    model.__dict__.update(engine.__dict__)
    return model


def load_bangla_model(variant="bosonlab") -> BanglaTTS:
    """Download (first use) + load one Bangla model. Singleton per variant."""
    variant = variant.lower() if isinstance(variant, str) else "bosonlab"
    if variant not in _bangla_caches:
        variant = "bosonlab"
    cache = _bangla_caches[variant]
    if cache["model"] is not None:
        print(f"♻️ [BN] Reusing already-loaded Bangla model ({variant}).")
        return cache["model"]
    with _bangla_lock:
        if cache["model"] is not None:
            print(f"♻️ [BN] Reusing already-loaded Bangla model ({variant}).")
            return cache["model"]
        if hf_hub_download is None:
            raise RuntimeError("huggingface_hub is required to download the Bangla model")
        other = "emtiazz" if variant == "bosonlab" else "bosonlab"
        evict_bangla_variant(other)
        _t0 = time.time()
        if variant == "bosonlab":
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
        else:
            print(f"[BN2] Model dir: {BANGLA2_DIR}", flush=True)
            _ensure_emtiazz_files()
            print("[BN2] Loading EMTIAZZ Bangla weights to CPU (T3 rebuild 704→4240)...", flush=True)
            model = _load_emtiazz_engine(BANGLA2_DIR)
        print(f"[BN] Bangla model ({variant}) ready in {time.time()-_t0:.1f}s", flush=True)
        cache["model"] = model
        return model


def resolve_model_for_language(model, language_id):
    """Route Bengali to the right Bangla singleton; everything else passes through.

    bn/bn-*  -> BosonLab model (vocab 2530)
    bn2/bn2-*-> EMTIAZZ model (vocab 4240)

    Never raises for missing Bangla deps — falls back to the given model so
    callers degrade gracefully (multilingual tokenizer will show its limits
    on Bengali text, but nothing crashes).
    """
    if not is_bangla(language_id):
        return model
    variant = variant_for(language_id)
    try:
        return load_bangla_model(variant)
    except Exception as e:
        print(f"[BN] Bangla model ({variant}) unavailable ({e}) — falling back to multilingual model", flush=True)
        return model