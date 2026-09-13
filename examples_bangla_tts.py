"""Headless Bengali TTS example (no browser, no Gradio UI).

Demonstrates the documented pipeline directly:
BanglaTTS.generate.

Usage:
    ./venv/bin/python examples_bangla_tts.py "আমি বাংলায় কথা বলি।" voice.wav out.wav
    ./venv/bin/python examples_bangla_tts.py @chapter.txt voice.wav out.wav
    (@file reads Bengali text from a file. Defaults shown below.)

Needs: installed venv + downloaded models-bangla/ (first Bengali UI run
fetches them, or run src/audiobook/bangla.py loader once).
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.audiobook.bangla import BanglaTTS, BANGLA_DIR
import numpy as np
import soundfile as sf

DEFAULT_TEXT = ("আমি বাংলায় কথা বলতে পারি। এটি একটি পরীক্ষামূলক বাক্য।")


def read_text(arg):
    if arg.startswith("@"):
        with open(os.path.expanduser(arg[1:]), encoding="utf-8-sig") as f:
            return f.read()
    return arg


def main(argv):
    text = read_text(argv[1]) if len(argv) > 1 else DEFAULT_TEXT
    ref = os.path.expanduser(argv[2]) if len(argv) > 2 else os.path.join(
        ROOT, "voice_library", "bangla_nur", "reference.wav")
    out = os.path.expanduser(argv[3]) if len(argv) > 3 else os.path.join(
        ROOT, "terminal_output", "bangla_example.wav")
    if not os.path.exists(ref):
        print(f"Reference voice not found: {ref}")
        print("Pass one explicitly: ./venv/bin/python examples_bangla_tts.py \"...\" <voice.wav> <out.wav>")
        return 1
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    t0 = time.time()
    print("Loading Bangla model...", flush=True)
    model = BanglaTTS.from_local(BANGLA_DIR, "cpu")
    print(f"Model ready in {time.time()-t0:.0f}s. Conditioning voice...", flush=True)
    conds = model.prepare_conditionals(ref, 0.5)
    print("Voice ready.", flush=True)

    # Text is used exactly as given (numbers, years, currency etc. should be
    # written out as spoken Bangla words beforehand — e.g. via an online AI).
    sents = [s for s in (x.strip() for x in text.replace("\r", "").splitlines()) if s]
    final = [s + "।" if not s.endswith(("।", "!", "?", ".")) else s for s in sents]
    print(f"{len(final)} sentence(s) to speak.", flush=True)

    parts = []
    for i, s in enumerate(final, 1):
        st = time.time()
        print(f"[{i}/{len(final)}] Generating ({len(s.split())} words)...", flush=True)
        wav = model.generate(s, conds, exaggeration=0.5, cfg_weight=0.5, temperature=0.8)
        arr = wav.squeeze(0).numpy()
        parts.append(arr)
        print(f"[{i}/{len(final)}] done: {len(arr)/model.sr:.1f}s audio in {time.time()-st:.0f}s", flush=True)

    full = np.concatenate(parts)
    sf.write(out, full, model.sr)
    print(f"SAVED: {out} ({len(full)/model.sr:.1f}s) in {time.time()-t0:.0f}s total", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
