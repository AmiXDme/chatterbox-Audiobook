"""Generate Bengali tongue-twister audiobook sample. Run: ./venv/bin/python gen_tongue_twister.py"""
import sys, time
sys.path.insert(0, '.')

from src.audiobook.bangla import BanglaTTS, BANGLA_DIR
import numpy as np
import soundfile as sf

TEXT = """একদিন অতিসূক্ষ্মাতিসূক্ষ্মবিষয়বিশ্লেষণবিশারদ অধ্যাপক মহাশয় তাঁর লশুনপলাণ্ডগুঞ্জনকুম্ভীশ্রাপথন্নসুতকান্নাভোজ্যান্যমধুমাংসমূত্ররেতোহমেধ্যাভক্ষভক্ষণেগায়ত্র্যাষ্টসহ নামের মহাবিশাল বইটি খুলে বললেন, “আজ থেকে যে এই বাক্যটি এক নিঃশ্বাসে পড়তে পারবে, তাকেই আমি সর্বশ্রেষ্ঠজিহ্বাব্যায়ামবিশারদ উপাধি দেব!”

এ কথা শুনে অঘটনঘটনপটীয়সী রাণী তাঁর অতিদুর্বোধ্যবহুমাত্রিকতাসম্পন্নবিশ্ববিখ্যাতদীর্ঘতমশব্দসংবলিত রাজকীয় ঘোষণাপত্র হাতে নিয়ে এমন দ্রুত উচ্চারণ করতে শুরু করলেন যে রাজ্যের সমস্ত ব্যাকরণবিশারদভাষাতত্ত্ববিদশব্দকোষসম্পাদক একসঙ্গে হতবাক হয়ে গেলেন। অধ্যাপক তখন বললেন, “যে ব্যক্তি এই সমগ্র অতিদীর্ঘদুর্বোধ্যবহুসংযুক্তসমাসবদ্ধজিহ্বাবিভ্রান্তিকরবাক্যসমষ্টি একবারে নির্ভুলভাবে পড়তে পারবে, সে-ই হবে পৃথিবীর সর্বশ্রেষ্ঠ অতিদ্রুতউচ্চারণক্ষমতাসম্পন্নবাক্যপাঠবিশারদ!”"""

REF = "/home/mint/Desktop/260911192714.wav"
OUT = "/home/mint/Desktop/tongue_twister_output.wav"

t0 = time.time()
print("Loading Bangla model...", flush=True)
model = BanglaTTS.from_local(BANGLA_DIR, "cpu")
print(f"Model ready in {time.time()-t0:.0f}s. Conditioning voice...", flush=True)
conds = model.prepare_conditionals(REF, 0.5)
print("Voice ready.", flush=True)

sents = [s for s in (x.strip() for x in TEXT.replace("\r", "").splitlines()) if s]
final = [s + "।" if not s.endswith(("।", "!", "?", ".")) else s for s in sents]
print(f"{len(final)} sentences to speak.", flush=True)

parts = []
for i, s in enumerate(final, 1):
    st = time.time()
    print(f"[{i}/{len(final)}] Generating ({len(s.split())} words)...", flush=True)
    wav = model.generate(s, conds, exaggeration=0.5, cfg_weight=0.5, temperature=0.8)
    arr = wav.squeeze(0).numpy()
    parts.append(arr)
    print(f"[{i}/{len(final)}] done: {len(arr)/model.sr:.1f}s audio in {time.time()-st:.0f}s", flush=True)
    sf.write(f"/home/mint/Desktop/tongue_twister_{i:02d}.wav", arr, model.sr)
    print(f"[{i}/{len(final)}] saved part {i}", flush=True)

full = np.concatenate(parts)
sf.write(OUT, full, model.sr)
print(f"SAVED: {OUT} ({len(full)/model.sr:.1f}s) in {time.time()-t0:.0f}s total", flush=True)
