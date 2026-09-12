# 🎧 Chatterbox Audiobook Generator

**Transform your text into high-quality audiobooks with the ResembleAI Chatterbox Multilingual model (23 languages), voice cloning, multi-character voices, and professional volume normalization. CPU-only — no GPU needed.**

> This README documents the full Linux port + realtime upgrade session: every fix, feature, and lesson learned is listed in [Session Changelog](#session-changelog-what-was-done) at the bottom.

## 🚀 Quick Start (Linux)

### 1. Install (first time only)
```bash
cd ~/Desktop/chatterbox-Audiobook
sudo apt install -y python3.12-venv   # Mint/Ubuntu need this once (python3-venv is not preinstalled)
./install-audiobook.sh
```
This creates `venv/`, installs CPU PyTorch + dependencies, and downloads the multilingual model (~1.5 GB) into `models-multilingual/`. The installer self-heals a missing `python3-venv` (auto-installs via apt), warns if disk space is under 6 GB, and offers to create an 8 GB swapfile (recommended on <16 GB RAM — prevents OOM kills).

### 2. Launch
```bash
./launch_audiobook.sh
```
Open the printed link (`http://127.0.0.1:7860`). Available scripts:

**No model preload by design:** the UI opens in seconds with zero models loaded. **You** pick the language first — the right model (multilingual or Bangla) loads on your first Generate/Create click via `load_model(language_id)`. Bengali-only users never download or load the multilingual weights at all.

**Strict one-model-at-a-time:** picking EN runs *only* multilingual; picking BN evicts multilingual and runs *only* Bangla (and vice versa) — caches cleared + RAM trimmed + UI state synced, logged as `[MEM] Evicted …`. Alternating languages reloads each switch; sticking to one never reloads.

| Script | Purpose |
|---|---|
| `install-audiobook.sh` | Full installer (venv, deps, model download) |
| `launch_audiobook.sh` | Main launcher (recommended) |
| `launch_local.sh` | Localhost only |
| `launch_network.sh` | LAN access (`0.0.0.0`, uses `hostname -I`) |

(Windows `.bat` launchers still exist and work on Windows.)

### 3. Double-click launch (no terminal typing)
- **Mint menu (recommended):** Super key → type `Chatterbox` → click. A `Chatterbox-Audiobook.desktop` entry is registered in `~/.local/share/applications/` (add to panel/desktop via right-click).
- **Source folder:** double-click `Chatterbox-Audiobook.desktop` inside the project folder (Nemo: right-click → Allow Launching on first use).
- **Sigma File Manager:** Sigma does not pass filenames to helper apps, so `.sh` double-click can't auto-run a terminal there. Use the Mint menu entry, or right-click empty space → Open in Terminal → `./launch_audiobook.sh`.

---

## 📁 Where Everything Lives (all inside the source folder)

| What | Path |
|---|---|
| Virtual environment | `venv/` |
| Multilingual TTS model, all 23 languages (~1.5 GB) | `models-multilingual/` (`ve.pt`, `t3_mtl23ls_v3.safetensors`, `s3gen.pt`, `grapheme_mtl_merged_expanded_v1.json`, `conds.pt`, `Cangjie5_TC.json`) |
| Voice Conversion model (~1 GB, downloaded on first VC use) | `models/` (`s3gen.pt`, `conds.pt`) |
| Your voice profiles | `voice_library/<name>/` (`voice.wav` + `config.json`) |
| Generated audiobooks (per-chunk WAVs + metadata) | `audiobook_projects/<project>/` |
| Voice conversion outputs | `vc_output/<source>_<timestamp>/` (per-chunk files + `*_converted_full.wav`) |
| Bengali TTS model (~2 GB, first Bengali use) | `models-bangla/` (`t3_cfg.safetensors`, `s3gen.safetensors`, `ve.safetensors`, `tokenizer.json`, `conds.pt`) |
| App config | `audiobook_config.json` |

Outside the folder, only two normal things: the `python3.12-venv` system package (via apt) and the pip download cache (`~/.cache/pip`, cleanable with `pip cache purge`).

---

## ✨ Features

### 📚 Audiobook Creation (single + multi-voice)
- 50-word smart chunks (sentence boundaries, line-break pauses preserved)
- Per-chunk WAVs saved immediately + resume support (re-run picks up missing chunks)
- Character voices via `[Character]` tags, per-character volume settings
- Volume normalization presets: audiobook −18 dB, podcast −16 dB, broadcast −23 dB

### ⚡ Realtime Generation Engine (single-voice, multi-voice AND voice conversion)
- **Live streaming UI** — partial audio, status, and timing refresh after every chunk (generator yields, original styling kept)
- **Timing panel** — start stamp, elapsed, ETA, projected end clock time (`ends ~14:35:10`)
- **Terminal heartbeat** — timestamped tick every 5 s (chunk X/Y, elapsed, ETA, tokens, **RAM MB**); pause-aware; job-id guarded; 30-min stale-thread guard; **token-aware live ETA** (T3 loop streams counts, so single-segment jobs show `tok 100/172 (2.5/s) ETA 29s` instead of a stuck `ETA 0s`); **⚠️ slow-chunk watchdog** (a chunk past 3× average is announced as slow-not-stuck)
- **🛑 Cancel** — stops after the current chunk, partial result kept (audiobooks can be resumed later)
- **⏸️ Pause / ▶️ Resume** (TTS tabs) — gate checked before every chunk, button label toggles live
- **Skip-and-continue** — a failed chunk is recorded and skipped instead of killing a hours-long run; failures listed at the end
- **Current chunk text** — live `Now: chunk 12/340 [Narrator]: "..."` display (HTML-escaped)
- **Model-phase logs** — voice-setup step timings (load/resample/embed/tokens/speaker), T3 token count + tok/s, 100-token AR milestones, S3Gen flow vs vocoder split, tokenizer chars→tokens, per-segment **×realtime factor**
- **Session odometer** — every finished job prints running totals (`📊 [SESSION] ... → 124.6 min audio in 18.42 h`)
- **Multi-voice conds cache** — voice conditioning embedded once per voice (not per chunk); 200-chunk books save minutes
- **Job framing everywhere** — `@logged_job` start/finish/duration on batch, legacy-audiobook, WAV-combine, regen, clean, analyze (nothing runs silent)
- **Unbuffered output** — all launchers run `python3 -u`, so every line appears instantly
- **Startup breadcrumbs** — torch/gradio/stack/UI-build/model-weight progress markers so cold starts never look dead
- **🔊 ASCII waveforms** — every finished job prints its sound-shape in the terminal (flat line = silent failure spotted without a player; solid wall = clipping) plus a **completion bell** for long runs
- **🐌 Slowest-chunk leaderboard** — each finale lists the top-5 slowest chunks with word counts, so problem text can be found and shortened
- **💾 Disk-space guard** — audiobook/VC jobs warn when free disk drops under 2 GB, before a mid-book write fails
- **Strict one-model-at-a-time** — EN runs only multilingual, BN runs only Bangla; switching evicts the other (cache cleared, RAM trimmed, UI state synced, logged as `[MEM] Evicted …`); multi-voice finale reports conditioning cache hits/misses

### 🔄 Voice Conversion Studio
- Any-length source audio → auto-split into **30 s chunks** → converted → stitched with 10 ms edge fades (no clicks)
- Per-chunk retry (2×), per-chunk downloadable files, original-vs-converted A/B players
- Target voice embedded once (first 10 s used); temp files cleaned automatically

### 🇧🇩 Bengali (Bangla) — 24th Language
The stock multilingual model covers 23 languages but **not Bengali**. This app adds it via the [**BosonLab/chatterbox-bangla**](https://huggingface.co/BosonLab/chatterbox-bangla) fine-tune (MIT license, ~99 h Bengali speech, vocab extended 704→2530):
- **Select `Bengali (Bangla)`** in any language dropdown — TTS tab, single/multi audiobooks, batch, regen all route automatically
- **Lazy singleton**: downloads to `models-bangla/` on first Bengali use, then reuses (terminal shows `[BN]` lines with sizes/durations)
- **Voice cloning works the same** — any ~10 s reference voice speaks Bengali
- **How it fits**: the fine-tune targets the English-class TTS (precomputed-conds API), so a thin `BanglaTTS` adapter makes it a drop-in everywhere; `punc_norm` respects the Bengali dari (`।`); English `from_local` auto-sizes T3 to any checkpoint vocab
- First Bengali click downloads ~2 GB once; failures fall back to multilingual with a terminal note (never a crash)

### 📱 Phone Access
- **LAN link + QR in terminal** — server binds `0.0.0.0`; on launch the terminal prints `http://<lan-ip>:7860` plus a scannable QR code (same WiFi, trusted networks only)
- **Public link** — disabled (`share=False` in code; re-enable only if you need off-WiFi access and accept the exposure)
- Phone is a remote control: generation still runs on the PC; mic recording works from mobile browsers

### 🖥️ Terminal Twin (`terminal_app.py`, `./launch_terminal.sh`)
The full Gradio UI cloned for the terminal — Bangla + English, quick TTS, single/multi audiobooks — **zero copied logic**: it imports and calls the exact same engine functions as the GUI buttons, so the two can never drift apart. Same models (lazy singletons, one-at-a-time), same `voice_library/` + `audiobook_projects/` (start a book in terminal, resume it in the browser and vice versa), same heartbeat/ETA/waveform terminal output. **Surprise inside: watch-folder batch mode** — drop `.txt` files in a folder, each becomes an audiobook. `Ctrl+C` cancels any job (partial work kept).

### 🎭 Voice Management, 🎚️ Normalization, 🔇 Return Pauses, 📋 Batch
Unchanged from before: voice library with clone-from-sample, professional loudness presets, 0.1 s pause per line break, batch multi-file processing. See original sections below.

### 🌍 Per-Language Sample Text
Switching the language dropdown auto-fills a native sample sentence (24 languages, incl. Bengali) — only when the box is empty or still holds a previous sample; your own typed text is never overwritten. Applies to the TTS, single-voice, and multi-voice tabs.

### 🈂️ Per-Language Text Processing (en / bn / hi)
English, Bengali and Hindi each get their own text pipeline (`src/audiobook/langtext.py`); all other languages keep exact legacy behavior:
- **Sentence enders**: Bengali/Hindi split on `।` (dari) as well as `. ! ?` — previously multi-sentence dari text fused into single chunks
- **Abbreviation protection**: `Dr.` / `ড.` / `डॉ.` etc. never end a sentence mid-chunk (placeholder hidden before splitting, restored before TTS/metadata so it never leaks into audio)
- **Symbol words**: `% ₹ ৳ $ & #` spoken natively (`50%` → `50 শতাংশ` / `50 प्रतिशत` / `50 percent`; `৳`=টাকা kept distinct from `₹`=রুপি)
- **Newlines untouched**: pause calculation downstream is unaffected; `[Character]` tags pass through unmodified

### 🔬 Bangla TN Engine (deterministic, spec-driven)
Staged protection engine with per-call registry — tags → URLs → emails → phones → numbers/decimals/versions/IPs → list markers → acronyms → abbreviations (more-specific always wins), then segmentation with priority STRONG (`। ! ?`) > MEDIUM (`.`, context-checked) > never `: ; ,`:
- **Protected, never split**: decimals (`৩.১৪`), versions (`২.১.৩`, `v2.1`), IPs, domains (`example.com`), emails, phones, `১.` list markers, bare `।`-handling with closing quotes glued (`।”`)
- **Natural spoken mode** (opt-in checkbox on both audiobook tabs, default OFF = digits preserved): full Bengali number engine (০–৯৯, হাজার/লাখ/কোটি, both groupings), ordinals (`১ম`→`প্রথম`), date-forms, classifiers, fractions, ranges, percents, currency with paise, times/dates, phones (digit-spelled), acronyms (`API`→`এপিআই`), units — unknown forms left untouched, never hallucinated.
- **Natural clock words**: `১:৩০`→`দেড়টা`, `৫:৩০`→`সাড়ে পাঁচটা`, `৫:১৫`→`সোয়া পাঁচটা`, `৫:৪৫`→`পৌনে ছয়টা` (24h folds to 12h; `দেড়টা` reserved for true 1:30)
  - Preserve: `ড. রহমান ৫০ শতাংশ ছাড়ে ৩টি বই টাকা ১,০০০ দিয়ে কিনলেন।`
  - Natural: `ড. রহমান পঞ্চাশ শতাংশ ছাড়ে তিনটি বই এক হাজার টাকা দিয়ে কিনলেন।`
- **Locales**: `bn`, `bn-BD`, `bn-IN` all normalize correctly (previously `bn-BD` silently fell back to English rules + wrong model)
- **Grapheme-safe**: word-boundary cuts only; conjuncts/hasanta/matras/nukta verified intact; ZWJ/ZWNJ never stripped; BOM stripped; NFC canonicalized
- **Audit CLI**: `python3 src/audiobook/lang_audit.py book.txt --locale bn-BD [--natural]` dumps RAW→PROTECTED→SEGMENTED→RESTORED with timings, protected-token table, unknown-token report, leak asserts
- **Golden suite**: `python3 tests/test_langtext.py` — 49 headless tests (no model/GPU/network): golden examples, natural-mode cases, leak/loss/determinism guards

---

## 🎤 Voice Reference Rules (model limits — not configurable)

| Input | Limit | Why |
|---|---|---|
| TTS reference voice | **10 s max** (first 10 s used) | `DEC_COND_LEN = 10 × 24,000 Hz` embedding window (`mtl_tts.py`) |
| → of which for T3 prompt tokens | **6 s** | 150-token slot ÷ 25 tok/s (`t3_config.py`, `s3tokenizer.py`) |
| VC target voice | **10 s max** | Same embedding window (`vc.py`) |
| VC source audio | **No cap** (we chunk at 30 s) | Full file tokenized; cost scales linearly (~25 tok/s) |
| Book text | **No cap** | 50-word chunking + resume |

Longer reference files don't crash — but extra audio is ignored (TTS/VC-target) or just slows things down. One clean 10 s clip beats five noisy minutes: the speaker embedding averages the whole clip, so silence/noise dilutes it. "Native" long-audio support would only mean smarter *selection* of the best 10 s (VAD + loudness), never bigger model windows (those are baked into the trained weights).

---

## 🛠️ Technical Requirements
- **OS:** Linux (Mint/Ubuntu tested) or Windows · **Python 3.10+**
- **RAM:** 8 GB+ recommended. 7–8 GB works for TTS/audiobooks **with swap** (installer offers an 8 GB swapfile); the VC model loads **lazily on first Convert click** (not at startup) and all loaders are **singletons** (no double-load spikes) so both weight sets are never resident unless you use VC
- **CPU:** 4+ cores fine · **Disk:** ~6 GB free (venv + torch + models) · **No GPU needed**

---

## 🆘 Troubleshooting (every issue hit + fixed this session)

| Symptom | Cause | Fix |
|---|---|---|
| `ensurepip is not available` / venv creation fails | Mint/Ubuntu omit `python3-venv` | `sudo apt install -y python3.12-venv` (installer now auto-installs it) |
| Stuck at `>>>` prompt | Typed `python3.12` (opens REPL) | `exit()` or `Ctrl+D`, then run real commands |
| `TypeError: Invalid file: None` on Generate | Clicked Generate with no voice selected | **Fixed in code** — now a red popup asks for a voice/text first |
| Words missing from Bengali output | (a) single call capped ~40 s audio — long tails cut silently; (b) emoji/foreign chars → UNK skipped | **Fixed in code** — TTS tab pre-splits long text into sentence groups; startup UNK scan warns (`⚠️ ... may be skipped/garbled`) with the exact words |
| Process `Killed` at startup/right after VC load | OOM: TTS+VC weights resident together | **Fixed in code** — VC loads lazily on first use + singleton caches prevent double-loads; if still killed, add swap (below) |
| Process `Killed` during big generations | RAM spike (weights + buffers) | Close apps; add swap: `sudo fallocate -l 8G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile` (+ fstab entry for reboots) |
| App loads model twice (`Loading...` ×2, then `Killed`) | `demo.load` re-fires per browser client | **Fixed in code** — singleton caches return `♻️ Reusing already-loaded model` |
| Installer dies at `pip install gradio qrcode` with garbled error | Transient pip glitch (venv left without gradio) | Re-run `./venv/bin/pip install gradio qrcode pydantic`, then launch (or re-run installer) |
| `venv/` nearly empty (no gradio, ~30 packages) | Interrupted installer run (`rm -rf venv` first) | Re-run `./install-audiobook.sh` to completion |
| PKUSEG warning (Chinese segmentation) | Optional dep missing | Harmless warning, ignore |
| HF Hub `unauthenticated requests` warning | No token set | Harmless; rate limits only matter for huge downloads |
| `.sh` double-click opens editor (Sigma) | Sigma never passes filenames to handlers (verified) | Use Mint menu entry, or right-click → Open in Terminal |
| First Bengali use downloads ~2 GB slowly | Expected one-time `models-bangla/` fetch | Watch `[BN]` lines; afterwards instant via singleton |
| No completion bell sound | Terminal bell depends on your terminal's settings | GNOME Terminal: Preferences → Sound → enable terminal bell (or watch for `[WAVEFORM]` + `📊 [SESSION]` lines instead) |
| `Cangjie5_TC.json` re-downloads each launch | Tokenizer cache miss (~2 MB) | Harmless, ignore |

---

## 📖 Original Feature Guide (unchanged)

### 🎭 Custom Audiobook Processing Pipeline
**Return Pause System** — every line break (`\n`) adds a 0.1 s pause automatically (accumulative, works single/multi/batch):
```
[Narrator] The sun was setting over the hills.

[Character1] "We need to find shelter soon."

[Character2] "I see a cave up ahead.
Let's hurry before it gets dark."


[Narrator] They rushed toward the cave, hearts pounding.
```
**Formatting tips:** `[Name]` tags per speaker · double returns for scene changes · extra returns before reveals · single returns between speakers. Multi-voice: always use identical `[Name]` spelling.

### 🎚️ Volume Setup
Voice Library tab → upload sample → set target (−18 dB default) → save. Multi-voice: enable normalization once, all characters match. Presets: audiobook −18, podcast −16, broadcast −23 dB RMS.

### 🎯 Workflow
1. Prepare text (chapters, `[Character]` tags, line-break pauses) 2. Select/clone voices 3. Configure volume 4. Generate (watch live chunk progress + ETA) 5. Pause/Cancel freely — resume anytime 6. Collect per-chunk WAVs + full file from `audiobook_projects/<name>/`

### 📋 Formats
In: `.txt/.md`, voice samples `.wav/.mp3/.flac`. Out: 16-bit mono `.wav` chunks + stitched full audio.

### ⚠️ Known Limitations
- Very short multi-voice snippets can be unstable (model limitation — use fuller sentences)
- `gradio_tts_app_audiobook_with_batch.py` is an unused alternate variant with its own old bugs; launchers use `gradio_tts_app_audiobook.py`
- Invalid-text multi-voice chunks are skipped (recorded) rather than silence-filled

---

## 🗂️ Session Changelog (what was done)

**Linux port:** fixed `text_processing→processing` import + missing `numpy` (`project_management.py`); `chatterbox→src.chatterbox` fallback import (`models.py`); removed duplicate `import os`; 4 executable `.sh` launchers; `.desktop` launcher + Mint-menu entry; stripped 20+ `🔍 DEBUG` prints.

**`src/` bug fixes:** `vc.py` generation un-trapped from `else` + `assert→ValueError`; `\r\n` pause double-count fixed; `extract_audio_segment(sample_rate=…)` parameter; `last_updated` real timestamp; Gradio tuple-audio save support (`voice_management.py`); auto-save overwrite fixed via `start_index` + global counter (single+multi); `language_id` added to `models.generate_with_retry`; redundant `import re` ×6 removed. (Note: an early `cfg_weight>0` token-guard was reverted — T3 hard-indexes batch[1], so tokens are always duplicated.)

**Realtime VC upgrade** (`gradio_tts_app_audiobook.py`): `vc_convert_chunked()` generator — 30 s chunks, edge-fade stitching, retry, cancel (partial kept), per-chunk files to `vc_output/`, A/B players, progress+ETA+`ends ~HH:MM:SS`, 5 s terminal heartbeat, timing panel — original UI styling.

**Realtime TTS upgrade (both tabs):** cores → streaming generators (5-tuple yields); wrappers stream through (temp-voice cleanup + metadata fix-up preserved); batch caller drains; cancel + pause/resume buttons; live chunk text; skip-and-continue failures; heartbeat with stale-guard. Also fixed: resume-path 4→2 arity crash, multi-voice permanent 4→2 arity crash, TTS-tab `None` voice/text crash (`gr.Error` popups), startup OOM (lazy VC load), installer self-heal (`python3-venv`, disk check), `qrcode` dep.

**Docs/Q&A in-session:** model locations, multilingual-only scope, 10 s/6 s reference math (150 tokens ÷ 25 tok/s), VC length rules, phone LAN+QR+public links, Mint/Sigma double-click behavior (Sigma limitation verified by test), AI-component breakdown (T3/S3/S3Gen/VE — no chatbot LLM, all local).

**Deep audit (4 agents, ~9k lines):** fixed audiobook-killing `signal.alarm`-in-worker-threads bug (gated to main thread); `@torch.inference_mode()` on voice-setup paths (stops grad-graph RAM creep); `load_project_btn` + volume-component cross-tab overwrites fixed; `delete_voice` Dropdown crash fixed; duplicate refresh/download bindings removed; startup dropdown loads fixed; launch scripts `cd` to own dir; pyproject deps completed + `requires-python>=3.10`; heartbeat job-ids + progress guards. Rejected after verification: per-event queue starvation (each listener has its own queue), upstream model-math tinkering, dead-module rewrites.

**Terminal mega-upgrade (8 agents):** `python3 -u` everywhere; model weight per-file logs; T3 100-token milestones + tok/s; S3Gen flow/vocoder split; tokenizer + VE single-line timings; English-TTS conds mirror; multi-voice conds cache; `@logged_job` on batch/legacy/combine/regen/clean/analyze; voice-save + startup breadcrumbs; session odometer; ×realtime on every chunk.

**Bengali support (BosonLab analysis):** stock 23-language model lacks Bengali → integrated `BosonLab/chatterbox-bangla` (MIT, ~99 h, vocab 2530) as 24th language. `tts.py from_local` auto-sizes T3 to checkpoint vocab; new `src/audiobook/bangla.py` (`BanglaTTS` adapter + lazy singleton loader into `models-bangla/` + graceful fallback); `bn` routed in TTS/single/multi/regen/legacy paths; `punc_norm` accepts `।`; `bn` in all language dropdowns.

**On-demand models (user idea):** removed `demo.load` model preload — UI opens model-free, `load_model(language_id)` lazy-loads multilingual-or-Bangla on first use; all 6 `from_pretrained` fallbacks routed through the singleton (fixes double-load race where a click during preload loaded weights twice).

**Per-language sample texts:** `DEFAULT_TEXTS` map (all 24 languages) auto-fills TTS/single/multi textboxes on language switch — only when empty or holding a previous sample, never overwriting typed text.

**ETA + surprise round:** token-aware heartbeat ETA (shared `T3_PROGRESS`, dual module-tree safe after finding editable-install vs local-import dict split); slow-chunk watchdog; conds-cache hit/miss scoreboard; ASCII waveforms on all 6 finales; completion bell; `_copy_ref_audio` sizes; app-ready startup timer.

**Surprise round 2:** slowest-chunks leaderboard on every finale (top-5 with word counts — find problem text fast); pre-job disk-space guard (warns under 2 GB free before a mid-book write fails).

**Missing-words fix (user report):** TTS tab pre-splits long segments into sentence groups (single 1000-token call silently cut tails); `_find_unk_words()` pre-scans input against the resolved model's vocab and warns with exact words (proven: full Bengali coverage, only emoji UNKs); `_coverage_ok()` guard flags any output suspiciously short for its input (per-chunk + finales + VC in/out ratio) with suspect counts in UI status.

**Natural-mode wave 2:** year-aware verbalization (`১৯৮৭ সালে`→`উনিশশো সাতাশি সালে`, quantities untouched); rule-ID tracing (`BN_CURR_001` etc. shown in audit); spaced-initials protection (`M. A. Rahman`); control-char sanitizer; long-sentence advisory in audit; 53 golden tests green.

**Per-language pipelines (user idea):** new `src/audiobook/langtext.py` — en/bn/hi profiles (enders, abbreviations, symbol words); dari added to all sentence splitters; abbreviation protect/restore cycle in single + multi + legacy cores; verified by headless tests (dari splits 3-from-1, `Dr.` never splits, symbols expand, newlines preserved).

**Bangla TN engine (700-section spec, 1 explore agent):** staged per-call protection registry (tags→URLs→emails→phones→numbers→lists→acronyms→abbreviations, specific-wins, collision-safe, leak-asserted); priority segmentation (STRONG ।!?, context-checked MEDIUM `.`, never `:;,`); bare-domain protection; `₹`/`৳` distinction; `bn-BD`/`bn-IN` normalization (model routing + profiles); dari reattach + batch-dari fixes; script-aware short-chunk filter; `tests/test_langtext.py` (33 golden tests, all green); `src/audiobook/lang_audit.py` CLI. Tests caught 4 real bugs during development (ESC-restore assert, ellipsis shatter, bare domains, quote-gluing).

**Natural spoken Bengali (opt-in):** `src/audiobook/bn_numbers.py` (০–৯৯, হাজার/লাখ/কোটি, both digit scripts/groupings, ordinals, date-forms, fractions, ranges, percents, currency+paise incl. `৳(৫০০)`, times/dates, phones, acronyms, units, v-versions, ratios); UI checkbox on both audiobook tabs (default OFF = preserve); nested-quote merge; `…` boundary; letter/roman lists; hashtags/mentions preserved; ratio vs time disambiguation; leading-minus; grapheme utils; BOM/NFC; unknown-token report + `--natural` in audit CLI; 49 golden tests green.

**Multi-voice tag routing fix:** voice tags now split on RAW text before any protection (previously `[রহিম]` was hidden from the splitter → whole book collapsed to Narrator); `[১]`-style footnotes excluded via letter-allowlist in both parsers; currency amounts protected as one unit (no more `টাকা ১,০০০` word flip).

**Exclusive one-model mode (user idea):** EN↔BN switching evicts the inactive weights (`[MEM] Evicted …`, gc + malloc_trim, UI state synced — verified settable server-side); one model resident ever; alternating reloads per switch, sticking never reloads.

---

## 📄 License
Licensed under the terms in `LICENSE`. Underlying TTS: ResembleAI Chatterbox (multilingual V3).

**Ready? `./install-audiobook.sh` once, `./launch_audiobook.sh` forever (or Mint menu → Chatterbox).** 🎉
