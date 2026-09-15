# Developer Handover — Chatterbox Audiobook Generator

## 1. What this thing is
A CPU-only **text-to-speech audiobook studio** (Gradio web UI) built on ResembleAI Chatterbox neural models, plus a Bengali fine-tune. Paste text → get narrated audiobooks with cloned voices, 24 languages, per-chunk resume, live progress. No GPU, no cloud, no API keys required.

## 2. Tech stack
- **Language:** Python 3.10+ (3.12 on this machine), `venv/` isolated env
- **UI:** Gradio 6.x (`gradio_tts_app_audiobook.py`, ~8,700 lines, single monolith)
- **TTS models:** Chatterbox Multilingual V3 (23 langs, `models-multilingual/`) + ONE Bengali fine-tune — BosonLab (`models-bangla/`, vocab 2530, dropdown `bn`) + optional Voice Conversion (`models/`)
- **ML:** PyTorch CPU, safetensors checkpoints, librosa/soundfile/scipy audio I/O
- **Launchers:** `install-audiobook.sh`, `launch_audiobook.sh`, `launch_local.sh`, `launch_network.sh`

## 3. Architecture map (key files)
| File | Owns |
|---|---|
| `gradio_tts_app_audiobook.py` | EVERYTHING UI + orchestration: tab layouts, all event wiring, single/multi/batch/regen/combine flows, realtime engine (heartbeat, cancel/pause, ETA), model singletons, LAN/QR startup |
| `src/audiobook/bangla.py` | Bengali support: `BanglaTTS` adapter, singleton loader (bosonlab), language router, evictor |
| `src/audiobook/gemini_normalizer.py` | Optional Gemini Bangla normalizer: reads the prompt from `prompts/Bangla_Audiobook_Master_Language_Prompt_v3.txt` (or any custom path), auto-chunks long input ≤30k chars, temp 0.0, safe fallback; `set_prompt_file()`/`get_system_prompt()` swap prompts live |
| `prompts/` | Canonical normalization prompt files (git-tracked, editable without code changes) |
| `src/audiobook/processing.py` | Text chunking (sentence/pause/line-break aware), audio save helpers |
| `src/audiobook/{models,project_management,voice_management,config,audio_processing}.py` | Refactor-library modules (mostly **unused** by the running app — see §6) |
| `src/chatterbox/mtl_tts.py` | Multilingual model class (23 langs + `bn` label) |
| `src/chatterbox/tts.py` | English-class model (base for `BanglaTTS`) |
| `src/chatterbox/vc.py` | Voice conversion model |
| `src/chatterbox/models/` | Vendored neural nets (T3 transformer, S3Gen, tokenizers, voice encoder) — **do not refactor internals** |

## 4. Core concepts (glossary)
- **Chunk** — ~50-word text unit; the atomic unit of generation, saving, resume, retry
- **Conds (conditionals)** — voice embedding computed once from a ~10 s reference clip, reused per chunk (cached per voice in multi-voice)
- **Singleton loaders** — `load_model()` / `load_bangla_model()` / `load_vc_model()` load weights exactly once (double-checked locking); `♻️ Reusing` means cache hit
- **Streaming generators** — long jobs `yield (audio, status, timing, chunktext, pausebtn)` per chunk so Gradio UI updates live
- **Heartbeat** — background thread printing a terminal tick every 5 s (progress, ETA, tok/s, RAM); driven by `TTS_LIVE` / `VC_LIVE` dicts
- **`language_id`** — code (`en`, `bn`, …); `bn` routes to the BosonLab Bangla singleton, everything else to multilingual. One-at-a-time is a 2-way rule (multilingual | bn).
- **Resume** — completed chunk WAVs on disk are skipped on re-run; failed chunks are recorded and skipped, never fatal

## 5. Critical invariants (break these and the app breaks)
1. **Yield arity**: streaming handlers must yield exactly the tuple size wired in their `.click(outputs=[...])` (5 for audiobook/VC flows). Bare `return <value>` inside a generator never reaches the UI.
2. **One model at a time**: never preload both weight sets; evict-then-load on language switch (`[MEM] Evicted …`).
3. **Module-tree rule**: the package exists as `src.chatterbox.*` AND `chatterbox.*` (editable install). Shared mutable state must handle both (see `T3_PROGRESS` dual-read); prefer `src.*` imports in new code.
4. **No `signal.alarm` in workers**: Gradio runs handlers in threads — alarms only on main thread (`_USE_ALARM` gate).
5. **`language_id` threading**: every generation entry point must resolve/forward it, or `bn` silently falls back to multilingual.
6. **Bangla text is used as typed**: no local normalization layer exists — `bn` text goes to the fine-tune verbatim. Normalize numbers/currency/years to spoken Bangla words before generation (typed manually or via an online AI), or the model voices them poorly.
7. **No blocking calls in handlers** except through the cancel/pause-gated loops.

## 6. Known dead code (leave alone unless reviving)
- `gradio_tts_app_audiobook_with_batch.py` — old alternate app, has its own bugs, nothing launches it
- `src/audiobook/{models,project_management}.py` — parallel refactor, not imported by the app (two one-line safety fixes applied, that's all)
- `create_multi_voice_audiobook()` (non-`_with_assignments`), `load_model_cpu()`, legacy `vc_convert()` — uncalled
- `launchers/*.py` — superseded by root `.sh` scripts (kept for reference)

## 7. Run / develop / test
```bash
./install-audiobook.sh   # venv + deps + models + swap prompt + launcher generation
./launch_audiobook.sh    # app on :7860 + LAN URL/QR
python3 -m py_compile gradio_tts_app_audiobook.py   # syntax gate (no test suite exists)
```
Headless checks need no GPU: tokenizer vocab assertions, safetensors header reads. Full generation test = one short sentence per touched path (minutes on CPU).

## 8. Deliberate limitations
- No auth on the Gradio server (LAN + `share=False` by default; do not expose publicly as-is)
- No automated test suite; verification is compile + targeted runtime checks
- Number-to-words expansion not implemented (digits pass through to TTS)
- Per-language pipelines cover en/bn/hi; other 21 languages use English rules
- Vendored model math (T3/S3Gen internals) intentionally untouched except logging
