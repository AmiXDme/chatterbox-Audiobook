# Pause Feature Documentation

## Overview

Chatterbox TTS Audiobook Edition ships with a **dual-engine pause system**:
one for English/other-languages, and a fully **Bengali-grammar-aware pause engine**
for Bangla `বাংলা` text. Pauses are inserted automatically — you never place a
silence tag by hand; the system reads **boundary cues** (line breaks, sentence
enders, paragraph dandas) and writes the silence itself.

| Engine | Trigger | Pause |
|---|---|---|
| English + others | every line break `\n` / `\r\n` | **0.1 s per return**, accumulates (10 returns = 1 s) |
| **Bangla** | `।` `?` `!` sentence ender | **0.6 s** |
| **Bangla** | `॥` (double danda / stanza end) | **1.2 s** |
| **Bangla** | single return `\n` (soft breath / line wrap) | **0.15 s** |
| **Bangla** | blank line `\n\n` (paragraph break) | **1.0 s** |
| **Bangla** | runs of 2+ returns | collapsed to one `\n\n` → always 1.0 s (never stacked) |

## Why Bangla is different (Bengali grammar)

Bengali prose marks sentence ends with the danda `।`, stanza/verse ends with
`॥`, and paragraphs with blank lines. A flat "0.1 s per line break" rule
sounds robotic for Bangla because line wraps inside a paragraph are **breaths**,
not full stops. The Bangla engine therefore treats formatting **grammatically**:

- `।` is a sentence boundary — a full **0.6 s thought pause**.
- `॥` closes a verse/stanza — the longest break, **1.2 s**.
- a single `\n` is usually a poet's line-wrap or a forced soft break — just a
  **0.15 s breath**.
- a blank line `\n\n` is a paragraph — **1.0 s**. Any run of 2+ returns is
  normalized to exactly one paragraph pause, so you never get accidental
  5-second dead air.

Punctuation is **kept** on the chunk text when it reaches the model, so the
synthesizer also sees the intonation cue — the pause and the prosody work
together, but independently.

## How It Works

### English / other languages (single, multi, batch)

1. The text processor counts every return: `process_text_for_pauses(text, 0.1)`.
2. Audio is combined with `create_silence_audio(total_pause, sample_rate)`.
3. Terminal shows the tally:

```
🔇 Detected 4 line breaks → 0.4s total pause time
🔇 Adding 0.4s pause (4 returns × 0.1s each)
```

### Bangla `বাংলা` (quick TTS, single audiobook, multi-voice, metadata samples)

1. **Repair + normalize** — `unicode_repair_bangla()` fixes broken grapheme
   clusters (যুক্তাক্ষর), then `bangla_normalize_text()` rewrites every
   digit shape to Bengali words *before* any language model sees them:
   `৳৩.১৪ → তিন টাকা চৌদ্দ পয়সা`, `১৩:৩০-এ → দেড়টায়`,
   `১৪৩৩ বঙ্গাব্দ → চৌদ্দশো তেত্রিশ বঙ্গাব্দ`.
2. **Cue extraction** — `bangla_chunk_text()` (max_tokens=500 quick / 850
   audiobook) walks paragraphs → lines → sentences and tags each boundary with
   its cue and converts the cue with `bangla_pause_duration(cue)`.
3. **Pause assignment** — each chunk carries `pause_before`; a cue with an
   empty buffer *adds onto the previous chunk* (stacking blank lines into one
   longer space), and a chunk ending in `।/॥` gets the fallback tail pause.
4. **Silence synthesis** — `create_silence_audio(pause, sr)` is emitted at each
   boundary. Terminal shows the boundary trace:

```
🔇 Adding 0.60s pause (boundary cue)
🔇 Total pause time distributed: 1.75s
```

Single/multi audiobook store the same pauses as per-chunk `pause_duration` and
print per-chunk lines like:

```
🔇 Chunk 5: Added 0.6s pause after speech
🔇 Chunk 5 (কথক): Added 1.0s pause after speech
```

Multi-voice Bangla keeps every character's pauses **inside that voice's own
audio**: `🔇 Line breaks detected in [কথক]: +0.15s pause (from 1 returns)`.

## Example

**Bengali input (`বাংলা`):**
```
বৃষ্টি পড়ল সারা রাত। সকালে সবুজ পাতা ঝলমল করছিল।?

পরের গ্রামে গিয়ে দেখি সেখানে মেলা বসেছে।
একটা দোকানে পিঠা বিক্রি হচ্ছে।॥
```

**Parsed cues → pauses:**

| Unit | Cue → Pause |
|---|---|
| `বৃষ্টি পড়ল সারা রাত।` | sentence end `।` → 0.6 s |
| `সকালে সবুজ পাতা ঝলমল করছিল।?` | `।?` → 0.6 s |
| blank line | paragraph `\n\n` → 1.0 s |
| `পরের গ্রামে গিয়ে দেখি সেখানে মেলা বসেছে।` | `।` → 0.6 s |
| `একটা দোকানে পিঠা বিক্রি হচ্ছে।॥` | verse end `॥` → 1.2 s |

**English input:** 4 line breaks → 0.4 s total (accumulative).

## Features Supported

### ✅ Quick Speech-to-Text (TTS tab)
- English: trailing pauses per return group.
- Bangla: full cue-aware breaks (`। 0.6 s`, `॥ 1.2 s`, `\n 0.15 s`, `\n\n 1.0 s`).

### ✅ Single-Voice Audiobook
- Bangla text normalized (grammar + numbers) before chunking; `pause_duration`
  stored per chunk and stitched after each chunk's WAV.

### ✅ Multi-Voice Audiobook
- `[Character]` / `[Name]` tags split first; each voice's block is chunked and
  paused independently, so the pause belongs to that speaker.

### ✅ Batch & Regeneration
- Same pause engine runs per file and per regenerated chunk.

## Technical Implementation

### Functions (Bangla engine — `src/audiobook/processing.py`)
- `BANGLA_PAUSE_SECONDS` — the cue→seconds table (the single place to tune).
- `bangla_pause_duration(cue)` — lookup, unknown cue → 0.
- `bangla_chunk_text(text, max_tokens, max_clusters)` — grammar-aware
  paragraphs→lines→sentences chunker that returns `{text, pause_before}` plans
  and keeps punctuation for prosody.
- `bangla_normalize_text(text)` — stage-1 deterministic rewrite (plus
  `unicode_repair_bangla()`); runs BEFORE Gemini so the model sees letters only.
- `process_voice_content_with_line_breaks(...)` — multi-voice Bangla branch.

### Functions (English engine)
- `process_text_for_pauses(text, 0.1)` — counts returns.
- `create_silence_audio(seconds, sample_rate)` — silence synthesis.
- `insert_pauses_between_chunks(...)` / `process_text_with_distributed_pauses(...)`
  — English audiobook combination.

### Files
- `src/audiobook/processing.py` — both engines + Bangla grammar rules.
- `gradio_tts_app_audiobook.py` — quick `/single/multi/batch` integration.
- `prompts/Bangla_Audiobook_Master_Language_Prompt_v4.txt` — Gemini rewrite rules.

### Compatibility
- ✅ Windows, macOS, Linux · ✅ CPU mode · ✅ All audio formats
- ✅ Existing voice profiles and projects · ✅ Batch workflows
- ✅ Mixed Bangla/English documents (each engine picks its own rules per text)

## Usage Guidelines

### Best Practices (Bangla)
1. End sentences with `।` (or `?`/`!`) — you get a natural 0.6 s thought break.
2. Leave **one blank line** between paragraphs — a deliberate 1.0 s pause.
3. Use a single `\n` only as a soft breath (e.g. poem lines).
4. `॥` for verse or scene coda — the longest pause.
5. Write numbers freely (digits or Bengali digits) — they become Bengali words
   automatically; dates, times, currency, percentages and ordinals included.
6. Multi-voice: `[Character]` tags first, identical spelling throughout.

### Best Practices (English/other)
- Line breaks are the pause control; double returns = longer pauses.

## Configuration

- **English pause**: `0.1` s per return — hardcoded in `process_text_for_pauses`.
- **Bangla pauses**: the `BANGLA_PAUSE_SECONDS` table in
  `src/audiobook/processing.py` — adjust `।`,`॥`,`\n`,`\n\n` values in one spot.
- **Sample rate**: 24,000 Hz default; auto-matches model output.

## Troubleshooting

**No pauses heard:** confirm your text actually has the cues (`\n` / blank line /
`।`). Bangla text with no sentence-enders and no returns has no boundaries —
add `।`.

**Bangla paragraphs gap too long/short:** tune `BANGLA_PAUSE_SECONDS['\n\n']`
(1.0 s default).

**English returns sound broken up:** for Bangla, single-line wraps are 0.15 s
breaths by grammar design — use `।` for sentence pauses or a blank line for a
paragraph.

**Multi-voice pauses in wrong voice:** each voice's block is chunked after tag
splitting — keep dialogue inside `[Name]` blocks.

## Future Enhancements
- User-configurable pause table in the UI.
- Comma-level micro-pauses (`,` → short breath) — not yet applied.
- Visual pause timeline in the UI.
- Pause preview before generation (see the Preview → Verify → Generate loop in
  the TTS tab).
- Sentence context (prosody) carry-over per language.

---

**Note**: The system is automatic — no configuration needed. Write naturally
(Bangla: `।` + blank lines; English: line breaks) and the pause system handles
the rhythm.