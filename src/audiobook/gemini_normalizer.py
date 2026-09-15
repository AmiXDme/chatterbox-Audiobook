"""Optional Gemini-powered Bangla text normalization for the web UI.

The local tn engine was removed (text is used as typed); when enabled, raw
Bangla text is sent to a Gemini model armed with the project's normalization
prompt (default: the full "Bangla Audiobook Language Engine — Master Prompt"
ruleset in prompts/, or any custom .txt you point at) and the returned
spoken-Bangla text feeds Chatterbox.

Pure stdlib (urllib) — no new dependency. Every caller must use this through
the safe wrapper so a missing key / offline machine / API error ALWAYS falls
back to raw text — never a crash.

Long input is auto-chunked at sentence/paragraph boundaries (each chunk ≤
MAX_INPUT_CHARS) so whole audiobooks are normalized, not skipped.
"""

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

MODEL = "gemini-2.5-flash"
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
MAX_INPUT_CHARS = 30000

# Where the project keeps its canonical normalization prompt.
DEFAULT_PROMPT_FILE = Path(__file__).resolve().parent.parent.parent / "prompts" / "Bangla_Audiobook_Master_Language_Prompt_v3.txt"

# Compact fallback used only when no prompt file is readable (never crashes).
_FALLBACK_PROMPT = """You are a Bangla (Bengali) audiobook text normalizer. You convert "raw" Bangla text into spoken Bangla text so a text-to-speech voice reads it correctly.

OUTPUT FORMAT
- Output ONLY the normalized text. No notes, explanation, quotes, commentary, bullet lists.
- Preserve paragraph breaks, blank lines, and sentence delimiters exactly (। ! ? .) — keep every one.
- Preserve line breaks exactly.

NEVER CHANGE / PASS THROUGH VERBATIM
- Voice tags: [Name], [রহিম], {Name}, <anything> — keep brackets and content exactly.
- URLs, emails, @mentions, #hashtags. Acronyms you do not know — leave as written.
- Do NOT translate, paraphrase, reorder, or spell-correct anything.
- Keep ZWJ/ZWNJ and joined characters intact. Dotted version strings stay as-is.

NUMBERS -> BANGLA WORDS (canonical)
০=শূন্য ১=এক ২=দুই ৩=তিন ৪=চার ৫=পাঁচ ৬=ছয় ৭=সাত ৮=আট ৯=নয় ১০=দশ ১১=এগারো ১২=বারো ১৩=তেরো ১৪=চৌদ্দ ১৫=পনেরো ১৬=ষোল ১৭=সতেরো ১৮=আঠারো ১৯=উনিশ ২০=বিশ ২১=একুশ ২২=বাইশ ২৩=তেইশ ২৪=চব্বিশ ২৫=পঁচিশ ২৬=ছাব্বিশ ২৭=সাতাশ ২৮=আটাশ ২৯=ঊনত্রিশ ৩০=ত্রিশ ৩১=একত্রিশ ৩২=বত্রিশ ৩৩=তেত্রিশ ৩৪=চৌত্রিশ ৩৫=পঁয়ত্রিশ ৩৬=ছত্রিশ ৩৭=সাঁইত্রিশ ৩৮=আটত্রিশ ৩৯=ঊনচল্লিশ ৪০=চল্লিশ ৪১=একচল্লিশ ৪২=বিয়াল্লিশ ৪৩=তেতাল্লিশ ৪৪=চুয়াল্লিশ ৪৫=পঁয়তাল্লিশ ৪৬=ছেচল্লিশ ৪৭=সাতচল্লিশ ৪৮=আটচল্লিশ ৪৯=ঊনপঞ্চাশ ৫০=পঞ্চাশ ৫১=একান্ন ৫২=বাহান্ন ৫৩=তিপ্পান্ন ৫৪=চুয়ান্ন ৫৫=পঞ্চান্ন ৫৬=ছাপ্পান্ন ৫৭=সাতান্ন ৫৮=আটান্ন ৫৯=ঊনষাট ৬০=ষাট ৬১=একষট্টি ৬২=বাষট্টি ৬৩=তেষট্টি ৬৪=চৌষট্টি ৬৫=পঁয়ষট্টি ৬৬=ছেষট্টি ৬৭=সাতষট্টি ৬৮=আটষট্টি ৬৯=ঊনসত্তর ৭০=সত্তর ৭১=একাত্তর ৭২=বাহাত্তর ৭৩=তিয়াত্তর ৭৪=চুয়াত্তর ৭৫=পঁচাত্তর ৭৬=ছিয়াত্তর ৭৭=সাতাত্তর ৭৮=আটাত্তর ৭৯=ঊনআশি ৮০=আশি ৮১=একাশি ৮২=বিরাশি ৮৩=তিরাশি ৮৪=চুরাশি ৮৫=পঁচাশি ৮৬=ছিয়াশি ৮৭=সাতাশি ৮৮=আটাশি ৮৯=ঊননব্বই ৯০=নব্বই ৯১=একানব্বই ৯২=বিরানব্বই ৯৩=তিরানব্বই ৯৪=চুরানব্বই ৯৫=পঁচানব্বই ৯৬=ছিয়ানব্বই ৯৭=সাতানব্বই ৯৮=আটানব্বই ৯৯=নিরানব্বই
SCALES: 100=একশ 200=দুইশ 1,000=হাজার 1,00,000=লাখ 1,00,00,000=কোটি. Negative: -৫ = মাইনাস পাঁচ.
PERCENT: ৫০% -> পঞ্চাশ শতাংশ. DECIMAL: ৩.১৪ -> তিন দশমিক এক চার; money: ৳৩.১৪ -> তিন টাকা চৌদ্দ পয়সা.
CURRENCY: ৳/Tk -> টাকা..পয়সা ₹->রুপি $->ডলার..সেন্ট €->ইউরো £->পাউন্ড ¥->ইয়েন.
CLOCK (:00=টা :15=সোয়া :30=সাড়ে :45=পৌনে; ১:৩০=দেড়টা; else HHটা MM). 24h folds to 12h.
DATES: ১২/০৯/২০২৬ -> বারো সেপ্টেম্বর দুই হাজার ছাব্বিশ. Date ordinals ১লা->পহেলা ২রা->দোসরা ৩রা->তেসরা ৪ঠা->চৌঠা ৫ই->পাঁচই ৬ই->ছয়ই ৭ই->সাতই ৮ই->আটই ৯ই->নয়ই ১০ই->দশই.
YEARS (1100-2099 near সালে/সাল/খ্রিস্টাব্দে): ১৯৮৭ -> উনিশশো সাতাশি; ২০২৬ -> দুই হাজার ছাব্বিশ. Never years for "মডেল ২০২৬".
ORDINALS: ১ম->প্রথম ২য়->দ্বিতীয় ৩য়->তৃতীয় ৪র্থ->চতুর্থ ৫ম->পঞ্চম ৬ষ্ঠ->ষষ্ঠ ৭ম->সপ্তম ৮ম->অষ্টম ৯ম->নবম ১০ম->দশম.
CLASSIFIERS: ৫টা->পাঁচটা ৪জন->চারজন ৩টি->তিনটি.
UNITS: km->কিলোমিটার cm->সেন্টিমিটার kg->কিলোগ্রাম g->গ্রাম L->লিটার s->সেকেন্ড min->মিনিট h->ঘণ্টা KB/MB/GB/TB килো/মেগা/গিগা/টেরাবাইট.
ACRONYMS: API->এপিআই GPU->জিপিইউ CPU->সিপিইউ AI->এআই TTS->টি টি এস NID->এনআইডি USB->ইউএসবি PDF->পিডিএফ TV->টিভি BBC->বিবিসি MBBS->এমবিবিএস MD->এমডি RAM->র্যাম.
PHONES: digit by digit. ০১৭১২৩৪৫৬৭৮ -> শূন্য এক সাত এক দুই তিন চার পাঁচ ছয় সাত আট.

VERIFY: never invent forms. If unsure, LEAVE THE TOKEN EXACTLY AS WRITTEN."""

_ACTIVE_PROMPT_FILE = {"path": ""}
_prompt_cache = {"mtime": 0.0, "text": None}


def api_key_ok(api_key) -> bool:
    return bool(api_key and str(api_key).strip().startswith("AIza"))


def set_prompt_file(path=None) -> None:
    """Set a custom prompt .txt path ('' or None restores the project default)."""
    _ACTIVE_PROMPT_FILE["path"] = (path or "").strip()
    _prompt_cache["mtime"] = 0.0  # force re-read


def get_system_prompt() -> str:
    """Return the active normalization prompt, re-reading the file if changed.

    Priority: custom path (if set) -> project default file -> built-in fallback.
    Never raises when a file is unreadable — falls back to the last good prompt.
    """
    candidates = [_ACTIVE_PROMPT_FILE.get("path") or None, str(DEFAULT_PROMPT_FILE)]
    for fp in candidates:
        if not fp:
            continue
        try:
            p = Path(fp)
            if not p.exists() or not p.is_file():
                continue
            mtime = p.stat().st_mtime
            if _prompt_cache["text"] is not None and _prompt_cache["mtime"] == mtime and str(p) == _ACTIVE_PROMPT_FILE.get("path"):
                return _prompt_cache["text"]
            text = p.read_text(encoding="utf-8").strip()
            if not text:
                continue
            _prompt_cache.update({"mtime": mtime, "text": text})
            return text
        except Exception as e:
            print(f"[GEMINI] Prompt file unreadable ({fp}: {e}) — using fallback", flush=True)
            continue
    return _FALLBACK_PROMPT


def _chunk_text(text: str, max_chars: int = MAX_INPUT_CHARS):
    """Split long text at sentence/paragraph boundaries so any book fits.

    Splits after । ! ? . and newlines (delimiters kept). A single over-long
    "sentence" is hard-split as a last resort. Always returns >= 1 chunk.
    """
    if not text:
        return [""]
    if len(text) <= max_chars:
        return [text]
    sentences = re.split(r"(?<=[।!?.\n])", text)
    chunks, cur = [], ""
    for s in sentences:
        if len(cur) + len(s) <= max_chars:
            cur += s
        else:
            if cur:
                chunks.append(cur)
            cur = s
            while len(cur) > max_chars:
                chunks.append(cur[:max_chars])
                cur = cur[max_chars:]
    if cur:
        chunks.append(cur)
    return chunks or [text]


def _call_gemini(prompt: str, api_key: str, timeout: float = 90.0) -> str:
    body = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": get_system_prompt() + "\n\n" + prompt}],
            }
        ],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 8192,
        },
    }
    req = urllib.request.Request(
        ENDPOINT.format(model=MODEL, key=api_key),
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    last_err = None
    for attempt in range(2):  # one automatic retry on transient failure
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", "replace")[:300]
            except Exception:
                pass
            last_err = RuntimeError(f"Gemini API HTTP {e.code}: {detail}")
        except urllib.error.URLError as e:
            last_err = RuntimeError(f"Gemini network error: {e.reason}")
    else:
        raise last_err

    candidates = data.get("candidates") or []
    if not candidates:
        raise RuntimeError(f"Gemini returned no candidates: {str(data)[:200]}")
    parts = candidates[0].get("content", {}).get("parts") or []
    if not parts or not parts[0].get("text"):
        raise RuntimeError("Gemini returned empty text")
    return parts[0]["text"]


def gemini_normalize(raw_text: str, api_key: str, model: str = MODEL, timeout: float = 90.0) -> str:
    """Send raw Bangla text to Gemini with the active ruleset; return spoken text.

    Long input is chunked at sentence boundaries and each chunk normalized, so
    whole audiobooks work. Preserves [Name] tags/URLs so multi-voice routing
    still works downstream. Raises RuntimeError on any failure — the caller
    decides the fallback.
    """
    if not api_key_ok(api_key):
        raise RuntimeError("No valid Gemini API key (must start with 'AIza')")
    if not raw_text or not raw_text.strip():
        raise RuntimeError("Empty text")
    chunks = _chunk_text(raw_text, MAX_INPUT_CHARS)
    if len(chunks) <= 1:
        result = _call_gemini(raw_text, api_key, timeout=timeout).strip()
        if not result:
            raise RuntimeError("Gemini returned empty text")
        return result
    out = []
    total = len(chunks)
    for i, chunk in enumerate(chunks, 1):
        result = _call_gemini(chunk, api_key, timeout=timeout).strip()
        if not result:
            raise RuntimeError(f"Gemini returned empty text for chunk {i}/{total}")
        print(f"[GEMINI] chunk {i}/{total} → {len(result)} chars", flush=True)
        out.append(result)
    return "".join(out)