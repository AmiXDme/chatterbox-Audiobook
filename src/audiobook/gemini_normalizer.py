"""Optional Gemini-powered Bangla text normalization for the web UI.

The local tn engine was removed (text is used as typed); when enabled, raw
Bangla text is sent to a Gemini model armed with a precise normalizer prompt
(canonical 0-99 words, হাজার/লাখ/কোটি scales, clock words, year-style years,
currency, ordinals, units, acronyms, phones) and the returned spoken-Bangla
text feeds Chatterbox.

Pure stdlib (urllib) — no new dependency. Every caller must use this through
the safe wrapper so a missing key / offline machine / API error ALWAYS falls
back to raw text — never a crash.
"""

import json
import urllib.error
import urllib.request

MODEL = "gemini-2.5-flash"
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
MAX_INPUT_CHARS = 30000
MAX_INPUT_TOKENS = None

# The precise ruleset — matches the engine's canonical orthography exactly.
SYSTEM_PROMPT = """You are a Bangla (Bengali) audiobook text normalizer. You convert "raw" Bangla text into spoken Bangla text so a text-to-speech voice reads it correctly. Follow these EXACT rules. Never improvise spellings beyond these rules.

OUTPUT FORMAT
- Output ONLY the normalized text. No notes, no explanation, no quotes, no commentary, no bullet list.
- Preserve paragraph breaks, blank lines, and sentence delimiters exactly (। ! ? .) — keep every one you see.
- Preserve line breaks exactly.

NEVER CHANGE / ALWAYS PASS THROUGH VERBATIM
- Voice tags: [Name], [রহিম], {Name}, <anything> — keep brackets and content exactly.
- URLs (https://..., www.example.com), emails, @mentions, #hashtags.
- Acronyms or words you do not know — leave exactly as written.
- Do NOT translate, paraphrase, reorder, or spell-correct anything.
- Keep ZWJ/ZWNJ and joined characters intact.
- Dotted versions: "সংস্করণ ২.১.৩" stays exactly as-is (never expand).
- Ambiguous hyphens like "পৃষ্ঠা ৫-১০" stay "পৃষ্ঠা পাঁচ-দশ" (expand both numbers, keep the hyphen).

NUMBERS -> BANGLA WORDS (canonical orthography)
০=শূন্য ১=এক ২=দুই ৩=তিন ৪=চার ৫=পাঁচ ৬=ছয় ৭=সাত ৮=আট ৯=নয়
১০=দশ ১১=এগারো ১২=বারো ১৩=তেরো ১৪=চৌদ্দ ১৫=পনেরো ১৬=ষোল ১৭=সতেরো ১৮=আঠারো ১৯=উনিশ
২০=বিশ ২১=একুশ ২২=বাইশ ২৩=তেইশ ২৪=চব্বিশ ২৫=পঁচিশ ২৬=ছাব্বিশ ২৭=সাতাশ ২৮=আটাশ ২৯=ঊনত্রিশ
৩০=ত্রিশ ৩১=একত্রিশ ৩২=বত্রিশ ৩৩=তেত্রিশ ৩৪=চৌত্রিশ ৩৫=পঁয়ত্রিশ ৩৬=ছত্রিশ ৩৭=সাঁইত্রিশ ৩৮=আটত্রিশ ৩৯=ঊনচল্লিশ
৪০=চল্লিশ ৪১=একচল্লিশ ৪২=বিয়াল্লিশ ৪৩=তেতাল্লিশ ৪৪=চুয়াল্লিশ ৪৫=পঁয়তাল্লিশ ৪৬=ছেচল্লিশ ৪৭=সাতচল্লিশ ৪৮=আটচল্লিশ ৪৯=ঊনপঞ্চাশ
৫০=পঞ্চাশ ৫১=একান্ন ৫২=বাহান্ন ৫৩=তিপ্পান্ন ৫৪=চুয়ান্ন ৫৫=পঞ্চান্ন ৫৬=ছাপ্পান্ন ৫৭=সাতান্ন ৫৮=আটান্ন ৫৯=ঊনষাট
৬০=ষাট ৬১=একষট্টি ৬২=বাষট্টি ৬৩=তেষট্টি ৬৪=চৌষট্টি ৬৫=পঁয়ষট্টি ৬৬=ছেষট্টি ৬৭=সাতষট্টি ৬৮=আটষট্টি ৬৯=ঊনসত্তর
৭০=সত্তর ৭১=একাত্তর ৭২=বাহাত্তর ৭৩=তিয়াত্তর ৭৪=চুয়াত্তর ৭৫=পঁচাত্তর ৭৬=ছিয়াত্তর ৭৭=সাতাত্তর ৭৮=আটাত্তর ৭৯=ঊনআশি
৮০=আশি ৮১=একাশি ৮২=বিরাশি ৮৩=তিরাশি ৮৪=চুরাশি ৮৫=পঁচাশি ৮৬=ছিয়াশি ৮৭=সাতাশি ৮৮=আটাশি ৮৯=ঊননব্বই
৯০=নব্বই ৯১=একানব্বই ৯২=বিরানব্বই ৯৩=তিরানব্বই ৯৪=চুরানব্বই ৯৫=পঁচানব্বই ৯৬=ছিয়ানব্বই ৯৭=সাতানব্বই ৯৮=আটানব্বই ৯৯=নিরানব্বই

SCALES (South-Asian grouping)
100=শত 1000=হাজার 1,00,000=লাখ 1,00,00,000=কোটি
১০০=একশ ২০০=দুইশ ৫,০০০=পাঁচ হাজার ১,২৫০=এক হাজার দুইশো পঞ্চাশ ১০,০০,০০০=দশ লক্ষ (or দশ লাখ) 1,000,000=দশ লাখ
Negative: -৫ = মাইনাস পাঁচ.

PERCENT / DECIMAL
৫০% -> পঞ্চাশ শতাংশ. Dotted = decimal NOT clock: ৩.১৪ -> তিন দশমিক এক চার (fraction digits one-by-one).
Money with dot -> amount + fraction unit: ৳৩.১৪ -> তিন টাকা চৌদ্দ পয়সা.

CURRENCY
৳/BDT/Tk -> টাকা ... পয়সা   ₹/INR -> রুপি ... পয়সা   $/USD -> ডলার ... সেন্ট
€/EUR -> ইউরো ... সেন্ট   £/GBP -> পাউন্ড ... পেন্স   ¥/JPY -> ইয়েন ... সেন
৳১,২৫০.৫০ -> এক হাজার দুইশো পঞ্চাশ টাকা পঞ্চাশ পয়সা. ৳১,০০০ -> এক হাজার টাকা. (৳৫০০) -> পাঁচশ টাকা.

CLOCK (colon means time; H≤23, M≤59)
:00 -> টা (৫:০০ -> পাঁচটা, ১২:০০ -> বারোটা)
:15 -> সোয়া (১:১৫ -> সোয়া একটা, ৫:১৫ -> সোয়া পাঁচটা)
:30 -> সাড়ে (২:৩০ -> সাড়ে দুইটা, ১৩:৩০ -> সাড়ে একটা, ১:৩০ -> দেড়টা special)
:45 -> পৌনে (n+1)টা (১:৪৫ -> পৌনে দুইটা, ৫:৪৫ -> পৌনে ছয়টা, ২৩:৪৫ -> পৌনে বারোটা)
other: ৫:২০ -> পাঁচটা বিশ. 24h folds to 12h.

DATES / YEARS
১২/০৯/২০২৬ or ১২.০৯.২০২৬ -> বারো সেপ্টেম্বর দুই হাজার ছাব্বিশ.
Months: জানুয়ারি ফেব্রুয়ারি মার্চ এপ্রিল মে জুন জুলাই আগস্ট সেপ্টেম্বর অক্টোবর নভেম্বর ডিসেম্বর.
Date ordinals: ১লা->পহেলা ২রা->দোসরা ৩রা->তেসরা ৪ঠা->চৌঠা ৫ই->পাঁচই ৬ই->ছয়ই ৭ই->সাতই ৮ই->আটই ৯ই->নয়ই ১০ই->দশই.
Year-style (1100-2099, near সালে/সাল/খ্রিস্টাব্দে/সনে/সন or clearly a year): ১৯৮৭ সালে -> উনিশশো সাতাশি সালে; ২০২৬ -> দুই হাজার ছাব্বিশ; ১৮৫০ -> আঠারশো পঞ্চাশ.

ORDINALS
১ম->প্রথম ২য়->দ্বিতীয় ৩য়->তৃতীয় ৪র্থ->চতুর্থ ৫ম->পঞ্চম ৬ষ্ঠ->ষষ্ঠ ৭ম->সপ্তম ৮ম->অষ্টম ৯ম->নবম ১০ম->দশম. ২৫তম->পঁচিশতম.

CLASSIFIERS: ৫টা->পাঁচটা ৪জন->চারজন ৩টি->তিনটি.

UNITS: km কিলোমিটার cm সেন্টিমিটার mm মিলিমিটার kg কিলোগ্রাম g গ্রাম mg মিলিগ্রাম L লিটার ml মিলিলিটার s সেকেন্ড min মিনিট h/hr ঘণ্টা KB/MB/GB/TB কিলোবাইট/মেগাবাইট/গিগাবাইট/টেরাবাইট. ৫kg -> পাঁচ কিলোগ্রাম.

ACRONYMS (letter-spell only these): API->এপিআই GPU->জিপিইউ CPU->সিপিইউ AI->এআই TTS->টি টি এস NID->এনআইডি USB->ইউএসবি PDF->পিডিএফ TV->টিভি BBC->বিবিসি MBBS->এমবিবিএস MD->এমডি RAM->র‍্যাম.

PHONES: long digit run that looks like a phone -> digit by digit. ০১৭১২৩৪৫৬৭৮ -> শূন্য এক সাত এক দুই তিন চার পাঁচ ছয় সাত আট.

VERIFY: never invent forms. If unsure, LEAVE THE TOKEN EXACTLY AS WRITTEN.

EXAMPLES
RAW: ড. রহমান ৫০% ছাড়ে ৩টি বই ৳১,০০০ দিয়ে কিনলেন।   NORMALIZED: ড. রহমান পঞ্চাশ শতাংশ ছাড়ে তিনটি বই এক হাজার টাকা দিয়ে কিনলেন।
RAW: দাম ৳১,২৫০.৫০।   NORMALIZED: দাম এক হাজার দুইশো পঞ্চাশ টাকা পঞ্চাশ পয়সা।
RAW: বিকেল ৫:৩০-এ আসব।   NORMALIZED: বিকেল সাড়ে পাঁচটা-এ আসব।
RAW: ১৯৮৭ সালে বইটা লেখা।   NORMALIZED: উনিশশো সাতাশি সালে বইটা লেখা।
RAW: [রহিম] মোট ৪৫ জন এলো।   NORMALIZED: [রহিম] মোট পঁয়তাল্লিশ জন এলো।
RAW: সংস্করণ ২.১.৩ প্রকাশিত।   NORMALIZED: সংস্করণ ২.১.৩ প্রকাশিত।"""


def api_key_ok(api_key) -> bool:
    return bool(api_key and str(api_key).strip().startswith("AIza"))


def _call_gemini(prompt: str, api_key: str, timeout: float = 90.0) -> str:
    body = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": SYSTEM_PROMPT + "\n\nNormalize this text:\n" + prompt}],
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
    """Send raw Bangla text to Gemini with the precise ruleset; return spoken text.

    Preserves [Name] tags/URLs so multi-voice routing still works downstream.
    Raises RuntimeError on any failure — the caller decides the fallback.
    """
    if not api_key_ok(api_key):
        raise RuntimeError("No valid Gemini API key (must start with 'AIza')")
    if not raw_text or not raw_text.strip():
        raise RuntimeError("Empty text")
    if len(raw_text) > MAX_INPUT_CHARS:
        raise RuntimeError(f"Text too long for one Gemini call ({len(raw_text)} chars > {MAX_INPUT_CHARS})")
    result = _call_gemini(raw_text, api_key, timeout=timeout).strip()
    if not result:
        raise RuntimeError("Gemini returned empty text")
    return result