"""
Per-language text processing for audiobook generation (Bangla-first TN layer).

Pipeline order (more-specific wins, never the reverse):

    escape pre-existing placeholders
    -> application tags ([Character])
    -> URLs -> emails -> phones
    -> numbers (decimals, versions, IPs, groupings, period-dates)
    -> line-start list markers
    -> acronyms/initials -> abbreviations
    -> symbol words (only on unprotected text)
    -> sentence segmentation (STRONG > MEDIUM)
    -> existing chunking -> restoration + leak assert

Design rules (from the Bangla master spec):
- PRESERVE by default. Never spell-correct, translate, paraphrase, or reorder.
- Placeholders are per-call registry entries (request-local), never global state.
- Restoration happens before TTS / metadata / UI; a final assert guarantees
  no placeholder leaks.
- Newlines are never touched (pause calculation depends on them).
- English + Hindi keep their profiles; unknown locales fall back to English.
"""

import re

# Placeholder alphabet (Unicode private-use, extremely unlikely in real books).
_PUA_ESC = "\uE002"   # escape marker for pre-existing PUA chars in source
_PUA_RE = re.compile(r"[\ue000-\uf8ff]")  # any private-use char in source text
_PUA_END = "\uE001"   # terminator for generated placeholders
_PFX = {  # one prefix char per protected-token type
    "TAG": "\uE010",
    "URL": "\uE011",
    "EMAIL": "\uE012",
    "NUM": "\uE013",
    "LIST": "\uE014",
    "ABBR": "\uE015",
    "ACR": "\uE016",
    "PHONE": "\uE017",
    "DOMAIN": "\uE018",
    "CURR": "\uE019",
}
_PH_RE = re.compile(r"[\ue002\ue010-\ue019]\d+[\ue001]")

_DIG = r"0-9০-৯"  # Latin + Bengali digits

# Ordered protection stages: (type, compiled pattern). More specific first.
# Patterns must not match across whitespace (placeholders stay atomic).
_PATTERNS = [
    ("TAG", re.compile(r"\[[^\]\n]{1,60}\]")),
    ("URL", re.compile(r"https?://[^\s<>\"“”‘’]+|www\.[^\s<>\"“”‘’]+")),
    ("EMAIL", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    # Currency amounts (symbol glued to digits): protected as ONE unit so
    # surface expansion can't flip word order (৳১,০০০ must NOT become টাকা ১,০০০).
    ("CURR", re.compile(r"[৳₹$€£¥]\s*\(?\s*[0-9০-৯][0-9০-৯,]*(?:\.[0-9০-৯]+)?\s*\)?"
                        r"|(?:BDT|INR|USD|EUR|GBP|JPY|Tk)\s*[0-9০-৯][0-9০-৯,\.]*")),
    # Hashtags + mentions: recognized, preserved as-is (never expanded).
    ("TAG", re.compile(r"#[^\s#.,;:!?।]+")),
    ("TAG", re.compile(r"@[A-Za-z0-9_]+")),
    # Bare domains (no scheme): strict TLD allowlist so "e.g." / "Ph.D."
    # never match (their tails aren't real TLDs).
    ("DOMAIN", re.compile(
        r"\b(?:[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?\.)+"
        r"(?:com|net|org|edu|gov|mil|int|io|bd|in|info|biz|me|tv|co|app|dev|xyz|site|online|news)\b"
    )),
    # Phones: +880... international form, or 11+ digit runs (BD mobiles).
    # Hyphen-only digit runs (২০২০-২০২৬ ranges) do NOT match: separators
    # allowed are spaces/parens only, so ranges fall through to bare numbers.
    # The 5-6+6 hyphen form (০১৭১২-৩৪৫৬৭৮) is explicitly included (10+ digits).
    ("PHONE", re.compile(r"\+\s?[\d০-৯][\d০-৯\s()]{7,}[\d০-৯]"
                         r"|\b[\d০-৯](?:[\d০-৯]{10,})\b"
                         r"|\b[\d০-৯]{4,6}-[\d০-৯]{6}\b")),
    ("NUM", re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")),  # IPv4 first (subset of dotted numbers)
    ("NUM", re.compile(r"[" + _DIG + r"]+[" + _DIG + r",]*[.,][" + _DIG + r"]+(?:\.["
                       + _DIG + r"]+)*")),  # decimals, versions, groupings, period-dates
    ("LIST", re.compile(r"(?m)^[ \t]*[0-9০-৯]+[.\)](?=\s|$)")),  # line-start markers only
    # Letter lists (ক. খ. at line start) and Roman-numeral lists (iv. / IV.).
    ("LIST", re.compile(r"(?m)^[ \t]*(?:[ক-হa-zA-Z]|[ivxlcdmIVXLCDM]+)[.\)](?=\s|$)")),
    ("ACR", re.compile(r"\b[A-Za-z]{1,4}(?:\.[A-Za-z]{1,4})+\.?")),  # Ph.D., M.A., U.S.
    # Spaced initials: "M. A. Rahman" (single capitals + periods before a name).
    # Lookahead is any Unicode letter ([^\W\d_] trick — safe for all scripts).
    ("ACR", re.compile(r"\b(?:[A-Za-z]\.\s*)+(?=[^\W\d_])")),
]

_TRAILING_PUNCT = ".,;:!?।)]}'\"“”‘’"

LANG_PROFILES = {
    "en": {
        "name": "English",
        "enders": {".", "!", "?"},
        "max_words": 50,
        "abbreviations": [
            "Mr.", "Mrs.", "Ms.", "Dr.", "St.", "Sr.", "Jr.", "Prof.",
            "Inc.", "Ltd.", "Corp.", "e.g.", "i.e.", "vs.", "etc.",
            "U.S.", "U.K.", "a.m.", "p.m.", "Ph.D.", "M.A.", "B.Sc.",
        ],
        "symbols": {
            "%": " percent ",
            "$": " dollars ",
            "₹": " rupees ",
            "&": " and ",
            "#": " number ",
        },
    },
    "bn": {
        "name": "Bengali",
        "enders": {".", "!", "?", "।"},
        "max_words": 50,
        "abbreviations": [
            "Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "St.",
            "ড.", "মো.", "প্রো.", "ইং.", "ডাঃ", "ডঃ", "মি.", "মিসেস",
            "পৃ.", "অর্থাৎ", "Ph.D.", "M.A.", "B.Sc.",
        ],
        # NOTE: ৳ (BDT, টাকা) and ₹ (INR, রুপি) are deliberately distinct.
        "symbols": {
            "%": " শতাংশ ",
            "৳": " টাকা ",
            "₹": " রুপি ",
            "$": " ডলার ",
            "&": " এবং ",
            "#": " নম্বর ",
        },
    },
    "hi": {
        "name": "Hindi",
        "enders": {".", "!", "?", "।"},
        "max_words": 50,
        "abbreviations": [
            "Dr.", "Mr.", "Mrs.", "Ms.", "Prof.",
            "डॉ.", "मो.", "प्रो.", "अर्थात",
        ],
        "symbols": {
            "%": " प्रतिशत ",
            "₹": " रुपये ",
            "$": " डॉलर ",
            "&": " और ",
            "#": " नंबर ",
        },
    },
}

# Sentence segmentation priority: STRONG (। ! ?) always splits; MEDIUM (.)
# splits only at whitespace/end/closing-quote (protected decimals, URLs,
# versions, abbreviations are already hidden). Single U+2026 ellipsis counts
# as one MEDIUM boundary (consistent with "..."). WEAK (: ;) and NONE (,)
# never split in this layer. A bare "." needs a boundary follower so stray
# mid-token dots prefer the longer sentence; closing quotes attach left.
_BOUND_RE = re.compile(r'[।!?]+["”\u2019\']*|\u2026(?=[\s"”\u2019\']|$)|\.(?=[\s"”\u2019\']|$)')


def normalize_locale(language_id) -> str:
    """Accept bn/bn-BD/bn_BD/BN (etc.), return base profile code; unknown -> en."""
    code = (language_id or "en").strip().lower().replace("_", "-")
    base = code.split("-")[0]
    if base in LANG_PROFILES:
        return base
    return "en"


_VOICE_TAG_LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)  # any Unicode letter


def is_voice_tag(name) -> bool:
    """True if a [bracket] name is a speaker voice, not a footnote/marker.

    Footnotes like [১]/[2], empty tags and overlong blobs are NOT voices —
    misrouting them creates phantom "characters" and drops narration.
    """
    name = (name or "").strip() if isinstance(name, str) else ""
    if not name or len(name) > 40:
        return False
    return bool(_VOICE_TAG_LETTER_RE.search(name))


def get_profile(language_id) -> dict:
    """Return the profile for a language code, falling back to English."""
    return LANG_PROFILES.get(normalize_locale(language_id), LANG_PROFILES["en"])


class ProtectionContext:
    """Request-local placeholder registry. One instance per protect_all() call."""

    def __init__(self, lang="en", natural=False):
        self.mapping = {}  # placeholder -> original text
        self.counts = {}  # type -> int
        self.lang = lang
        self.natural = natural  # opt-in spoken expansion (default: preserve)

    def add(self, ptype: str, original: str) -> str:
        n = self.counts.get(ptype, 0)
        self.counts[ptype] = n + 1
        ph = f"{_PFX[ptype]}{n}{_PUA_END}"
        self.mapping[ph] = original
        return ph


def _protect_spans(text: str, ctx: ProtectionContext) -> str:
    """Apply ordered regex stages right-to-left so offsets stay valid."""
    # Escape any pre-existing private-use chars so restore never corrupts them.
    if _PUA_RE.search(text):
        text = re.sub(r"[\ue000-\uE01F]",
                      lambda m: _new_escape_placeholder(ctx, m.group(0)), text)
    for ptype, pattern in _PATTERNS:
        def _one(m, _pt=ptype):
            original = m.group(0)
            # Strip trailing sentence punctuation from URL/email/domain matches
            # (https://example.com/বই। -> URL + dari), keep it in the text.
            if _pt in ("URL", "EMAIL", "DOMAIN"):
                stripped = original.rstrip(_TRAILING_PUNCT)
                if stripped and stripped != original:
                    return ctx.add(_pt, stripped) + original[len(stripped):]
            return ctx.add(_pt, original)
        # Right-to-left replacement keeps earlier match offsets valid.
        matches = list(pattern.finditer(text))
        for m in reversed(matches):
            s, e = m.span()
            text = text[:s] + _one(m) + text[e:]
    return text


def _new_escape_placeholder(ctx: ProtectionContext, original: str) -> str:
    n = ctx.counts.get("ESC", 0)
    ctx.counts["ESC"] = n + 1
    ph = f"{_PUA_ESC}{n}{_PUA_END}"
    ctx.mapping[ph] = original
    return ph


def protect_all(text: str, language_id="en", natural=False):
    """Full ordered protection. Returns (protected_text, ctx).

    Order: BOM/NFC ingest -> escape -> tags -> URLs -> emails -> phones ->
    numbers -> list markers -> acronyms -> abbreviations. Symbols are NOT
    touched here (they run after protection in normalize_text).
    `natural=True` only arms opt-in spoken expansion at finalize time.
    """
    import unicodedata
    if not text:
        return text, ProtectionContext(language_id, natural)
    ctx = ProtectionContext(normalize_locale(language_id), natural)
    # Ingest safety: strip leading BOM, canonicalize to NFC (visually
    # identical by definition — never a spelling change), drop C0/C1
    # control chars that can choke synthesizers (keep \n \t for pauses).
    text = text.lstrip("\ufeff")
    import unicodedata
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
    text = _protect_spans(text, ctx)
    # Abbreviations (locale list, longest first) after structural patterns.
    for abbr in sorted(get_profile(language_id)["abbreviations"], key=len, reverse=True):
        if "." not in abbr:
            continue
        start = 0
        while True:
            i = text.find(abbr, start)
            if i < 0:
                break
            # Skip occurrences already inside a placeholder (no PUA overlap possible,
            # but guard anyway) and require non-letter neighbours to avoid
            # masking decimals/versions already protected (defensive).
            text = text[:i] + ctx.add("ABBR", abbr) + text[i + len(abbr):]
            start = i + 1
    return text, ctx


def restore_all(text: str, ctx: ProtectionContext) -> str:
    """Restore every placeholder from this call's registry. Unknown markers kept."""

    def _rep(m):
        ph = m.group(0)
        return ctx.mapping.get(ph, ph)

    return _PH_RE.sub(_rep, text)


def assert_no_placeholders(text: str, where: str = "output"):
    """Final leak assert: no UNRESOLVED placeholder may survive to TTS/metadata/UI.

    Note: this checks typed markers (ESC/TAG/URL/... + index + terminator),
    NOT the raw PUA range — source text may legitimately contain PUA chars,
    which escape/restore round-trips faithfully.
    """
    m = _PH_RE.search(text or "")
    if m:
        raise ValueError(
            f"Placeholder leak in {where}: {m.group(0)!r} remains in text. "
            "Refusing to send to TTS."
        )


def finalize_text(text: str, ctx: ProtectionContext, where: str = "tts", trace=None) -> str:
    """Restore + leak-assert. The only sanctioned exit before TTS/metadata/UI.

    With ctx.natural (opt-in), restored sentences additionally pass through
    spoken expansion (numbers/dates/currency/acronyms -> Bengali words).
    Fired rule IDs accumulate in `trace` when given a list (debug/audit).
    """
    text = restore_all(text, ctx)
    assert_no_placeholders(text, where)
    if getattr(ctx, "natural", False) and normalize_locale(getattr(ctx, "lang", "en")) == "bn":
        text = naturalize_sentence(text, "bn", trace=trace)
    return text


def segment_sentences(text: str, language_id="en"):
    """Split PROTECTED text into sentences. Never returns empties.

    Delimiters stay attached to their sentence; ellipsis ("...") counts as
    ONE boundary; closing quotes attach left. Anything ambiguous is left
    whole (prefer-longer policy).
    """
    get_profile(language_id)  # validates locale; boundary set is uniform here
    sentences = []
    start = 0
    for m in _BOUND_RE.finditer(text or ""):
        piece = text[start:m.end()].strip()
        if piece:
            sentences.append(piece)
        start = m.end()
    tail = (text[start:] or "").strip()
    if tail:
        sentences.append(tail)
    sentences = [s for s in sentences if s.strip()]
    # Pass 1: glue pure-punctuation fragments back (a lone । or ” is never
    # a sentence — it belongs to its neighbour).
    glued = []
    for s in sentences:
        if glued and re.fullmatch(r"[।!?.,;:…\"”’'\s]+", s):
            glued[-1] = (glued[-1] + s).strip()
        else:
            glued.append(s)
    # Pass 2: nested quotes — a short tail inside still-open “...” belongs to
    # the outer sentence. Bounded (short tails only) so unbalanced quotes in
    # the wild can't fuse the rest of the book.
    merged, depth = [], 0
    for s in glued:
        if (merged and depth > 0
                and (re.fullmatch(r"[”’\"'\s]+", s) or len(s) < 80)):
            merged[-1] = (merged[-1] + " " + s).strip()
        else:
            merged.append(s)
        depth += s.count("“") + s.count("‘") - s.count("”") - s.count("’")
        depth = max(0, depth)
    return [s for s in merged if s.strip()]


def naturalize_sentence(text: str, lang="bn", trace=None) -> str:
    """Opt-in spoken expansion for Bengali (default pipeline never calls this).

    Order matters (specific-first): currency -> percent -> time -> date ->
    year-words -> ordinals/classifiers -> units-adjacent -> acronyms ->
    bare integers. Each step's output is Bengali words, so later
    digit-regexes can't re-fire. Unknown/malformed forms are left untouched
    (preserve-on-uncertainty). Fired rules append (rule_id, before, after)
    to `trace` when given a list.
    """
    if normalize_locale(lang) != "bn" or not text:
        return text
    try:
        from src.audiobook import bn_numbers as _bn
    except ImportError:
        from . import bn_numbers as _bn

    def _sub(pattern, fn, s, flags=0, rule_id=None):
        def _rep(m):
            try:
                v = fn(m)
            except Exception:
                v = None
            if v and v != m.group(0):
                if trace is not None and rule_id:
                    trace.append((rule_id, m.group(0), v))
                return v
            return m.group(0)
        return re.sub(pattern, _rep, s, flags=flags)

    s = text
    # Currency with amount: ৳১,২৫০.৫০ / BDT 500 (also unit-first forms and
    # parenthesized accounting ৳(৫০০)). The close-paren is conditional on an
    # open paren so trailing spaces are never consumed (spacing preserved).
    s = _sub(r"[৳₹$€£¥]\s*(?:\(\s*)?[0-9০-৯][0-9০-৯,]*(?:\.[0-9০-৯]+)?(?:\s*\))?"
             r"|(?:BDT|INR|USD|EUR|GBP|JPY|Tk)\s*[0-9০-৯][0-9০-৯,\.]*"
             r"|[0-9০-৯][0-9০-৯,\.]*\s*(?:টাকা|রুপি|ডলার)",
             lambda m: _bn.verbalize_currency(m.group(0)), s, rule_id="BN_CURR_001")
    # Percent: ৫০% / 50 %.
    s = _sub(r"[0-9০-৯][0-9০-৯,]*\s*%",
             lambda m: _bn.verbalize_percent(m.group(0)), s, rule_id="BN_PERCENT_001")
    # Time HH:MM with colon only (dotted "৩.১৪" is a decimal, never a clock;
    # plausible clocks only; versions already protected).
    s = _sub(r"(?<![\d.])([0-9০-৯]{1,2}):([0-9০-৯]{2})(?![\d.])",
             lambda m: _bn.verbalize_time(m.group(0)), s, rule_id="BN_TIME_001")
    # Dates D/M/YYYY (any of / . - separators).
    s = _sub(r"(?<![\d.])([0-9০-৯]{1,2})[/.\-]([0-9০-৯]{1,2})[/.\-]([0-9০-৯]{2,4})(?![\d.])",
             lambda m: _bn.verbalize_date(m.group(0)), s, rule_id="BN_DATE_001")
    # Ratio N:N leftovers (plausible clocks already claimed above as times):
    # ৩:২ -> তিন বনাম দুই.
    def _ratio(m):
        a, b = _bn.parse_integer(m.group(1)), _bn.parse_integer(m.group(2))
        if a is None or b is None:
            return m.group(0)
        return _bn.verbalize_integer(a) + " বনাম " + _bn.verbalize_integer(b)
    s = _sub(r"(?<![\d:.])([0-9০-৯][0-9০-৯,]*)\s*:\s*([0-9০-৯][0-9০-৯,]*)(?![\d:.])",
             _ratio, s, rule_id="BN_RATIO_001")
    # Year with year-word (সালে/সাল/খ্রিস্টাব্দ/সন): ১৯৮৭ সালে -> উনিশশো সাতাশি সালে.
    # Mc-safe lookahead (not \b): Bengali vowel signs are non-\w.
    def _yr(m):
        n = _bn.parse_integer(m.group(1))
        w = _bn.verbalize_year(n) if n is not None else None
        return (w + m.group(2) + m.group(3)) if w else m.group(0)
    s = _sub(r"([0-9০-৯]{4})(\s*)(সালে|সাল|খ্রিস্টাব্দে|খ্রিস্টাব্দ|সনে|সন)(?![\w\u0980-\u09FF])",
             _yr, s, rule_id="BN_YEAR_001")
    # Ordinals + date-ordinals: ১ম ২য় ১লা ২রা... NOTE: trailing (?!\w),
    # not \b — Bengali vowel signs (Mc, e.g. া) are non-\w so \b misfires.
    s = _sub(r"[0-9০-৯]+(?:ম|য়|র্থ|ষ্ঠ|লা|রা|ঠা|ই|তম)(?!\w)",
             lambda m: _bn.verbalize_ordinal(m.group(0)), s, rule_id="BN_ORD_001")
    # Classifiers: ৫টা / ৩ টি / ৪জন (number verbalized, classifier kept,
    # original spacing kept: "৩ টি" stays spaced, "৩টি" stays glued).
    def _cls(m):
        n = _bn.parse_integer(m.group(1))
        if n is None:
            return m.group(0)
        return _bn.verbalize_integer(n) + m.group(2) + m.group(3)
    s = _sub(r"([0-9০-৯][0-9০-৯,]*)(\s*)(টা|টি|জন|খানা|খানি|গুলো|গুলি)(?!\w)", _cls, s,
             rule_id="BN_CLS_001")
    # Leading minus (not ranges like ৫-১০): -৫ -> মাইনাস পাঁচ.
    # Lookbehind covers Bengali block so ঢাকা-১২০৫ is NOT read as minus-amount.
    s = _sub(r"(?<![\d০-৯A-Za-z\u0980-\u09FF\u200c\u200d])[-−](?=[\d০-৯])", lambda m: "মাইনাস ", s,
             rule_id="BN_NEG_001")
    # Units adjacent to numbers: ৫kg / ১০ কিমি... Latin abbrevs mapped, Bengali kept.
    def _unit(m):
        v = _bn.verbalize_unit(m.group(1), m.group(2))
        return v if v else m.group(0)
    s = _sub(r"([0-9০-৯][0-9০-৯,]*)\s*(km|cm|mm|kg|mg|L|ml|s|min|h|hr|KB|MB|GB|TB)\b", _unit, s,
             rule_id="BN_UNIT_001")
    # Known undotted acronyms: API/GPU/... (unknown ALL-CAPS left alone).
    s = _sub(r"\b(API|GPU|CPU|AI|TTS|NID|USB|PDF|TV|BBC|MBBS|MD|RAM)\b",
             lambda m: _bn.verbalize_acronym(m.group(1)) or m.group(0), s,
             rule_id="BN_ACR_001")
    # Phone-like runs: digit-spell (no grouping guess). Tightened so
    # hyphenated ranges (২০২০-২০২৬) never match — only +, spaces, parens.
    s = _sub(r"\+?[০-৯0-9][০-৯0-9\s()]{7,}[০-৯0-9]|\b[\d০-৯](?:[\d০-৯]{10,})\b|\b[\d০-৯]{4,6}-[\d০-৯]{6}\b",
             lambda m: _bn.verbalize_phone(m.group(0)) or m.group(0), s,
             rule_id="BN_PHONE_001")
    # Version with v-prefix: v2.1 -> ভার্সন দুই দশমিক এক.
    def _ver(m):
        n = _bn.parse_integer(m.group(1).split(".")[0]) if "." not in m.group(1) else None
        body = _bn.verbalize_decimal(m.group(1)) if "." in m.group(1) else (
            _bn.verbalize_integer(n) if n is not None else None)
        return ("ভার্সন " + body) if body else m.group(0)
    s = _sub(r"\bv\s*([0-9০-৯][0-9০-৯.]*)", _ver, s, rule_id="BN_VER_001")
    def _num(m):
        tok = m.group(0)
        if "." in tok:
            v = _bn.verbalize_decimal(tok)
            return v if v else tok
        n = _bn.parse_integer(tok)
        return _bn.verbalize_integer(n) if n is not None else tok
    s = _sub(r"(?<![\w\u0980-\u09FF\u200c\u200d.])([0-9০-৯][0-9০-৯,]*\.?[0-9০-৯]*)(?![\w\u0980-\u09FF\u200c\u200d.])", _num, s,
             rule_id="BN_INT_001")
    return re.sub(r"[ \t]+", " ", s).strip()


def split_graphemes(text: str):
    """Split into extended-grapheme-ish clusters (base + combining marks +
    ZWJ/ZWNJ-joined sequences). stdlib only; used by tests and the audit."""
    import unicodedata
    clusters, cur, prev_joiner = [], "", False
    for ch in text or "":
        cat = unicodedata.category(ch)
        if not cur:
            cur = ch
        elif cat in ("Mn", "Me") or ch in "\u200d\u200c" or prev_joiner:
            cur += ch
        else:
            clusters.append(cur)
            cur = ch
        prev_joiner = ch in "\u200d\u200c"
    if cur:
        clusters.append(cur)
    return clusters


def find_unknown_tokens(text: str, lang="bn"):
    """Heuristic scan for tokens the engine doesn't understand.

    Returns dict with possible acronyms (ALL-CAPS), dotted tokens outside the
    abbreviation lists, and currency/unit symbols adjacent to numbers that
    have no mapping. Advisory only — never rewrites.
    """
    profile = get_profile(lang)
    known_abbr = set(profile["abbreviations"])
    found = {"acronyms": [], "dotted": [], "symbols": []}
    for tok in re.findall(r"[A-Z]{2,}", text or ""):
        if tok not in found["acronyms"]:
            found["acronyms"].append(tok)
    for tok in re.findall(r"\S+\.\S*", text or ""):
        core = tok.strip(".,;:!?।\"”‘’'()[]")
        # Skip anything already recognized as URL/email (they contain :// or @).
        if "://" in core or "@" in core:
            continue
        if "." in core and core not in known_abbr and core not in found["dotted"]:
            found["dotted"].append(core)
    for sym in sorted(set(re.findall(r"[৳₹$€£¥%#°‰™®©+\-*/=]", text or ""))):
        if sym not in profile["symbols"] and sym not in ("-", "/", "*", "+", "="):
            found["symbols"].append(sym)
    return found


def normalize_text(text: str, language_id="en", symbols=True) -> str:
    """Light per-language cleanup. Runs AFTER protection. Never touches newlines.

    symbols=False skips the symbol-word map (natural mode verbalizes symbols
    with their digits instead, e.g. ৫০% -> পঞ্চাশ শতাংশ, not ৫০ শতাংশ).
    """
    profile = get_profile(language_id)
    if symbols:
        for sym, word in profile["symbols"].items():
            text = text.replace(sym, word)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    # Dari needs a following space for splitters — unless a closing quote
    # follows (।” stays glued so the quote attaches left, not stranded).
    if "।" in profile["enders"]:
        text = re.sub("।(?=[^\\s\"”\u2019'])", "। ", text)
    return text.strip(" \t")


def prepare_text_for_language(text: str, language_id="en"):
    """Legacy pre-pass (abbrev-only placeholder). Prefer protect_all for new code."""
    if not text:
        return text
    text = normalize_text(text, language_id)
    text = protect_abbreviations(text, language_id)
    return text


# --- Legacy single-char abbreviation API (kept for backward compatibility) ---

_PLACEHOLDER = "\uE000"


def protect_abbreviations(text: str, language_id="en") -> str:
    """Hide abbreviation periods from sentence splitters (longest first)."""
    for abbr in sorted(get_profile(language_id)["abbreviations"], key=len, reverse=True):
        if "." in abbr:
            text = text.replace(abbr, abbr.replace(".", _PLACEHOLDER))
    return text


def restore_abbreviations(text: str) -> str:
    """Restore protected abbreviation periods. No-op when none present."""
    if _PLACEHOLDER in text:
        text = text.replace(_PLACEHOLDER, ".")
    return text
