"""
Bengali number verbalization (opt-in NATURAL mode only).

Default pipeline PRESERVEs digits untouched. These helpers convert numeric
forms to spoken Bengali words ONLY when explicitly requested
(verbalize_* / naturalize_sentence). Data and algorithm are separate:
lexicon tables below, grouping/parsing logic in functions.

South-Asian grouping: last 3 digits (hundreds), then pairs
(hazar -> lakh -> crore -> arab).
"""

import re

BN_DIGITS = "০১২৩৪৫৬৭৮৯"
_LATN_DIGITS = "0123456789"
_BN_TO_LATN = {b: l for b, l in zip(BN_DIGITS, _LATN_DIGITS)}
_LATN_TO_BN = {l: b for b, l in zip(BN_DIGITS, _LATN_DIGITS)}


def bn_to_latn_digit(text: str) -> str:
    """০১২ -> 012 (no other changes)."""
    return "".join(_BN_TO_LATN.get(c, c) for c in text)


def latn_to_bn_digit(text: str) -> str:
    """012 -> ০১২ (no other changes)."""
    return "".join(_LATN_TO_BN.get(c, c) for c in text)


# Canonical 0-99 (single accepted orthography per spec variant policy;
# variants like ছয়/ছয় both understood, one emitted).
_BN_0_99 = [
    "শূন্য", "এক", "দুই", "তিন", "চার", "পাঁচ", "ছয়", "সাত", "আট", "নয়",
    "দশ", "এগারো", "বারো", "তেরো", "চৌদ্দ", "পনেরো", "ষোল", "সতেরো",
    "আঠারো", "উনিশ", "বিশ", "একুশ", "বাইশ", "তেইশ", "চব্বিশ", "পঁচিশ",
    "ছাব্বিশ", "সাতাশ", "আটাশ", "ঊনত্রিশ", "ত্রিশ", "একত্রিশ", "বত্রিশ",
    "তেত্রিশ", "চৌত্রিশ", "পঁয়ত্রিশ", "ছত্রিশ", "সাঁইত্রিশ", "আটত্রিশ",
    "ঊনচল্লিশ", "চল্লিশ", "একচল্লিশ", "বিয়াল্লিশ", "তেতাল্লিশ",
    "চুয়াল্লিশ", "পঁয়তাল্লিশ", "ছেচল্লিশ", "সাতচল্লিশ", "আটচল্লিশ",
    "ঊনপঞ্চাশ", "পঞ্চাশ", "একান্ন", "বাহান্ন", "তিপ্পান্ন", "চুয়ান্ন",
    "পঞ্চান্ন", "ছাপ্পান্ন", "সাতান্ন", "আটান্ন", "ঊনষাট", "ষাট",
    "একষট্টি", "বাষট্টি", "তেষট্টি", "চৌষট্টি", "পঁয়ষট্টি", "ছেষট্টি",
    "সাতষট্টি", "আটষট্টি", "ঊনসত্তর", "সত্তর", "একাত্তর", "বাহাত্তর",
    "তিয়াত্তর", "চুয়াত্তর", "পঁচাত্তর", "ছিয়াত্তর", "সাতাত্তর",
    "আটাত্তর", "ঊনআশি", "আশি", "একাশি", "বিরাশি", "তিরাশি", "চুরাশি",
    "পঁচাশি", "ছিয়াশি", "সাতাশি", "আটাশি", "ঊননব্বই", "নব্বই",
    "একানব্বই", "বিরানব্বই", "তিরানব্বই", "চুরানব্বই", "পঁচানব্বই",
    "ছিয়ানব্বই", "সাতানব্বই", "আটানব্বই", "নিরানব্বই",
]

# (divisor, word) largest-first for South-Asian grouping.
_SCALES = [
    (10000000, "কোটি"),
    (100000, "লাখ"),
    (1000, "হাজার"),
    (100, "শত"),
]

_ORDINALS_1_10 = {
    1: "প্রথম", 2: "দ্বিতীয়", 3: "তৃতীয়", 4: "চতুর্থ", 5: "পঞ্চম",
    6: "ষষ্ঠ", 7: "সপ্তম", 8: "অষ্টম", 9: "নবম", 10: "দশম",
}

# Numeric date-ordinals ১লা/২রা/... (surface -> spoken).
_DATE_ORDINALS = {
    "১": "পহেলা", "২": "দোসরা", "৩": "তেসরা", "৪": "চৌঠা", "৫": "পাঁচই",
    "৬": "ছয়ই", "৭": "সাতই", "৮": "আটই", "৯": "নয়ই", "১০": "দশই",
    "1": "পহেলা", "2": "দোসরা", "3": "তেসরা", "4": "চৌঠা", "5": "পাঁচই",
    "6": "ছয়ই", "7": "সাতই", "8": "আটই", "9": "নয়ই", "10": "দশই",
}
_DATE_ORDINAL_SUFFIX = ("লা", "রা", "ঠা", "ই")

_BN_MONTHS_NUM = {
    1: "জানুয়ারি", 2: "ফেব্রুয়ারি", 3: "মার্চ", 4: "এপ্রিল",
    5: "মে", 6: "জুন", 7: "জুলাই", 8: "আগস্ট",
    9: "সেপ্টেম্বর", 10: "অক্টোবর", 11: "নভেম্বর", 12: "ডিসেম্বর",
}

# Undotted acronyms -> Bengali letter-spelled forms (spec list + safe common).
_ACRONYMS = {
    "API": "এপিআই", "GPU": "জিপিইউ", "CPU": "সিপিইউ", "AI": "এআই",
    "TTS": "টি টি এস", "NID": "এনআইডি", "USB": "ইউএসবি", "PDF": "পিডিএফ",
    "TV": "টিভি", "BBC": "বিবিসি", "MBBS": "এমবিবিএস", "MD": "এমডি",
    "RAM": "র‍্যাম",
}

# Latin unit abbreviations adjacent to numbers -> spoken (unambiguous only;
# bare "m" intentionally excluded per contextual-ambiguity rule).
_UNITS = {
    "km": "কিলোমিটার", "cm": "সেন্টিমিটার", "mm": "মিলিমিটার",
    "kg": "কিলোগ্রাম", "g": "গ্রাম", "mg": "মিলিগ্রাম",
    "L": "লিটার", "ml": "মিলিলিটার", "s": "সেকেন্ড", "min": "মিনিট",
    "h": "ঘণ্টা", "hr": "ঘণ্টা", "KB": "কিলোবাইট", "MB": "মেগাবাইট",
    "GB": "গিগাবাইট", "TB": "টেরাবাইট",
}

_CURRENCY = {
    "৳": ("টাকা", "পয়সা"), "BDT": ("টাকা", "পয়সা"), "Tk": ("টাকা", "পয়সা"),
    "₹": ("রুপি", "পয়সা"), "INR": ("রুপি", "পয়সা"),
    "$": ("ডলার", "সেন্ট"), "USD": ("ডলার", "সেন্ট"),
    "€": ("ইউরো", "সেন্ট"), "EUR": ("ইউরো", "সেন্ট"),
    "£": ("পাউন্ড", "পেন্স"), "GBP": ("পাউন্ড", "পেন্স"),
    "¥": ("ইয়েন", "সেন"), "JPY": ("ইয়েন", "সেন"),
}


def parse_integer(token: str):
    """Parse Western/South-Asian grouped Latin+Bengali digits -> int, else None."""
    if token is None:
        return None
    s = "".join(_BN_TO_LATN.get(c, c) for c in token.strip())
    s = s.replace(",", "").replace(" ", "").replace("_", "")
    if not s or not s.lstrip("+-").isdigit():
        return None
    try:
        return int(s)
    except ValueError:
        return None


def verbalize_integer(n: int) -> str:
    """Integer -> spoken Bengali (South-Asian scales). n >= 0."""
    if n < 0:
        return "মাইনাস " + verbalize_integer(-n)
    if n < 100:
        return _BN_0_99[n]
    parts = []
    for div, word in _SCALES:
        if n >= div:
            q, n = divmod(n, div)
            parts.append(verbalize_integer(q) + " " + word)
    if n:
        parts.append(_BN_0_99[n])
    return " ".join(parts)


def verbalize_decimal(token: str):
    """'৩.১৪' -> 'তিন দশমিক এক চার'. Returns None if not a decimal."""
    s = "".join(_BN_TO_LATN.get(c, c) for c in token.strip())
    m = re.fullmatch(r"([+-]?\d+)\.(\d+)", s)
    if not m:
        return None
    out = verbalize_integer(int(m.group(1))) if m.group(1).lstrip("+-") else ""
    frac = " ".join(_BN_0_99[int(d)] for d in m.group(2))
    if m.group(1).startswith("-"):
        out = "মাইনাস " + out if out else "মাইনাস"
    return (out + " দশমিক " + frac).strip() if out else "দশমিক " + frac


def verbalize_percent(token: str):
    """'৫০%' / '50 %' -> 'পঞ্চাশ শতাংশ'. None if not a percent."""
    m = re.fullmatch(r"\s*([0-9০-৯][0-9০-৯,]*)\s*%\s*", token)
    if not m:
        return None
    n = parse_integer(m.group(1))
    return verbalize_integer(n) + " শতাংশ" if n is not None else None


def _split_amount(token: str):
    """'১,২৫০.৫০' -> (1250, '৫০'). Returns (int|None, frac_str|None)."""
    s = "".join(_BN_TO_LATN.get(c, c) for c in token.strip())
    if "." in s:
        whole, frac = s.split(".", 1)
        w = parse_integer(whole) if whole else 0
        return w, frac if frac and frac.isdigit() else None
    w = parse_integer(s)
    return w, None


def verbalize_currency(token: str):
    """'৳১,২৫০.৫০' -> 'এক হাজার দুইশত পঞ্চাশ টাকা পঞ্চাশ পয়সা'. None if no match."""
    token = token.replace("(", " ").replace(")", " ")
    m = re.fullmatch(
        r"\s*([৳₹$€£¥]|BDT|INR|USD|EUR|GBP|JPY|Tk)\s*([0-9০-৯][0-9০-৯,\.]*)\s*"
        r"|\s*([0-9০-৯][0-9০-৯,\.]*)\s*(টাকা|রুপি|ডলার|পয়সা)\s*",
        token,
    )
    if not m:
        return None
    _FRAC_OF = {"টাকা": "পয়সা", "রুপি": "পয়সা", "ডলার": "সেন্ট", "পয়সা": "পয়সা"}
    if m.group(1):
        unit, frac_unit = _CURRENCY.get(m.group(1), (m.group(1), ""))
        whole, frac = _split_amount(m.group(2))
    else:
        unit = m.group(4)
        frac_unit = _FRAC_OF.get(unit, "")
        whole, frac = _split_amount(m.group(3))
    if whole is None:
        return None
    out = verbalize_integer(whole) + " " + unit
    if frac:
        # Paise/cents read as a number (পঞ্চাশ পয়সা), not digit-by-digit.
        fwords = verbalize_integer(int(frac)) if frac.isdigit() else ""
        if fwords and frac_unit:
            out += " " + fwords + " " + frac_unit
    return out


def verbalize_time(token: str):
    """'৫:৩০' -> 'সাড়ে পাঁচটা'; natural Bengali clock words; H<24, M<60.

    Colon only (dotted forms like '৩.১৪' are decimals, never clocks).
    :00 -> Hটা · :15 -> সোয়া Hটা · :30 -> দেড়টা (1:30) / সাড়ে Hটা ·
    :45 -> পৌনে (H+1)টা · else Hটা M. 24h folds to 12h (১৩:৩০->সাড়ে একটা).
    """
    m = re.fullmatch(r"\s*([0-9০-৯]{1,2}):([0-9০-৯]{2})\s*", token)
    if not m:
        return None
    h = parse_integer(m.group(1))
    mi = parse_integer(m.group(2))
    if h is None or mi is None or h > 23 or mi > 59:
        return None
    h12 = h % 12 or 12
    hw = verbalize_integer(h12)
    if mi == 0:
        return hw + "টা"
    if mi == 15:
        return "সোয়া " + hw + "টা"
    if mi == 30:
        # দেড়টা is 1:30 specifically; 13:30 stays explicit (সাড়ে একটা).
        return "দেড়টা" if h == 1 else "সাড়ে " + hw + "টা"
    if mi == 45:
        nxt = h12 % 12 + 1
        return "পৌনে " + verbalize_integer(nxt) + "টা"
    return hw + "টা " + verbalize_integer(mi)


def verbalize_date(token: str):
    """'১২/০৯/২০২৬' or '১২.০৯.২০২৬' -> 'বারো সেপ্টেম্বর দুই হাজার ছাব্বিশ'."""
    m = re.fullmatch(r"\s*([0-9০-৯]{1,2})[/.\-]([0-9০-৯]{1,2})[/.\-]([0-9০-৯]{2,4})\s*", token)
    if not m:
        return None
    d, mo, y = (parse_integer(g) for g in m.groups())
    if d is None or mo is None or y is None:
        return None
    if y < 100:
        y += 2000
    if not (1 <= mo <= 12 and 1 <= d <= 31):
        return None
    return f"{verbalize_integer(d)} {_BN_MONTHS_NUM[mo]} {verbalize_integer(y)}"


def verbalize_ordinal(token: str):
    """'১ম'->'প্রথম', '২রা'->'দোসরা', '২৫তম'->fallback. None if no match."""
    m = re.fullmatch(r"\s*([0-9০-৯]+)\s*(ম|য়|র্থ|ষ্ঠ|লা|রা|ঠা|ই|তম)\s*", token)
    if not m:
        return None
    n = parse_integer(m.group(1))
    if n is None:
        return None
    if m.group(2) in ("লা", "রা", "ঠা", "ই"):
        key = bn_to_latn_digit(m.group(1)).lstrip("0") or "0"
        latin = str(n)
        if latin in _DATE_ORDINALS_FLAT or m.group(1) in _DATE_ORDINALS:
            return _DATE_ORDINALS.get(m.group(1), _DATE_ORDINALS_FLAT.get(latin))
        return verbalize_integer(n) + m.group(2)
    if n in _ORDINALS_1_10 and m.group(2) in ("ম", "য়", "র্থ", "ষ্ঠ"):
        canon = {1: "ম", 2: "য়", 3: "য়", 4: "র্থ", 5: "ম",
                 6: "ষ্ঠ", 7: "ম", 8: "ম", 9: "ম", 10: "ম"}
        if m.group(2) == canon[n]:
            return _ORDINALS_1_10[n]
    if m.group(2) == "তম":
        return verbalize_integer(n) + "তম"
    return verbalize_integer(n) + m.group(2)


_DATE_ORDINALS = {
    "১": "পহেলা", "২": "দোসরা", "৩": "তেসরা", "৪": "চৌঠা", "৫": "পাঁচই",
    "৬": "ছয়ই", "৭": "সাতই", "৮": "আটই", "৯": "নয়ই", "১০": "দশই",
}
_DATE_ORDINALS_FLAT = {
    "1": "পহেলা", "2": "দোসরা", "3": "তেসরা", "4": "চৌঠা", "5": "পাঁচই",
    "6": "ছয়ই", "7": "সাতই", "8": "আটই", "9": "নয়ই", "10": "দশই",
}

# Year words that license year-style reading (উনিশশো সাতাশি, not quantity).
_YEAR_WORDS = ("সালে", "সাল", "খ্রিস্টাব্দে", "খ্রিস্টাব্দ", "সনে", "সন")


def verbalize_year(n: int):
    """4-digit year -> natural Bengali year words. None if not year-shaped.

    1987 -> উনিশশো সাতাশি, 2026 -> দুই হাজার ছাব্বিশ, 2000 -> দুই হাজার.
    Only 1100-2099 get year-style; anything else falls back to quantity
    reading (caller decides).
    """
    if n is None or not (1100 <= n <= 2099):
        return None
    if 2000 <= n <= 2099:
        rest = n - 2000
        out = "দুই হাজার"
        if rest:
            out += " " + verbalize_integer(rest)
        return out
    hi, lo = divmod(n, 100)
    out = verbalize_integer(hi) + "শো"
    if lo:
        out += " " + verbalize_integer(lo)
    return out

_ORDINALS_1_10 = {
    1: "প্রথম", 2: "দ্বিতীয়", 3: "তৃতীয়", 4: "চতুর্থ", 5: "পঞ্চম",
    6: "ষষ্ঠ", 7: "সপ্তম", 8: "অষ্টম", 9: "নবম", 10: "দশম",
}


def verbalize_acronym(token: str):
    """Known undotted acronyms -> Bengali letter-spelled forms. None if unknown."""
    return _ACRONYMS.get(token.strip())


def verbalize_unit(number_token: str, unit_token: str):
    """'৫'+'kg' -> 'পাঁচ কিলোগ্রাম'. None if unit unknown or number unparsable."""
    n = parse_integer(number_token)
    word = _UNITS.get(unit_token.strip())
    if n is None or word is None:
        return None
    return verbalize_integer(n) + " " + word


def verbalize_phone(token: str):
    """Digit string -> space-separated Bengali digit words (no grouping guess)."""
    digits = "".join(
        _BN_TO_LATN.get(c, "") for c in token if c in _LATN_DIGITS + BN_DIGITS
    )
    if not digits:
        return None
    return " ".join(_BN_0_99[int(d)] for d in digits)
