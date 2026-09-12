"""Golden tests for the Bangla TN layer (src/audiobook/langtext.py).

Headless: no model, no GPU, no network. Run with plain python:
    python3 tests/test_langtext.py
(pytest also collects every test_* function.)
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.audiobook.langtext import (
    protect_all,
    restore_all,
    finalize_text,
    assert_no_placeholders,
    normalize_locale,
    segment_sentences,
    normalize_text,
    LANG_PROFILES,
)

PASS = []


def check(name, fn):
    try:
        fn()
    except AssertionError as e:
        print(f"FAIL {name}: {e}")
        raise SystemExit(1)
    PASS.append(name)


def seg(text, lang="bn"):
    """Full pipeline: protect -> normalize -> segment -> finalize each."""
    ptext, ctx = protect_all(text, lang)
    ptext = normalize_text(ptext, lang)
    out = []
    for sent in segment_sentences(ptext, lang):
        out.append(finalize_text(sent, ctx, where="test"))
    # Nothing protected may be lost: every placeholder must resolve.
    assert_no_placeholders(" ".join(out), where="test")
    return out


# --- Spec §521-536 golden examples -------------------------------------

def test_danda_three_sentences():
    out = seg("আমি আজ ঢাকায় গেলাম। তারপর বন্ধুর সঙ্গে দেখা করলাম। রাতে বাড়ি ফিরলাম।")
    assert len(out) == 3, out


def test_mixed_abbrev_quote():
    out = seg("ড. রহমান আজ Zoom meeting করলেন। তিনি বললেন, “API ঠিকমতো কাজ করছে।”")
    assert len(out) == 2, out
    assert "ড. রহমান" in out[0], out


def test_number_currency():
    # Preserve mode: glued currency stays byte-identical (no টাকা/১,০০০ flip).
    out = seg("আজ ৫০ শতাংশ ছাড় আছে। দাম ৳১,০০০।")
    assert len(out) == 2, out
    assert "৫০ শতাংশ" in out[0], out
    assert "৳১,০০০" in out[1], out
    # Natural mode: full verbalization with correct word order.
    from src.audiobook.langtext import naturalize_sentence
    assert naturalize_sentence("দাম ৳১,০০০।", "bn") == "দাম এক হাজার টাকা।"


def test_time_question():
    out = seg("আমি ৫:৩০-এ আসব। তুমি কি তখন থাকবে?")
    assert len(out) == 2, out


def test_date_time():
    out = seg("১২ সেপ্টেম্বর ২০২৬ তারিখে অনুষ্ঠান হবে। সময় বিকেল ৫টা।")
    assert len(out) == 2, out


def test_list_markers():
    out = seg("১. বাংলা\n২. হিন্দি\n৩. ইংরেজি")
    joined = " ".join(out)
    assert "১." in joined and "২." in joined and "৩." in joined, out


def test_legal():
    out = seg("ধারা ৫(২)(ক) অনুযায়ী আবেদন করতে হবে। পরে কর্তৃপক্ষ সিদ্ধান্ত নেবে।")
    assert len(out) == 2, out
    assert "৫(২)(ক)" in out[0], out


def test_academic():
    out = seg("পৃ. ১২-এ এই বিষয়টি আলোচনা করা হয়েছে। পরে আরও উদাহরণ দেওয়া হয়েছে।")
    assert len(out) == 2, out


def test_literary_ellipsis():
    out = seg("হঠাৎ সে থেমে গেল... তারপর ধীরে বলল, “আমি জানি।”")
    assert len(out) == 2, out
    assert "..." in out[0], out


def test_dialogue():
    out = seg("—তুমি যাবে?\n—হ্যাঁ, আমি যাব।")
    assert len(out) == 2, out
    assert out[0].startswith("—"), out


def test_repetition():
    out = seg("না না, আমি যাব না!")
    assert len(out) == 1, out
    assert "না না" in out[0], out


def test_interjection():
    out = seg("আহা! কী সুন্দর দৃশ্য!")
    assert len(out) == 2, out


def test_mixed_script_version():
    out = seg("আমি আজ Python 3.12 ব্যবহার করেছি। Version 3.13 এখনো ইনস্টল করিনি।")
    assert len(out) == 2, out
    assert "3.12" in out[0] and "3.13" in out[1], out


def test_url():
    out = seg("আমি https://example.com দেখেছি। সাইটটি ভালো।")
    assert len(out) == 2, out
    assert "https://example.com" in out[0], out


def test_email():
    out = seg("test@example.com-এ মেইল করুন। তারপর আমাকে জানান।")
    assert len(out) == 2, out
    assert "test@example.com" in out[0], out


def test_phone():
    out = seg("আমার নম্বর +৮৮০১৭১২৩৪৫৬৭৮। প্রয়োজনে ফোন করবেন।")
    assert len(out) == 2, out
    assert "+৮৮০১৭১২৩৪৫৬৭৮" in out[0], out


# --- Spec §438-445: false-split guards ----------------------------------

def test_en_abbrev():
    out = seg("Dr. Rahman এলেন।", lang="en")
    assert len(out) == 1 and out[0].startswith("Dr. Rahman"), out


def test_bn_abbrev():
    out = seg("ড. রহমান এলেন।")
    assert len(out) == 1 and out[0].startswith("ড. রহমান"), out


def test_decimal():
    out = seg("মূল্য ৩.১৪ টাকা। ভালো দাম।")
    assert len(out) == 2, out
    assert "৩.১৪" in out[0], out


def test_url_period():
    out = seg("দেখুন example.com। ভালো সাইট।")
    assert "example.com" in out[0], out


def test_version_period():
    out = seg("সংস্করণ ২.১.৩। নতুন ফিচার।")
    assert len(out) == 2, out
    assert "২.১.৩" in out[0], out


# --- Currency distinction (spec: ৳ distinct from ₹) ---------------------

def test_currency_distinct():
    t, ctx = protect_all("দাম ৫০৳ এবং ৫০₹।", "bn")
    t = normalize_text(t, "bn")
    sents = [finalize_text(s, ctx, where="test") for s in segment_sentences(t, "bn")]
    joined = " ".join(sents)
    assert "টাকা" in joined and "রুপি" in joined, joined


# --- Locales --------------------------------------------------------------

def test_locale_normalization():
    assert normalize_locale("bn-BD") == "bn"
    assert normalize_locale("bn_BD") == "bn"
    assert normalize_locale("bn-IN") == "bn"
    assert normalize_locale("BN") == "bn"
    assert normalize_locale("xx") == "en"
    assert normalize_locale(None) == "en"


# --- Unicode / grapheme safety --------------------------------------------

def test_conjuncts_survive():
    src = "ক্ষতিপূরণ জ্ঞান শ্রমিক স্ত্রী চন্দ্রবিন্দু ঙ্ক। শেষ।"
    out = seg(src)
    assert "ক্ষতিপূরণ" in out[0] and "জ্ঞান" in out[0], out
    assert "ন্দ্র" in out[0], out  # chandrabindu + conjunct intact


def test_nukta_chandrabindu():
    out = seg("চাঁদ উঠেছে। হাঁস ডাকছে। গাড়ি চলছে।")
    assert len(out) == 3, out
    assert "চাঁদ" in out[0] and "হাঁস" in out[1] and "গাড়ি" in out[2], out


def test_emoji_preserved():
    out = seg("আমি ভালোবাসি ❤️ বাংলা। শেষ।")
    assert "❤️" in out[0], out


def test_newlines_preserved():
    src = "প্রথম লাইন।\n\nদ্বিতীয় লাইন।"
    ptext, ctx = protect_all(src, "bn")
    ptext = normalize_text(ptext, "bn")
    assert ptext.count("\n") == 2, repr(ptext)


def test_character_tag_preserved():
    src = "[রহিম] ড. রহমান বললেন, “আমি আসব।”"
    ptext, ctx = protect_all(src, "bn")
    assert "[রহিম]" in restore_all(ptext, ctx), ptext


# --- Leak / loss / determinism ---------------------------------------------

def test_no_placeholder_leak():
    nasty = "ড. রহমান https://x.com/a.b test@y.z ৩.১৪ ১. item ৫০% ৳৫ [রহিম] শেষ।"
    ptext, ctx = protect_all(nasty, "bn")
    assert re.search(r"[\ue000-\uE01F]", ptext), "expected placeholders during staging"
    done = finalize_text(ptext, ctx, where="test")
    assert not re.search(r"[\ue002\ue010-\ue017]\d+\ue001", done), done


def test_source_pua_escaped():
    src = "a\ue000b। শেষ।"
    ptext, ctx = protect_all(src, "bn")
    done = finalize_text(ptext, ctx, where="test")
    assert "\ue000" in done and "." not in done.replace("।", "").replace("শেষ", ""), repr(done)


def test_text_loss_roundtrip():
    srcs = [
        "ড. রহমান আজ ৫০% ছাড়ে laptop কিনেছেন। দাম ছিল ৳১,০০০।",
        "[রহিম] তিনি বললেন, “আমি ৫:৩০-এ আসব।”",
        "API v2.1.3 দেখুন https://example.com/x। ঠিক আছে।",
    ]
    for src in srcs:
        ptext, ctx = protect_all(src, "bn")
        # Every protected span must restore byte-identical.
        for ph, orig in ctx.mapping.items():
            assert orig in src or ph.startswith("\ue002"), (ph, orig)
        done = finalize_text(ptext, ctx, where="test")
        # Stripping only added normalize spaces; words must all survive.
        for w in src.split():
            core = w.strip(".,;:!?।“”‘’\"'()[]")
            if core and not re.fullmatch(r"[.]+", core):
                assert core in done or core.replace("%", "শতাংশ") in done or \
                    core in ("৫০%", "৳১,০০০", "৫:৩০-এ"), (w, done)


def test_determinism():
    src = "ড. রহমান https://example.com ৩.১৪ ৫০% ৳৫ [রহিম] শেষ। আরেকটা।"
    outs = set()
    for _ in range(20):
        ptext, ctx = protect_all(src, "bn")
        outs.add(finalize_text(segment_sentences(normalize_text(ptext, "bn"), "bn")[0], ctx))
    assert len(outs) == 1, outs


def test_empty_and_pause_safety():
    assert segment_sentences("", "bn") == []
    assert segment_sentences("   ", "bn") == []
    out = seg("!!! ???")
    assert all(s.strip() for s in out), out


# --- Natural mode (opt-in spoken expansion, default pipeline untouched) ---

def _nat(text):
    from src.audiobook.langtext import naturalize_sentence
    return naturalize_sentence(text, "bn")


def test_natural_off_by_default():
    from src.audiobook.langtext import protect_all, normalize_text, finalize_text
    ptext, ctx = protect_all("দাম ৫০ টাকা।", "bn")
    done = finalize_text(ptext, ctx, where="test")
    assert "৫০" in done and "পঞ্চাশ" not in done, done


def test_natural_integers():
    assert _nat("মোট ৪৫ জন এলো।") == "মোট পঁয়তাল্লিশ জন এলো।"
    assert _nat("দাম ১,০০,০০০ টাকা।") == "দাম এক লাখ টাকা।"
    assert _nat("জনসংখ্যা 1000000।") == "জনসংখ্যা দশ লাখ।"


def test_natural_ordinals():
    assert _nat("সে ১ম হলো।") == "সে প্রথম হলো।"
    assert _nat("সে ২রা এলো।") == "সে দোসরা এলো।"
    assert _nat("২৫তম বার্ষিকী।") == "পঁচিশতম বার্ষিকী।"


def test_natural_classifiers():
    assert _nat("সে ৫টা বই কিনল।") == "সে পাঁচটা বই কিনল।"
    assert _nat("৪জন এলো।") == "চারজন এলো।"


def test_natural_currency_paise():
    assert _nat("দাম ৳১,২৫০.৫০।") == "দাম এক হাজার দুই শত পঞ্চাশ টাকা পঞ্চাশ পয়সা।"


def test_natural_time_date():
    assert _nat("বিকেল ৫:৩০-এ আসব।") == "বিকেল সাড়ে পাঁচটা-এ আসব।"
    assert _nat("তারিখ ১২/০৯/২০২৬।") == "তারিখ বারো সেপ্টেম্বর দুই হাজার ছাব্বিশ।"
    assert _nat("অনুপাত ৩:২।") == "অনুপাত তিন বনাম দুই।"
    # Dotted versions are never partially verbalized (no ২.এক-style corruption).
    assert _nat("সংস্করণ ২.১.৩ প্রকাশিত।") == "সংস্করণ ২.১.৩ প্রকাশিত।"
    assert _nat("মূল্য ৩.১৪ টাকা।") == "মূল্য তিন টাকা চৌদ্দ পয়সা।"


def test_natural_clock_words():
    from src.audiobook.bn_numbers import verbalize_time as _vt
    assert _vt("১:৩০") == "দেড়টা"
    assert _vt("২:৩০") == "সাড়ে দুইটা"
    assert _vt("১:১৫") == "সোয়া একটা"
    assert _vt("১:৪৫") == "পৌনে দুইটা"
    assert _vt("৫:০০") == "পাঁচটা"
    assert _vt("১৩:৩০") == "সাড়ে একটা"
    assert _vt("২৩:৪৫") == "পৌনে বারোটা"
    assert _vt("১২:০০") == "বারোটা"
    assert _vt("৫:২০") == "পাঁচটা বিশ"
    assert _vt("২৫:০০") is None
    assert _vt("৩.১৪") is None


def test_natural_acronym_unit():
    assert _nat("এই API কাজ করে।") == "এই এপিআই কাজ করে।"
    assert _nat("ওজন ৫kg।") == "ওজন পাঁচ কিলোগ্রাম।"
    assert _nat("Zoom meeting আছে।") == "Zoom meeting আছে।"  # unknown left alone


def test_natural_negative_range():
    assert _nat("তাপমাত্রা -৫।") == "তাপমাত্রা মাইনাস পাঁচ।"
    assert "থেকে" not in _nat("পৃষ্ঠা ৫-১০ পড়ো।")  # ambiguous hyphen preserved
    assert _nat("পৃষ্ঠা ৫-১০ পড়ো।") == "পৃষ্ঠা পাঁচ-দশ পড়ো।"


def test_natural_determinism():
    a = _nat("ড. রহমান ৫০% ছাড়ে ৩টি বই ৳১,০০০ দিয়ে কিনলেন।")
    b = _nat("ড. রহমান ৫০% ছাড়ে ৩টি বই ৳১,০০০ দিয়ে কিনলেন।")
    assert a == b
    assert "ড." in a and "%" not in a and "৩" not in a


# --- Grapheme / Unicode safety ---------------------------------------------

def test_grapheme_split():
    from src.audiobook.langtext import split_graphemes
    g = split_graphemes("ক্ষতিপূরণ")
    assert "".join(g) == "ক্ষতিপূরণ", g
    # conjunct ক্ষ (ক+্+ষ) must stay one cluster, not three code points
    assert len(g) < len("ক্ষতিপূরণ"), g
    assert split_graphemes("") == []


def test_zwj_preserved():
    from src.audiobook.langtext import protect_all, restore_all
    src = "র‍্যাম ক্‌ষ। শেষ।"  # contains ZWJ/ZWNJ
    ptext, ctx = protect_all(src, "bn")
    done = restore_all(ptext, ctx)
    assert "\u200d" in done and "\u200c" in done, repr(done)


def test_bom_stripped_nfc_stable():
    from src.audiobook.langtext import protect_all, restore_all
    src = "﻿আমি এলাম।"  # leading BOM
    ptext, ctx = protect_all(src, "bn")
    done = restore_all(ptext, ctx)
    assert not done.startswith("﻿"), repr(done)
    assert "আমি এলাম।" in done, repr(done)


def test_letter_list_roman_footnote():
    from src.audiobook.langtext import protect_all, normalize_text, segment_sentences, finalize_text
    for src, must in [
        ("ক. বাংলা\nখ. ইংরেজি।", ["ক.", "খ."]),
        ("iv. ভূমিকা। পরের অংশ।", ["iv."]),
        ("দেখুন [১]। শেষ।", ["[১]"]),
        ("টেস্ট #বইমেলা শুরু। শেষ।", ["#বইমেলা"]),
    ]:
        ptext, ctx = protect_all(src, "bn")
        sents = [finalize_text(s, ctx, where="test")
                 for s in segment_sentences(normalize_text(ptext, "bn"), "bn")]
        joined = " ".join(sents)
        for m in must:
            assert m in joined, (src, sents)


def test_ellipsis_char_boundary():
    from src.audiobook.langtext import protect_all, normalize_text, segment_sentences, finalize_text
    ptext, ctx = protect_all("সে থামল… তারপর বলল।", "bn")
    sents = [finalize_text(s, ctx, where="test")
             for s in segment_sentences(normalize_text(ptext, "bn"), "bn")]
    assert len(sents) == 2, sents


def test_nested_quotes():
    from src.audiobook.langtext import protect_all, normalize_text, segment_sentences, finalize_text
    ptext, ctx = protect_all("তিনি বললেন, “সে বলল, ‘আমি আসব।’”।", "bn")
    sents = [finalize_text(s, ctx, where="test")
             for s in segment_sentences(normalize_text(ptext, "bn"), "bn")]
    assert len(sents) == 1, sents
    assert "‘আমি আসব।’" in sents[0], sents


def test_unknown_token_report():
    from src.audiobook.langtext import find_unknown_tokens
    rep = find_unknown_tokens("API ও GPU ঠিক আছে। যোগাযোগ Foo.Bar sn@il।", "bn")
    assert "API" in rep["acronyms"] and "GPU" in rep["acronyms"], rep
    assert any("Foo.Bar" in d for d in rep["dotted"]), rep


def test_year_style():
    from src.audiobook.langtext import naturalize_sentence as N
    assert N("১৯৮৭ সালে পুরস্কার।") == "উনিশশো সাতাশি সালে পুরস্কার।"
    assert N("২০২৬ সাল।") == "দুই হাজার ছাব্বিশ সাল।"
    assert N("১৮৫৭ সালে যুদ্ধ।") == "আঠারোশো সাতান্ন সালে যুদ্ধ।"
    # No year word -> quantity reading preserved (no false year-style).
    assert N("মোট ১৯৮৭টি বই।") == "মোট এক হাজার নয় শত সাতাশিটি বই।"


def test_initials_protected():
    from src.audiobook.langtext import protect_all, restore_all
    ptext, ctx = protect_all("M. A. Rahman এলেন। তারপর গেলেন।", "bn")
    assert "M." in ctx.mapping.get(
        next(k for k in ctx.mapping if "M." in ctx.mapping[k]), ""), ctx.mapping
    done = restore_all(ptext, ctx)
    assert "M. A. Rahman" in done, done


def test_control_chars_sanitized():
    from src.audiobook.langtext import protect_all, restore_all
    ptext, ctx = protect_all("আমি\x07এলাম। শেষ।", "bn")
    done = restore_all(ptext, ctx)
    assert "\x07" not in done and "আমি" in done and "এলাম" in done, repr(done)


def test_rule_trace():
    from src.audiobook.langtext import naturalize_sentence
    trace = []
    out = naturalize_sentence("দাম ৫০%।", "bn", trace=trace)
    assert "পঞ্চাশ শতাংশ" in out, out
    assert any(t[0] == "BN_PERCENT_001" for t in trace), trace
    # Preserve-mode equivalent collects nothing.
    trace2 = []
    out2 = naturalize_sentence("কোনো নম্বর নেই।", "bn", trace=trace2)
    assert trace2 == [], trace2


if __name__ == "__main__":
    import traceback
    fns = sorted(
        (n, f) for n, f in list(globals().items())
        if n.startswith("test_") and callable(f)
    )
    print(f"Running {len(fns)} golden tests (no model, no network)...")
    failed = 0
    for n, f in fns:
        try:
            f()
            print(f"  PASS {n}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL {n}: {e}")
            traceback.print_exc(limit=3)
    print(f"{len(fns)-failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
