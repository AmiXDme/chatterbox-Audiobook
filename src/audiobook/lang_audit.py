"""Bangla TN audit tool: shows RAW -> PROTECTED -> SEGMENTED -> RESTORED.

Usage:
    python3 src/audiobook/lang_audit.py <file.txt> [--locale bn-BD] [--natural]

No model, no GPU, no network. Exit code 0 = clean, 1 = leak detected.
--natural also shows opt-in spoken expansion (numbers/dates/currency).
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.audiobook.langtext import (
    protect_all,
    restore_all,
    assert_no_placeholders,
    normalize_locale,
    normalize_text,
    segment_sentences,
    find_unknown_tokens,
)


def audit_file(path: str, locale: str, natural: bool = False) -> int:
    with open(path, encoding="utf-8-sig") as f:
        raw = f.read()
    print(f"locale: {locale} -> profile: {normalize_locale(locale)} | natural={natural}")
    print(f"input: {len(raw)} chars, {len(raw.split())} words")
    print("=" * 70)

    t0 = time.time()
    protected, ctx = protect_all(raw, locale, natural=natural)
    t_protect = time.time() - t0
    print(f"--- PROTECTED TOKENS ({t_protect:.2f}s) ---")
    if not ctx.mapping:
        print("(none)")
    for ph, orig in ctx.mapping.items():
        print(f"  {ph.encode('unicode_escape').decode()}  <-  {orig[:60]!r}")

    t0 = time.time()
    # Natural mode skips surface symbol expansion so glued forms (৳১,০০০)
    # reach the verbalizer intact instead of pre-split (টাকা ১,০০০).
    normalized = normalize_text(protected, locale, symbols=not natural)
    t_norm = time.time() - t0
    t0 = time.time()
    sentences = segment_sentences(normalized, locale)
    t_seg = time.time() - t0
    print(f"--- SEGMENTED: {len(sentences)} sentence(s) "
          f"(normalize {t_norm:.2f}s, segment {t_seg:.2f}s) ---")
    t0 = time.time()
    out = []
    all_trace = []
    long_sents = []
    # finalize() would auto-naturalize via ctx.natural; disable that here so
    # the explicit traced call below is the single expansion point (no doubles).
    ctx.natural = False
    for i, s in enumerate(sentences, 1):
        from src.audiobook.langtext import finalize_text, naturalize_sentence
        trace = []
        restored = finalize_text(s, ctx, where=f"audit-sentence-{i}")
        if natural:
            traced = naturalize_sentence(restored, "bn", trace=trace)
            all_trace.extend((i, r, b, a) for r, b, a in trace)
            restored = traced
        out.append(restored)
        nwords = len(restored.split())
        if nwords > 60:
            long_sents.append((i, nwords))
        mark = " [natural]" if natural and restored != restore_all(s, ctx) else ""
        print(f"  [{i}]{mark} {restored[:100]!r}")
    print(f"--- RESTORED OK in {time.time()-t0:.2f}s: no placeholder leaks ---")
    if natural:
        print("--- RULE TRACE (rule_id: before -> after) ---")
        if not all_trace:
            print("(no natural expansions fired)")
        for i, r, b, a in all_trace[:30]:
            print(f"  [{i}] {r}: {b!r} -> {a!r}")
    if long_sents:
        print("--- ADVISORY: sentences over 60 words (slow TTS, consider splitting) ---")
        for i, nwords in long_sents[:10]:
            print(f"  sentence [{i}]: {nwords} words")

    full = "\n".join(out)
    assert_no_placeholders(full, where="audit-full")

    print("--- UNKNOWN TOKENS (advisory: possible acronyms/abbrevs/symbols) ---")
    rep = find_unknown_tokens(raw, locale)
    if not any(rep.values()):
        print("(none)")
    for kind, items in rep.items():
        for tok in items[:15]:
            print(f"  {kind}: {tok!r}")

    print("--- INPUT SUMMARY ---")
    print(f"chars={len(raw)} words={len(raw.split())} sentences={len(sentences)} "
          f"protected={len(ctx.mapping)}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Audit Bangla TN stages for a text file.")
    ap.add_argument("file", help="UTF-8 text file to audit")
    ap.add_argument("--locale", default="bn-BD", help="bn, bn-BD, bn-IN, en, hi (default: bn-BD)")
    ap.add_argument("--natural", action="store_true",
                    help="also show opt-in spoken expansion (numbers/dates/currency)")
    args = ap.parse_args(argv)
    try:
        return audit_file(args.file, args.locale, natural=args.natural)
    except ValueError as e:
        print(f"AUDIT FAILED: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
