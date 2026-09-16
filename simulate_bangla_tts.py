#!/usr/bin/env python
"""Simulated Bangla TTS preview: what the real pipeline will say.

Shows the deterministic normalization, the Gemini-input text, the chunk plan,
and the pause timeline for any Bangla text — before a single second of real
model audio is generated. Run:

    venv/bin/python simulate_bangla_tts.py "আপনার পাঠ্য এখানে..."
    echo "বাংলা লেখা" | venv/bin/python simulate_bangla_tts.py
"""
import sys

sys.path.insert(0, 'src')
from audiobook import processing as P


def token_estimate(txt):
    return max(80, int(len(txt) * 1.35))


def main():
    if len(sys.argv) > 1:
        text = sys.argv[1]
    else:
        text = sys.stdin.read().strip()
        if not text:
            print('Usage: simulate_bangla_tts.py "বাংলা টেক্সট"  (or pipe stdin)')
            return

    print('=' * 64)
    print('🎭 SIMULATED BANGLA TTS PIPELINE')
    print('=' * 64)

    print('\n[1] RAW INPUT')
    print('   ', text.replace('\n', '↵\n    '))

    repaired = P.unicode_repair_bangla(text)
    if repaired != text:
        print('\n[1b] UNICODE REPAIR (যুক্তাক্ষর)')
        print('   ', repaired.replace('\n', '↵\n    '))

    normalized = P.bangla_normalize_text(text)
    if normalized != text:
        print('\n[2] GRAMMAR + DIGIT NORMALIZATION (what Gemini receives)')
        print('   ', normalized.replace('\n', '↵\n    '))

    print('\n[3] CHUNK PLAN (grammar-aware)')
    chunks = P.bangla_chunk_text(normalized, max_tokens=850)
    total_pause = 0.0
    steps = []
    for i, ck in enumerate(chunks, 1):
        ctxt = ck.get('text', '')
        pause = float(ck.get('pause_before') or 0.0)
        total_pause += pause
        cue = ''
        if pause == 0.15:
            cue = 'soft breath \\n'
        elif pause == 0.6:
            cue = 'sentence ।'
        elif pause == 1.0:
            cue = 'paragraph \\n\\n'
        elif pause == 1.2:
            cue = 'verse ॥'
        print(f'   Chunk {i:02d}  est {token_estimate(ctxt):>4} tok  '
              f'pause_before {pause:>4.2f}s  {cue}')
        print(f'        {ctxt[:110]}{"…" if len(ctxt) > 110 else ""}')
        steps.append((ctxt, pause))

    print(f'\n[4] PAUSE TIMELINE  (total distributed: {total_pause:.2f}s)')
    w = 1.0  # ~1 char per 1.0s of speech; pause scaled 20x for visibility
    for i, (ctxt, pause) in enumerate(steps, 1):
        chars = min(72, len(ctxt))
        p = int(pause * 20)
        print(f'   {i:02d} |{"▓" * chars}|{"·" * p}{pause:.2f}s')

    print(f'\n[5] SUMMARY')
    print(f'   chunks           : {len(chunks)}')
    print(f'   total pause      : {total_pause:.2f}s  '
          '(।.6s  ॥1.2s  \\n.15s  \\n\\n1.0s)')
    print(f'   est total tokens : {sum(token_estimate(c.get("text", "")) for c in chunks)}')


if __name__ == '__main__':
    main()