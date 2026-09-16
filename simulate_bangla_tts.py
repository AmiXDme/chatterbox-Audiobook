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

    print('=' * 68)
    print('🎭 SIMULATED BANGLA TTS — full pipeline, role by role')
    print('=' * 68)

    print('\n📤 [WHAT YOU GAVE]  (your raw text, untouched)')
    print('   ', text.replace('\n', '↵\n    '))
    print('   └─ digits, symbols (৳ % :) [Name] tags, line breaks')

    repaired = P.unicode_repair_bangla(text)
    if repaired != text:
        print('\n🔧 [CHATTERBOX ENGINE — STEP 1]  unicode_repair_bangla()')
        print(
            '   repairs broken যুক্তাক্ষর/grapheme clusters so the synthesizer '
            'pronounces every conjunct')
        print('   ', repaired.replace('\n', '↵\n    '))

    normalized = P.bangla_normalize_text(text)
    print('\n🔧 [CHATTERBOX ENGINE — STEP 2]  bangla_normalize_text()   (no AI)')
    print('   numbers → words  • dates/times/currency/%/years → natural Bangla')
    print('   • possessive/locative glued (টাকার, সাতটায়, সেপ্টেম্বরে)')
    if normalized != text:
        print('   ', normalized.replace('\n', '↵\n    '))
    else:
        print('   (no changes needed — already pure Bangla words)')

    print('\n🧠 [AI GEMINI — project Master-Prompt]   (live, optional)')
    print('   receives ONLY the letter text above; rewrites into natural, ')
    print('   story-toned narration, keeps [Name] tags, never re-numbers.')
    print('   ├─ simulation mode: Gemini call skipped (offline)')
    print('   ├─ live mode: runs on the engine output + shows live panel')
    print('   └─ output feeds the chunker below')

    print('\n📚 [CHATTERBOX CHUNKER]  bangla_chunk_text(max_tokens=850)')
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

    print('\n🎙️ [TTS MODEL — one pass per chunk]  xvector voice → T3 Bengali')
    print('   each chunk becomes 24 kHz Bengali speech (real run ≈ 6 s audio / min)')
    print('   voice routing: [কথক] → narrator voice, [রাহিম] → his voice')

    print(f'\n🕑 [PAUSE TIMELINE you would HEAR]  (total {total_pause:.2f}s of')
    print('   distributed: । .6s  ॥1.2s  \\n .15s  \\n\\n 1.0s)')
    for i, (ctxt, pause) in enumerate(steps, 1):
        chars = min(72, len(ctxt))
        p = int(pause * 20)
        print(f'   {i:02d} |{"▓" * chars}|{"·" * p}{pause:.2f}s')

    print(f'\n📦 [ASSEMBLY + DELIVERABLES]')
    print(f'   chunks           : {len(chunks)}')
    print(f'   total pause      : {total_pause:.2f}s')
    print(f'   est total tokens : {sum(token_estimate(c.get("text", "")) for c in chunks)}')
    print('   output           : per-chunk WAVs + stitched full audio')
    print('                     -> audiobook_projects/<project>/  (single voice)')
    print('                     -> per-[Name] voice folders             (multi-voice)')


if __name__ == '__main__':
    main()