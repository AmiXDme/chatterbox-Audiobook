#!/usr/bin/env python3
"""Terminal twin of the Gradio UI — Bangla + English only.

Covers: quick TTS, single-voice audiobook, multi-voice audiobook,
plus a watch-folder batch mode (surprise). No browser needed.

Architecture: ZERO copied logic. Everything below calls the exact same
functions the Gradio buttons call (imported from gradio_tts_app_audiobook),
so GUI and terminal share models, folders, projects, resume, and realtime
terminal logs. Nothing in the main app was modified for this file.
"""

import os
import re
import shutil
import subprocess
import sys
import time

print("Loading engines (same startup as the app, ~20-90s)...", flush=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# NOTE: importing the main module builds (but never launches) the Gradio UI.
from gradio_tts_app_audiobook import (
    generate,
    create_audiobook_with_volume_settings,
    create_multi_voice_audiobook_with_volume_settings,
    get_voice_choices,
    load_voice_for_tts,
    get_voice_config,
    SAVED_VOICE_LIBRARY_PATH,
    TTS_CANCEL_EVENT,
    TTS_PAUSE_EVENT,
)
from src.audiobook.langtext import is_voice_tag
import soundfile as sf

MODEL = None  # lazy singleton lives inside the engines, like the GUI
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "terminal_output")
os.makedirs(OUT_DIR, exist_ok=True)
LANGS = (("English", "en"), ("Bengali (Bangla)", "bn"))


# ---------- small helpers ----------

def pick(options, prompt):
    """options: [(display, value)]. Returns chosen value (None allowed)."""
    for i, (display, _value) in enumerate(options, 1):
        print(f"  {i}. {display}")
    while True:
        raw = input(f"{prompt} [1-{len(options)}]: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1][1]
        print("  Invalid choice, try again.")


def ask_yn(prompt, default=False):
    hint = "Y/n" if default else "y/N"
    raw = input(f"{prompt} [{hint}]: ").strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes")


def clean_path(raw):
    """Normalize a pasted/drag-dropped path: strip quotes, unescape spaces."""
    p = (raw or "").strip()
    if len(p) >= 2 and p[0] == p[-1] and p[0] in ("'", '"'):
        p = p[1:-1]
    p = p.replace("\\ ", " ")
    return os.path.expanduser(p)


def ask_text(prompt):
    """Type text directly, @/path/to/file.txt, or drag-and-drop a .txt file
    (dropped paths are detected automatically, @ optional)."""
    raw = input(f"{prompt} (type, @file.txt, or drag-and-drop): ").strip()
    if raw.startswith("@"):
        path = clean_path(raw[1:])
        with open(path, encoding="utf-8-sig") as f:
            print(f"  Loaded {os.path.basename(path)}")
            return f.read()
    maybe = clean_path(raw)
    if maybe.lower().endswith(".txt") and os.path.isfile(maybe):
        with open(maybe, encoding="utf-8-sig") as f:
            print(f"  Loaded {os.path.basename(maybe)}")
            return f.read()
    return raw


def choose_language():
    print("Language:")
    labels = [f"{name} ({code})" for name, code in LANGS]
    idx = pick(list(zip(labels, [c for _, c in LANGS])), "Language")
    return idx if idx else "en"


def choose_voice(profile_only=False):
    """Returns (profile_name_or_None, audio_path, exag, cfg, temp, display)."""
    choices = get_voice_choices(SAVED_VOICE_LIBRARY_PATH)
    if profile_only:
        choices = [(d, v) for d, v in choices if v]
    print("Voice:")
    name = pick(choices, "Voice")
    if not name:
        audio = clean_path(input("Audio file path (or drag-and-drop): ").strip())
        if not os.path.isfile(audio):
            raise SystemExit(f"❌ File not found: {audio}")
        return None, audio, 0.5, 0.5, 0.8, os.path.basename(audio)
    audio, exag, cfg, temp, _comp, status = load_voice_for_tts(SAVED_VOICE_LIBRARY_PATH, name)
    print(f"  {status}")
    if not audio:
        raise SystemExit("❌ Voice has no audio file.")
    cfgd = get_voice_config(SAVED_VOICE_LIBRARY_PATH, name) or {}
    return name, audio, exag, cfg, temp, cfgd.get("display_name", name)


def play(path):
    for player, args in (("ffplay", ["-nodisp", "-autoexit", "-loglevel", "quiet"]),
                         ("mpv", ["--no-video"]), ("aplay", [])):
        if shutil.which(player):
            print(f"  Playing via {player} (Ctrl+C skips)...")
            try:
                subprocess.run([player, *args, path], check=False)
            except KeyboardInterrupt:
                print("  (playback skipped)")
            return
    print(f"  No CLI player found — play it yourself: {path}")


def stamp(name):
    safe = re.sub(r"[^\w\-]+", "_", name).strip("_")[:40] or "out"
    return os.path.join(OUT_DIR, f"{safe}_{time.strftime('%Y%m%d_%H%M%S')}.wav")


def drain(gen, prefix=""):
    """Drive a streaming engine generator, printing live statuses. Returns last yield."""
    last = None
    for item in gen:
        last = item
        try:
            status = item[1] if len(item) > 1 else ""
        except Exception:
            status = ""
        if status:
            line = str(status).replace("\n", " | ")
            print(f"  {prefix}{line[:160]}", flush=True)
    return last


# ---------- flows (same engines as the GUI buttons) ----------

def flow_quick():
    print("\n--- Quick speech ---")
    text = ask_text("Text")
    if not text.strip():
        print("Empty text, cancelled.")
        return
    lang = choose_language()
    _name, audio, exag, cfg, temp, display = choose_voice()
    print(f"Generating ({display}, {lang})...")
    try:
        result = generate(MODEL, text, audio, exag, temp, 0, cfg,
                          language_id=lang, progress=None)
    except KeyboardInterrupt:
        print("\n🛑 Cancelled.")
        return
    if not result:
        print("❌ No audio produced.")
        return
    sr, arr = result
    path = stamp(f"quick_{lang}")
    sf.write(path, arr, sr)
    print(f"✅ Saved: {path} ({len(arr)/sr:.1f}s)")
    if ask_yn("Play it?", default=True):
        play(path)


def flow_single():
    print("\n--- Single-voice audiobook ---")
    text = ask_text("Text")
    if not text.strip():
        print("Empty text, cancelled.")
        return
    lang = choose_language()
    voice_name, _audio, _e, _c, _t, display = choose_voice(profile_only=True)
    print(f"Book voice: {display}")
    project = input("Project name: ").strip() or f"book_{time.strftime('%Y%m%d_%H%M%S')}"
    natural = ask_yn("Natural spoken numbers (Bengali digits→words)?", default=False)
    print("Generating — Ctrl+C cancels (partial chunks kept, resume anytime)...")
    try:
        last = drain(create_audiobook_with_volume_settings(
            MODEL, text, SAVED_VOICE_LIBRARY_PATH, voice_name, project,
            enable_norm=True, target_level=-18.0, language_id=lang,
            progress=None, natural=natural), prefix="")
    except KeyboardInterrupt:
        TTS_CANCEL_EVENT.set()
        print("\n🛑 Cancel requested — stopping after current chunk. Resume via this menu later.")
        return
    if last:
        print(f"Done: {str(last[1])[:300]}")


def flow_multi():
    print("\n--- Multi-voice audiobook ---")
    text = ask_text("Text with [Character] tags")
    if not text.strip():
        print("Empty text, cancelled.")
        return
    chars = []
    for m in re.finditer(r"\[([^\]]+)\]", text):
        name = m.group(1).strip()
        if is_voice_tag(name) and name not in chars:
            chars.append(name)
    if not chars:
        print("No [Character] tags found — use single-voice mode instead.")
        return
    print(f"Characters found: {', '.join(chars)}")
    lib = [v for _d, v in get_voice_choices(SAVED_VOICE_LIBRARY_PATH) if v]
    assignments = {}
    for ch in chars:
        print(f"Voice for [{ch}]:")
        assignments[ch] = pick([(f"🎭 {v}", v) for v in lib], f"Voice for [{ch}]")
    lang = choose_language()
    project = input("Project name: ").strip() or f"multi_{time.strftime('%Y%m%d_%H%M%S')}"
    natural = ask_yn("Natural spoken numbers (Bengali digits→words)?", default=False)
    print("Generating — Ctrl+C cancels (partial chunks kept)...")
    try:
        last = drain(create_multi_voice_audiobook_with_volume_settings(
            MODEL, text, SAVED_VOICE_LIBRARY_PATH, project, assignments,
            enable_norm=True, target_level=-18.0, language_id=lang,
            progress=None, natural=natural), prefix="")
    except KeyboardInterrupt:
        TTS_CANCEL_EVENT.set()
        print("\n🛑 Cancel requested — stopping after current chunk.")
        return
    if last:
        print(f"Done: {str(last[1])[:300]}")


def flow_watch():
    """SURPRISE: drop .txt files in a folder -> each becomes an audiobook."""
    print("\n--- Watch-folder batch (surprise mode) ---")
    folder = clean_path(input("Folder to watch (empty = ./terminal_inbox, or drag-and-drop): ").strip() or "./terminal_inbox")
    folder = os.path.abspath(folder)
    os.makedirs(folder, exist_ok=True)
    print(f"Drop .txt files into:\n  {folder}\nPress Enter to scan (Ctrl+C quits)...")
    try:
        input()
    except KeyboardInterrupt:
        return
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(".txt"))
    if not files:
        print("No .txt files found.")
        return
    print(f"Found {len(files)} file(s).")
    lang = choose_language()
    lib = [v for _d, v in get_voice_choices(SAVED_VOICE_LIBRARY_PATH) if v]
    print("Voice for all files:")
    voice_name = pick([(f"🎭 {v}", v) for v in lib], "Voice")
    for i, fn in enumerate(files, 1):
        path = os.path.join(folder, fn)
        with open(path, encoding="utf-8-sig") as f:
            text = f.read()
        project = os.path.splitext(fn)[0][:40]
        print(f"\n[{i}/{len(files)}] {fn} -> project '{project}'")
        try:
            drain(create_audiobook_with_volume_settings(
                MODEL, text, SAVED_VOICE_LIBRARY_PATH, voice_name, project,
                enable_norm=True, target_level=-18.0, language_id=lang,
                progress=None, natural=False), prefix="")
        except KeyboardInterrupt:
            TTS_CANCEL_EVENT.set()
            print("🛑 Batch stopped. Finished files are kept.")
            return
        TTS_CANCEL_EVENT.clear()
    print("\n✅ Batch complete.")


def main():
    print("=" * 60)
    print("  Chatterbox Terminal  (bn/en: quick, single, multi + watch)")
    print("  Same engines, models, voices and projects as the browser UI.")
    print("=" * 60)
    enc = sys.stdin.encoding or "unknown"
    print(f"  Terminal encoding: {enc}")
    if enc.lower().replace("-", "") not in ("utf8", "utf8sig"):
        print("  ⚠️ WARNING: terminal is not UTF-8 — Bengali input may arrive "
              "garbled. Run with: PYTHONUTF8=1 python3 terminal_app.py "
              "or: export LC_ALL=C.UTF-8")
    while True:
        print("\n1. Quick speech  2. Single-voice audiobook  3. Multi-voice audiobook")
        print("4. Watch-folder batch  5. Quit")
        try:
            choice = input("Choose [1-5]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            return
        try:
            if choice == "1":
                flow_quick()
            elif choice == "2":
                flow_single()
            elif choice == "3":
                flow_multi()
            elif choice == "4":
                flow_watch()
            elif choice == "5":
                print("Bye!")
                return
            else:
                print("Pick 1-5.")
        except SystemExit as e:
            print(e)
        except Exception:
            import traceback
            print("❌ Error (full traceback — paste this when reporting):")
            traceback.print_exc()
        finally:
            TTS_CANCEL_EVENT.clear()
            TTS_PAUSE_EVENT.clear()


if __name__ == "__main__":
    main()
