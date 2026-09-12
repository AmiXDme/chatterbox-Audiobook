#!/bin/bash
set -e

# Always run from this script's own directory (works via symlink, file manager, any CWD)
cd "$(dirname "$0")"

if [ ! -f "venv/bin/activate" ]; then
    echo "ERROR: Virtual environment not found!"
    echo "Please run ./install-audiobook.sh first to set up the environment."
    exit 1
fi

source venv/bin/activate
echo "Starting Chatterbox Terminal (bn/en: quick, single, multi + watch)..."
echo "No browser needed. Ctrl+C cancels jobs (partial work is kept)."
echo ""
python3 -u terminal_app.py
