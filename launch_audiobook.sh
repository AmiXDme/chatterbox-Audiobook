#!/bin/bash
set -e

# Always run from this script's own directory (works via symlink, file manager, any CWD)
cd "$(dirname "$0")"

echo "Checking for virtual environment..."
if [ ! -f "venv/bin/activate" ]; then
    echo "ERROR: Virtual environment not found!"
    echo "Please run ./install-audiobook.sh first to set up the environment."
    echo ""
    echo "Make sure you're in the chatterbox repository directory."
    exit 1
fi

echo "Checking repository structure..."
if [ ! -f "gradio_tts_app_audiobook.py" ]; then
    echo "ERROR: gradio_tts_app_audiobook.py not found!"
    echo "Please make sure you're in the chatterbox repository root."
    exit 1
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo ""
echo "Starting Chatterbox TTS Audiobook Edition..."
echo "Features: Voice Library, Character Management, Audiobook Tools"
echo "Audio Cleaning Available in 'Clean Samples' Tab"
echo "This may take a moment to load the models..."
echo ""
echo "Current directory: $(pwd)"
echo "Python environment: ${VIRTUAL_ENV:-not set}"
echo "Voice library will be created at: $(pwd)/voice_library"
echo ""

python3 -u gradio_tts_app_audiobook.py

echo ""
echo "Chatterbox TTS Audiobook Edition has stopped."
echo "Deactivating virtual environment..."
deactivate 2>/dev/null || true
echo ""
echo "Thanks for using Chatterbox TTS Audiobook Edition!"
echo "Your voice profiles are saved in the voice_library folder."
echo "Audio cleaning features are in the 'Clean Samples' tab!"
