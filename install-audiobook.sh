#!/bin/bash
set -e

# Always run from this script's own directory (works via symlink, file manager, any CWD)
cd "$(dirname "$0")"

echo "========================================"
echo "  Chatterbox TTS - Installation Setup"
echo "========================================"
echo ""
echo "This will install Chatterbox TTS in a virtual environment"
echo "to keep it isolated from other Python projects."
echo ""
echo "Requirements:"
echo "- Python 3.10 or higher"
echo "- No GPU required (CPU-only mode)"
echo "- Git (if you want to pull updates)"
echo ""
echo "Current directory: $(pwd)"
echo ""
read -p "Press Enter to continue..."

echo ""
echo "[1/9] Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 is not installed or not in PATH"
    echo "Please install Python 3.10+ from https://python.org"
    exit 1
fi
python3 --version

echo ""
echo "[2/9] Checking if we're in the correct directory..."
if [ ! -f "pyproject.toml" ]; then
    echo "ERROR: pyproject.toml not found!"
    echo "Please make sure you're running this from the chatterbox repository root."
    exit 1
fi

if [ ! -d "src/chatterbox" ]; then
    echo "ERROR: src/chatterbox directory not found!"
    echo "Please make sure you're in the correct chatterbox repository."
    exit 1
fi

echo "Repository structure verified"

echo ""
echo "[3/9] Creating virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists. Removing old one..."
    rm -rf venv
fi

# Self-heal: Debian/Ubuntu/Mint ship python3 WITHOUT the venv module by default.
# Detect it and try to install automatically before failing.
if ! python3 -c "import venv, ensurepip" 2>/dev/null; then
    PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    echo "python3-venv module is missing (required to create virtual environments)."
    if command -v apt &> /dev/null && command -v sudo &> /dev/null; then
        echo "Attempting automatic install of python${PYVER}-venv (may ask for your password)..."
        sudo apt update && (sudo apt install -y "python${PYVER}-venv" || sudo apt install -y python3-venv)
    else
        echo "ERROR: automatic install not possible (no apt/sudo)."
        echo "Please install it manually, then re-run this script:"
        echo "  sudo apt install python${PYVER}-venv"
        exit 1
    fi
    if ! python3 -c "import venv, ensurepip" 2>/dev/null; then
        echo "ERROR: python3-venv is still missing."
        echo "Install it manually, then re-run this script:"
        echo "  sudo apt install python${PYVER}-venv"
        exit 1
    fi
    echo "python3-venv installed successfully."
fi

# Disk space sanity check (venv + torch + models need several GB)
if command -v df &> /dev/null; then
    FREE_KB=$(df -k . | awk 'NR==2 {print $4}')
    if [ "$FREE_KB" -lt 6291456 ]; then
        echo "WARNING: less than 6GB free disk space here. Install may fail."
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
fi

python3 -m venv venv || { echo "ERROR: virtual environment creation failed."; exit 1; }
echo "Virtual environment created."

echo ""
echo "[SWAP] Checking swap space (prevents out-of-memory kills on low-RAM machines)..."
SWAP_KB=$(free -k | awk '/^Swap:/ {print $2}')
if [ "${SWAP_KB:-0}" -gt 4000000 ]; then
    echo "Swap OK ($(free -h | awk '/^Swap:/ {print $2}') total) - nothing to do."
else
    echo "Little or no swap detected. On machines with <16GB RAM the app can be"
    echo "killed by the system (OOM killer) while loading/generating models."
    if command -v sudo &> /dev/null && [ -t 0 ]; then
        read -p "Create an 8GB swapfile now? (requires sudo) [Y/n] " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            if [ ! -f /swapfile ]; then
                if !(sudo fallocate -l 8G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile); then
                    echo "fallocate failed, trying dd instead (slower, a few minutes)..."
                    sudo dd if=/dev/zero of=/swapfile bs=1M count=8192 status=progress && sudo chmod 600 /swapfile && sudo mkswap /swapfile || echo "WARNING: swapfile creation failed - continuing without swap."
                fi
            fi
            sudo swapon /swapfile 2>/dev/null || true
            if ! grep -qs "^/swapfile " /etc/fstab; then
                echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab > /dev/null && echo "Swapfile added to /etc/fstab (survives reboots)."
            fi
            echo "Swap now: $(free -h | awk '/^Swap:/ {print $2}') total."
        fi
    else
        echo "Skipping automatic setup (no interactive sudo). To add swap manually later:"
        echo "  sudo fallocate -l 8G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile"
    fi
fi

echo ""
echo "[4/9] Activating virtual environment..."
source venv/bin/activate

echo ""
echo "[5/9] Upgrading pip..."
python -m pip install --upgrade pip
echo "Pinning setuptools (needed for perth/pkg_resources compatibility)..."
pip install "setuptools<81"

echo ""
echo "[6/9] Installing compatible PyTorch (CPU-only)..."
echo "This may take a while..."
echo "Installing PyTorch (CPU-only, latest compatible version)..."
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

echo ""
echo "[7/9] Installing Chatterbox TTS and dependencies..."
pip install -e .
pip install gradio qrcode

echo ""
echo "[8/9] Installing and configuring pydantic..."
echo "Installing pydantic..."
pip install pydantic

echo ""
echo "[9/9] Testing installation..."
echo "Testing PyTorch (CPU-only)..."
python -u -c "import torch; print('PyTorch version:', torch.__version__); print('CPU mode: OK')"

echo ""
echo "Testing Chatterbox import..."
python -u -c "from chatterbox.mtl_tts import ChatterboxMultilingualTTS as ChatterboxTTS; print('Chatterbox Multilingual TTS imported successfully!')"

echo ""
echo "Testing pydantic compatibility..."
python -u -c "import pydantic; print('Pydantic version:', pydantic.__version__)"

echo ""
echo "[10/10] Downloading multilingual TTS model files..."
echo "This downloads the Chatterbox Multilingual V3 model into models-multilingual/"
mkdir -p models-multilingual
python -u -c "
from huggingface_hub import hf_hub_download
files=['ve.pt','t3_mtl23ls_v3.safetensors','s3gen.pt','grapheme_mtl_merged_expanded_v1.json','conds.pt','Cangjie5_TC.json']
print(f'Downloading {len(files)} model files to models-multilingual/ (GBs, quiet per file)...', flush=True)
for f in files:
    print(f'Downloading {f} ...', flush=True)
    hf_hub_download(repo_id='ResembleAI/chatterbox', filename=f, local_dir='models-multilingual')
    print(f'Downloaded: {f}', flush=True)
print('Multilingual model files downloaded.', flush=True)
"

echo ""
echo "========================================"
echo "        Installation Complete!"
echo "========================================"
echo ""
echo "Virtual environment created at: $(pwd)/venv"
echo ""
echo "Final system check..."
python -c "import torch; print('Mode: CPU-only (no GPU needed)')"

echo ""
echo "========================================"
echo "           Ready for Audiobooks!"
echo "========================================"
echo ""
echo "To start Chatterbox TTS:"
echo "1. Run ./launch_audiobook.sh (recommended)"
echo "2. Or manually: source venv/bin/activate && python3 gradio_tts_app_audiobook.py"
echo ""
echo "Creating double-click launchers (Desktop entry + Mint menu)..."
APP_DIR="$(pwd)"
cat > Chatterbox-Audiobook.desktop <<EOF
[Desktop Entry]
Type=Application
Name=Chatterbox Audiobook
Comment=Launch Chatterbox TTS Audiobook Generator (23 languages + Bangla, CPU mode)
Exec=bash -c 'cd "$APP_DIR" && ./launch_audiobook.sh; echo ""; read -p "Press Enter to close this window..."'
Path=$APP_DIR
Terminal=true
Categories=AudioVideo;Audio;
StartupNotify=false
EOF
chmod +x Chatterbox-Audiobook.desktop
mkdir -p ~/.local/share/applications
cp Chatterbox-Audiobook.desktop ~/.local/share/applications/
gio set ~/.local/share/applications/Chatterbox-Audiobook.desktop metadata::trusted true 2>/dev/null || true
update-desktop-database ~/.local/share/applications 2>/dev/null || true
echo "Launchers ready: ./Chatterbox-Audiobook.desktop + Mint menu (Super key -> Chatterbox)."
echo ""
echo "Perfect for:"
echo "- Voice cloning for audiobook narration"
echo "- Multiple character voices"
echo "- Consistent voice quality across chapters"
echo "- Professional audiobook production"
echo ""
echo "Installation finished successfully!"
