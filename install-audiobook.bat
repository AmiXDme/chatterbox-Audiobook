@echo off
echo ========================================
echo   Chatterbox TTS - Installation Setup
echo ========================================
echo.
echo This will install Chatterbox TTS in a virtual environment
echo to keep it isolated from other Python projects.
echo.
echo Requirements:
echo - Python 3.10 or higher
echo - No GPU required (CPU-only mode)
echo - Git (if you want to pull updates)
echo.
echo Current directory: %CD%
echo.
pause

echo.
echo [1/9] Checking Python installation...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

echo.
echo [2/9] Checking if we're in the correct directory...
if not exist "pyproject.toml" (
    echo ERROR: pyproject.toml not found!
    echo Please make sure you're running this from the chatterbox repository root.
    echo Expected files: pyproject.toml, gradio_tts_app.py, src/chatterbox/
    pause
    exit /b 1
)

if not exist "src\chatterbox" (
    echo ERROR: src\chatterbox directory not found!
    echo Please make sure you're in the correct chatterbox repository.
    pause
    exit /b 1
)

echo Repository structure verified ✓

echo.
echo [3/9] Creating virtual environment...
if exist "venv" (
    echo Virtual environment already exists. Removing old one...
    rmdir /s /q venv
)
python -m venv venv

echo.
echo [4/9] Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo [5/9] Upgrading pip...
python -m pip install --upgrade pip
echo Pinning setuptools (needed for perth/pkg_resources compatibility)...
pip install "setuptools<81"

echo.
echo [6/9] Installing compatible PyTorch (CPU-only)...
echo This may take a while...
echo Installing PyTorch (CPU-only, latest compatible version)...
pip install torch torchaudio

echo.
echo [7/9] Installing Chatterbox TTS and dependencies...
pip install -e .
pip install gradio

echo.
echo [8/9] Installing and configuring pydantic...
echo Installing pydantic...
pip install pydantic

echo.
echo [9/9] Testing installation...
echo Testing PyTorch (CPU-only)...
python -c "import torch; print('PyTorch version:', torch.__version__); print('CPU mode: OK')"

if %errorlevel% neq 0 (
    echo WARNING: PyTorch test failed. Retesting...
    python -c "import torch; print('PyTorch version:', torch.__version__); print('CPU mode: OK')"
)

echo.
echo Testing Chatterbox import...
python -c "from chatterbox.mtl_tts import ChatterboxMultilingualTTS as ChatterboxTTS; print('Chatterbox Multilingual TTS imported successfully!')"

if %errorlevel% neq 0 (
    echo WARNING: Chatterbox import failed. This might be a dependency issue.
    echo The installation will continue, but you may need to troubleshoot.
    echo Common fixes:
    echo 1. Run install.bat again
    echo 2. Restart your computer
)

echo.
echo Testing pydantic compatibility...
python -c "import pydantic; print('Pydantic version:', pydantic.__version__)"

echo.
echo [10/10] Downloading multilingual TTS model files...
echo This downloads the Chatterbox Multilingual V3 model into models-multilingual\
if not exist "models-multilingual" mkdir "models-multilingual"
python -c "from huggingface_hub import hf_hub_download; files=['ve.pt','t3_mtl23ls_v3.safetensors','s3gen.pt','grapheme_mtl_merged_expanded_v1.json','conds.pt','Cangjie5_TC.json']; [hf_hub_download(repo_id='ResembleAI/chatterbox', filename=f, local_dir='models-multilingual') for f in files]; print('Multilingual model files downloaded.')"

if %errorlevel% neq 0 (
    echo WARNING: Multilingual model download had issues.
    echo The app will retry downloading automatically on first run.
)

echo.
echo ========================================
echo        Installation Complete!
echo ========================================
echo.
echo Virtual environment created at: %CD%\venv
echo.

echo Final system check...
python -c "import torch; print('Mode: CPU-only (no GPU needed)')"

echo.
echo ========================================
echo           Ready for Audiobooks!
echo ========================================
echo.
echo To start Chatterbox TTS:
echo 1. Run launch_audiobook.bat (recommended)
echo 2. Or manually: venv\Scripts\activate.bat then python gradio_tts_app_audiobook.py
echo.
echo Perfect for:
echo - Voice cloning for audiobook narration
echo - Multiple character voices
echo - Consistent voice quality across chapters
echo - Professional audiobook production
echo.
echo Note: If you encounter pydantic compatibility issues later,
echo you can run update.bat to specifically update pydantic.
echo.
echo Installation finished successfully!
pause 
