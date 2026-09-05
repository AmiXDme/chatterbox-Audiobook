@echo off
setlocal

rem Performance Section
rem =================================
rem CPU-only mode - no CUDA settings needed

echo Checking for virtual environment...
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run install.bat first to set up the environment.
    echo.
    echo Make sure you're in the chatterbox repository directory.
    pause
    exit /b 1
)

echo Checking repository structure...
if not exist "gradio_tts_app_audiobook.py" (
    echo ERROR: gradio_tts_app_audiobook.py not found!
    echo Please make sure you're in the chatterbox repository root.
    pause
    exit /b 1
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Starting Chatterbox TTS Audiobook Edition (PUBLIC SHARING MODE)...
echo Features: Voice Library, Character Management, Audiobook Tools
echo Audio Cleaning Available in "Clean Samples" Tab
echo.
echo SECURITY MODE: PUBLIC SHARING ENABLED
echo - Accessible from anywhere via temporary public URL
echo - ⚠️  WARNING: Your application will be publicly accessible!
echo - ⚠️  Anyone with the link can access your voice library!
echo - ⚠️  Use only for demonstration or temporary sharing!
echo - Public URL will be displayed when ready
echo.
echo This may take a moment to load the models...
echo.
echo Current directory: %CD%
echo Python environment: %VIRTUAL_ENV%
echo Voice library will be created at: %CD%\voice_library
echo.

python gradio_tts_app_audiobook.py

echo.
echo Chatterbox TTS Audiobook Edition (PUBLIC SHARING) has stopped.
echo Deactivating virtual environment...
deactivate
echo.
echo Thanks for using Chatterbox TTS Audiobook Edition! 🎧✨
echo Your voice profiles are saved in the voice_library folder.
echo Audio cleaning features are in the "Clean Samples" tab!
echo.
echo ⚠️  Remember: Your session was publicly accessible!
echo ⚠️  Make sure you're comfortable with what was shared!
pause 