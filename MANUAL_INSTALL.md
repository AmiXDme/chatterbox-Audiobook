# Manual Installation Guide

This guide is for users who prefer to install Chatterbox TTS manually instead of using the `install-audiobook.bat` file.

## Prerequisites

- **Python 3.10+** (Required)
- **Git** (Optional, for updates)

## Installation Steps

### 1. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. Upgrade pip

```bash
python -m pip install --upgrade pip
```

### 3. Install Dependencies

```bash
# Install all dependencies from requirements.txt
pip install -r requirements.txt
```

### 4. Install Chatterbox TTS Package

```bash
# Install the Chatterbox TTS package in development mode
pip install -e .
```

## CPU-Only Installation

This version is CPU-only. No GPU or CUDA required. Just install normally:

```bash
pip install -r requirements.txt
```

## Troubleshooting

**Common Issues:**

**Pydantic Compatibility:**
- The requirements.txt pins pydantic to version 2.10.6 for stability
- If you encounter issues, try: `pip install pydantic==2.10.6 --force-reinstall`

**Import Errors:**
- Make sure you're in the project root directory
- Verify the virtual environment is activated
- Run `pip install -e .` again if needed

### Verification

Test your installation:

```bash
# Test PyTorch
python -c "import torch; print('PyTorch:', torch.__version__); print('CPU mode: OK')"

# Test Chatterbox TTS
python -c "from chatterbox.mtl_tts import ChatterboxMultilingualTTS as ChatterboxTTS; print('Chatterbox Multilingual TTS imported successfully!')"
```

## Running the Application

After successful installation:

```bash
# Option 1: Use launcher scripts (recommended)
# For local-only access:
launch_local.bat

# For network access:
launch_network.bat

# For public sharing:
launch_huggingface.bat

# Option 2: Direct execution
python gradio_tts_app_audiobook.py
```

## Notes

- This manual installation provides the same functionality as `install-audiobook.bat`
- The batch file installer includes additional error checking and troubleshooting
- If you encounter issues, you can always fall back to using `install-audiobook.bat`
- Virtual environment is highly recommended to avoid dependency conflicts 