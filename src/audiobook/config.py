"""
Configuration management for the audiobook generator.

Handles loading and saving of application configuration including voice library paths.
"""

import json
import os
from pathlib import Path


# Default configuration values
DEFAULT_VOICE_LIBRARY = "voice_library"
CONFIG_FILE = "audiobook_config.json"


def load_config() -> str:
    """Load configuration including voice library path.
    
    Returns:
        str: Path to the voice library directory
    """
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
            return config.get('voice_library_path', DEFAULT_VOICE_LIBRARY)
        except Exception:
            return DEFAULT_VOICE_LIBRARY
    return DEFAULT_VOICE_LIBRARY


def save_config(voice_library_path: str) -> str:
    """Save configuration including voice library path.
    
    Args:
        voice_library_path: Path to the voice library directory
        
    Returns:
        str: Success or error message
    """
    config = {
        'voice_library_path': voice_library_path,
        'last_updated': str(Path().resolve())  # timestamp
    }
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        return f"✅ Configuration saved - Voice library path: {voice_library_path}"
    except Exception as e:
        return f"❌ Error saving configuration: {str(e)}"


def update_voice_library_path(new_path: str) -> tuple[str, str]:
    """Update the voice library path in configuration.
    
    Args:
        new_path: New path to the voice library
        
    Returns:
        tuple: (status_message, updated_path)
    """
    if not new_path.strip():
        return "❌ Voice library path cannot be empty", ""
    
    # Create directory if it doesn't exist
    try:
        os.makedirs(new_path, exist_ok=True)
        save_result = save_config(new_path)
        return save_result, new_path
    except Exception as e:
        return f"❌ Error updating voice library path: {str(e)}", "" 