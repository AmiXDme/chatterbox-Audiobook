"""
Voice management utilities for audiobook generation.

Handles voice profile CRUD operations, voice library management, and voice selection.
"""

import os
import json
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any


def ensure_voice_library_exists(voice_library_path: str) -> None:
    """Ensure the voice library directory exists.
    
    Args:
        voice_library_path: Path to voice library directory
    """
    os.makedirs(voice_library_path, exist_ok=True)


def get_voice_profiles(voice_library_path: str) -> List[Dict[str, Any]]:
    """Get all voice profiles from the voice library.
    
    Args:
        voice_library_path: Path to voice library directory
        
    Returns:
        List of voice profile dictionaries
    """
    ensure_voice_library_exists(voice_library_path)
    profiles = []
    
    try:
        for item in os.listdir(voice_library_path):
            profile_dir = os.path.join(voice_library_path, item)
            if os.path.isdir(profile_dir):
                config_file = os.path.join(profile_dir, "config.json")
                if os.path.exists(config_file):
                    try:
                        with open(config_file, 'r', encoding='utf-8') as f:
                            profile = json.load(f)
                            profile['voice_name'] = item
                            profiles.append(profile)
                    except Exception as e:
                        print(f"Warning: Could not load profile {item}: {e}")
    except Exception as e:
        print(f"Warning: Could not read voice library: {e}")
    
    return profiles


def get_voice_choices(voice_library_path: str) -> List[str]:
    """Get list of available voice names for UI dropdowns.
    
    Args:
        voice_library_path: Path to voice library directory
        
    Returns:
        List of voice names
    """
    profiles = get_voice_profiles(voice_library_path)
    return [profile['voice_name'] for profile in profiles]


def get_audiobook_voice_choices(voice_library_path: str) -> List[str]:
    """Get voice choices formatted for audiobook interface.
    
    Args:
        voice_library_path: Path to voice library directory
        
    Returns:
        List of voice names with display formatting
    """
    choices = get_voice_choices(voice_library_path)
    if not choices:
        return ["No voices available - Please add voices first"]
    return choices


def get_voice_config(voice_library_path: str, voice_name: str) -> Dict[str, Any]:
    """Get configuration for a specific voice.
    
    Args:
        voice_library_path: Path to voice library directory
        voice_name: Name of the voice
        
    Returns:
        Voice configuration dictionary
    """
    profile_dir = os.path.join(voice_library_path, voice_name)
    config_file = os.path.join(profile_dir, "config.json")
    
    default_config = {
        'voice_name': voice_name,
        'display_name': voice_name,
        'description': '',
        'exaggeration': 1.0,
        'cfg_weight': 1.0,
        'temperature': 0.7
    }
    
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                return {**default_config, **config}
        except Exception as e:
            print(f"Warning: Could not load config for {voice_name}: {e}")
    
    return default_config


def load_voice_for_tts(voice_library_path: str, voice_name: str) -> Tuple[Optional[str], Dict[str, Any]]:
    """Load voice audio file and configuration for TTS generation.
    
    Args:
        voice_library_path: Path to voice library directory
        voice_name: Name of the voice to load
        
    Returns:
        tuple: (audio_file_path, voice_config)
    """
    if not voice_name:
        return None, {}
    
    profile_dir = os.path.join(voice_library_path, voice_name)
    if not os.path.exists(profile_dir):
        return None, {}
    
    # Look for audio file
    audio_file = None
    for ext in ['.wav', '.mp3', '.flac']:
        potential_file = os.path.join(profile_dir, f"voice{ext}")
        if os.path.exists(potential_file):
            audio_file = potential_file
            break
    
    # Get voice configuration
    config = get_voice_config(voice_library_path, voice_name)
    
    return audio_file, config


def save_voice_profile(
    voice_library_path: str,
    voice_name: str,
    display_name: str,
    description: str,
    audio_file: Any,
    exaggeration: float,
    cfg_weight: float,
    temperature: float
) -> str:
    """Save a new voice profile to the library.
    
    Args:
        voice_library_path: Path to voice library directory
        voice_name: Internal voice name (used for directory)
        display_name: Display name for UI
        description: Voice description
        audio_file: Audio file data from Gradio
        exaggeration: Exaggeration parameter
        cfg_weight: CFG weight parameter
        temperature: Temperature parameter
        
    Returns:
        Status message
    """
    if not voice_name.strip():
        return "❌ Voice name cannot be empty"
    
    # Sanitize voice name for directory
    safe_voice_name = "".join(c for c in voice_name if c.isalnum() or c in (' ', '-', '_')).strip()
    safe_voice_name = safe_voice_name.replace(' ', '_')
    
    if not safe_voice_name:
        return "❌ Voice name contains only invalid characters"
    
    ensure_voice_library_exists(voice_library_path)
    
    profile_dir = os.path.join(voice_library_path, safe_voice_name)
    os.makedirs(profile_dir, exist_ok=True)
    
    try:
        # Save audio file
        if audio_file is not None:
            audio_path = os.path.join(profile_dir, "voice.wav")
            if isinstance(audio_file, str):
                # File path provided
                shutil.copy2(audio_file, audio_path)
            elif hasattr(audio_file, 'name'):
                # Gradio file object
                shutil.copy2(audio_file.name, audio_path)
            else:
                return "❌ Invalid audio file format"
        
        # Save configuration
        config = {
            'voice_name': safe_voice_name,
            'display_name': display_name or safe_voice_name,
            'description': description or '',
            'exaggeration': float(exaggeration),
            'cfg_weight': float(cfg_weight),
            'temperature': float(temperature)
        }
        
        config_path = os.path.join(profile_dir, "config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        
        return f"✅ Voice profile '{display_name}' saved successfully"
        
    except Exception as e:
        return f"❌ Error saving voice profile: {str(e)}"


def load_voice_profile(voice_library_path: str, voice_name: str) -> Tuple[str, str, str, float, float, float]:
    """Load voice profile data for editing.
    
    Args:
        voice_library_path: Path to voice library directory
        voice_name: Name of voice to load
        
    Returns:
        tuple: (display_name, description, audio_path, exaggeration, cfg_weight, temperature)
    """
    if not voice_name:
        return "", "", "", 1.0, 1.0, 0.7
    
    config = get_voice_config(voice_library_path, voice_name)
    audio_file, _ = load_voice_for_tts(voice_library_path, voice_name)
    
    return (
        config.get('display_name', voice_name),
        config.get('description', ''),
        audio_file or "",
        config.get('exaggeration', 1.0),
        config.get('cfg_weight', 1.0),
        config.get('temperature', 0.7)
    )


def delete_voice_profile(voice_library_path: str, voice_name: str) -> str:
    """Delete a voice profile from the library.
    
    Args:
        voice_library_path: Path to voice library directory
        voice_name: Name of voice to delete
        
    Returns:
        Status message
    """
    if not voice_name:
        return "❌ No voice selected for deletion"
    
    profile_dir = os.path.join(voice_library_path, voice_name)
    
    if not os.path.exists(profile_dir):
        return f"❌ Voice profile '{voice_name}' not found"
    
    try:
        shutil.rmtree(profile_dir)
        return f"✅ Voice profile '{voice_name}' deleted successfully"
    except Exception as e:
        return f"❌ Error deleting voice profile: {str(e)}"


def refresh_voice_list(voice_library_path: str) -> List[str]:
    """Refresh and return the current voice list.
    
    Args:
        voice_library_path: Path to voice library directory
        
    Returns:
        Updated list of voice names
    """
    return get_voice_choices(voice_library_path)


def refresh_voice_choices(voice_library_path: str) -> List[str]:
    """Refresh voice choices for regular dropdowns.
    
    Args:
        voice_library_path: Path to voice library directory
        
    Returns:
        Updated list of voice choices
    """
    return get_voice_choices(voice_library_path)


def refresh_audiobook_voice_choices(voice_library_path: str) -> List[str]:
    """Refresh voice choices for audiobook interface.
    
    Args:
        voice_library_path: Path to voice library directory
        
    Returns:
        Updated list of audiobook voice choices
    """
    return get_audiobook_voice_choices(voice_library_path)


def create_assignment_interface_with_dropdowns(
    voice_counts: Dict[str, int], 
    voice_library_path: str
) -> List[Any]:
    """Create voice assignment interface components.
    
    Args:
        voice_counts: Dictionary mapping character names to word counts
        voice_library_path: Path to voice library directory
        
    Returns:
        List of interface components
    """
    # This would typically return Gradio components
    # For now, return character names and available voices
    characters = list(voice_counts.keys())
    available_voices = get_voice_choices(voice_library_path)
    
    # Return data that can be used to create dropdowns
    return [
        {
            'character': char,
            'word_count': voice_counts[char],
            'available_voices': available_voices
        }
        for char in characters[:6]  # Limit to 6 characters
    ] 