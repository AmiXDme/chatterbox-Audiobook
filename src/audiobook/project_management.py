"""
Project management utilities for audiobook generation.

Handles project creation, loading, metadata, file organization, and project lifecycle.
"""

import os
import json
import shutil
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime

from .text_processing import chunk_text_by_sentences, parse_multi_voice_text, chunk_multi_voice_segments
from .audio_processing import save_audio_chunks, auto_remove_silence, normalize_audio_levels, analyze_audio_quality
from .voice_management import load_voice_for_tts, get_voice_config
from .models import generate_with_retry, load_model_cpu


# Constants
MAX_CHUNKS_FOR_INTERFACE = 100
MAX_CHUNKS_FOR_AUTO_SAVE = 100


def save_project_metadata(
    project_dir: str,
    project_name: str,
    text_content: str,
    voice_info: dict,
    chunks: list,
    project_type: str = "single_voice"
) -> None:
    """Save project metadata to JSON file.
    
    Args:
        project_dir: Project directory path
        project_name: Name of the project
        text_content: Original text content
        voice_info: Voice configuration information
        chunks: List of text chunks
        project_type: Type of project (single_voice or multi_voice)
    """
    metadata = {
        'project_name': project_name,
        'project_type': project_type,
        'created_at': datetime.now().isoformat(),
        'text_content': text_content,
        'voice_info': voice_info,
        'chunks': chunks,
        'total_chunks': len(chunks),
        'status': 'in_progress'
    }
    
    metadata_path = os.path.join(project_dir, 'metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)


def load_project_metadata(project_dir: str) -> dict:
    """Load project metadata from JSON file.
    
    Args:
        project_dir: Project directory path
        
    Returns:
        Project metadata dictionary
    """
    metadata_path = os.path.join(project_dir, 'metadata.json')
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load metadata for {project_dir}: {e}")
    return {}


def get_existing_projects(output_dir: str = "audiobook_projects") -> List[Dict[str, Any]]:
    """Get list of existing audiobook projects.
    
    Args:
        output_dir: Directory containing projects
        
    Returns:
        List of project information dictionaries
    """
    projects = []
    
    if not os.path.exists(output_dir):
        return projects
    
    try:
        for item in os.listdir(output_dir):
            project_dir = os.path.join(output_dir, item)
            if os.path.isdir(project_dir):
                metadata = load_project_metadata(project_dir)
                
                if metadata:
                    # Use metadata information
                    project_info = {
                        'name': metadata.get('project_name', item),
                        'path': project_dir,
                        'type': metadata.get('project_type', 'unknown'),
                        'created_at': metadata.get('created_at', ''),
                        'total_chunks': metadata.get('total_chunks', 0),
                        'status': metadata.get('status', 'unknown')
                    }
                else:
                    # Fallback to directory scanning
                    audio_files = [f for f in os.listdir(project_dir) if f.endswith('.wav')]
                    project_info = {
                        'name': item,
                        'path': project_dir,
                        'type': 'legacy',
                        'created_at': '',
                        'total_chunks': len(audio_files),
                        'status': 'completed' if audio_files else 'empty'
                    }
                
                projects.append(project_info)
                
    except Exception as e:
        print(f"Warning: Error scanning projects directory: {e}")
    
    # Sort by creation date (newest first)
    def get_sort_key(project):
        created_at = project.get('created_at', '')
        if created_at:
            try:
                return datetime.fromisoformat(created_at)
            except:
                pass
        return datetime.min
    
    projects.sort(key=get_sort_key, reverse=True)
    return projects


def get_project_choices() -> List[str]:
    """Get project names for UI dropdowns.
    
    Returns:
        List of project names
    """
    projects = get_existing_projects()
    if not projects:
        return ["No projects found"]
    
    # Format: "project_name (type - chunks)"
    choices = []
    for project in projects:
        name = project['name']
        project_type = project['type']
        chunk_count = project['total_chunks']
        formatted = f"{name} ({project_type} - {chunk_count} chunks)"
        choices.append(formatted)
    
    return choices


def load_project_for_regeneration(project_name: str) -> Tuple[str, str, str, str]:
    """Load project data for regeneration interface.
    
    Args:
        project_name: Name of the project to load
        
    Returns:
        tuple: (text_content, voice_name, project_type, status_message)
    """
    if not project_name or project_name == "No projects found":
        return "", "", "", "No project selected"
    
    # Extract actual project name from formatted string
    actual_name = project_name.split(' (')[0] if ' (' in project_name else project_name
    
    projects = get_existing_projects()
    project_info = None
    
    for project in projects:
        if project['name'] == actual_name:
            project_info = project
            break
    
    if not project_info:
        return "", "", "", f"❌ Project '{actual_name}' not found"
    
    # Load project metadata
    metadata = load_project_metadata(project_info['path'])
    
    if not metadata:
        return "", "", "", f"❌ Could not load project metadata for '{actual_name}'"
    
    text_content = metadata.get('text_content', '')
    voice_info = metadata.get('voice_info', {})
    project_type = metadata.get('project_type', 'single_voice')
    
    # Extract voice name based on project type
    if project_type == 'single_voice':
        voice_name = voice_info.get('voice_name', '')
    else:
        voice_name = 'Multi-voice project'
    
    return text_content, voice_name, project_type, f"✅ Loaded project '{actual_name}'"


def create_audiobook(
    model: Any,
    text_content: str,
    voice_library_path: str,
    selected_voice: str,
    project_name: str,
    resume: bool = False,
    autosave_interval: int = 10
) -> Tuple[str, List[str], str]:
    """Create a single-voice audiobook project.
    
    Args:
        model: TTS model instance
        text_content: Text to convert to audio
        voice_library_path: Path to voice library
        selected_voice: Name of selected voice
        project_name: Name for the project
        resume: Whether to resume existing project
        autosave_interval: Chunks between auto-saves
        
    Returns:
        tuple: (status_message, audio_file_paths, project_path)
    """
    if not model:
        model = load_model_cpu()
    
    # Load voice configuration
    audio_prompt_path, voice_config = load_voice_for_tts(voice_library_path, selected_voice)
    
    if not audio_prompt_path:
        return f"❌ Could not load voice '{selected_voice}'", [], ""
    
    # Get voice parameters
    exaggeration = voice_config.get('exaggeration', 1.0)
    temperature = voice_config.get('temperature', 0.7)
    cfg_weight = voice_config.get('cfg_weight', 1.0)
    
    # Chunk the text
    chunks = chunk_text_by_sentences(text_content, max_words=50)
    
    # Create project directory
    safe_project_name = "".join(c for c in project_name if c.isalnum() or c in (' ', '-', '_')).strip()
    safe_project_name = safe_project_name.replace(' ', '_')
    project_dir = os.path.join("audiobook_projects", safe_project_name)
    os.makedirs(project_dir, exist_ok=True)
    
    # Save project metadata
    voice_info = {
        'voice_name': selected_voice,
        'audio_prompt_path': audio_prompt_path,
        'exaggeration': exaggeration,
        'temperature': temperature,
        'cfg_weight': cfg_weight
    }
    
    save_project_metadata(project_dir, project_name, text_content, voice_info, chunks, "single_voice")
    
    # Generate audio for chunks
    audio_chunks = []
    generated_files = []
    
    try:
        for i, chunk in enumerate(chunks):
            print(f"Generating chunk {i+1}/{len(chunks)}")
            
            # Generate audio
            wav, device_used = generate_with_retry(
                model, chunk, audio_prompt_path, exaggeration, temperature, cfg_weight
            )
            
            # Convert to numpy array if needed
            if hasattr(wav, 'squeeze'):
                audio_array = wav.squeeze(0).numpy()
            else:
                audio_array = wav
            
            audio_chunks.append(audio_array)
            
            # Auto-save periodically
            if (i + 1) % autosave_interval == 0 or i == len(chunks) - 1:
                # Save current batch
                batch_files = save_audio_chunks(
                    audio_chunks, model.sr, safe_project_name, "audiobook_projects"
                )
                generated_files.extend(batch_files)
                audio_chunks = []  # Reset for next batch
        
        # Update metadata to completed
        metadata = load_project_metadata(project_dir)
        metadata['status'] = 'completed'
        metadata['completed_at'] = datetime.now().isoformat()
        
        metadata_path = os.path.join(project_dir, 'metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        return f"✅ Audiobook '{project_name}' created successfully! Generated {len(chunks)} audio chunks.", generated_files, project_dir
        
    except Exception as e:
        return f"❌ Error creating audiobook: {str(e)}", generated_files, project_dir


def create_multi_voice_audiobook_with_assignments(
    model: Any,
    text_content: str,
    voice_library_path: str,
    project_name: str,
    voice_assignments: Dict[str, str],
    resume: bool = False,
    autosave_interval: int = 10
) -> Tuple[str, List[str], str]:
    """Create a multi-voice audiobook project with character voice assignments.
    
    Args:
        model: TTS model instance
        text_content: Text with character markers
        voice_library_path: Path to voice library
        project_name: Name for the project
        voice_assignments: Character to voice mappings
        resume: Whether to resume existing project
        autosave_interval: Chunks between auto-saves
        
    Returns:
        tuple: (status_message, audio_file_paths, project_path)
    """
    if not model:
        model = load_model_cpu()
    
    # Parse multi-voice text
    segments = parse_multi_voice_text(text_content)
    chunked_segments = chunk_multi_voice_segments(segments, max_words=50)
    
    # Create project directory
    safe_project_name = "".join(c for c in project_name if c.isalnum() or c in (' ', '-', '_')).strip()
    safe_project_name = safe_project_name.replace(' ', '_')
    project_dir = os.path.join("audiobook_projects", safe_project_name)
    os.makedirs(project_dir, exist_ok=True)
    
    # Save project metadata
    voice_info = {
        'voice_assignments': voice_assignments,
        'characters': list(voice_assignments.keys())
    }
    
    save_project_metadata(project_dir, project_name, text_content, voice_info, chunked_segments, "multi_voice")
    
    # Generate audio for segments
    audio_chunks = []
    generated_files = []
    
    try:
        for i, segment in enumerate(chunked_segments):
            character = segment['character']
            text = segment['text']
            
            # Get assigned voice for character
            assigned_voice = voice_assignments.get(character)
            if not assigned_voice:
                print(f"Warning: No voice assigned for character '{character}', skipping segment")
                continue
            
            # Load voice configuration
            audio_prompt_path, voice_config = load_voice_for_tts(voice_library_path, assigned_voice)
            
            if not audio_prompt_path:
                print(f"Warning: Could not load voice '{assigned_voice}' for character '{character}'")
                continue
            
            print(f"Generating segment {i+1}/{len(chunked_segments)} - {character}: {text[:50]}...")
            
            # Generate audio
            wav, device_used = generate_with_retry(
                model, text, audio_prompt_path,
                voice_config.get('exaggeration', 1.0),
                voice_config.get('temperature', 0.7),
                voice_config.get('cfg_weight', 1.0)
            )
            
            # Convert to numpy array if needed
            if hasattr(wav, 'squeeze'):
                audio_array = wav.squeeze(0).numpy()
            else:
                audio_array = wav
            
            audio_chunks.append(audio_array)
            
            # Auto-save periodically
            if (i + 1) % autosave_interval == 0 or i == len(chunked_segments) - 1:
                # Save current batch
                batch_files = save_audio_chunks(
                    audio_chunks, model.sr, safe_project_name, "audiobook_projects"
                )
                generated_files.extend(batch_files)
                audio_chunks = []  # Reset for next batch
        
        # Update metadata to completed
        metadata = load_project_metadata(project_dir)
        metadata['status'] = 'completed'
        metadata['completed_at'] = datetime.now().isoformat()
        
        metadata_path = os.path.join(project_dir, 'metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        return f"✅ Multi-voice audiobook '{project_name}' created successfully! Generated {len(chunked_segments)} audio segments.", generated_files, project_dir
        
    except Exception as e:
        return f"❌ Error creating multi-voice audiobook: {str(e)}", generated_files, project_dir


def cleanup_project_temp_files(project_name: str) -> str:
    """Clean up temporary files for a project.
    
    Args:
        project_name: Name of the project
        
    Returns:
        Status message
    """
    if not project_name:
        return "❌ No project specified"
    
    # Extract actual project name
    actual_name = project_name.split(' (')[0] if ' (' in project_name else project_name
    safe_name = "".join(c for c in actual_name if c.isalnum() or c in (' ', '-', '_')).strip()
    safe_name = safe_name.replace(' ', '_')
    
    project_dir = os.path.join("audiobook_projects", safe_name)
    
    if not os.path.exists(project_dir):
        return f"❌ Project directory not found: {safe_name}"
    
    try:
        temp_files = []
        for file in os.listdir(project_dir):
            if 'temp' in file.lower() or 'trimmed' in file.lower():
                temp_files.append(os.path.join(project_dir, file))
        
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        return f"✅ Cleaned up {len(temp_files)} temporary files for project '{actual_name}'"
        
    except Exception as e:
        return f"❌ Error cleaning up project files: {str(e)}"


def auto_clean_project_audio(
    project_name: str,
    silence_threshold: float = -50.0,
    min_silence_duration: float = 0.5
) -> str:
    """Automatically clean audio for all chunks in a project.
    
    Args:
        project_name: Name of the project
        silence_threshold: Silence threshold in dB
        min_silence_duration: Minimum silence duration to remove
        
    Returns:
        Status message
    """
    if not project_name:
        return "❌ No project specified"
    
    # Extract actual project name
    actual_name = project_name.split(' (')[0] if ' (' in project_name else project_name
    safe_name = "".join(c for c in actual_name if c.isalnum() or c in (' ', '-', '_')).strip()
    safe_name = safe_name.replace(' ', '_')
    
    project_dir = os.path.join("audiobook_projects", safe_name)
    
    if not os.path.exists(project_dir):
        return f"❌ Project directory not found: {safe_name}"
    
    try:
        # Get all WAV files in the project
        audio_files = [f for f in os.listdir(project_dir) 
                      if f.endswith('.wav') and not 'cleaned' in f.lower() and not 'temp' in f.lower()]
        
        if not audio_files:
            return f"❌ No audio files found in project '{actual_name}'"
        
        cleaned_count = 0
        failed_count = 0
        total_time_saved = 0.0
        
        for audio_file in audio_files:
            file_path = os.path.join(project_dir, audio_file)
            
            # Clean the audio
            status_msg, cleaned_path = auto_remove_silence(
                file_path, silence_threshold, min_silence_duration
            )
            
            if "✅" in status_msg:
                cleaned_count += 1
                # Extract time saved from status message
                if "Removed" in status_msg:
                    try:
                        # Parse "Removed X.XXs" from status message
                        import re
                        match = re.search(r'Removed (\d+\.?\d*)s', status_msg)
                        if match:
                            total_time_saved += float(match.group(1))
                    except:
                        pass
            else:
                failed_count += 1
                print(f"Failed to clean {audio_file}: {status_msg}")
        
        if cleaned_count > 0:
            return (
                f"✅ Auto-cleaned {cleaned_count}/{len(audio_files)} audio files for project '{actual_name}'. "
                f"Total silence removed: {total_time_saved:.2f}s. "
                f"Failed: {failed_count}"
            )
        else:
            return f"❌ Failed to clean any audio files for project '{actual_name}'"
        
    except Exception as e:
        return f"❌ Error auto-cleaning project audio: {str(e)}"


def analyze_project_audio_quality(project_name: str) -> str:
    """Analyze audio quality for all chunks in a project.
    
    Args:
        project_name: Name of the project
        
    Returns:
        Analysis results
    """
    if not project_name:
        return "❌ No project specified"
    
    # Extract actual project name
    actual_name = project_name.split(' (')[0] if ' (' in project_name else project_name
    safe_name = "".join(c for c in actual_name if c.isalnum() or c in (' ', '-', '_')).strip()
    safe_name = safe_name.replace(' ', '_')
    
    project_dir = os.path.join("audiobook_projects", safe_name)
    
    if not os.path.exists(project_dir):
        return f"❌ Project directory not found: {safe_name}"
    
    try:
        # Get all WAV files in the project
        audio_files = [f for f in os.listdir(project_dir) 
                      if f.endswith('.wav') and not 'temp' in f.lower()]
        
        if not audio_files:
            return f"❌ No audio files found in project '{actual_name}'"
        
        total_duration = 0.0
        total_rms = 0.0
        peak_levels = []
        analyzed_count = 0
        
        for audio_file in audio_files:
            file_path = os.path.join(project_dir, audio_file)
            
            # Analyze the audio
            metrics = analyze_audio_quality(file_path)
            
            if 'error' not in metrics:
                total_duration += metrics.get('duration', 0)
                total_rms += metrics.get('rms_level', 0)
                peak_levels.append(metrics.get('peak_level', 0))
                analyzed_count += 1
        
        if analyzed_count > 0:
            avg_rms = total_rms / analyzed_count
            max_peak = max(peak_levels) if peak_levels else 0
            avg_peak = sum(peak_levels) / len(peak_levels) if peak_levels else 0
            
            # Convert to dB
            avg_rms_db = 20 * np.log10(avg_rms) if avg_rms > 0 else -np.inf
            max_peak_db = 20 * np.log10(max_peak) if max_peak > 0 else -np.inf
            avg_peak_db = 20 * np.log10(avg_peak) if avg_peak > 0 else -np.inf
            
            return (
                f"📊 Audio Quality Analysis for '{actual_name}':\n"
                f"• Files analyzed: {analyzed_count}/{len(audio_files)}\n"
                f"• Total duration: {total_duration:.2f} seconds\n"
                f"• Average RMS level: {avg_rms_db:.1f} dB\n"
                f"• Average peak level: {avg_peak_db:.1f} dB\n"
                f"• Maximum peak level: {max_peak_db:.1f} dB\n"
                f"• Recommended: Keep peaks below -3 dB for headroom"
            )
        else:
            return f"❌ Failed to analyze any audio files for project '{actual_name}'"
        
    except Exception as e:
        return f"❌ Error analyzing project audio quality: {str(e)}"


def get_project_chunks(project_name: str) -> List[Dict[str, Any]]:
    """Get list of audio chunks for a project.
    
    Args:
        project_name: Name of the project
        
    Returns:
        List of chunk information dictionaries
    """
    if not project_name or project_name == "No projects found":
        return []
    
    # Extract actual project name
    actual_name = project_name.split(' (')[0] if ' (' in project_name else project_name
    safe_name = "".join(c for c in actual_name if c.isalnum() or c in (' ', '-', '_')).strip()
    safe_name = safe_name.replace(' ', '_')
    
    project_dir = os.path.join("audiobook_projects", safe_name)
    
    if not os.path.exists(project_dir):
        return []
    
    try:
        chunks = []
        audio_files = [f for f in os.listdir(project_dir) if f.endswith('.wav') and not 'temp' in f.lower()]
        
        # Sort files by chunk number
        def extract_chunk_num_from_filename(filename: str) -> int:
            # Extract number from filename like "project_001.wav"
            parts = filename.replace('.wav', '').split('_')
            for part in reversed(parts):
                if part.isdigit():
                    return int(part)
            return 0
        
        audio_files.sort(key=extract_chunk_num_from_filename)
        
        for i, filename in enumerate(audio_files):
            file_path = os.path.join(project_dir, filename)
            chunk_info = {
                'chunk_num': i + 1,
                'filename': filename,
                'file_path': file_path,
                'size': os.path.getsize(file_path) if os.path.exists(file_path) else 0
            }
            chunks.append(chunk_info)
        
        return chunks
        
    except Exception as e:
        print(f"Error getting project chunks: {e}")
        return [] 