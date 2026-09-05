"""
Audio processing utilities for audiobook generation.

Handles audio saving, combining, trimming, and file operations.
"""

import os
import wave
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Any
import tempfile
import shutil

# Optional audio processing imports
try:
    import librosa
    import soundfile as sf
    AUDIO_PROCESSING_AVAILABLE = True
except ImportError:
    AUDIO_PROCESSING_AVAILABLE = False


def save_audio_chunks(
    audio_chunks: List[np.ndarray], 
    sample_rate: int, 
    project_name: str, 
    output_dir: str = "audiobook_projects"
) -> List[str]:
    """Save audio chunks as numbered WAV files.
    
    Args:
        audio_chunks: List of audio arrays
        sample_rate: Audio sample rate
        project_name: Name of the project
        output_dir: Output directory for projects
        
    Returns:
        List of saved file paths
    """
    if not project_name.strip():
        project_name = "untitled_audiobook"
    
    # Sanitize project name
    safe_project_name = "".join(c for c in project_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
    safe_project_name = safe_project_name.replace(' ', '_')
    
    # Create output directory
    project_dir = os.path.join(output_dir, safe_project_name)
    os.makedirs(project_dir, exist_ok=True)
    
    saved_files = []
    
    for i, audio_chunk in enumerate(audio_chunks, 1):
        filename = f"{safe_project_name}_{i:03d}.wav"
        filepath = os.path.join(project_dir, filename)
        
        # Save as WAV file
        with wave.open(filepath, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            
            # Convert float32 to int16
            audio_int16 = (audio_chunk * 32767).astype(np.int16)
            wav_file.writeframes(audio_int16.tobytes())
        
        saved_files.append(filepath)
    
    return saved_files


def combine_audio_files(file_paths: List[str], output_path: str, output_format: str = "wav") -> str:
    """Combine multiple audio files into a single file.
    
    Args:
        file_paths: List of audio file paths to combine
        output_path: Output file path
        output_format: Output format (wav or mp3)
        
    Returns:
        Success message or error
    """
    if not file_paths:
        return "❌ No audio files to combine"
    
    try:
        # Read all audio files and combine
        combined_audio = []
        sample_rate = None
        
        for file_path in file_paths:
            if not os.path.exists(file_path):
                continue
                
            with wave.open(file_path, 'rb') as wav_file:
                if sample_rate is None:
                    sample_rate = wav_file.getframerate()
                
                frames = wav_file.readframes(wav_file.getnframes())
                audio_data = np.frombuffer(frames, dtype=np.int16)
                combined_audio.append(audio_data)
        
        if not combined_audio:
            return "❌ No valid audio files found"
        
        # Concatenate all audio
        final_audio = np.concatenate(combined_audio)
        
        # Save combined audio
        if output_format.lower() == "wav":
            with wave.open(output_path, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(final_audio.tobytes())
        else:
            # For MP3, we'd need additional dependencies like pydub
            return "❌ MP3 export not implemented yet"
        
        return f"✅ Combined {len(file_paths)} files into {output_path}"
        
    except Exception as e:
        return f"❌ Error combining audio files: {str(e)}"


def save_trimmed_audio(audio_data: Any, original_file_path: str, chunk_num: int) -> Tuple[str, str]:
    """Save trimmed audio data to a new file.
    
    Args:
        audio_data: Audio data from Gradio component
        original_file_path: Original audio file path
        chunk_num: Chunk number
        
    Returns:
        tuple: (success_message, file_path)
    """
    if audio_data is None:
        return "❌ No audio data provided", ""
    
    try:
        # Extract sample rate and audio array from gradio format
        if isinstance(audio_data, tuple) and len(audio_data) == 2:
            sample_rate, audio_array = audio_data
        else:
            return "❌ Invalid audio data format", ""
        
        # Create temporary file path
        base_dir = os.path.dirname(original_file_path)
        base_name = os.path.splitext(os.path.basename(original_file_path))[0]
        trimmed_path = os.path.join(base_dir, f"{base_name}_trimmed.wav")
        
        # Save trimmed audio
        with wave.open(trimmed_path, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            
            # Convert to int16 if needed
            if audio_array.dtype != np.int16:
                if audio_array.dtype == np.float32 or audio_array.dtype == np.float64:
                    audio_array = (audio_array * 32767).astype(np.int16)
                else:
                    audio_array = audio_array.astype(np.int16)
            
            wav_file.writeframes(audio_array.tobytes())
        
        return f"✅ Trimmed audio saved for chunk {chunk_num}", trimmed_path
        
    except Exception as e:
        return f"❌ Error saving trimmed audio: {str(e)}", ""


def extract_audio_segment(
    audio_data: Any, 
    start_time: Optional[float] = None, 
    end_time: Optional[float] = None
) -> Tuple[str, Any]:
    """Extract a segment from audio data based on time stamps.
    
    Args:
        audio_data: Audio data tuple (sample_rate, audio_array)
        start_time: Start time in seconds
        end_time: End time in seconds
        
    Returns:
        tuple: (status_message, extracted_audio_data)
    """
    if audio_data is None:
        return "❌ No audio data provided", None
    
    try:
        if isinstance(audio_data, tuple) and len(audio_data) == 2:
            sample_rate, audio_array = audio_data
        else:
            return "❌ Invalid audio data format", None
        
        if start_time is None and end_time is None:
            return "❌ Please specify start time or end time", None
        
        # Convert time to sample indices
        start_sample = int(start_time * sample_rate) if start_time is not None else 0
        end_sample = int(end_time * sample_rate) if end_time is not None else len(audio_array)
        
        # Validate bounds
        start_sample = max(0, start_sample)
        end_sample = min(len(audio_array), end_sample)
        
        if start_sample >= end_sample:
            return "❌ Invalid time range", None
        
        # Extract segment
        extracted_audio = audio_array[start_sample:end_sample]
        
        return "✅ Audio segment extracted", (sample_rate, extracted_audio)
        
    except Exception as e:
        return f"❌ Error extracting audio segment: {str(e)}", None


def handle_audio_trimming(audio_data: Any) -> Tuple[str, Any]:
    """Handle audio trimming from Gradio component.
    
    Args:
        audio_data: Audio data from Gradio component
        
    Returns:
        tuple: (status_message, processed_audio_data)
    """
    if audio_data is None:
        return "No audio data", None
    
    try:
        # Process audio data from Gradio
        if isinstance(audio_data, tuple) and len(audio_data) == 2:
            sample_rate, audio_array = audio_data
            
            # Validate audio array
            if audio_array is None or len(audio_array) == 0:
                return "Empty audio data", None
            
            return "Audio ready for processing", audio_data
        else:
            return "Invalid audio format", None
            
    except Exception as e:
        return f"Error processing audio: {str(e)}", None


def cleanup_temp_files(file_paths: List[str]) -> None:
    """Clean up temporary files.
    
    Args:
        file_paths: List of file paths to delete
    """
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Warning: Could not delete {file_path}: {e}")


def analyze_audio_quality(file_path: str) -> dict:
    """Analyze audio file quality metrics.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Dictionary with quality metrics
    """
    try:
        if AUDIO_PROCESSING_AVAILABLE:
            # Use librosa for more detailed analysis
            y, sr = librosa.load(file_path, sr=None)
            
            # Calculate advanced metrics
            rms = np.sqrt(np.mean(y**2))
            peak = np.max(np.abs(y))
            duration = len(y) / sr
            
            # Calculate spectral centroid (brightness)
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_centroid_mean = np.mean(spectral_centroids)
            
            # Calculate zero crossing rate (useful for speech analysis)
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            zcr_mean = np.mean(zcr)
            
            return {
                'duration': duration,
                'sample_rate': sr,
                'rms_level': float(rms),
                'peak_level': float(peak),
                'dynamic_range': float(peak / (rms + 1e-6)),
                'spectral_centroid': float(spectral_centroid_mean),
                'zero_crossing_rate': float(zcr_mean),
                'has_advanced_analysis': True
            }
        else:
            # Fallback to basic wave analysis
            with wave.open(file_path, 'rb') as wav_file:
                sample_rate = wav_file.getframerate()
                n_frames = wav_file.getnframes()
                duration = n_frames / sample_rate
                
                frames = wav_file.readframes(n_frames)
                audio_data = np.frombuffer(frames, dtype=np.int16)
                
                # Normalize to float
                audio_data = audio_data.astype(np.float32) / 32768.0
                
                # Calculate basic metrics
                rms = np.sqrt(np.mean(audio_data**2))
                peak = np.max(np.abs(audio_data))
                
                return {
                    'duration': duration,
                    'sample_rate': sample_rate,
                    'rms_level': float(rms),
                    'peak_level': float(peak),
                    'dynamic_range': float(peak / (rms + 1e-6)),
                    'has_advanced_analysis': False
                }
            
    except Exception as e:
        return {'error': str(e)}


def auto_remove_silence(
    file_path: str, 
    silence_threshold: float = -50.0, 
    min_silence_duration: float = 0.5
) -> Tuple[str, str]:
    """Automatically remove silence from audio file using advanced audio processing.
    
    Args:
        file_path: Path to audio file
        silence_threshold: Silence threshold in dB
        min_silence_duration: Minimum silence duration to remove in seconds
        
    Returns:
        tuple: (status_message, output_file_path)
    """
    if not AUDIO_PROCESSING_AVAILABLE:
        # Fallback behavior - just copy the file with a warning
        try:
            output_path = file_path.replace('.wav', '_cleaned.wav')
            shutil.copy2(file_path, output_path)
            return "⚠️ Audio processing libraries not available. File copied without cleaning. Install librosa and soundfile for real audio processing.", output_path
        except Exception as e:
            return f"❌ Error copying file: {str(e)}", ""
    
    try:
        # Load audio with librosa
        y, sr = librosa.load(file_path, sr=None)
        
        if len(y) == 0:
            return "❌ Audio file is empty", ""
        
        # Convert threshold from dB to amplitude
        # silence_threshold is in dB (e.g., -50 dB)
        threshold_amplitude = 10 ** (silence_threshold / 20)
        
        # Calculate frame length based on minimum silence duration
        frame_length = int(min_silence_duration * sr)
        hop_length = frame_length // 4  # 75% overlap
        
        # Calculate RMS energy for each frame
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Create time array for frames
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
        
        # Find frames above threshold (non-silent)
        non_silent_frames = rms > threshold_amplitude
        
        if not np.any(non_silent_frames):
            return "❌ Entire audio file is below silence threshold", ""
        
        # Find continuous segments of non-silent audio
        # Add padding to avoid cutting speech too close
        padding_frames = max(1, int(0.1 * sr / hop_length))  # 100ms padding
        
        # Expand non-silent regions
        expanded_mask = np.copy(non_silent_frames)
        for i in range(len(non_silent_frames)):
            if non_silent_frames[i]:
                start_pad = max(0, i - padding_frames)
                end_pad = min(len(expanded_mask), i + padding_frames + 1)
                expanded_mask[start_pad:end_pad] = True
        
        # Convert frame indices back to sample indices
        non_silent_samples = np.zeros(len(y), dtype=bool)
        for i, is_voice in enumerate(expanded_mask):
            if is_voice:
                start_sample = int(times[i] * sr) if i < len(times) else len(y)
                end_sample = int(times[i + 1] * sr) if i + 1 < len(times) else len(y)
                start_sample = max(0, min(start_sample, len(y)))
                end_sample = max(0, min(end_sample, len(y)))
                non_silent_samples[start_sample:end_sample] = True
        
        # Extract non-silent audio
        cleaned_audio = y[non_silent_samples]
        
        if len(cleaned_audio) == 0:
            return "❌ No audio remaining after silence removal", ""
        
        # Save cleaned audio
        output_path = file_path.replace('.wav', '_cleaned.wav')
        sf.write(output_path, cleaned_audio, sr)
        
        # Calculate statistics
        original_duration = len(y) / sr
        cleaned_duration = len(cleaned_audio) / sr
        removed_duration = original_duration - cleaned_duration
        percentage_removed = (removed_duration / original_duration) * 100
        
        return (
            f"✅ Silence removal completed! "
            f"Removed {removed_duration:.2f}s ({percentage_removed:.1f}%) of silence. "
            f"Final duration: {cleaned_duration:.2f}s",
            output_path
        )
        
    except Exception as e:
        return f"❌ Error removing silence: {str(e)}", ""


def normalize_audio_levels(
    file_path: str,
    target_lufs: float = -23.0,
    peak_limit: float = -1.0
) -> Tuple[str, str]:
    """Normalize audio levels to broadcast standards.
    
    Args:
        file_path: Path to audio file
        target_lufs: Target loudness in LUFS (default: -23 for broadcast)
        peak_limit: Peak limit in dB (default: -1.0)
        
    Returns:
        tuple: (status_message, output_file_path)
    """
    if not AUDIO_PROCESSING_AVAILABLE:
        return "❌ Audio processing libraries not available. Install librosa and soundfile.", ""
    
    try:
        # Load audio
        y, sr = librosa.load(file_path, sr=None)
        
        if len(y) == 0:
            return "❌ Audio file is empty", ""
        
        # Simple peak normalization (more advanced LUFS would require pyloudnorm)
        current_peak = np.max(np.abs(y))
        target_peak = 10 ** (peak_limit / 20)  # Convert dB to linear
        
        if current_peak > 0:
            # Normalize to target peak
            normalized_audio = y * (target_peak / current_peak)
        else:
            normalized_audio = y
        
        # Save normalized audio
        output_path = file_path.replace('.wav', '_normalized.wav')
        sf.write(output_path, normalized_audio, sr)
        
        # Calculate gain applied
        gain_db = 20 * np.log10(target_peak / current_peak) if current_peak > 0 else 0
        
        return (
            f"✅ Audio normalized! Applied {gain_db:+.2f} dB gain. "
            f"Peak level now at {peak_limit:.1f} dB.",
            output_path
        )
        
    except Exception as e:
        return f"❌ Error normalizing audio: {str(e)}", "" 