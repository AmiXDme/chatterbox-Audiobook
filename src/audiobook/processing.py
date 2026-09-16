"""
Text processing utilities for audiobook generation.

Handles text chunking, validation, multi-voice parsing, and text cleanup.
"""

import re
import os
import wave
import unicodedata
import numpy as np

try:
    from .langtext import (
        protect_all, normalize_text, finalize_text, is_voice_tag,
    )
except ImportError:  # repo-root on sys.path
    try:
        from src.audiobook.langtext import (
            protect_all, normalize_text, finalize_text, is_voice_tag,
        )
    except ImportError:  # langtext unavailable: voice features degrade safely
        protect_all = normalize_text = finalize_text = is_voice_tag = None
from pathlib import Path
from typing import List, Dict, Tuple, Any


def chunk_text_by_sentences(text: str, max_words: int = 50) -> List[str]:
    """Split text into chunks, breaking at sentence boundaries after reaching max_words.
    
    Args:
        text: Input text to chunk
        max_words: Maximum words per chunk
        
    Returns:
        List of text chunks
    """
    # Split text into sentences using regex to handle multiple punctuation marks.
    # Includes the Bengali/Hindi dari (।); harmless for Latin-only text.
    sentences = re.split(r'([.!?।]+\s*)', text)
    
    chunks = []
    current_chunk = ""
    current_word_count = 0
    
    i = 0
    while i < len(sentences):
        sentence = sentences[i].strip()
        if not sentence:
            i += 1
            continue
            
        # Add punctuation if it exists (dari । reattaches like . ! ?)
        if i + 1 < len(sentences) and re.match(r'[.!?।]+\s*', sentences[i + 1]):
            sentence += sentences[i + 1]
            i += 2
        else:
            i += 1
        
        sentence_words = len(sentence.split())
        
        # If adding this sentence would exceed max_words, start new chunk
        if current_word_count > 0 and current_word_count + sentence_words > max_words:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_word_count = sentence_words
        else:
            current_chunk += " " + sentence if current_chunk else sentence
            current_word_count += sentence_words
    
    # Add the last chunk if it exists
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


def adaptive_chunk_text(text: str, max_words: int = 50, reduce_on_error: bool = True) -> List[str]:
    """Adaptively chunk text with error handling.
    
    Args:
        text: Input text to chunk
        max_words: Maximum words per chunk
        reduce_on_error: Whether to reduce chunk size on errors
        
    Returns:
        List of text chunks
    """
    return chunk_text_by_sentences(text, max_words)


def load_text_file(file_path: str) -> Tuple[str, str]:
    """Load text content from a file with encoding detection.
    
    Args:
        file_path: Path to the text file
        
    Returns:
        tuple: (text_content, status_message)
    """
    if not file_path:
        return "", "No file selected"
    
    try:
        # Try UTF-8 first
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # Fallback to latin-1 for older files
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()
        
        if not content.strip():
            return "", "File is empty"
        
        return content.strip(), f"✅ Loaded {len(content.split())} words from file"
    
    except FileNotFoundError:
        return "", "❌ File not found"
    except Exception as e:
        return "", f"❌ Error reading file: {str(e)}"


def validate_audiobook_input(text_content: str, selected_voice: str, project_name: str) -> Tuple[bool, str]:
    """Validate input for single-voice audiobook creation.
    
    Args:
        text_content: Text to validate
        selected_voice: Selected voice name
        project_name: Project name
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if not text_content or not text_content.strip():
        return False, "❌ Please provide text content or upload a text file"
    
    if not selected_voice:
        return False, "❌ Please select a voice"
    
    if not project_name or not project_name.strip():
        return False, "❌ Please provide a project name"
    
    word_count = len(text_content.split())
    if word_count < 10:
        return False, "❌ Text content too short (minimum 10 words)"
    
    if word_count > 50000:
        return False, "❌ Text content too long (maximum 50,000 words for performance)"
    
    return True, ""


def parse_multi_voice_text(text: str) -> List[Dict[str, str]]:
    """Parse text with multi-voice format markers.
    
    Expected format:
    [CHARACTER_NAME] dialogue text (no colon needed, tag and dialogue can be on the same line)
    
    Args:
        text: Input text with character markers
        
    Returns:
        List of segments with character and text, e.g.,
        [{'character': 'Character1', 'text': 'Dialogue for character 1.'}, ...]
    """
    segments = []
    
    # Regex to find [CharacterName] tags
    # It captures the character name and the text that follows until the next tag or end of string
    # Using re.split to capture text between tags and the tags themselves
    parts = re.split(r'(\[[^\]]+\])', text)
    
    current_character = None
    buffer = ""
    
    for part in parts:
        if not part:
            continue # Skip empty parts that can result from re.split
            
        part_stripped = part.strip()
        # A [bracket] span is a character tag only if the name holds a letter;
        # footnotes like [১]/[2] fall through as ordinary text content.
        if (re.match(r'^\[[^\]]+\]$', part_stripped)
                and (is_voice_tag is None or is_voice_tag(part_stripped[1:-1]))):
            if current_character and buffer.strip():
                segments.append({
                    'character': current_character,
                    'text': buffer.strip()
                })
            current_character = part_stripped[1:-1] # Remove brackets
            buffer = ""
        else: # It's text content
            if current_character is None and part_stripped: # Text before any character tag
                # Assign to a default "Narrator" if no character tag precedes it.
                # This can be adjusted based on desired behavior for untagged leading text.
                segments.append({
                    'character': "Narrator", # Or None, if untagged leading text should be handled differently
                    'text': part_stripped
                })
                buffer = "" # Clear buffer as this part is processed
            elif current_character:
                buffer += part # Append to current character's text buffer
            # If no current_character and it's not leading text, this part might be ignored
            # or could be appended to a default narrator if strict tagging isn't enforced.
            # Current logic: only appends if current_character is set.
                
    # Add any remaining text in the buffer for the last character
    if current_character and buffer.strip():
        segments.append({
            'character': current_character,
            'text': buffer.strip()
        })
    elif not current_character and buffer.strip() and not segments: # Only if it's the *only* content
        # If the entire text has no tags, assign it all to Narrator
        segments.append({
            'character': "Narrator",
            'text': buffer.strip()
        })
        
    # Filter out any segments where the text is empty after stripping
    final_segments = [seg for seg in segments if seg['text']]
    
    # Debug: Print parsed segments by the module
    # print("Parsed Segments by text_processing.py module:", final_segments)
    return final_segments


def clean_character_name_from_text(text: str, voice_name: str) -> str:
    """Clean character name markers from text.
       The new parse_multi_voice_text in this module should handle name/dialogue separation.
       This function primarily acts as a pass-through or for minor cleanup.
    
    Args:
        text: Text that may contain character markers
        voice_name: Voice name (largely ignored by this simplified version)
        
    Returns:
        Cleaned text
    """
    # The parsing logic should have already separated the character name.
    # This function just ensures the text is stripped.
    return text.strip()


def chunk_multi_voice_segments(segments: List[Dict[str, str]], max_words: int = 50) -> List[Dict[str, str]]:
    """Chunk multi-voice segments while preserving character assignments.
    
    Args:
        segments: List of character segments
        max_words: Maximum words per chunk
        
    Returns:
        List of chunked segments with character assignments
    """
    chunked_segments = []
    
    for segment in segments:
        character = segment['character']
        text = segment['text']
        
        # Chunk the text for this character
        text_chunks = chunk_text_by_sentences(text, max_words)
        
        # Create segment for each chunk
        for chunk in text_chunks:
            chunked_segments.append({
                'character': character,
                'text': chunk
            })
    
    return chunked_segments


def validate_multi_voice_text(text_content: str, voice_library_path: str) -> Tuple[bool, str, List[str]]:
    """Validate multi-voice text format and extract characters.
    
    Args:
        text_content: Text to validate
        voice_library_path: Path to voice library
        
    Returns:
        tuple: (is_valid, error_message, character_list)
    """
    if not text_content or not text_content.strip():
        return False, "❌ Please provide text content", []
    
    # Parse segments to extract characters
    segments = parse_multi_voice_text(text_content)
    
    if not segments:
        return False, "❌ No valid character segments found. Use format: [CHARACTER_NAME]: dialogue", []
    
    # Extract unique characters
    characters = list(set(segment['character'] for segment in segments))
    
    if len(characters) < 2:
        return False, "❌ Multi-voice requires at least 2 different characters", characters
    
    if len(characters) > 6:
        return False, "❌ Too many characters (maximum 6 for performance)", characters
    
    # Check if we have enough text
    total_words = sum(len(segment['text'].split()) for segment in segments)
    if total_words < 20:
        return False, "❌ Not enough text content (minimum 20 words)", characters
    
    return True, "", characters


def validate_multi_audiobook_input(text_content: str, voice_library_path: str, project_name: str) -> Tuple[bool, str]:
    """Validate input for multi-voice audiobook creation.
    
    Args:
        text_content: Text to validate
        voice_library_path: Path to voice library
        project_name: Project name
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if not project_name or not project_name.strip():
        return False, "❌ Please provide a project name"
    
    is_valid, error_msg, _ = validate_multi_voice_text(text_content, voice_library_path)
    return is_valid, error_msg


def analyze_multi_voice_text(text_content: str, voice_library_path: str) -> Tuple[bool, str, Dict[str, int]]:
    """Analyze multi-voice text and return character statistics.
    
    Args:
        text_content: Text to analyze
        voice_library_path: Path to voice library
        
    Returns:
        tuple: (is_valid, message, character_counts)
    """
    is_valid, error_msg, characters = validate_multi_voice_text(text_content, voice_library_path)
    
    if not is_valid:
        return False, error_msg, {}
    
    # Parse segments and count words per character
    segments = parse_multi_voice_text(text_content)
    character_counts = {}
    
    for segment in segments:
        character = segment['character']
        word_count = len(segment['text'].split())
        character_counts[character] = character_counts.get(character, 0) + word_count
    
    total_words = sum(character_counts.values())
    message = f"✅ Found {len(characters)} characters with {total_words} total words"
    
    return True, message, character_counts


def _filter_problematic_short_chunks(chunks: List[str], voice_assignments: Dict[str, str]) -> List[str]:
    """Filter out problematic short chunks that might cause TTS issues.
    
    Args:
        chunks: List of text chunks
        voice_assignments: Character to voice mappings
        
    Returns:
        Filtered list of chunks
    """
    filtered_chunks = []
    min_length = 10  # Minimum character length
    
    for chunk in chunks:
        # Skip very short chunks
        if len(chunk.strip()) < min_length:
            continue
        
        # Skip chunks that are just punctuation or whitespace.
        # Script-aware: keep any chunk containing letters in ANY script
        # (Bengali/Devanagari/CJK/…). The old [a-zA-Z] test silently dropped
        # entire non-Latin books as "punctuation".
        if not re.search(r'[^\W\d_]', chunk, re.UNICODE):
            # ...unless it is a speakable digit run (postal codes, years).
            if not re.search(r'\d', chunk):
                continue
        
        filtered_chunks.append(chunk)
    
    return filtered_chunks


# PHASE 4 REFACTOR: Adding audio processing functions to this module
# Originally from gradio_tts_app_audiobook.py save_audio_chunks() function

def save_audio_chunks(audio_chunks: List[np.ndarray], sample_rate: int, project_name: str, output_dir: str = "audiobook_projects", start_index: int = 1) -> Tuple[List[str], str]:
    """
    Save audio chunks as numbered WAV files
    
    Args:
        audio_chunks: List of audio numpy arrays
        sample_rate: Sample rate for audio files
        project_name: Name of the project
        output_dir: Directory to save project files
        start_index: Starting number for file naming (default 1)
        
    Returns:
        tuple: (list of saved file paths, project directory path)
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
    
    for i, audio_chunk in enumerate(audio_chunks):
        filename = f"{safe_project_name}_{start_index + i:03d}.wav"
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
    
    return saved_files, project_dir


# PHASE 4 REFACTOR: Adding extract_audio_segment function from gradio_tts_app_audiobook.py
def extract_audio_segment(audio_data, start_time: float = None, end_time: float = None, sample_rate: int = 24000) -> tuple:
    """Extract a segment from audio data.
    
    Args:
        audio_data: Numpy array of audio data
        start_time: Start time in seconds (None for beginning)
        end_time: End time in seconds (None for end)
        sample_rate: Audio sample rate in Hz (default 24000)
        
    Returns:
        tuple: (status_message, extracted_audio_data)
    """
    try:
        
        if audio_data is None or len(audio_data) == 0:
            return "❌ No audio data to extract from", None
            
        total_duration = len(audio_data) / sample_rate
        
        start_sample = int(start_time * sample_rate) if start_time else 0
        end_sample = int(end_time * sample_rate) if end_time else len(audio_data)
        
        # Validate bounds
        start_sample = max(0, min(start_sample, len(audio_data)))
        end_sample = max(start_sample, min(end_sample, len(audio_data)))
        
        extracted_audio = audio_data[start_sample:end_sample]
        
        if len(extracted_audio) == 0:
            return "❌ Invalid time range - no audio extracted", None
            
        extracted_duration = len(extracted_audio) / sample_rate
        return f"✅ Extracted {extracted_duration:.2f}s of audio", extracted_audio
        
    except Exception as e:
        return f"❌ Error extracting audio segment: {str(e)}", None


def process_text_for_pauses(text: str, pause_duration: float = 0.1) -> tuple:
    """Process text to count returns and calculate total pause time.
    
    Args:
        text: Input text to process
        pause_duration: Duration in seconds per line break (default 0.1)
        
    Returns:
        tuple: (processed_text, return_count, total_pause_duration)
    """
    # Count line breaks — normalize first to avoid double-counting \r\n
    normalized = text.replace('\r\n', '\n')
    return_count = normalized.count('\n') + normalized.count('\r')
    total_pause_duration = return_count * pause_duration
    
    # Clean up text for TTS (normalize line breaks but keep content)
    processed_text = text.replace('\r\n', '\n').replace('\r', '\n')
    # Replace multiple consecutive newlines with single space to avoid empty chunks
    processed_text = re.sub(r'\n+', ' ', processed_text).strip()
    
    print(f"🔇 Detected {return_count} line breaks → {total_pause_duration:.1f}s total pause time")
    
    return processed_text, return_count, total_pause_duration


def create_silence_audio(duration: float, sample_rate: int = 24000) -> np.ndarray:
    """Create silence audio of specified duration.
    
    Args:
        duration: Duration in seconds
        sample_rate: Sample rate for the audio
        
    Returns:
        numpy array of silence audio
    """
    num_samples = int(duration * sample_rate)
    return np.zeros(num_samples, dtype=np.float32)


def insert_pauses_between_chunks(audio_chunks: List[np.ndarray], 
                                return_count: int, 
                                sample_rate: int = 24000,
                                pause_duration: float = 0.1) -> np.ndarray:
    """Insert pauses between audio chunks based on return count.
    
    Args:
        audio_chunks: List of audio chunk arrays
        return_count: Number of returns detected in original text
        sample_rate: Sample rate for audio
        pause_duration: Duration per return in seconds
        
    Returns:
        Combined audio with pauses inserted
    """
    if not audio_chunks:
        return np.array([], dtype=np.float32)
    
    if return_count == 0:
        # No pauses needed, just concatenate
        return np.concatenate(audio_chunks)
    
    # Calculate how to distribute pauses
    # For simplicity, we'll add all pause time at the end
    # In a more sophisticated approach, we could distribute pauses throughout
    total_pause_time = return_count * pause_duration
    pause_audio = create_silence_audio(total_pause_time, sample_rate)
    
    print(f"🔇 Adding {total_pause_time:.1f}s pause ({return_count} returns × {pause_duration}s each)")
    
    # Concatenate audio chunks with pause at the end
    combined_audio = np.concatenate(audio_chunks)
    final_audio = np.concatenate([combined_audio, pause_audio])
    
    return final_audio


def process_text_with_distributed_pauses(text: str, max_words: int = 50, 
                                        pause_duration: float = 0.1) -> tuple:
    """Process text and distribute pauses throughout chunks based on line breaks.
    
    Args:
        text: Input text to process
        max_words: Maximum words per chunk
        pause_duration: Duration per line break in seconds
        
    Returns:
        tuple: (chunks_with_pauses, total_return_count, total_pause_duration)
    """
    # First, process text to understand pause requirements
    processed_text, return_count, total_pause_duration = process_text_for_pauses(text, pause_duration)
    
    # Split into lines to track where pauses should be
    lines = text.split('\n')
    chunks_with_pauses = []
    
    current_chunk = ""
    current_word_count = 0
    pauses_for_chunk = 0
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            pauses_for_chunk += 1  # Empty line counts as a pause
            continue
            
        line_words = len(line.split())
        
        # If adding this line would exceed max_words, finalize current chunk
        if current_word_count > 0 and current_word_count + line_words > max_words:
            if current_chunk.strip():
                chunks_with_pauses.append({
                    'text': current_chunk.strip(),
                    'pauses': pauses_for_chunk
                })
            current_chunk = line
            current_word_count = line_words
            pauses_for_chunk = 0
        else:
            current_chunk += " " + line if current_chunk else line
            current_word_count += line_words
        
        # Add pause if not the last line
        if i < len(lines) - 1:
            pauses_for_chunk += 1
    
    # Add the last chunk if it exists
    if current_chunk.strip():
        chunks_with_pauses.append({
            'text': current_chunk.strip(),
            'pauses': pauses_for_chunk
        })
    
    return chunks_with_pauses, return_count, total_pause_duration


def map_line_breaks_to_chunks(original_text: str, chunks: List[str], pause_duration: float = 0.1) -> tuple:
    """Map line breaks from original text to corresponding chunks.
    
    Args:
        original_text: Original text with line breaks
        chunks: List of text chunks created by sentence chunking
        pause_duration: Duration per line break in seconds
        
    Returns:
        tuple: (chunk_pause_map, total_pause_duration)
            chunk_pause_map: Dict mapping chunk index to pause duration
            total_pause_duration: Total pause time across all chunks
    """
    chunk_pause_map = {}
    total_pause_duration = 0.0
    
    # Create a version of original text for matching (remove extra whitespace but keep structure)
    normalized_original = re.sub(r'\s+', ' ', original_text.replace('\n', ' ')).strip()
    
    # Track position in original text
    original_position = 0
    
    for chunk_idx, chunk in enumerate(chunks):
        chunk_normalized = chunk.strip()
        if not chunk_normalized:
            continue
            
        # Find this chunk in the original text
        chunk_start = normalized_original.find(chunk_normalized, original_position)
        if chunk_start == -1:
            # Fallback: try to find it without position constraint
            chunk_start = normalized_original.find(chunk_normalized)
        
        if chunk_start == -1:
            # Can't find chunk, no pauses for this one
            continue
            
        chunk_end = chunk_start + len(chunk_normalized)
        
        # Count line breaks in the corresponding section of original text
        # Map back to original text position
        orig_text_section_start = 0
        orig_text_section_end = len(original_text)
        
        # Find the corresponding section in original text
        words_before = len(normalized_original[:chunk_start].split())
        words_in_chunk = len(chunk_normalized.split())
        
        # Find the section in original text that corresponds to this chunk
        original_words = original_text.split()
        if words_before < len(original_words):
            # Find the start position in original text
            words_section = ' '.join(original_words[words_before:words_before + words_in_chunk])
            section_start = original_text.find(words_section)
            if section_start != -1:
                section_end = section_start + len(words_section)
                # Count line breaks in this section and the gap after it (until next chunk)
                next_chunk_start = section_end
                if chunk_idx < len(chunks) - 1:
                    next_chunk_text = chunks[chunk_idx + 1].strip()
                    next_chunk_pos = original_text.find(next_chunk_text, section_end)
                    if next_chunk_pos != -1:
                        next_chunk_start = next_chunk_pos
                
                # Count line breaks from end of current chunk to start of next chunk
                gap_text = original_text[section_end:next_chunk_start]
                line_breaks = gap_text.count('\n')
                
                if line_breaks > 0:
                    pause_time = line_breaks * pause_duration
                    chunk_pause_map[chunk_idx] = pause_time
                    total_pause_duration += pause_time
        
        original_position = chunk_end
    
    return chunk_pause_map, total_pause_duration 


def chunk_text_by_sentences_local(text, max_words=50):
    """Local copy of sentence chunking to avoid circular imports."""
    
    # Split into sentences using common sentence endings
    # Sentence enders include the Bengali/Hindi dari (।) alongside Latin marks.
    sentences = re.split(r'(?<=[.!?।])\s+', text.strip())
    
    chunks = []
    current_chunk = ""
    current_word_count = 0
    
    for sentence in sentences:
        if not sentence.strip():
            continue
            
        sentence_words = len(sentence.split())
        
        # If adding this sentence would exceed max_words and we have content, start a new chunk
        if current_word_count > 0 and current_word_count + sentence_words > max_words:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_word_count = sentence_words
        else:
            current_chunk += " " + sentence if current_chunk else sentence
            current_word_count += sentence_words
    
    # Add the last chunk if it exists
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

def chunk_text_with_line_break_priority(text: str, max_words: int = 50, pause_duration: float = 0.1) -> tuple:
    """Chunk text with line breaks taking priority over sentence breaks.
    
    This function first splits on line breaks, then applies sentence chunking
    within each line break segment if needed.
    
    Args:
        text: Input text with line breaks
        max_words: Maximum words per chunk
        pause_duration: Duration per line break in seconds
        
    Returns:
        tuple: (chunks_with_pauses, total_pause_duration)
            chunks_with_pauses: List of dicts with 'text' and 'pause_duration' keys
            total_pause_duration: Total pause time across all chunks
    """
    chunks_with_pauses = []
    total_pause_duration = 0.0
    
    # Split text by line breaks, keeping track of consecutive breaks
    line_segments = re.split(r'(\n+)', text)
    
    for i, segment in enumerate(line_segments):
        if not segment:
            continue
            
        # Check if this segment is line breaks
        if re.match(r'\n+', segment):
            # Count the number of line breaks for pause calculation
            line_break_count = segment.count('\n')
            pause_time = line_break_count * pause_duration
            
            # Add pause to the previous chunk if it exists
            if chunks_with_pauses:
                chunks_with_pauses[-1]['pause_duration'] += pause_time
                total_pause_duration += pause_time
                print(f"🔇 Line breaks detected: +{pause_time:.1f}s pause (from {line_break_count} returns)")
            continue
        
        # This is actual text content - chunk it by sentences if needed
        text_content = segment.strip()
        if not text_content:
            continue
            
        # Apply sentence chunking to this segment
        text_chunks = chunk_text_by_sentences_local(text_content, max_words)
        
        # Add these chunks with initial pause duration of 0
        for chunk in text_chunks:
            if chunk.strip():
                chunks_with_pauses.append({
                    'text': chunk.strip(),
                    'pause_duration': 0.0
                })
    
    return chunks_with_pauses, total_pause_duration 


def parse_multi_voice_text_local(text):
    """Local copy of multi-voice text parsing to avoid circular imports."""
    
    # Pattern to match [CharacterName] at the beginning of lines
    pattern = r'^\[([^\]]+)\]\s*(.*?)(?=^\[|\Z)'
    matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL)
    
    if not matches:
        # If no voice tags found, treat as single narrator
        return [("Narrator", text.strip())]
    
    segments = []
    for character_name, content in matches:
        # DON'T strip content to preserve line breaks for pause processing
        # Only strip leading/trailing spaces, but preserve newlines
        content = content.rstrip(' \t').lstrip(' \t')
        if content:
            segments.append((character_name.strip(), content))
    
    return segments

def chunk_multi_voice_text_with_line_break_priority(text: str, max_words: int = 30, pause_duration: float = 0.1, language_id: str = "en", natural: bool = False) -> tuple:
    """Chunk multi-voice text with line breaks taking priority over sentence breaks.
    
    Args:
        text: Input text with voice tags and line breaks (RAW — voice tags are
            parsed here first; per-voice content is protected/normalized inside
            process_voice_content_with_line_breaks, then restored before emit)
        max_words: Maximum words per chunk
        pause_duration: Duration per line break in seconds
        language_id: Locale for per-voice text processing
        natural: Opt-in spoken expansion (default preserve)
        
    Returns:
        tuple: (segments_with_pauses, total_pause_duration)
            segments_with_pauses: List of dicts with 'voice', 'text', and 'pause_duration' keys
            total_pause_duration: Total pause time across all segments
    """
    segments_with_pauses = []
    total_pause_duration = 0.0
    
    # Find all voice segments with their positions, preserving everything in between
    voice_pattern = r'(\[([^\]]+)\]\s*)'
    split_parts = re.split(voice_pattern, text)
    
    current_voice = None
    
    i = 0
    while i < len(split_parts):
        part = split_parts[i]
        
        # A [bracket] span is a voice tag only if the name holds a letter.
        # Footnotes like [১]/[2] fall through as ordinary narration content.
        if (i + 2 < len(split_parts) and re.match(r'\[([^\]]+)\]\s*', part)
                and (is_voice_tag is None or is_voice_tag(split_parts[i + 1]))):
            # This is a voice tag, extract the voice name
            current_voice = split_parts[i + 1]  # The captured voice name
            
            # The content is in the next part after the voice tag and whitespace
            content_part = split_parts[i + 2] if i + 2 < len(split_parts) else ""
            
            # Process the content with line break awareness
            if content_part:
                processed_segments = process_voice_content_with_line_breaks(
                    current_voice, content_part, max_words, pause_duration,
                    language_id=language_id, natural=natural
                )
                
                for segment in processed_segments:
                    segments_with_pauses.append(segment)
                    total_pause_duration += segment['pause_duration']
            
            i += 3  # Skip voice tag, voice name, and content
        else:
            # This is content between voice tags or before first voice tag
            if current_voice and part.strip():
                # Content continuation for current voice
                processed_segments = process_voice_content_with_line_breaks(
                    current_voice, part, max_words, pause_duration,
                    language_id=language_id, natural=natural
                )
                
                for segment in processed_segments:
                    segments_with_pauses.append(segment)
                    total_pause_duration += segment['pause_duration']
            elif not current_voice and part.strip():
                # Content before any voice tag - treat as narrator
                processed_segments = process_voice_content_with_line_breaks(
                    "Narrator", part, max_words, pause_duration,
                    language_id=language_id, natural=natural
                )
                
                for segment in processed_segments:
                    segments_with_pauses.append(segment)
                    total_pause_duration += segment['pause_duration']
            
            i += 1
    
    return segments_with_pauses, total_pause_duration


def process_voice_content_with_line_breaks(voice_name: str, content: str, max_words: int, pause_duration: float, language_id: str = "en", natural: bool = False) -> list:
    """Process voice content while preserving line breaks for pauses.

    Per-voice content is protected + normalized here (tags were already split
    on RAW text above), and each emitted chunk is restored + leak-asserted,
    so placeholders never leave this function.
    """
    segments = []

    # Protect + normalize this voice block (pauses use raw \n, counted below).
    _pctx = None
    if protect_all is not None:
        content, _pctx = protect_all(content, language_id, natural=natural)
        content = normalize_text(content, language_id, symbols=not natural)

    # Bangla: use the Bangla-first engine (grapheme-cluster safe, token-budgeted,
    # dari/॥/blank-line/soft-return pause cues preserved from raw \n). The classic
    # word-count split below stays for other languages.
    if language_id is not None and str(language_id).lower().startswith("bn"):
        for _bk in bangla_chunk_text(content, max_tokens=850):
            if _bk.get("text"):
                segments.append({
                    'voice': voice_name,
                    'text': _bk["text"].strip(),
                    'pause_duration': float(_bk.get("pause_before") or 0.0),
                })
        return segments

    # Split content by line breaks, keeping the line breaks
    line_segments = re.split(r'(\n+)', content)
    
    for i, line_segment in enumerate(line_segments):
        if not line_segment:
            continue
            
        # Check if this segment is line breaks
        if re.match(r'\n+', line_segment):
            # Count the number of line breaks for pause calculation
            line_break_count = line_segment.count('\n')
            pause_time = line_break_count * pause_duration
            
            # Add pause to the previous segment if it exists and has the same voice
            if segments and segments[-1]['voice'] == voice_name:
                segments[-1]['pause_duration'] += pause_time
                print(f"🔇 Line breaks detected in [{voice_name}]: +{pause_time:.1f}s pause (from {line_break_count} returns)")
            continue
        
        # This is actual text content - chunk it by sentences if needed
        text_content = line_segment.strip()
        if not text_content:
            continue
            
        # Apply sentence chunking to this segment
        text_chunks = chunk_text_by_sentences_local(text_content, max_words)
        
        # Add these chunks with voice assignment and initial pause duration of 0.
        # Restored + leak-asserted: no placeholder ever leaves this function.
        for chunk in text_chunks:
            if chunk.strip():
                if finalize_text is not None and _pctx is not None:
                    chunk = finalize_text(chunk, _pctx, where="multi-chunk")
                segments.append({
                    'voice': voice_name,
                    'text': chunk.strip(),
                    'pause_duration': 0.0
                })
    
    return segments 


# ============================================================================
# Bangla-first text engine
# ----------------------------------------------------------------------------
# The base Chatterbox TTS is English-born; these helpers make the text layer
# treat Bengali as Bengali (abugida, dari । sentence ender, lakh/crore digits,
# agglutinative morphology, grapheme-cluster boundaries) instead of
# "English without spaces". Everything below is deterministic and never raises.

_BN_RE = re.compile(r'[\u0980-\u09FF]')
_BN_SENT_END = re.compile(r'[।॥?!]$')

# Space is a phrase boundary, not a word boundary, in Bangla: attachable case
# postpositions / plural classifiers glue onto the noun. Listed longest-first,
# and only split when the remaining stem is >= 2 clusters (avoids shredding
# conjunct stacks like ক্ষ / জ্ঞ).
_BN_ATTACH = [
    'গুলিকেই', 'গুলোকেই', 'গুলোটাই', 'গুলোটাকে', 'গুলিটার', 'গুলোরই',
    'গুলোই', 'গুলোতে', 'গুলোকে', 'গুলোর', 'গুলির', 'গুলো', 'গুলা', 'গুলি',
    'খানেক', 'খানা', 'খানি', 'জনদের', 'য়ের', 'দের', 'ের',
]
_BN_ATTACH_RE = re.compile('|'.join(re.escape(s) for s in _BN_ATTACH))

# Pause hierarchy for chunk boundaries (seconds). , and — are deliberately NOT
# chunk boundaries: they're intra-clause cues the acoustic model renders.
BANGLA_PAUSE_SECONDS = {
    '।': 0.6, '?': 0.6, '!': 0.6,
    '॥': 1.2,
    '\n\n': 1.0,
    '\n': 0.15,
}

# Bangla numerals 0-99 (spoken). Used by bangla_digit_normalize.
_BN0_99 = {
    0: 'শূন্য', 1: 'এক', 2: 'দুই', 3: 'তিন', 4: 'চার', 5: 'পাঁচ', 6: 'ছয়',
    7: 'সাত', 8: 'আট', 9: 'নয়', 10: 'দশ', 11: 'এগারো', 12: 'বারো',
    13: 'তেরো', 14: 'চৌদ্দ', 15: 'পনেরো', 16: 'ষোল', 17: 'সতেরো',
    18: 'আঠারো', 19: 'উনিশ', 20: 'বিশ', 21: 'একুশ', 22: 'বাইশ', 23: 'তেইশ',
    24: 'চব্বিশ', 25: 'পঁচিশ', 26: 'ছাব্বিশ', 27: 'সাতাশ', 28: 'আটাশ',
    29: 'ঊনত্রিশ', 30: 'ত্রিশ', 31: 'একত্রিশ', 32: 'বত্রিশ', 33: 'তেত্রিশ',
    34: 'চৌত্রিশ', 35: 'পঁয়ত্রিশ', 36: 'ছত্রিশ', 37: 'সাঁইত্রিশ',
    38: 'আটত্রিশ', 39: 'ঊনচল্লিশ', 40: 'চল্লিশ', 41: 'একচল্লিশ',
    42: 'বিয়াল্লিশ', 43: 'তেতাল্লিশ', 44: 'চুয়াল্লিশ', 45: 'পঁয়তাল্লিশ',
    46: 'ছেচল্লিশ', 47: 'সাতচল্লিশ', 48: 'আটচল্লিশ', 49: 'ঊনপঞ্চাশ',
    50: 'পঞ্চাশ', 51: 'একান্ন', 52: 'বাহান্ন', 53: 'তিপ্পান্ন', 54: 'চুয়ান্ন',
    55: 'পঞ্চান্ন', 56: 'ছাপ্পান্ন', 57: 'সাতান্ন', 58: 'আটান্ন', 59: 'ঊনষাট',
    60: 'ষাট', 61: 'একষট্টি', 62: 'বাষট্টি', 63: 'তেষট্টি', 64: 'চৌষট্টি',
    65: 'পঁয়ষট্টি', 66: 'ছেষট্টি', 67: 'সাতষট্টি', 68: 'আটষট্টি',
    69: 'ঊনসত্তর', 70: 'সত্তর', 71: 'একাত্তর', 72: 'বাহাত্তর', 73: 'তিয়াত্তর',
    74: 'চুয়াত্তর', 75: 'পঁচাত্তর', 76: 'ছিয়াত্তর', 77: 'সাতাত্তর',
    78: 'আটাত্তর', 79: 'ঊনআশি', 80: 'আশি', 81: 'একাশি', 82: 'বিরাশি',
    83: 'তিরাশি', 84: 'চুরাশি', 85: 'পঁচাশি', 86: 'ছিয়াশি', 87: 'সাতাশি',
    88: 'আটাশি', 89: 'ঊননব্বই', 90: 'নব্বই', 91: 'একানব্বই', 92: 'বিরানব্বই',
    93: 'তিরানব্বই', 94: 'চুরানব্বই', 95: 'পঁচানব্বই', 96: 'ছিয়ানব্বই',
    97: 'সাতানব্বই', 98: 'আটানব্বই', 99: 'নিরানব্বই',
}
_BN_HUND = ['', 'একশো', 'দুইশো', 'তিনশো', 'চারশো', 'পাঁচশো', 'ছয়শো', 'সাতশো', 'আটশো', 'নয়শো']
_BN0_9 = [_BN0_99[i] for i in range(10)]
_BN_DIGIT_RE = re.compile(r'[0-9০-৯]+(?:[,.][0-9০-৯]+)*')

# Common Bengali words that genuinely belong (function words, pronouns, high-
# frequency verbs/nouns, polite forms). If the tokenizer reports them OOV they
# are usually a corpus artifact -- never letter-space them into garble.
_BN_COMMON = frozenset("""
আমি তুমি আপনি সে তিনি আমরা তোমরা আপনারা তারা ইনি উনি এরা ওরা
এই ওই সেই যে যা কে কী কেউ কিছু কার কাদের কাকে কীসে কীভাবে কেন কখন কোথায়
এখানে সেখানে ওখানে এখন তখন আবার পরে আগে প্রথম শেষ শেষে শুরু মাঝে মধ্যে সামনে পেছনে উপরে নিচে ভিতরে সাথে জন্য হয়ে হয়ে যায় গেল করে করেছে করছি করছেন হবে হল হয় হয়নি হয়েছে হচ্ছে
আছে ছিল ছিলাম আছে ছিলাম নেই হয়েছিল
না ও তো তাই সত্যি ঠিক আছে হ্যাঁ হয় হুম ঠিক নয় কই গিয়ে নিয়ে করে দিয়ে বেশি কম একটু অনেক খুব মোটে সত্যিই মনে কথা বলে বলল বললেন বলেন বলেছেন শুনে শোনা দেখে দেখা পড়ে পড়ছি লিখে লিখছি চলছে চলল গেল এল এলো দাও নাও নাওয়া খেলে খুলে দিয়ে নিয়ে হয়ে থাকা থাকবে থাকুন
সে দিন রাত সকাল দুপুর বিকাল সন্ধ্যা ভোর বেলা বছর মাস সপ্তাহ আবার কাল আজ কাল পরশু
প্রশ্ন উত্তর কথা গল্প বই লেখা কথা নাম ঠিকানা বাড়ি শহর গ্রাম দেশ ভাষা মানুষ জীবন মন প্রাণ ভালোবাসা
ভালো খারাপ সুন্দর বড় ছোট নতুন পুরনো দূর নিকট সোজা বাঁকা সাদা কালো লাল হলুদ সবুজ নীল
এক দুই তিন কিন্তু এবং আর বা নাকি যেন মনে হয় যেন কী
""".split())

# Extra honorifics / very common particles never letter-spaced regardless.
_BN_COMMON |= frozenset(['জনাব', 'জনাবা', 'শ্রী', 'শ্রীমতি', 'ডা', 'টি', 'টির',
                          'এসএমএস', 'ঢাকা', 'কলকাতা'])


def is_bangla_text(text):
    """True if the string contains any Bengali-script character."""
    return bool(text) and bool(_BN_RE.search(text))


def unicode_repair_bangla(text):
    """NFC-normalize; drop ZWJ/ZWNJ rendering hints and stray joiners.

    Keep hasanta conjuncts intact on the codepoint level (a cluster is one
    unit); stripping U+200C/U+200D only removes optional glyph-joining hints
    that fragment English-trained BPE vocabularies into UNK pieces.
    """
    if not text or not is_bangla_text(text):
        return text
    text = unicodedata.normalize('NFC', text)
    text = text.replace('\u200c', '').replace('\u200d', '')
    text = text.replace('\ufeff', '').replace('\u00a0', ' ')
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    return text


def _clusters(text):
    """Yield Bangla-safe grapheme clusters (base + combining marks + joiners)."""
    buf = ""
    for ch in text:
        cat = unicodedata.category(ch)
        if cat in ('Mn', 'Mc', 'Cf'):
            buf += ch
        else:
            if buf:
                yield buf
            buf = ch
    if buf:
        yield buf


def bangla_word_segments(text):
    """Morph-aware word tokens for scanning, NOT for altering spoken text.

    Splits on spaces/punctuation, then strips only word-FINAL attached case
    postpositions / plural classifiers so the UNK scan sees the stem.
    Conjunct stacks, glued chains, and short stems are never shredded.
    """
    if not is_bangla_text(text):
        return [w for w in text.split() if w]
    parts = re.split(r'([\u0980-\u09FF]+)', text)
    words = []
    for part in parts:
        if not part:
            continue
        if not _BN_RE.search(part):
            words.extend([w for w in re.split(r'[\s\W_]+', part) if w])
            continue
        stem = part
        while True:
            hit = None
            for suf in _BN_ATTACH:
                if stem.endswith(suf):
                    hit = suf
                    break
            if not hit or len(list(_clusters(stem[:-len(hit)]))) < 2:
                break
            stem = stem[:-len(hit)]
        words.append(stem)
    return [w for w in words if w]


def _bn_num_words(n):
    """Convert an int to spoken Bangla with lakh/crore Indian grouping."""
    if n < 0:
        return 'মাইনাস ' + _bn_num_words(-n)
    if n < 100:
        return _BN0_99[n]
    parts = []
    crore = n // 10000000
    n %= 10000000
    lakh = n // 100000
    n %= 100000
    thousand = n // 1000
    n %= 1000
    hundred = n // 100
    rest = n % 100
    if crore:
        parts.append((_bn_num_words(crore) if crore >= 100 else _BN0_99[crore]) + ' কোটি')
    if lakh:
        parts.append((_bn_num_words(lakh) if lakh >= 100 else _BN0_99[lakh]) + ' লাখ')
    if thousand:
        parts.append((_bn_num_words(thousand) if thousand >= 100 else _BN0_99[thousand]) + ' হাজার')
    if hundred:
        parts.append(_BN_HUND[hundred])
    if rest:
        parts.append(_BN0_99[rest])
    return ' '.join(parts) if parts else _BN0_99[n]


_BN_MONTHS = ['', 'জানুয়ারি', 'ফেব্রুয়ারি', 'মার্চ', 'এপ্রিল', 'মে', 'জুন',
              'জুলাই', 'আগস্ট', 'সেপ্টেম্বর', 'অক্টোবর', 'নভেম্বর', 'ডিসেম্বর']

_BN_CURRENCY_NAMES = {
    '৳': 'টাকা', 'Tk': 'টাকা', 'TK': 'টাকা', 'BDT': 'টাকা',
    '₹': 'রুপি', '$': 'ডলার', '€': 'ইউরো', '£': 'পাউন্ড', '¥': 'ইয়েন',
}
_BN_COIN_NAMES = {
    'টাকা': 'পয়সা', 'রুপি': 'পয়সা', 'ডলার': 'সেন্ট',
    'ইউরো': 'সেন্ট', 'পাউন্ড': 'পেনি', 'ইয়েন': 'সেন',
}

_BN_ACRONYM = {
    'API': 'এপিআই', 'GPU': 'জিপিইউ', 'CPU': 'সিপিইউ', 'AI': 'এআই',
    'TTS': 'টি টি এস', 'ASR': 'এ এস আর', 'NID': 'এনআইডি', 'BIN': 'বিন',
    'NIN': 'এন আই এন', 'USB': 'ইউএসবি', 'PDF': 'পিডিএফ', 'HTML': 'এইচটিএমএল',
    'URL': 'ইউআরএল', 'RAM': 'র্যাম', 'TV': 'টিভি', 'BBC': 'বিবিসি',
    'MBBS': 'এমবিবিএস', 'MD': 'এমডি', 'GPS': 'জিপিএস', 'ATM': 'এটিএম',
    'SMS': 'এসএমএস', 'SIM': 'সিম', 'OTP': 'ওটিপি', 'PIN': 'পিন',
}

_BN_UNIT = {
    'km/h': 'কিলোমিটার প্রতি ঘণ্টা', 'km': 'কিলোমিটার', 'cm': 'সেন্টিমিটার',
    'mm': 'মিলিমিটার', 'kg': 'কিলোগ্রাম', 'mg': 'মিলিগ্রাম', 'g': 'গ্রাম',
    'ml': 'মিলিলিটার', 'L': 'লিটার', 'l': 'লিটার', 's': 'সেকেন্ড',
    'min': 'মিনিট', 'hr': 'ঘণ্টা', 'h': 'ঘণ্টা', 'kcal': 'কিলোক্যালরি',
    'KB': 'কিলোবাইট', 'MB': 'মেগাবাইট', 'GB': 'গিগাবাইট', 'TB': 'টেরাবাইট',
    '°C': 'ডিগ্রি সেলসিয়াস', '℃': 'ডিগ্রি সেলসিয়াস', '°F': 'ডিগ্রি ফারেনহাইট',
    '%': 'শতাংশ',
}

_BN_DIG_AS = str.maketrans('০১২৩৪৫৬৭৮৯', '0123456789')


def _bn_to_ascii_digits(s):
    """০-৯ -> 0-9, keep every other character untouched."""
    return s.translate(_BN_DIG_AS)


def _bn_read_amount(amount_str):
    """'5' | '৫,০০০' | '3.14' -> spoken Bangla (মাইনাস for a - prefix)."""
    sign = ''
    if amount_str.startswith('-'):
        sign = 'মাইনাস '
        amount_str = amount_str[1:]
    whole, dot, frac = amount_str.partition('.')
    whole_d = _bn_to_ascii_digits(whole).replace(',', '')
    if frac:
        frac_d = _bn_to_ascii_digits(frac).replace(',', '')
        words = [sign + _bn_num_words(int(whole_d)), 'দশমিক']
        for ch in frac_d:
            words.append(_BN0_9[int(ch)])
        return ' '.join(words)
    # Leading-zero codes (phones, roll numbers) read digit-by-digit like a
    # narrator; a single "0" is just শূন্য.
    if len(whole_d) > 1 and whole_d.startswith('0'):
        return sign + ' '.join(_BN0_9[int(ch)] for ch in whole_d)
    return sign + _bn_num_words(int(whole_d))


def _bn_year_words(year):
    """Narrated year near সালে/সাল: 1987 -> উনিশশো সাতাশি, 2026 -> দুই হাজার ছাব্বিশ."""
    if 1100 <= year <= 1999:
        century = year // 100
        rest = year % 100
        base = _BN0_99[century] + 'শো'
        return base + ((' ' + _BN0_99[rest]) if rest else '')
    if 2000 <= year <= 2099:
        rest = year % 100
        base = 'দুই হাজার'
        return base + ((' ' + _BN0_99[rest]) if rest else '')
    return _bn_num_words(year)


def _bn_ordinal(num, suffix):
    """Speech form of ordinal suffixes: ৪র্থ, ২য়, ১ম, ৫ই, ৩রা, ৪ঠা, Nতম."""
    basic = {1: 'প্রথম', 2: 'দ্বিতীয়', 3: 'তৃতীয়', 4: 'চতুর্থ', 5: 'পঞ্চম',
             6: 'ষষ্ঠ', 7: 'সপ্তম', 8: 'অষ্টম', 9: 'নবম', 10: 'দশম'}
    date_form = {1: 'পহেলা', 2: 'দোসরা', 3: 'তেসরা', 4: 'চৌঠা'}
    n = int(_bn_to_ascii_digits(num))
    if suffix == 'ম':
        return basic.get(n, _bn_num_words(n) + 'তম')
    if suffix == 'র্থ':
        return basic.get(n, _bn_num_words(n) + 'র্থ')
    if suffix in ('য়', 'তম'):
        return basic.get(n, _bn_num_words(n) + 'তম')
    if suffix == 'শে':
        return _bn_num_words(n) + 'ে'
    if suffix in ('লা', 'রা', 'ঠা', 'ই'):
        return date_form.get(n, _bn_num_words(n) + 'ই')
    return _bn_num_words(n)


def _bn_time_words(hour, minute, has_am_pm, am_pm):
    """Clock phrase: 7:30 -> সাড়ে সাতটা, 1:30 -> দেড়টা, 6:23 -> ছয়টা বেজে তেইশ মিনিট,
    6:40 -> সাতটা বাজতে বিশ মিনিট বাকি. Standard hour stems retained; durations
    never use ঘণ্টা.
    """
    h = int(_bn_to_ascii_digits(str(hour)))
    m = int(_bn_to_ascii_digits(str(minute)))
    if not (0 <= h <= 24 and 0 <= m <= 59):
        return None
    h12 = h % 12 or 12
    if h == 0:
        h12 = 12
    if m == 0:
        phrase = _bn_num_words(h12) + 'টা'
    elif m == 15:
        phrase = 'সোয়া ' + _bn_num_words(h12) + 'টা'
    elif m == 30 and h12 == 1:
        phrase = 'দেড়টা'
    elif m == 30:
        phrase = 'সাড়ে ' + _bn_num_words(h12) + 'টা'
    elif m == 45:
        nxt = (h % 12) + 1
        phrase = 'পৌনে ' + _bn_num_words(nxt) + 'টা'
    elif m <= 30:
        phrase = _bn_num_words(h12) + 'টা বেজে ' + _bn_num_words(m) + ' মিনিট'
    else:
        nxt = (h12 % 12) + 1
        phrase = _bn_num_words(nxt) + 'টা বাজতে ' + _bn_num_words(60 - m) + ' মিনিট বাকি'
    if not has_am_pm:
        return phrase
    if am_pm and am_pm.lower() in ('am', 'pm'):
        if h < 6:
            daypart = 'ভোরে'
        elif h < 12:
            daypart = 'সকালে'
        elif h < 15:
            daypart = 'দুপুরে'
        elif h < 18:
            daypart = 'বিকালে'
        elif h < 20:
            daypart = 'সন্ধ্যায়'
        else:
            daypart = 'রাতে'
    else:
        daypart = {'সকাল': 'সকালে', 'বিকাল': 'বিকালে', 'সন্ধ্যা': 'সন্ধ্যায়',
                   'রাত': 'রাতে'}.get(am_pm, '')
    return (daypart + ' ' + phrase) if daypart else phrase


def bangla_normalize_text(text):
    """Deterministic Bangla text normalization (stage 1, superset of digits).

    Runs BEFORE any Gemini pass so the language model sees letters, never digit
    shapes. Covers: percentages, currencies (৳ ₹ $ € £ ¥ Tk/BDT), clock times
    (সাড়ে/সোয়া/পৌনে/দেড় + 24h folding), dates (dd/mm/yyyy incl. Bangla
    digits), ordinals (৪র্থ/২য়/৩রা/৫ই), context years (২০২৬ সালে), units
    (km/kg/L/°C...), decimals (৩.১৪ -> তিন দশমিক এক চার), lakh/crore integer
    grouping, leading-zero phones read digit-by-digit, negatives, and a small
    trusted acronym dictionary. Forms match the Master-Prompt ruleset exactly
    (ষোল/আটাশ/তিপ্পান্ন canonical). Never raises.
    """
    if not text or not is_bangla_text(text):
        return text
    text = unicode_repair_bangla(text)

    for key, rep in _BN_ACRONYM.items():
        text = re.sub(r'(?<![A-Za-z])' + key + r'(?![A-Za-z])', rep, text)

    text = re.sub(r'([0-9০-৯]+)\s*%', lambda m: _bn_read_amount(m.group(1)) + ' শতাংশ', text)

    def _cur(m):
        amt, sym, suf = m.group('amt'), m.group('sym'), m.group('cur')
        if sym:
            cur = _BN_CURRENCY_NAMES.get(sym, 'টাকা')
        else:
            cur = _BN_CURRENCY_NAMES.get(suf, 'টাকা')
        frac = ''
        whole_s = amt
        if '.' in amt:
            whole_s, _, frac = amt.partition('.')
        whole_i = _bn_to_ascii_digits(whole_s).replace(',', '')
        n = int(whole_i)
        coin = _BN_COIN_NAMES.get(cur, 'পয়সা')
        if frac:
            two = (_bn_to_ascii_digits(frac).replace(',', '') + '00')[:2]
            p = int(two)
            if p:
                return (_bn_read_amount(whole_s) + ' ' + cur + ' ' +
                        _BN0_99[p] + ' ' + coin)
            return _bn_read_amount(whole_s) + ' ' + cur
        if n == 0:
            return _BN0_99[0] + ' ' + cur
        return _bn_num_words(n) + ' ' + cur

    text = re.sub(
        r'(?P<sym>৳|₹|Tk|TK|BDT|\$|€|£|¥)\s*(?P<amt>[0-9০-৯]+(?:[,.][0-9০-৯]+)*)'
        r'|(?P<amt2>[0-9০-৯]+(?:\.[0-9০-৯]+)?)\s*(?P<cur>টাকা|৳|Tk|TK|BDT)\b',
        lambda m: _cur(m) if (m.group('amt') or m.group('amt2')) else m.group(0),
        text,
    )

    text = re.sub(
        r'(?P<h>[0-9০-৯]{1,2}):(?P<m>[0-9০-৯]{1,2})(?::[0-9০-৯]{1,2})?'
        r'(?:(?P<ap> ?(?:AM|PM|am|pm|সকাল|বিকাল|সন্ধ্যা|রাত)))?',
        lambda m: (lambda p, ap: p if p else m.group(0))(
            _bn_time_words(m.group('h'), m.group('m'), bool((m.group('ap') or '').strip()),
                           (m.group('ap') or '').strip()),
            (m.group('ap') or '').strip()),
        text,
    )

    text = re.sub(
        r'([0-9০-৯]{1,2})(ম|য়|র্থ|লা|রা|ঠা|ই|শে|তম)',
        lambda m: _bn_ordinal(m.group(1), m.group(2)),
        text,
    )

    def _date(m):
        day = int(_bn_to_ascii_digits(m.group('d')))
        mon = int(_bn_to_ascii_digits(m.group('mo')))
        if not (1 <= day <= 31 and 1 <= mon <= 12):
            return m.group(0)
        base = _bn_num_words(day) + ' ' + _BN_MONTHS[mon]
        if m.group('y'):
            yr = int(_bn_to_ascii_digits(m.group('y')))
            if yr < 70:
                yr += 2000
            elif yr < 100:
                yr += 1900
            return base + ' ' + _bn_year_words(yr)
        return base

    # Slash dates may omit the year; dot/dash forms LOOK like decimals/ranges so
    # they only count as dates when a year is present (no false 3.14 -> 3rd June).
    text = re.sub(
        r'(?P<d>[0-9০-৯]{1,2})/(?P<mo>[0-9০-৯]{1,2})(?:/(?P<y>[0-9০-৯]{2,4}))?',
        _date, text,
    )
    text = re.sub(
        r'(?P<d>[0-9০-৯]{1,2})[.\-](?P<mo>[0-9০-৯]{1,2})[.\-](?P<y>[0-9০-৯]{2,4})',
        _date, text,
    )

    # Ranges: ৫-৭ জন -> পাঁচ থেকে সাত জন
    text = re.sub(
        r'(?P<a>[0-9০-৯]+)\s*[-–—]\s*(?P<b>[0-9০-৯]+)',
        lambda m: (_bn_read_amount(m.group('a')) + ' থেকে ' + _bn_read_amount(m.group('b'))),
        text,
    )

    def _yr(m):
        yr = int(_bn_to_ascii_digits(m.group('n')))
        return _bn_year_words(yr) + m.group('ctx')

    text = re.sub(
        r'(?P<n>[0-9০-৯]+)(?P<ctx>\s*(?:সাল|সালে|সন|সনে|খ্রিস্টাব্দ|খ্রিস্টাব্দে))',
        lambda m: _bn_year_words(int(_bn_to_ascii_digits(m.group('n')))) + m.group('ctx'),
        text,
    )

    def _unit(m):
        return _bn_read_amount(m.group('amt')) + ' ' + _BN_UNIT[m.group('u')]

    text = re.sub(
        r'(?P<amt>-?(?:[0-9০-৯]+(?:[.,][0-9০-৯]+)*))\s*'
        r'(?P<u>km/h|km|cm|mm|mg|kg|ml|kcal|KB|MB|GB|TB|min|°C|℃|°F|L|l|s|g|m|h)\b',
        _unit, text,
    )

    text = re.sub(
        r'(?<![A-Za-z])-?(?:[0-9০-৯]+(?:,[0-9০-৯]{1,3})*(?:\.[0-9০-৯]+)?)(?![A-Za-z])',
        lambda m: _bn_read_amount(m.group(0)), text,
    )

    return text


def bangla_digit_normalize(text):
    """Backwards-compatible name; see bangla_normalize_text (full stage 1)."""
    return bangla_normalize_text(text)


def bangla_pause_duration(cue):
    """Seconds of silence for a boundary cue (unknown → 0)."""
    return BANGLA_PAUSE_SECONDS.get(cue, 0.0)


def _bangla_letter_runs(text):
    """Full raw Bangla word tokens (incl. attached suffixes), letter-only.
    Excludes digit sequences so numbers are never letter-spaced."""
    for part in re.split(r'([\u0980-\u09FF]+)', text):
        if part and re.search(r'[\u0985-\u09CE]', part):
            yield part


def letterspace_unknown_bangla(model, text, language_id="bn", max_words=8):
    """Opt-in, Bangla-only: spell out-of-vocab words letter-by-letter like a
    narrator ("ক-ম-ল") instead of letting the model garble or skip them.

    Only fully unknown word tokens are touched (tokenizer reports [UNK]); if
    more than max_words are unknown the text is left unchanged (over-spacing
    would degrade prosody). Returns (text2, {original: spaced}). Never raises.
    """
    if not text or not is_bangla_text(text):
        return text, {}
    tok = getattr(model, "tokenizer", None)
    if tok is None:
        return text, {}
    vocab = tok.tokenizer.get_vocab() if hasattr(tok, "tokenizer") else {}
    unk_id = vocab.get("[UNK]")
    if unk_id is None:
        return text, {}

    candidates = []
    for w in _bangla_letter_runs(text):
        if w in _BN_COMMON:
            continue
        if len(list(_clusters(w))) < 2:
            continue
        try:
            try:
                ids = tok.encode(w, language_id=language_id)
            except TypeError:
                ids = tok.encode(w)
        except Exception:
            continue
        ids = list(ids[0]) if hasattr(ids, "dim") else list(ids)
        if any(int(i) == unk_id for i in ids):
            candidates.append(w)

    if not candidates:
        return text, {}
    if len(candidates) > max_words:
        print(f"🔤 [BN] {len(candidates)} unknown words — above the {max_words} "
              f"letter-spacing cap, leaving text unchanged.", flush=True)
        return text, {}

    out = text
    touched = {}
    for w in candidates:
        spaced = ' '.join(_clusters(w))
        out = re.sub(re.escape(w), spaced, out)
        touched[w] = spaced
    return out, touched


def _bn_count_tokens(s, tokenizer):
    try:
        if tokenizer is not None:
            t = tokenizer.text_to_tokens(s)
            n = int(t.shape[-1])
            if n:
                return n
    except Exception:
        pass
    return max(1, (len(s) // 2) + 1)


def bangla_chunk_text(text, tokenizer=None, max_tokens=500, max_clusters=2400):
    """Bangla-aware segmentation returning [{text, pause_before}].

    - Never cuts inside a grapheme cluster (hasanta conjunct / vowel matra).
    - Chunk budget is T3 *text tokens* (not English "words") when a tokenizer
      is passed, else grapheme clusters — so the 1000-token AR cap is respected.
    - Boundary cues follow BANGLA_PAUSE_SECONDS (dari 0.6s, ॥ 1.2s, blank line
      1.0s, soft newline 0.15s). Punctuation is kept on the chunk so the model
      gets the intonation cue; the pause is emitted BEFORE the next chunk.
    """
    text = unicode_repair_bangla(text)
    if not text:
        return []
    text = re.sub(r'\n{2,}', '\n\n', text)

    cues = []  # (text_or_None, cue)
    for piece in re.split(r'(\n\n+)', text):
        if not piece:
            continue
        if '\n\n' in piece:
            cues.append((None, '\n\n'))
            continue
        for line in re.split(r'(\n)', piece):
            if not line:
                continue
            if line == '\n':
                cues.append((None, '\n'))
                continue
            for sent in re.split(r'(?<=[।॥?!])\s*', line):
                if not sent.strip():
                    continue
                cue = '।' if sent.rstrip().endswith(('।', '?', '!', '॥')) else None
                cues.append((sent.strip(), cue))

    chunks = []
    buf = ''
    buf_tok = 0
    for raw_sent, cue in cues:
        if raw_sent is None:
            if buf:
                chunks.append({'text': buf, 'pause_before': bangla_pause_duration(cue) if cue else 0.0})
                buf = ''
                buf_tok = 0
            elif chunks:
                chunks[-1]['pause_before'] += bangla_pause_duration(cue)
            elif cue:
                chunks.append({'text': '', 'pause_before': bangla_pause_duration(cue)})
            continue
        n_tok = _bn_count_tokens(raw_sent, tokenizer)
        if buf and buf_tok + n_tok > max_tokens:
            chunks.append({'text': buf, 'pause_before': 0.0})
            buf = ''
            buf_tok = 0
        if n_tok > max_tokens:
            cls = list(_clusters(raw_sent))
            step = max(80, int(max_tokens * 1.2))
            start = 0
            while start < len(cls):
                end = min(start + step, len(cls))
                cut = end
                if end < len(cls):
                    for i in range(end, max(start, end - 80), -1):
                        if i < len(cls) and cls[i] in (' ', '\u00a0'):
                            cut = i + 1
                            break
                sub = ''.join(cls[start:cut]).strip()
                if sub:
                    if start:
                        chunks.append({'text': '', 'pause_before': 0.0})
                    chunks.append({'text': sub, 'pause_before': 0.0})
                start = cut
            continue
        buf += ((' ' if buf else '') + raw_sent)
        buf_tok += n_tok
        if len(list(_clusters(buf))) >= max_clusters and buf_tok >= max_tokens:
            chunks.append({'text': buf, 'pause_before': 0.0})
            buf = ''
            buf_tok = 0

    if buf:
        chunks.append({'text': buf, 'pause_before': 0.0})

    if not chunks:
        return []

    resolved = [c for c in chunks if c['text']]
    for i in range(len(resolved)):
        if not resolved[i]['pause_before']:
            m = re.search(r'([।॥?!])$', resolved[i]['text'].strip())
            if m and i > 0:
                resolved[i]['pause_before'] = BANGLA_PAUSE_SECONDS[m.group(1)]
    return resolved 