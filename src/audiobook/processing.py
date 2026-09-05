"""
Text processing utilities for audiobook generation.

Handles text chunking, validation, multi-voice parsing, and text cleanup.
"""

import re
import os
import wave
import numpy as np
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
    # Split text into sentences using regex to handle multiple punctuation marks
    sentences = re.split(r'([.!?]+\s*)', text)
    
    chunks = []
    current_chunk = ""
    current_word_count = 0
    
    i = 0
    while i < len(sentences):
        sentence = sentences[i].strip()
        if not sentence:
            i += 1
            continue
            
        # Add punctuation if it exists
        if i + 1 < len(sentences) and re.match(r'[.!?]+\s*', sentences[i + 1]):
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
        if re.match(r'^\[[^\]]+\]$', part_stripped): # It's a character tag
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
        
        # Skip chunks that are just punctuation or whitespace
        if not re.search(r'[a-zA-Z]', chunk):
            continue
        
        filtered_chunks.append(chunk)
    
    return filtered_chunks


# PHASE 4 REFACTOR: Adding audio processing functions to this module
# Originally from gradio_tts_app_audiobook.py save_audio_chunks() function

def save_audio_chunks(audio_chunks: List[np.ndarray], sample_rate: int, project_name: str, output_dir: str = "audiobook_projects") -> Tuple[List[str], str]:
    """
    Save audio chunks as numbered WAV files
    
    Args:
        audio_chunks: List of audio numpy arrays
        sample_rate: Sample rate for audio files
        project_name: Name of the project
        output_dir: Directory to save project files
        
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
    
    return saved_files, project_dir


# PHASE 4 REFACTOR: Adding extract_audio_segment function from gradio_tts_app_audiobook.py
def extract_audio_segment(audio_data, start_time: float = None, end_time: float = None) -> tuple:
    """Extract a segment from audio data.
    
    Args:
        audio_data: Numpy array of audio data
        start_time: Start time in seconds (None for beginning)
        end_time: End time in seconds (None for end)
        
    Returns:
        tuple: (status_message, extracted_audio_data)
    """
    try:
        sample_rate = 24000  # Default sample rate
        
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
    # Count line breaks (both \n and \r\n)
    return_count = text.count('\n') + text.count('\r')
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
    import re
    
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
    import re
    
    # Split into sentences using common sentence endings
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    
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
    import re
    
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
    import re
    
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

def chunk_multi_voice_text_with_line_break_priority(text: str, max_words: int = 30, pause_duration: float = 0.1) -> tuple:
    """Chunk multi-voice text with line breaks taking priority over sentence breaks.
    
    Args:
        text: Input text with voice tags and line breaks
        max_words: Maximum words per chunk
        pause_duration: Duration per line break in seconds
        
    Returns:
        tuple: (segments_with_pauses, total_pause_duration)
            segments_with_pauses: List of dicts with 'voice', 'text', and 'pause_duration' keys
            total_pause_duration: Total pause time across all segments
    """
    import re
    
    # Add debugging output for the input text
    print(f"🔍 DEBUG: chunk_multi_voice_text_with_line_break_priority input:")
    print(f"🔍 DEBUG: Input text length: {len(text)} characters")
    print(f"🔍 DEBUG: Line breaks in input: {text.count(chr(10))} \\n chars, {text.count(chr(13))} \\r chars")
    print(f"🔍 DEBUG: First 200 chars: {repr(text[:200])}")
    
    # NEW APPROACH: Process line breaks in the full text before voice parsing
    # Split the entire text by voice segments while preserving line breaks
    segments_with_pauses = []
    total_pause_duration = 0.0
    
    # Find all voice segments with their positions, preserving everything in between
    voice_pattern = r'(\[([^\]]+)\]\s*)'
    split_parts = re.split(voice_pattern, text)
    
    print(f"🔍 DEBUG: Split text into {len(split_parts)} parts")
    for i, part in enumerate(split_parts):
        print(f"🔍 DEBUG: Part {i}: {repr(part[:50])}")
    
    current_voice = None
    
    i = 0
    while i < len(split_parts):
        part = split_parts[i]
        
        # Check if this part is a voice tag match
        if i + 2 < len(split_parts) and re.match(r'\[([^\]]+)\]\s*', part):
            # This is a voice tag, extract the voice name
            current_voice = split_parts[i + 1]  # The captured voice name
            print(f"🔍 DEBUG: Found voice tag: '{current_voice}'")
            
            # The content is in the next part after the voice tag and whitespace
            content_part = split_parts[i + 2] if i + 2 < len(split_parts) else ""
            
            # Process the content with line break awareness
            if content_part:
                processed_segments = process_voice_content_with_line_breaks(
                    current_voice, content_part, max_words, pause_duration
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
                    current_voice, part, max_words, pause_duration
                )
                
                for segment in processed_segments:
                    segments_with_pauses.append(segment)
                    total_pause_duration += segment['pause_duration']
            elif not current_voice and part.strip():
                # Content before any voice tag - treat as narrator
                processed_segments = process_voice_content_with_line_breaks(
                    "Narrator", part, max_words, pause_duration
                )
                
                for segment in processed_segments:
                    segments_with_pauses.append(segment)
                    total_pause_duration += segment['pause_duration']
            
            i += 1
    
    print(f"🔍 DEBUG: Final result: {len(segments_with_pauses)} segments, {total_pause_duration:.1f}s total pause time")
    
    return segments_with_pauses, total_pause_duration


def process_voice_content_with_line_breaks(voice_name: str, content: str, max_words: int, pause_duration: float) -> list:
    """Process voice content while preserving line breaks for pauses."""
    import re
    
    segments = []
    
    # Split content by line breaks, keeping the line breaks
    line_segments = re.split(r'(\n+)', content)
    
    print(f"🔍 DEBUG: Processing voice '{voice_name}' content split into {len(line_segments)} line segments")
    
    for i, line_segment in enumerate(line_segments):
        if not line_segment:
            continue
            
        # Check if this segment is line breaks
        if re.match(r'\n+', line_segment):
            # Count the number of line breaks for pause calculation
            line_break_count = line_segment.count('\n')
            pause_time = line_break_count * pause_duration
            
            print(f"🔍 DEBUG: Found {line_break_count} line breaks, calculating {pause_time:.1f}s pause")
            
            # Add pause to the previous segment if it exists and has the same voice
            if segments and segments[-1]['voice'] == voice_name:
                segments[-1]['pause_duration'] += pause_time
                print(f"🔇 Line breaks detected in [{voice_name}]: +{pause_time:.1f}s pause (from {line_break_count} returns)")
            else:
                print(f"🔍 DEBUG: No previous segment to add pause to, or voice mismatch")
            continue
        
        # This is actual text content - chunk it by sentences if needed
        text_content = line_segment.strip()
        if not text_content:
            continue
            
        print(f"🔍 DEBUG: Processing text content: '{text_content[:50]}...'")
        
        # Apply sentence chunking to this segment
        text_chunks = chunk_text_by_sentences_local(text_content, max_words)
        
        print(f"🔍 DEBUG: chunk_text_by_sentences_local produced {len(text_chunks)} chunks")
        
        # Add these chunks with voice assignment and initial pause duration of 0
        for chunk in text_chunks:
            if chunk.strip():
                segments.append({
                    'voice': voice_name,
                    'text': chunk.strip(),
                    'pause_duration': 0.0
                })
                print(f"🔍 DEBUG: Added segment: voice='{voice_name}', text='{chunk.strip()[:30]}...', pause=0.0")
    
    return segments 