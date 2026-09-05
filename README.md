# 🎧 Chatterbox Audiobook Generator

**This is a work in progress. You can consider this a pre-launch repo at the moment, but if you find bugs, please put them in the issues area. Thank you.**
**Transform your text into high-quality audiobooks with advanced TTS models, voice cloning, and professional volume normalization.**

## 🚀 Quick Start

### 1. Install Dependencies
```bash
./install-audiobook.bat
```

### 3. Launch the Application
```bash
./launch_audiobook.bat
```

The web interface will open automatically in your browser at `http://localhost:7860`

---

## ✨ Features

### 📚 **Audiobook Creation**
- **Single Voice**: Generate entire audiobooks with one consistent voice
- **Multi-Voice**: Create dynamic audiobooks with multiple characters
- **Custom Voices**: Clone voices from audio samples for personalized narration
- **Professional Volume Normalization**: Ensure consistent audio levels across all voices
- **📋 Text Queuing System** ⭐ *NEW*: Upload books in any size chapters and generate continuously
- **🔄 Chunk-Based Processing** ⭐ *NEW*: Improved reliability for longer text generations

### 🎵 **Audio Processing**
- **Smart Cleanup**: Remove unwanted silence and audio artifacts
- **Volume Normalization**: Professional-grade volume balancing for all voices
- **Real-time Audio Analysis**: Live volume level monitoring and feedback
- **Preview System**: Test settings before applying to entire projects
- **Batch Processing**: Process multiple projects efficiently
- **Quality Control**: Advanced audio optimization tools
- **🎯 Enhanced Audio Quality** ⭐ *NEW*: Improved P-top and minimum P parameters for better voice generation

### 🎭 **Voice Management**
- **Voice Library**: Organize and manage your voice collection
- **Voice Cloning**: Create custom voices from audio samples
- **Volume Settings**: Configure target volume levels for each voice
- **Professional Presets**: Industry-standard volume levels (audiobook, podcast, broadcast)
- **Character Assignment**: Map specific voices to story characters

### 📊 **Volume Normalization System** ⭐ *NEW*
- **Professional Standards**: Audiobook (-18 dB), Podcast (-16 dB), Broadcast (-23 dB) presets
- **Consistent Character Voices**: All characters maintain the same volume level
- **Real-time Analysis**: Color-coded volume status with RMS and peak level display
- **Retroactive Normalization**: Apply volume settings to existing voice projects
- **Multi-Voice Support**: Batch normalize all voices in multi-character audiobooks
- **Soft Limiting**: Intelligent audio limiting to prevent distortion

### 📖 **Text Processing**
- **Chapter Support**: Automatic chapter detection and organization
- **Multi-Voice Parsing**: Parse character dialogue automatically
- **Text Validation**: Ensure proper formatting before generation
- **📋 Queue Management** ⭐ *NEW*: Batch process multiple text files sequentially
- **🔇 Return Pause System** ⭐ *NEW*: Automatic pause insertion based on line breaks for natural speech flow

---

## 🎭 Custom Audiobook Processing Pipeline ⭐ *NEW*

Our advanced text processing pipeline transforms your written content into natural-sounding audiobooks with intelligent pause placement and character flow management.

### 🔇 **Return Pause System**

**Automatic pause insertion based on your text formatting** - Every line break (`\n`) in your text automatically adds a 0.1-second pause to the generated audio, creating natural speech rhythms without manual intervention.

#### **How It Works**
- **Line Break Detection**: System automatically counts all line breaks in your text
- **Pause Calculation**: Each return adds exactly 0.1 seconds of silence
- **Accumulative Pauses**: Multiple consecutive line breaks create longer pauses
- **Universal Support**: Works with single-voice, multi-voice, and batch processing

#### **Example Text Formatting**
```
[Narrator] The sun was setting over the hills.

[Character1] "We need to find shelter soon."

[Character2] "I see a cave up ahead.
Let's hurry before it gets dark."


[Narrator] They rushed toward the cave, hearts pounding.
```
**Result**: Natural pauses between dialogue, emphasis pauses for dramatic effect, and smooth character transitions.

### 📝 **Text Formatting Best Practices**

#### **🎭 Multi-Voice Dialogue Structure**
```
[Character Name] Dialogue content here.

[Another Character] Response content here.
Multiple lines can be used for the same character.

[Narrator] Descriptive text and scene setting.
```

#### **🎪 Natural Flow Techniques**
- **Paragraph Breaks**: Use double line breaks for scene transitions
- **Emphasis Pauses**: Add extra returns before important revelations
- **Character Separation**: Single returns between different speakers
- **Breathing Room**: Natural pauses for complex concepts or emotional moments

#### **📖 Single Voice Formatting**
```
Chapter content flows naturally here.

New paragraphs create natural pauses.


Extended pauses can emphasize dramatic moments.

Regular text continues with normal pacing.
```

### 🔄 **Processing Pipeline Features**

#### **🧠 Intelligent Text Analysis**
- **Line Break Preservation**: Maintains your formatting intentions throughout processing
- **Character Assignment**: Automatically maps voice tags to selected voice profiles
- **Chunk Optimization**: Breaks long texts into optimal segments while preserving pause timing
- **Error Recovery**: Validates text and provides helpful formatting suggestions

#### **⚡ Real-Time Processing**
- **Live Feedback**: Console output shows exactly how many pauses are being added
- **Debug Information**: Detailed logging of pause detection and application
- **Progress Tracking**: Monitor pause processing alongside audio generation
- **Quality Assurance**: Automatic validation of pause placement

#### **🎚️ Professional Output**
- **Seamless Integration**: Pauses blend naturally with generated speech
- **Volume Consistency**: Silence segments match the audio output specifications
- **Format Compatibility**: Works with all supported audio formats and quality settings
- **Project Preservation**: Pause information saved in project metadata for regeneration

### 💡 **Pro Tips for Better Audiobooks**

#### **🎯 Dialogue Formatting**
- **Character Consistency**: Always use the same character name format `[Name]`
- **Natural Breaks**: Place returns where a human reader would naturally pause
- **Scene Transitions**: Use multiple returns (2-3) for major scene changes
- **Emotional Beats**: Add single returns before/after emotional dialogue

#### **📚 Chapter Structure**
```
Chapter 1: The Beginning

Opening paragraph with scene setting.

"Character dialogue with natural flow."

Descriptive narrative continues.


Major scene transition with extended pause.

New section begins here.
```

#### **🎪 Advanced Techniques**
- **Cliffhangers**: Use extended pauses before revealing crucial information
- **Action Sequences**: Shorter, punchy sentences with minimal pauses for intensity
- **Contemplative Moments**: Longer pauses for reflection and character development
- **Comedic Timing**: Strategic pauses before punchlines or comedic reveals

### 🔍 **Debug Output Examples**

When generating your audiobook, watch for these helpful console messages:
```
🔇 Detected 15 line breaks → 1.5s total pause time
🔇 Line breaks detected in [Character1]: +0.3s pause (from 3 returns)
🔇 Chunk 2 (Narrator): Added 0.2s pause after speech
```

This real-time feedback helps you understand exactly how your formatting translates to audio timing.

---

## 🆕 Recent Improvements

### 🎯 **Audio Quality Enhancements**
We've significantly improved audio generation quality by optimizing the underlying TTS parameters:

- **Enhanced P-top and Minimum P Settings**: Fine-tuned probability parameters for more natural speech patterns
- **Reduced Audio Artifacts**: Better handling of pronunciation and intonation
- **Improved Voice Consistency**: More stable voice characteristics across long generations
- **Better Pronunciation**: Enhanced handling of complex words and names

**📝 Note for Existing Users**: 
- Older voice profiles will continue to work as before
- To take advantage of the new audio quality improvements, consider re-creating voice profiles
- Existing projects remain fully compatible

### 📋 **Text Queuing System**
Perfect for processing large books or multiple chapters:

- **Batch Upload**: Upload multiple text files of any size
- **Sequential Processing**: Automatically processes files one after another
- **Progress Tracking**: Monitor generation progress across all queued items
- **Flexible Chapter Sizes**: No restrictions on individual file length
- **Unattended Generation**: Set up large projects and let them run automatically

### 🔄 **Chunk-Based TTS System**
Enhanced the core text-to-speech engine for better reliability:

- **Background Chunking**: Automatically splits long texts into optimal chunks
- **Memory Management**: Better handling of large text inputs
- **Error Recovery**: Improved resilience during long generation sessions
- **Consistent Quality**: Maintains voice quality across chunk boundaries
- **Progress Feedback**: Real-time updates on generation progress

---

## 🎚️ Volume Normalization Guide

### **Individual Voice Setup**
1. Go to **Voice Library** tab
2. Upload your voice sample and configure settings
3. Set target volume level (default: -18 dB for audiobooks)
4. Choose from professional presets or use custom levels
5. Save voice profile with volume settings

### **Multi-Voice Projects**
1. Navigate to **Multi-Voice Audiobook Creation** tab
2. Enable volume normalization for all voices
3. Set target level for consistent character voices
4. All characters will be automatically normalized during generation

### **Text Queuing Workflow** ⭐ *NEW*
1. Go to **Production Studio** tab
2. Select "Batch Processing" mode
3. Upload multiple text files (chapters, sections, etc.)
4. Choose your voice and settings
5. Start batch processing - files will generate sequentially
6. Monitor progress and download completed audiobooks

### **Professional Standards**
- **📖 Audiobook Standard**: -18 dB RMS (recommended for most audiobooks)
- **🎙️ Podcast Standard**: -16 dB RMS (for podcast-style content)
- **🔇 Quiet/Comfortable**: -20 dB RMS (for quiet listening environments)
- **🔊 Loud/Energetic**: -14 dB RMS (for dynamic, energetic content)
- **📺 Broadcast Standard**: -23 dB RMS (for broadcast television standards)

---

## 📁 Project Structure

```
📦 Your Audiobook Projects
├── 🎤 speakers/           # Voice library and samples
├── 📚 audiobook_projects/ # Generated audiobooks
├── 🔧 src/audiobook/      # Core processing modules
└── 📄 Generated files...  # Audio chunks and final outputs
```

---

## 🎯 Workflow

1. **📝 Prepare Text**: Format your story with proper chapter breaks and strategic line breaks for natural pauses
2. **🎤 Select Voices**: Choose or clone voices for your characters  
3. **🎚️ Configure Volume**: Set professional volume levels and normalization
4. **⚙️ Configure Settings**: Adjust quality, speed, and processing options
5. **🎧 Generate Audio**: Create your audiobook with advanced TTS and automatic pause insertion
6. **🧹 Clean & Optimize**: Use smart cleanup tools for perfect audio
7. **📦 Export**: Get your finished audiobook ready for distribution

### 🎭 **Enhanced Multi-Voice Workflow**
1. **📝 Format Dialogue**: Use `[Character]` tags and strategic line breaks for natural flow
2. **🔇 Add Return Pauses**: Place line breaks where you want natural speech pauses (0.1s each)
3. **🎤 Assign Voices**: Map each character to their voice profile
4. **⚡ Process with Intelligence**: Watch console output for pause detection feedback
5. **🎧 Review & Adjust**: Listen to generated audio and refine formatting if needed

### 📋 **Batch Processing Workflow** ⭐ *NEW*
1. **📚 Organize Chapters**: Split your book into individual text files
2. **📋 Queue Setup**: Upload all files to the batch processing system
3. **🎤 Voice Selection**: Choose voice and configure settings once
4. **🔄 Automated Generation**: Let the system process all files sequentially
5. **📊 Monitor Progress**: Track completion status in real-time
6. **📦 Collect Results**: Download all generated audiobook chapters

---

## 🛠️ Technical Requirements

- **Python 3.10+**
- **8GB+ RAM** (16GB recommended for large projects)
- **Modern web browser** for the interface
- **CPU-only** - No GPU or CUDA required

---

## ⚠️ Known Issues & Compatibility

### **Multi-Voice Generation**
- Short sentences or sections may occasionally cause issues during multi-voice generation
- This is a limitation of the underlying TTS models rather than the implementation
- **Workaround**: Use longer, more detailed sentences for better stability
- Single-voice generation is not affected by this issue

### **Voice Profile Compatibility**
- **Existing Voices**: All older voice profiles remain fully functional
- **New Features**: To benefit from improved audio quality, consider re-creating voice profiles
- **Project Compatibility**: Existing audiobook projects work without modification
- **Regeneration**: Individual chunks can be regenerated with improved quality settings

### **Batch Processing Considerations**
- Large batch jobs may take significant time depending on text length and hardware
- Monitor system resources during extended batch processing sessions
- Consider processing very large books in smaller batches for better control

---

## 📋 Supported Formats

### Input
- **Text**: `.txt`, `.md`, formatted stories and scripts
- **Audio Samples**: `.wav`, `.mp3`, `.flac` for voice cloning
- **Batch Files**: Multiple text files for queue processing

### Output
- **Audio**: High-quality `.wav` files with professional volume levels
- **Projects**: Organized folder structure with chapters
- **Exports**: Ready-to-use audiobook files
- **Batch Results**: Multiple completed audiobooks from queue processing

---

## 🆘 Support

- **Features Guide**: See `AUDIOBOOK_FEATURES.md` for detailed capabilities
- **Development Notes**: Check `development/` folder for technical details
- **Issues**: Report problems via GitHub issues

---

## 📄 License

This project is licensed under the terms specified in `LICENSE`.

---

**🎉 Ready to create amazing audiobooks with professional volume levels and enhanced audio quality? Run `./launch_audiobook.bat` and start generating!** 
