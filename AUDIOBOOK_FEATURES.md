# 🎧 Chatterbox TTS - Audiobook Edition Features

## 🚀 New Voice Management System

The Audiobook Edition adds powerful voice management capabilities perfect for creating consistent character voices across your audiobook projects.

## ✨ Key Features

### 📚 Voice Library Tab
- **Organized Voice Storage**: Keep all your character voices in one place
- **Custom Voice Profiles**: Save voice settings with names, descriptions, and reference audio
- **Easy Voice Selection**: Quick dropdown to switch between saved voices
- **Voice Testing**: Test voices before saving or using them

### 🎭 Character Voice Management
- **Voice Profiles**: Each voice includes:
  - Voice name (for file organization)
  - Display name (human-readable)
  - Description (character notes)
  - Reference audio file
  - Optimized settings (exaggeration, CFG/pace, temperature)

### 🎙️ Voice Testing & Configuration
- **Live Testing**: Test voice settings with custom text
- **Parameter Tuning**: Fine-tune exaggeration, CFG/pace, and temperature
- **Instant Feedback**: Hear changes immediately
- **Save Optimized Settings**: Store perfect settings for each character

## 🛠️ How to Use

### 1. Launch the Audiobook Edition
```bash
# Use the audiobook launcher
launch_audiobook.bat
```

### 2. Set Up Your Voice Library
1. Go to the **"📚 Voice Library"** tab
2. Set your voice library folder path (default: `voice_library`)
3. Click **"📁 Update Library Path"**

### 3. Create a Voice Profile
1. **Upload Reference Audio**: Upload 10-30 seconds of clear speech
2. **Configure Settings**:
   - **Exaggeration**: 0.3-0.7 for most voices
   - **CFG/Pace**: Lower = slower, more deliberate
   - **Temperature**: Higher = more variation
3. **Test the Voice**: Use the test text to hear how it sounds
4. **Save Profile**: Give it a name and description, then save

### 4. Use Saved Voices
1. **Select Voice**: Choose from dropdown in Voice Library
2. **Load Voice**: Click "📥 Load Voice" to load settings
3. **Generate Speech**: Switch to TTS tab and generate with loaded voice

## 📁 Voice Library Structure

```
voice_library/
├── narrator_male_deep/
│   ├── config.json          # Voice settings
│   └── reference.wav        # Reference audio
├── character_female_young/
│   ├── config.json
│   └── reference.mp3
└── villain_gravelly/
    ├── config.json
    └── reference.wav
```

## 🎯 Audiobook Workflow

### Step 1: Character Planning
- List all characters in your audiobook
- Gather reference audio for each (record or find samples)
- Plan voice characteristics (age, personality, accent)

### Step 2: Voice Creation
- Create a voice profile for each character
- Test and refine settings for consistency
- Save with descriptive names (e.g., "Harry_confident", "Hermione_intelligent")

### Step 3: Production
- Load character voice before generating their dialogue
- Use consistent settings throughout the book
- Test voice regularly to maintain quality

### Step 4: Quality Control
- Use the same test phrase for all characters
- Ensure voices are distinguishable
- Adjust settings if characters sound too similar

## 💡 Pro Tips

### Voice Creation
- **Reference Audio**: Use clean, noise-free recordings
- **Length**: 10-30 seconds is optimal
- **Content**: Natural speech, not overly dramatic
- **Quality**: Higher quality audio = better cloning

### Settings Optimization
- **Exaggeration**:
  - 0.3-0.5: Subtle, natural voices
  - 0.5-0.7: Standard character voices
  - 0.7-1.0: Dramatic or distinctive voices
  
- **CFG/Pace**:
  - 0.3-0.4: Slow, deliberate (elderly, wise characters)
  - 0.5: Standard pace
  - 0.6-0.8: Faster pace (young, energetic characters)

- **Temperature**:
  - 0.5-0.8: Consistent delivery
  - 0.8-1.2: More natural variation
  - 1.2+: Creative but less predictable

### Organization
- **Naming Convention**: Use descriptive names (character_trait_type)
- **Descriptions**: Include character details and usage notes
- **Backup**: Keep your voice_library folder backed up
- **Version Control**: Save multiple versions for different emotions

## 🔧 Advanced Features

### Voice Library Management
- **Import/Export**: Copy voice_library folder between projects
- **Sharing**: Share voice profiles with other audiobook creators
- **Backup**: Regular backups of your voice library
- **Organization**: Folder structure for different projects

### Batch Processing (Future)
- Process entire chapters with character voice switching
- Automatic voice detection based on speaker tags
- Export management for audiobook production

## 🎵 Example Character Voices

### Narrator
- **Settings**: Exaggeration 0.4, CFG 0.5, Temp 0.7
- **Description**: Clear, neutral, professional tone
- **Use**: Chapter narration, scene descriptions

### Hero Character
- **Settings**: Exaggeration 0.6, CFG 0.6, Temp 0.8
- **Description**: Confident, determined, slightly higher energy
- **Use**: Main character dialogue

### Wise Mentor
- **Settings**: Exaggeration 0.3, CFG 0.3, Temp 0.6
- **Description**: Slow, deliberate, thoughtful delivery
- **Use**: Advisor character, important wisdom

### Comic Relief
- **Settings**: Exaggeration 0.8, CFG 0.7, Temp 1.0
- **Description**: Energetic, expressive, variable delivery
- **Use**: Funny sidekick, lighthearted moments

## 🛡️ Best Practices

1. **Consistency**: Always use the same voice profile for each character
2. **Testing**: Test voices regularly during production
3. **Backup**: Keep voice profiles backed up
4. **Documentation**: Maintain character voice notes
5. **Quality**: Use high-quality reference audio
6. **Organization**: Use clear naming conventions

---

**Ready to create amazing audiobooks with consistent character voices? Launch the Audiobook Edition and start building your voice library! 🎧✨** 