#!/usr/bin/env python3
"""
Local-only launcher for Chatterbox TTS Audiobook Edition
Security: Local access only (127.0.0.1:7860)
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, '.')

# Import the main application
try:
    # Import all the necessary components from the main app
    from gradio_tts_app_audiobook import *
    
    # Launch with local-only configuration
    if __name__ == "__main__":
        print("🔒 Launching in LOCAL ONLY mode...")
        print("📍 Finding available port (starting from 7860)...")
        print("🚫 No network or public access enabled")
        
        demo.queue(
            max_size=50,
            default_concurrency_limit=1,
        ).launch(
            share=False,
            server_name="127.0.0.1", 
            server_port=None,  # Let Gradio find an available port
            inbrowser=False  # Don't auto-open browser
        )

except Exception as e:
    print(f"❌ Error launching application: {e}")
    print("📁 Make sure you're in the correct directory with gradio_tts_app_audiobook.py")
    input("Press Enter to exit...") 