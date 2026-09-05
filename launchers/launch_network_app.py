#!/usr/bin/env python3
"""
Network launcher for Chatterbox TTS Audiobook Edition
Security: Local network access (0.0.0.0:7860)
"""

import sys
import os
import socket

# Add current directory to path
sys.path.insert(0, '.')

def get_local_ip():
    """Get the local IP address"""
    try:
        # Connect to a remote server to get local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except:
        return "Unable to determine IP"

# Import the main application
try:
    # Import all the necessary components from the main app
    from gradio_tts_app_audiobook import *
    
    # Launch with network configuration
    if __name__ == "__main__":
        local_ip = get_local_ip()
        print("🏠 Launching in NETWORK mode...")
        print("📍 Finding available port (starting from 7860)...")
        print(f"🌐 Your network IP: {local_ip}")
        print("🚫 No public sharing enabled")
        
        demo.queue(
            max_size=50,
            default_concurrency_limit=1,
        ).launch(
            share=False,
            server_name="0.0.0.0", 
            server_port=None,  # Let Gradio find an available port
            inbrowser=False  # Don't auto-open browser
        )

except Exception as e:
    print(f"❌ Error launching application: {e}")
    print("📁 Make sure you're in the correct directory with gradio_tts_app_audiobook.py")
    input("Press Enter to exit...") 