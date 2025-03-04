#!/usr/bin/env python3
"""
Install Kokoro TTS for speech-mcp

This script installs Kokoro and its dependencies for use with speech-mcp.
It creates a virtual environment and installs the required packages.
"""

import os
import sys
import subprocess
import argparse
import platform
import venv
import tempfile
import shutil
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Install Kokoro TTS for speech-mcp")
    parser.add_argument("--venv", type=str, default=None, help="Path to virtual environment to create/use")
    parser.add_argument("--no-venv", action="store_true", help="Install in the current Python environment")
    parser.add_argument("--force", action="store_true", help="Force reinstallation even if already installed")
    args = parser.parse_args()
    
    print("=== Kokoro TTS Installer for speech-mcp ===")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 9):
        print(f"Error: Python 3.9 or higher is required. You have Python {python_version.major}.{python_version.minor}")
        sys.exit(1)
    
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    print(f"Platform: {platform.platform()}")
    
    # Determine virtual environment path
    venv_path = None
    if not args.no_venv:
        if args.venv:
            venv_path = Path(args.venv).expanduser().resolve()
        else:
            # Use the default location
            venv_path = Path.home() / ".speech-mcp" / "kokoro-venv"
        
        print(f"Using virtual environment at: {venv_path}")
        
        # Create virtual environment if it doesn't exist
        if not venv_path.exists() or args.force:
            print("Creating virtual environment...")
            venv.create(venv_path, with_pip=True)
        else:
            print("Virtual environment already exists.")
    else:
        print("Installing in the current Python environment.")
    
    # Get the Python executable to use
    if venv_path:
        if platform.system() == "Windows":
            python_exe = venv_path / "Scripts" / "python.exe"
        else:
            python_exe = venv_path / "bin" / "python"
    else:
        python_exe = sys.executable
    
    print(f"Using Python executable: {python_exe}")
    
    # Install Kokoro and dependencies
    print("\nInstalling Kokoro and dependencies...")
    
    # Create a temporary requirements file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp:
        tmp_path = tmp.name
        tmp.write("kokoro>=0.8.4\n")
        tmp.write("soundfile\n")
        tmp.write("torch\n")
        tmp.write("misaki[en]\n")
    
    try:
        # Install the requirements
        cmd = [str(python_exe), "-m", "pip", "install", "-r", tmp_path]
        print(f"Running: {' '.join(cmd)}")
        subprocess.check_call(cmd)
        
        # Check if installation was successful
        try:
            cmd = [str(python_exe), "-c", "import kokoro; print(f'Kokoro version: {kokoro.__version__}')"]
            print("\nVerifying installation...")
            subprocess.check_call(cmd)
            print("\nKokoro installation successful!")
        except subprocess.CalledProcessError:
            print("\nWarning: Kokoro installation verification failed.")
            print("You may need to install it manually or check for errors.")
    finally:
        # Clean up
        os.unlink(tmp_path)
    
    # Print instructions
    print("\n=== Installation Complete ===")
    if venv_path:
        print(f"\nKokoro has been installed in a virtual environment at: {venv_path}")
        print("\nTo use Kokoro with speech-mcp, you need to:")
        print(f"1. Activate the virtual environment:")
        if platform.system() == "Windows":
            print(f"   {venv_path}\\Scripts\\activate")
        else:
            print(f"   source {venv_path}/bin/activate")
        print("2. Run your speech-mcp application")
    else:
        print("\nKokoro has been installed in your current Python environment.")
        print("You can now use it with speech-mcp.")
    
    print("\nAlternatively, you can use pip with optional dependencies:")
    print("  pip install speech-mcp[kokoro]     # Basic Kokoro support with English")
    print("  pip install speech-mcp[ja]         # Add Japanese support")
    print("  pip install speech-mcp[zh]         # Add Chinese support")
    print("  pip install speech-mcp[all]        # All languages and features")
    
    print("\nEnjoy using Kokoro TTS with speech-mcp!")

if __name__ == "__main__":
    main()