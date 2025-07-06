#!/usr/bin/env python3
"""
Helper script to copy the Gemma 3 N model file from a source location
Uses the MediaPipe-compatible Task bundle from LiteRT Community
"""

import os
import shutil
import sys
import subprocess
from pathlib import Path

def check_huggingface_cli():
    """Check if huggingface-cli is available"""
    try:
        result = subprocess.run(["huggingface-cli", "--help"], 
                              capture_output=True, text=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def download_from_huggingface():
    """Download the model file using huggingface-cli"""
    repo_id = "google/gemma-3n-E2B-it-litert-lm-preview"
    filename = "gemma-3n-E2B-it-int4.litertlm"
    target_filename = "gemma-3n-E2B-it-int4.task"
    
    assets_dir = Path("assets")
    assets_dir.mkdir(exist_ok=True)
    
    print(f"🔄 Downloading model from Hugging Face: {repo_id}")
    print(f"   File: {filename}")
    
    try:
        # Download to assets directory
        result = subprocess.run([
            "huggingface-cli", "download", 
            repo_id, filename,
            "--local-dir", str(assets_dir),
            "--local-dir-use-symlinks", "False"
        ], capture_output=True, text=True, check=True)
        
        # Rename the downloaded file to the target filename
        downloaded_file = assets_dir / filename
        target_file = assets_dir / target_filename
        
        if downloaded_file.exists():
            # Rename from .litertlm to .task
            downloaded_file.rename(target_file)
            print(f"✅ Model file downloaded and renamed successfully!")
            print(f"   Size: {target_file.stat().st_size / (1024**3):.2f} GB")
            return True
        else:
            print("❌ Downloaded file not found")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Error downloading from Hugging Face: {e}")
        print(f"   stderr: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def find_model_file():
    """Search for the model file in common locations"""
    possible_paths = [
        "../test3/assets/gemma-3n-E2B-it-int4.task",
        "../../test3/assets/gemma-3n-E2B-it-int4.task",
        "./assets/gemma-3n-E2B-it-int4.task"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

def copy_model(source_path=None):
    """Copy the model file to the assets directory"""
    assets_dir = Path("assets")
    assets_dir.mkdir(exist_ok=True)
    target_path = assets_dir / "gemma-3n-E2B-it-int4.task"
    
    if target_path.exists():
        print(f"✅ Model file already exists: {target_path}")
        print(f"   Size: {target_path.stat().st_size / (1024**3):.2f} GB")
        return True
    
    # Try downloading from Hugging Face first if CLI is available
    if check_huggingface_cli():
        print("🤗 Hugging Face CLI detected, attempting download...")
        if download_from_huggingface():
            return True
        else:
            print("⚠️  Hugging Face download failed, trying local copy...")
    
    if not source_path:
        source_path = find_model_file()
    
    if not source_path or not os.path.exists(source_path):
        print("❌ Model file not found!")
        print("\nOptions to get the model file:")
        print("1. Install huggingface-cli and run this script again:")
        print("   pip install --upgrade 'huggingface_hub[cli]'")
        print("   python copy_model.py")
        print("\n2. Manually copy the model file:")
        print("   cp /path/to/gemma3-4b-it-int4-web.task ./assets/")
        print("\n3. Manual download from Hugging Face:")
        print("   Repository: google/gemma-3n-E4B-it-litert-preview")
        print("   File: gemma-3n-E4B-it-int4.task")
        return False
    
    print(f"🔄 Copying model file from: {source_path}")
    print(f"   To: {target_path}")
    
    try:
        shutil.copy2(source_path, target_path)
        print(f"✅ Model file copied successfully!")
        print(f"   Size: {target_path.stat().st_size / (1024**3):.2f} GB")
        return True
    except Exception as e:
        print(f"❌ Error copying file: {e}")
        return False

if __name__ == "__main__":
    os.chdir(Path(__file__).parent)
    
    source = sys.argv[1] if len(sys.argv) > 1 else None
    if copy_model(source):
        print("\n🚀 Ready to run the demo!")
        print("   python serve.py")
    else:
        print("\n📖 See assets/README.md for download instructions")