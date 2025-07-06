#!/usr/bin/env python3
"""
Helper script to download the correct Gemma 3n model file from Hugging Face
Downloads the MediaPipe-compatible Task bundle from LiteRT Community
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
            print(f"✅ Downloaded file: {downloaded_file}")
            print(f"   Renaming to: {target_file}")
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

if __name__ == "__main__":
    os.chdir(Path(__file__).parent)
    
    if not check_huggingface_cli():
        print("❌ huggingface-cli not found!")
        print("\nPlease install huggingface-cli:")
        print("   pip install --upgrade 'huggingface_hub[cli]'")
        sys.exit(1)
    
    if download_from_huggingface():
        print("\n🚀 Ready to test the model!")
        print("   Update demo.html to use the new model file:")
        print("   baseOptions: { modelAssetPath: \"./assets/gemma-3n-E2B-it-int4.task\" }")
    else:
        print("\n📖 Failed to download the model file")