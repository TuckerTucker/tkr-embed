"""
Download quantized models from Hugging Face
"""

import os
import sys
from pathlib import Path
from typing import List, Optional
import argparse

try:
    from huggingface_hub import hf_hub_download, repo_exists
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    print("⚠️  huggingface_hub not installed. Installing...")
    os.system("pip install huggingface_hub")
    from huggingface_hub import hf_hub_download, repo_exists

class ModelDownloader:
    """Download quantized models from Hugging Face"""
    
    def __init__(self, repo_id: str, local_dir: str = "./models"):
        self.repo_id = repo_id
        self.local_dir = Path(local_dir)
        self.local_dir.mkdir(exist_ok=True)
        
        # Available model variants
        self.models = {
            "q4_0": "rzen-embed-v1-7b-q4_0.gguf",     # ~4GB, 16GB M1
            "q8_0": "rzen-embed-v1-7b-q8_0.gguf",     # ~7GB, 32GB M1  
            "f16": "rzen-embed-v1-7b-f16.gguf",       # ~14GB, 64GB+ systems
        }
    
    def check_repo_exists(self) -> bool:
        """Check if Hugging Face repository exists"""
        try:
            return repo_exists(self.repo_id)
        except Exception as e:
            print(f"❌ Error checking repository: {e}")
            return False
    
    def download_model(self, variant: str, force: bool = False) -> Optional[str]:
        """Download specific model variant"""
        if variant not in self.models:
            print(f"❌ Unknown variant: {variant}")
            print(f"Available variants: {list(self.models.keys())}")
            return None
        
        filename = self.models[variant]
        local_path = self.local_dir / filename
        
        # Check if already exists
        if local_path.exists() and not force:
            print(f"✅ {filename} already exists (use --force to re-download)")
            return str(local_path)
        
        try:
            print(f"📥 Downloading {filename}...")
            downloaded_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=filename,
                local_dir=str(self.local_dir),
                local_dir_use_symlinks=False
            )
            
            print(f"✅ Downloaded: {downloaded_path}")
            return downloaded_path
            
        except Exception as e:
            print(f"❌ Failed to download {filename}: {e}")
            return None
    
    def download_all(self, force: bool = False) -> List[str]:
        """Download all model variants"""
        downloaded = []
        
        print(f"📦 Downloading from repository: {self.repo_id}")
        print(f"📁 Local directory: {self.local_dir}")
        
        for variant in self.models:
            path = self.download_model(variant, force)
            if path:
                downloaded.append(path)
        
        return downloaded
    
    def download_recommended(self, memory_gb: int, force: bool = False) -> Optional[str]:
        """Download recommended model based on system memory"""
        if memory_gb <= 16:
            variant = "q4_0"
            print(f"🎯 Recommended for {memory_gb}GB RAM: Q4_0 quantization")
        elif memory_gb <= 32:
            variant = "q8_0"
            print(f"🎯 Recommended for {memory_gb}GB RAM: Q8_0 quantization")
        else:
            variant = "f16"
            print(f"🎯 Recommended for {memory_gb}GB RAM: F16 (full precision)")
        
        return self.download_model(variant, force)
    
    def list_downloaded(self) -> List[str]:
        """List already downloaded models"""
        downloaded = []
        for variant, filename in self.models.items():
            path = self.local_dir / filename
            if path.exists():
                size_gb = path.stat().st_size / (1024**3)
                downloaded.append(f"{variant}: {filename} ({size_gb:.1f}GB)")
        return downloaded
    
    def get_model_info(self) -> dict:
        """Get information about available models"""
        return {
            "repository": self.repo_id,
            "local_directory": str(self.local_dir),
            "variants": {
                "q4_0": {
                    "filename": self.models["q4_0"],
                    "size": "~4GB",
                    "recommended_for": "16GB M1 MacBook Pro",
                    "quality": "Good balance of speed/quality"
                },
                "q8_0": {
                    "filename": self.models["q8_0"], 
                    "size": "~7GB",
                    "recommended_for": "32GB M1 MacBook Pro",
                    "quality": "High quality, near F16"
                },
                "f16": {
                    "filename": self.models["f16"],
                    "size": "~14GB", 
                    "recommended_for": "64GB+ systems",
                    "quality": "Full precision"
                }
            }
        }

def main():
    parser = argparse.ArgumentParser(description="Download quantized models from Hugging Face")
    parser.add_argument("--repo", default="your-username/RzenEmbed-v1-7B-GGUF", 
                       help="Hugging Face repository ID")
    parser.add_argument("--variant", choices=["q4_0", "q8_0", "f16", "all"],
                       help="Model variant to download")
    parser.add_argument("--memory", type=int, help="System RAM in GB (for automatic recommendation)")
    parser.add_argument("--local-dir", default="./models", help="Local directory for models")
    parser.add_argument("--force", action="store_true", help="Force re-download existing files")
    parser.add_argument("--list", action="store_true", help="List downloaded models")
    parser.add_argument("--info", action="store_true", help="Show model information")
    
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = ModelDownloader(args.repo, args.local_dir)
    
    # Handle different commands
    if args.info:
        info = downloader.get_model_info()
        print("📋 Model Information:")
        print(f"Repository: {info['repository']}")
        print(f"Local directory: {info['local_directory']}")
        print("\nAvailable variants:")
        for variant, details in info['variants'].items():
            print(f"  {variant}: {details['filename']}")
            print(f"    Size: {details['size']}")
            print(f"    Recommended for: {details['recommended_for']}")
            print(f"    Quality: {details['quality']}")
        return
    
    if args.list:
        downloaded = downloader.list_downloaded()
        if downloaded:
            print("📁 Downloaded models:")
            for model in downloaded:
                print(f"  ✅ {model}")
        else:
            print("📁 No models downloaded yet")
        return
    
    # Check if repository exists
    if not downloader.check_repo_exists():
        print(f"❌ Repository not found: {args.repo}")
        print("Make sure the repository exists and is public, or you have access")
        return
    
    # Download models
    if args.memory:
        downloader.download_recommended(args.memory, args.force)
    elif args.variant == "all":
        downloader.download_all(args.force)
    elif args.variant:
        downloader.download_model(args.variant, args.force)
    else:
        print("🤖 No specific variant requested. Use --variant, --memory, or --help")
        
        # Auto-detect system memory
        try:
            import psutil
            memory_gb = round(psutil.virtual_memory().total / (1024**3))
            print(f"🔍 Detected {memory_gb}GB system RAM")
            print(f"💡 Suggestion: python download_models.py --memory {memory_gb}")
        except ImportError:
            print("💡 Suggestion: python download_models.py --variant q4_0")

if __name__ == "__main__":
    main()
