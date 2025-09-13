#!/bin/bash
# convert_model.sh - Download pre-quantized models from Hugging Face or convert locally

set -e

MODEL_NAME="RzenEmbed-v1-7B"
MODEL_DIR="./models"
HF_REPO="your-username/RzenEmbed-v1-7B-GGUF"  # Replace with your HF repo

echo "🚀 Setting up $MODEL_NAME models..."

# Create directories
mkdir -p "$MODEL_DIR"
mkdir -p "./logs"

# Function to download from Hugging Face
download_from_hf() {
    echo "📥 Downloading pre-quantized models from Hugging Face..."
    
    # Install huggingface_hub if not available
    pip install huggingface_hub >/dev/null 2>&1 || echo "Installing huggingface_hub..."
    
    python3 << EOF
from huggingface_hub import hf_hub_download
import os

repo_id = "$HF_REPO"
models = [
    "rzen-embed-v1-7b-q4_0.gguf",
    "rzen-embed-v1-7b-q8_0.gguf",
    "rzen-embed-v1-7b-f16.gguf"
]

print("Downloading from Hugging Face repository:", repo_id)

for model in models:
    try:
        print(f"  Downloading {model}...")
        hf_hub_download(
            repo_id=repo_id,
            filename=model,
            local_dir="$MODEL_DIR",
            local_dir_use_symlinks=False
        )
        print(f"  ✅ {model} downloaded")
    except Exception as e:
        print(f"  ❌ Failed to download {model}: {e}")

print("Download complete!")
EOF
}

# Function to convert locally (fallback)
convert_locally() {
    echo "🔧 Converting models locally..."
    TEMP_DIR="./temp"
    mkdir -p "$TEMP_DIR"
    
    # Download original model if not exists
    if [ ! -d "$TEMP_DIR/$MODEL_NAME" ]; then
        echo "📥 Downloading original $MODEL_NAME..."
        cd "$TEMP_DIR"
        git clone "https://huggingface.co/RzenAI/$MODEL_NAME"
        cd ..
    fi

    # Setup llama.cpp if not exists
    if [ ! -d "./llama.cpp" ]; then
        echo "🔧 Setting up llama.cpp..."
        git clone https://github.com/ggerganov/llama.cpp
        cd llama.cpp
        LLAMA_METAL=1 make -j$(nproc 2>/dev/null || sysctl -n hw.logicalcpu)
        cd ..
    fi

    # Convert to F16 GGUF first
    echo "🔄 Converting to F16 GGUF..."
    python llama.cpp/convert.py \
        --outfile "$MODEL_DIR/rzen-embed-v1-7b-f16.gguf" \
        --outtype f16 \
        "$TEMP_DIR/$MODEL_NAME/"

    # Create quantized versions
    echo "⚡ Creating quantized versions..."
    
    echo "  - Creating Q4_0..."
    ./llama.cpp/quantize \
        "$MODEL_DIR/rzen-embed-v1-7b-f16.gguf" \
        "$MODEL_DIR/rzen-embed-v1-7b-q4_0.gguf" \
        q4_0

    echo "  - Creating Q8_0..."
    ./llama.cpp/quantize \
        "$MODEL_DIR/rzen-embed-v1-7b-f16.gguf" \
        "$MODEL_DIR/rzen-embed-v1-7b-q8_0.gguf" \
        q8_0
        
    # Cleanup temp files
    rm -rf "$TEMP_DIR"
}

# Try Hugging Face first, fallback to local conversion
if download_from_hf; then
    echo "✅ Downloaded pre-quantized models from Hugging Face"
else
    echo "⚠️  Hugging Face download failed, converting locally..."
    convert_locally
fi
# Configuration for 16GB M1 MacBook Pro
model:
  path: "./models/rzen-embed-v1-7b-q4_0.gguf"
  context_length: 4096
  gpu_layers: -1
  threads: 8
  batch_size: 32

quantization:
  enabled: false  # File is pre-quantized
  type: "q4_0"   # Documentation

server:
  host: "127.0.0.1"
  port: 8008
  workers: 1

embedding:
  normalize: true
  pooling: "mean"
  max_tokens: 512

logging:
  level: "INFO"
  file: "./logs/embedding_server.log"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
EOF

# Config for 32GB M1
cat > config_32gb.yaml << EOF
# Configuration for 32GB M1 MacBook Pro
model:
  path: "./models/rzen-embed-v1-7b-q8_0.gguf"
  context_length: 8192
  gpu_layers: -1
  threads: 8
  batch_size: 64

quantization:
  enabled: false  # File is pre-quantized
  type: "q8_0"   # Documentation

server:
  host: "127.0.0.1"
  port: 8008
  workers: 1

embedding:
  normalize: true
  pooling: "mean"
  max_tokens: 1024

logging:
  level: "INFO"
  file: "./logs/embedding_server.log"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
EOF

echo "✅ Configuration files created: config_16gb.yaml, config_32gb.yaml"

# Step 7: Cleanup temp files (optional)
read -p "🗑️  Remove temporary files? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf "$TEMP_DIR"
    echo "🧹 Temporary files cleaned up"
fi

echo ""
echo "🎉 Conversion complete! Ready to use:"
echo "   For 16GB M1: python server.py config_16gb.yaml"
echo "   For 32GB M1: python server.py config_32gb.yaml"
echo ""
echo "📁 Model files location: $MODEL_DIR/"
echo "📊 Storage used: $(du -sh $MODEL_DIR | cut -f1)"
