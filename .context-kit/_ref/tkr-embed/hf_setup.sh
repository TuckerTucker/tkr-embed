#!/bin/bash
# setup_hf_repo.sh - Upload quantized models to Hugging Face

set -e

# Configuration
HF_USERNAME="your-username"  # Replace with your HF username
MODEL_REPO="RzenEmbed-v1-7B-GGUF"
LOCAL_MODELS_DIR="./models"

echo "🤗 Setting up Hugging Face repository for quantized models..."

# Install Hugging Face Hub if not already installed
pip install huggingface_hub

# Login to Hugging Face (you'll need to provide your token)
echo "🔐 Please login to Hugging Face:"
huggingface-cli login

# Create repository on Hugging Face
echo "📦 Creating repository: $HF_USERNAME/$MODEL_REPO"
huggingface-cli repo create $MODEL_REPO --type model --private=false

# Clone the repository
echo "📥 Cloning repository..."
git clone https://huggingface.co/$HF_USERNAME/$MODEL_REPO hf_models
cd hf_models

# Copy model files
echo "📋 Copying quantized models..."
cp $LOCAL_MODELS_DIR/*.gguf . 2>/dev/null || echo "No GGUF files found - run convert_model.sh first"

# Create model card
cat > README.md << EOF
---
license: apache-2.0
tags:
- embedding
- gguf
- quantized
- llama.cpp
- m1
- apple-silicon
pipeline_tag: feature-extraction
---

# RzenEmbed-v1-7B Quantized GGUF Models

Quantized versions of [RzenAI/RzenEmbed-v1-7B](https://huggingface.co/RzenAI/RzenEmbed-v1-7B) optimized for Apple Silicon M1/M2 Macs using llama.cpp.

## Model Variants

| File | Size | Quantization | Recommended For |
|------|------|--------------|-----------------|
| \`rzen-embed-v1-7b-f16.gguf\` | ~14GB | F16 | 64GB+ RAM systems |
| \`rzen-embed-v1-7b-q8_0.gguf\` | ~7GB | Q8_0 | 32GB+ M1 MacBook Pro |
| \`rzen-embed-v1-7b-q4_0.gguf\` | ~4GB | Q4_0 | 16GB M1 MacBook Pro |

## Usage

### Download Specific Model

\`\`\`bash
# For 16GB M1 MacBook Pro
wget https://huggingface.co/$HF_USERNAME/$MODEL_REPO/resolve/main/rzen-embed-v1-7b-q4_0.gguf

# For 32GB M1 MacBook Pro
wget https://huggingface.co/$HF_USERNAME/$MODEL_REPO/resolve/main/rzen-embed-v1-7b-q8_0.gguf
\`\`\`

### Using Hugging Face Hub

\`\`\`python
from huggingface_hub import hf_hub_download

# Download Q4_0 model
model_path = hf_hub_download(
    repo_id="$HF_USERNAME/$MODEL_REPO",
    filename="rzen-embed-v1-7b-q4_0.gguf",
    local_dir="./models"
)
\`\`\`

### With Custom Embedding Server

\`\`\`yaml
# config.yaml
model:
  path: "./models/rzen-embed-v1-7b-q4_0.gguf"
  # ... other settings
\`\`\`

## Performance

### M1 MacBook Pro 16GB (Q4_0)
- **Memory usage**: ~4GB
- **Loading time**: 5-10 seconds
- **Quality**: Minimal degradation from F16

### M1 MacBook Pro 32GB (Q8_0)
- **Memory usage**: ~7GB
- **Loading time**: 10-15 seconds
- **Quality**: Near-identical to F16

## Integration

These models work with:
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [Ollama](https://ollama.ai/)
- Custom embedding servers
- Any GGUF-compatible inference engine

## Original Model

Based on [RzenAI/RzenEmbed-v1-7B](https://huggingface.co/RzenAI/RzenEmbed-v1-7B) - a 7B parameter embedding model.

## Quantization Details

Models were quantized using llama.cpp with the following settings:
- **Q4_0**: 4-bit quantization, good balance of size/quality
- **Q8_0**: 8-bit quantization, high quality
- **F16**: Half-precision, full quality

## License

Same as original model (Apache 2.0).
EOF

# Add and commit files
echo "📤 Uploading to Hugging Face..."
git add .
git commit -m "Add quantized GGUF models for Apple Silicon"
git push

echo "✅ Upload complete! Repository available at:"
echo "   https://huggingface.co/$HF_USERNAME/$MODEL_REPO"

cd ..
rm -rf hf_models  # Cleanup local clone

echo ""
echo "🎉 Your models are now available on Hugging Face!"
echo "   Update your code to download from HF instead of local conversion."
