# Custom Embedding Server

A high-performance embedding server using llama.cpp with YAML configuration support, optimized for Apple Silicon M1/M2 Macs.

## Features

- 🚀 **Native M1/M2 optimization** with Metal GPU acceleration
- ⚡ **Pre-quantized GGUF models** for fast loading
- 🔧 **YAML/JSON configuration** for easy customization
- 🌐 **FastAPI server** with OpenAI-compatible API
- 📊 **Batch processing** with configurable batch sizes
- 🎯 **Multiple quantization levels** (Q4_0, Q8_0, F16)

## Quick Start

### 1. Clone Repository

```bash
git clone <your-repo-url>
cd embedding-server
```

### 2. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# For M1/M2 Macs, install with Metal support
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### 3. Download Pre-quantized Models

**Option A: Automatic download (recommended)**
```bash
# Download recommended model for your system
python download_models.py --memory 16  # For 16GB M1
python download_models.py --memory 32  # For 32GB M1

# Or download specific variant
python download_models.py --variant q4_0  # 4GB model
python download_models.py --variant q8_0  # 7GB model
```

**Option B: Manual download**
```bash
# Run conversion/download script
./convert_model.sh
```

**Option C: Direct from Hugging Face**
```bash
# Install huggingface_hub
pip install huggingface_hub

# Download Q4_0 model (4GB)
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='your-username/RzenEmbed-v1-7B-GGUF',
    filename='rzen-embed-v1-7b-q4_0.gguf',
    local_dir='./models'
)
"
```

### 4. Start Server

```bash
# For 16GB M1 MacBook Pro
python server.py config_16gb.yaml

# For 32GB M1 MacBook Pro  
python server.py config_32gb.yaml
```

### 5. Test the Server

```bash
# Run examples
python examples.py

# Or test manually
curl -X POST "http://127.0.0.1:8008/embed" \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello, world!"}'
```

## Configuration

### Model Files (Not in Git)

After running `convert_model.sh`, you'll have:

```
models/
├── rzen-embed-v1-7b-f16.gguf   # ~14GB - Full precision
├── rzen-embed-v1-7b-q8_0.gguf  # ~7GB  - High quality
└── rzen-embed-v1-7b-q4_0.gguf  # ~4GB  - Balanced quality/speed
```

### Configuration Files

- `config_16gb.yaml` - Optimized for 16GB M1 MacBook Pro (Q4_0 quantization)
- `config_32gb.yaml` - Optimized for 32GB M1 MacBook Pro (Q8_0 quantization)

### Custom Configuration

Create your own config file:

```yaml
model:
  path: "./models/rzen-embed-v1-7b-q4_0.gguf"
  context_length: 4096
  gpu_layers: -1  # Use all GPU layers
  batch_size: 32

embedding:
  normalize: true
  max_tokens: 512

server:
  host: "127.0.0.1"
  port: 8008

logging:
  level: "INFO"
  file: "./logs/embedding_server.log"
```

## API Endpoints

### Single Embedding
```bash
POST /embed
{
  "text": "Your text here"
}
```

### Batch Embeddings
```bash
POST /embed/batch
{
  "texts": ["Text 1", "Text 2", "Text 3"],
  "batch_size": 32
}
```

### Text Similarity
```bash
POST /similarity
{
  "text1": "First text",
  "text2": "Second text"
}
```

### Model Info
```bash
GET /info
```

## Performance

### M1 MacBook Pro 16GB
- **Model**: Q4_0 quantization (~4GB)
- **Loading time**: 5-10 seconds
- **Memory usage**: ~6GB total
- **Inference speed**: ~50-100 tokens/second

### M1 MacBook Pro 32GB
- **Model**: Q8_0 quantization (~7GB)
- **Loading time**: 10-15 seconds  
- **Memory usage**: ~9GB total
- **Inference speed**: ~100-200 tokens/second

## Hardware Requirements

### Minimum (16GB M1)
- **RAM**: 16GB (8GB+ available)
- **Storage**: 25GB free space
- **Model**: Q4_0 quantization

### Recommended (32GB M1)
- **RAM**: 32GB (16GB+ available)
- **Storage**: 30GB free space
- **Model**: Q8_0 quantization

## Why Models Aren't in Git

Model files are **excluded from version control** because:

- 📦 **Size**: 4-14GB per model variant
- 💰 **Cost**: GitHub LFS charges for bandwidth
- ⚡ **Speed**: Faster repository clones
- 🔄 **Updates**: Models don't change like code
- 🎯 **Flexibility**: Choose quantization for your hardware

## Troubleshooting

### Model Not Found
```bash
# Re-run conversion
./convert_model.sh
```

### Out of Memory
```bash
# Use smaller quantization
python server.py config_16gb.yaml  # Uses Q4_0
```

### Slow Performance
```bash
# Check Metal acceleration
export OLLAMA_GPU_LAYERS=-1
python server.py config.yaml
```

## Development

### Adding New Models

1. Update `convert_model.sh` with new model URL
2. Add new configuration file
3. Test with `examples.py`

### Custom Quantization

```bash
# Create custom quantization
./llama.cpp/quantize input.gguf output.gguf q5_0
```

## License

MIT License - see LICENSE file for details.
