# MLX-Based Architecture Plan for Multimodal Embeddings

## Project: tkr-embed with OpenSearch-AI/Ops-MM-embedding-v1-7B

### 🎯 Executive Summary

Build a high-performance multimodal embedding server optimized for Apple Silicon using MLX framework with the OpenSearch-AI/Ops-MM-embedding-v1-7B model. The system will handle text, images, documents, and video frames with unified embeddings for cross-modal retrieval.

## 📐 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      Client Applications                      │
└──────────────────────┬──────────────────────────────────────┘
                       │ REST API / WebSocket
┌──────────────────────▼──────────────────────────────────────┐
│                    FastAPI Server Layer                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Text API   │  │  Image API   │  │  Video API   │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
└─────────┼──────────────────┼──────────────────┼─────────────┘
          │                  │                  │
┌─────────▼──────────────────▼──────────────────▼─────────────┐
│               Multimodal Processing Pipeline                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Tokenizer  │  │Image Processor│  │Video Processor│      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
└─────────┼──────────────────┼──────────────────┼─────────────┘
          │                  │                  │
┌─────────▼──────────────────▼──────────────────▼─────────────┐
│                    MLX Model Engine                          │
│  ┌────────────────────────────────────────────────────┐     │
│  │     Ops-MM-embedding-v1-7B (Quantized in MLX)      │     │
│  │         Unified Multimodal Embedding Space          │     │
│  └────────────────────────────────────────────────────┘     │
│                  Apple Silicon (M1/M2/M3)                    │
└───────────────────────────────────────────────────────────────┘
```

## 🏗️ Core Components

### 1. MLX Model Management

```python
# model_manager.py
import mlx.core as mx
import mlx.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any
import json

class OpsMMEmbeddingMLX:
    """MLX implementation of Ops-MM-embedding-v1-7B"""
    
    def __init__(
        self,
        model_path: str = "OpenSearch-AI/Ops-MM-embedding-v1-7B",
        quantization: str = "q4",  # q4, q8, or none
        device: str = "gpu"
    ):
        self.model_path = model_path
        self.quantization = quantization
        self.device = device
        self.model = None
        self.processor = None
        
    async def load_model(self):
        """Load and quantize model for Apple Silicon"""
        # Load base model
        from mlx_lm import load, generate
        
        self.model, self.tokenizer = load(
            self.model_path,
            tokenizer_config={"trust_remote_code": True}
        )
        
        # Apply quantization if specified
        if self.quantization == "q4":
            self.quantize_4bit()
        elif self.quantization == "q8":
            self.quantize_8bit()
            
    def quantize_4bit(self):
        """4-bit quantization for 16GB M1"""
        from mlx.nn import quantize
        self.model = quantize(self.model, bits=4, group_size=64)
        
    def quantize_8bit(self):
        """8-bit quantization for 32GB M1"""
        from mlx.nn import quantize
        self.model = quantize(self.model, bits=8, group_size=128)
```

### 2. Multimodal Input Processing

```python
# processors.py
import mlx.core as mx
from PIL import Image
import numpy as np
from typing import List, Union, Tuple

class MultimodalProcessor:
    """Process various input modalities for embedding"""
    
    def __init__(self, model_config: Dict[str, Any]):
        self.image_size = model_config.get("image_size", 336)
        self.max_text_length = model_config.get("max_text_length", 512)
        
    def process_text(self, text: Union[str, List[str]]) -> mx.array:
        """Process text inputs"""
        if isinstance(text, str):
            text = [text]
        
        # Tokenization happens in model
        return text
    
    def process_image(self, image_path: str) -> mx.array:
        """Process single image"""
        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.image_size, self.image_size))
        
        # Convert to MLX array
        image_array = np.array(image).astype(np.float32) / 255.0
        return mx.array(image_array)
    
    def process_video_frames(
        self, 
        video_path: str, 
        max_frames: int = 8
    ) -> List[mx.array]:
        """Extract and process video frames"""
        import cv2
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, frame_count - 1, max_frames, dtype=int)
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.image_size, self.image_size))
                frame_array = mx.array(frame.astype(np.float32) / 255.0)
                frames.append(frame_array)
        
        cap.release()
        return frames
    
    def process_document(self, doc_path: str) -> Tuple[str, List[mx.array]]:
        """Process documents with text and images"""
        # Implementation depends on document type (PDF, DOCX, etc.)
        pass
```

### 3. Embedding Generation Engine

```python
# embedding_engine.py
import mlx.core as mx
import mlx.nn as nn
from typing import List, Dict, Any, Optional
import numpy as np

class EmbeddingEngine:
    """Core embedding generation with MLX"""
    
    def __init__(self, model: OpsMMEmbeddingMLX, processor: MultimodalProcessor):
        self.model = model
        self.processor = processor
        self.embedding_dim = 1024  # Ops-MM output dimension
        
    async def generate_text_embeddings(
        self, 
        texts: List[str],
        normalize: bool = True
    ) -> mx.array:
        """Generate embeddings for text inputs"""
        # Process texts
        inputs = self.processor.process_text(texts)
        
        # Generate embeddings
        with mx.no_grad():
            embeddings = self.model.encode_text(inputs)
        
        if normalize:
            embeddings = mx.nn.functional.normalize(embeddings, axis=-1)
            
        return embeddings
    
    async def generate_image_embeddings(
        self,
        image_paths: List[str],
        normalize: bool = True
    ) -> mx.array:
        """Generate embeddings for images"""
        images = [self.processor.process_image(path) for path in image_paths]
        images = mx.stack(images)
        
        with mx.no_grad():
            embeddings = self.model.encode_image(images)
        
        if normalize:
            embeddings = mx.nn.functional.normalize(embeddings, axis=-1)
            
        return embeddings
    
    async def generate_multimodal_embeddings(
        self,
        text: Optional[str] = None,
        image_path: Optional[str] = None,
        normalize: bool = True
    ) -> mx.array:
        """Generate unified multimodal embeddings"""
        inputs = {}
        
        if text:
            inputs["text"] = self.processor.process_text(text)
        if image_path:
            inputs["image"] = self.processor.process_image(image_path)
            
        with mx.no_grad():
            embeddings = self.model.encode_multimodal(**inputs)
        
        if normalize:
            embeddings = mx.nn.functional.normalize(embeddings, axis=-1)
            
        return embeddings
    
    async def generate_video_embeddings(
        self,
        video_path: str,
        max_frames: int = 8,
        pooling: str = "mean",
        normalize: bool = True
    ) -> mx.array:
        """Generate embeddings for video"""
        frames = self.processor.process_video_frames(video_path, max_frames)
        frame_embeddings = []
        
        for frame in frames:
            with mx.no_grad():
                emb = self.model.encode_image(mx.expand_dims(frame, axis=0))
                frame_embeddings.append(emb)
        
        # Pool frame embeddings
        frame_embeddings = mx.stack(frame_embeddings, axis=0)
        
        if pooling == "mean":
            video_embedding = mx.mean(frame_embeddings, axis=0)
        elif pooling == "max":
            video_embedding = mx.max(frame_embeddings, axis=0)
        else:  # first
            video_embedding = frame_embeddings[0]
        
        if normalize:
            video_embedding = mx.nn.functional.normalize(video_embedding, axis=-1)
            
        return video_embedding
```

### 4. FastAPI Server Implementation

```python
# server.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import mlx.core as mx
import numpy as np
import asyncio
from contextlib import asynccontextmanager

class TextEmbeddingRequest(BaseModel):
    texts: List[str]
    normalize: bool = True

class MultimodalEmbeddingRequest(BaseModel):
    text: Optional[str] = None
    image_url: Optional[str] = None
    normalize: bool = True

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    shape: List[int]
    model: str = "Ops-MM-embedding-v1-7B"

# Global model instance
model_instance = None
engine_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage model lifecycle"""
    global model_instance, engine_instance
    
    # Startup
    print("🚀 Loading Ops-MM-embedding-v1-7B with MLX...")
    
    # Detect system memory and choose quantization
    import psutil
    memory_gb = psutil.virtual_memory().total // (1024**3)
    
    if memory_gb <= 16:
        quantization = "q4"
        print(f"💾 Using 4-bit quantization for {memory_gb}GB system")
    elif memory_gb <= 32:
        quantization = "q8"
        print(f"💾 Using 8-bit quantization for {memory_gb}GB system")
    else:
        quantization = None
        print(f"💾 Using full precision for {memory_gb}GB system")
    
    model_instance = OpsMMEmbeddingMLX(quantization=quantization)
    await model_instance.load_model()
    
    processor = MultimodalProcessor({"image_size": 336})
    engine_instance = EmbeddingEngine(model_instance, processor)
    
    print("✅ Model loaded and ready!")
    
    yield
    
    # Shutdown
    print("👋 Shutting down model...")
    del model_instance
    del engine_instance
    mx.metal.clear_cache()

app = FastAPI(
    title="Multimodal Embedding Server (MLX)",
    version="1.0.0",
    lifespan=lifespan
)

@app.post("/embed/text", response_model=EmbeddingResponse)
async def embed_text(request: TextEmbeddingRequest):
    """Generate text embeddings"""
    try:
        embeddings = await engine_instance.generate_text_embeddings(
            request.texts,
            normalize=request.normalize
        )
        
        # Convert to Python list
        embeddings_list = embeddings.tolist()
        
        return EmbeddingResponse(
            embeddings=embeddings_list,
            shape=list(embeddings.shape)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embed/image", response_model=EmbeddingResponse)
async def embed_image(file: UploadFile = File(...)):
    """Generate image embeddings"""
    try:
        # Save uploaded file temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        
        embeddings = await engine_instance.generate_image_embeddings(
            [tmp_path],
            normalize=True
        )
        
        # Cleanup
        import os
        os.unlink(tmp_path)
        
        return EmbeddingResponse(
            embeddings=embeddings.tolist(),
            shape=list(embeddings.shape)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embed/multimodal", response_model=EmbeddingResponse)
async def embed_multimodal(request: MultimodalEmbeddingRequest):
    """Generate unified multimodal embeddings"""
    try:
        embeddings = await engine_instance.generate_multimodal_embeddings(
            text=request.text,
            image_path=request.image_url,
            normalize=request.normalize
        )
        
        return EmbeddingResponse(
            embeddings=embeddings.tolist(),
            shape=list(embeddings.shape)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embed/video", response_model=EmbeddingResponse)
async def embed_video(
    file: UploadFile = File(...),
    max_frames: int = 8,
    pooling: str = "mean"
):
    """Generate video embeddings from frames"""
    try:
        # Save uploaded video temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        
        embeddings = await engine_instance.generate_video_embeddings(
            tmp_path,
            max_frames=max_frames,
            pooling=pooling,
            normalize=True
        )
        
        # Cleanup
        import os
        os.unlink(tmp_path)
        
        return EmbeddingResponse(
            embeddings=[embeddings.tolist()],
            shape=list(embeddings.shape)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_instance is not None,
        "framework": "MLX",
        "device": "Apple Silicon GPU"
    }

@app.get("/info")
async def model_info():
    """Get model information"""
    import mlx
    return {
        "model": "Ops-MM-embedding-v1-7B",
        "framework": "MLX",
        "mlx_version": mlx.__version__,
        "quantization": model_instance.quantization if model_instance else None,
        "embedding_dim": 1024,
        "supported_modalities": ["text", "image", "video", "multimodal"]
    }
```

## 🔧 Configuration Management

### Configuration Schema (config.yaml)

```yaml
# config.yaml - MLX Multimodal Embedding Server Configuration
model:
  name: "OpenSearch-AI/Ops-MM-embedding-v1-7B"
  quantization: "auto"  # auto, q4, q8, or none
  cache_dir: "./models"
  device: "gpu"

processing:
  image_size: 336
  max_text_length: 512
  max_video_frames: 8
  video_pooling: "mean"  # mean, max, or first

server:
  host: "0.0.0.0"
  port: 8000
  workers: 1
  max_batch_size: 32
  timeout: 30

optimization:
  use_cache: true
  cache_size: 1000
  warmup_on_start: true
  
monitoring:
  enable_metrics: true
  log_level: "INFO"
  log_file: "./logs/mlx_server.log"
```

### Memory Profiles

```yaml
# profiles/m1_16gb.yaml
model:
  quantization: "q4"
  
processing:
  max_batch_size: 16
  
optimization:
  cache_size: 500

---
# profiles/m1_32gb.yaml  
model:
  quantization: "q8"
  
processing:
  max_batch_size: 32
  
optimization:
  cache_size: 1000

---
# profiles/m1_64gb.yaml
model:
  quantization: "none"
  
processing:
  max_batch_size: 64
  
optimization:
  cache_size: 2000
```

## 📊 Performance Optimization

### 1. MLX-Specific Optimizations

```python
# optimizations.py
import mlx.core as mx
from functools import lru_cache
import hashlib

class MLXOptimizer:
    """MLX-specific performance optimizations"""
    
    @staticmethod
    def enable_metal_optimizations():
        """Configure Metal performance shaders"""
        mx.metal.set_memory_limit(0.8)  # Use 80% of available memory
        mx.metal.set_cache_limit(2 * 1024 * 1024 * 1024)  # 2GB cache
    
    @staticmethod
    def batch_processing(items, batch_size=32):
        """Efficient batch processing with MLX"""
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            yield mx.stack(batch)
    
    @staticmethod
    @lru_cache(maxsize=1000)
    def cached_embedding(text_hash: str):
        """Cache frequently requested embeddings"""
        # Cache implementation
        pass
```

### 2. Memory Management

```python
# memory_manager.py
import mlx.core as mx
import psutil
from typing import Dict, Any

class MemoryManager:
    """Manage memory for different M1 configurations"""
    
    def __init__(self):
        self.total_memory = psutil.virtual_memory().total
        self.memory_profile = self.detect_profile()
    
    def detect_profile(self) -> Dict[str, Any]:
        """Auto-detect optimal settings based on system memory"""
        gb = self.total_memory // (1024**3)
        
        if gb <= 16:
            return {
                "quantization": "q4",
                "batch_size": 16,
                "cache_size": 500,
                "max_context": 2048
            }
        elif gb <= 32:
            return {
                "quantization": "q8",
                "batch_size": 32,
                "cache_size": 1000,
                "max_context": 4096
            }
        else:
            return {
                "quantization": None,
                "batch_size": 64,
                "cache_size": 2000,
                "max_context": 8192
            }
    
    def optimize_for_inference(self):
        """Optimize MLX for inference workloads"""
        # Set memory limits
        mx.metal.set_memory_limit(0.75)  # 75% of system memory
        
        # Enable fast math
        mx.metal.set_fast_math(True)
        
        # Configure threading
        mx.set_num_threads(8)  # M1 has 8 cores
```

## 🚀 Deployment Guide

### Installation Script

```bash
#!/bin/bash
# setup_mlx_server.sh

echo "🚀 Setting up MLX Multimodal Embedding Server"

# 1. Check system
if [[ $(uname -m) != "arm64" ]]; then
    echo "❌ This script is for Apple Silicon only"
    exit 1
fi

# 2. Install dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install mlx mlx-lm
pip install fastapi uvicorn[standard]
pip install pillow opencv-python-headless
pip install psutil pyyaml
pip install huggingface-hub transformers

# 3. Create directory structure
echo "📁 Creating project structure..."
mkdir -p models logs configs profiles

# 4. Download model (optional - can be done at runtime)
echo "🤖 Model will be downloaded on first run"

# 5. Create service script
cat > start_server.sh << 'EOF'
#!/bin/bash
# Detect memory and use appropriate profile
MEMORY_GB=$(sysctl -n hw.memsize | awk '{print int($1/1024/1024/1024)}')

if [ $MEMORY_GB -le 16 ]; then
    CONFIG="profiles/m1_16gb.yaml"
elif [ $MEMORY_GB -le 32 ]; then
    CONFIG="profiles/m1_32gb.yaml"
else
    CONFIG="profiles/m1_64gb.yaml"
fi

echo "Starting server with config: $CONFIG"
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
EOF

chmod +x start_server.sh

echo "✅ Setup complete! Run ./start_server.sh to start the server"
```

### Docker Support (Optional)

```dockerfile
# Dockerfile.mlx
FROM ghcr.io/ml-explore/mlx:latest

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Start server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 📈 Performance Benchmarks

### Expected Performance on Apple Silicon

| Device | Model Config | Load Time | Inference Speed | Memory Usage |
|--------|-------------|-----------|-----------------|--------------|
| M1 16GB | Q4 quantized | 5-8s | 150-200 tok/s | ~4GB |
| M1 32GB | Q8 quantized | 8-12s | 120-180 tok/s | ~8GB |
| M2 Pro | Q8 quantized | 5-8s | 200-250 tok/s | ~8GB |
| M2 Max | Full precision | 10-15s | 180-220 tok/s | ~16GB |

## 🔄 Implementation Roadmap

### Phase 1: Core Infrastructure (Week 1)
- [ ] Set up MLX environment
- [ ] Implement model loading with quantization
- [ ] Create basic text embedding endpoint
- [ ] Test on M1 hardware

### Phase 2: Multimodal Support (Week 2)
- [ ] Add image processing pipeline
- [ ] Implement multimodal embedding fusion
- [ ] Add video frame extraction
- [ ] Create unified API endpoints

### Phase 3: Optimization (Week 3)
- [ ] Implement caching layer
- [ ] Add batch processing
- [ ] Optimize memory management
- [ ] Profile and tune performance

### Phase 4: Production Features (Week 4)
- [ ] Add authentication/rate limiting
- [ ] Implement monitoring/metrics
- [ ] Create deployment scripts
- [ ] Write comprehensive documentation

### Phase 5: Testing & Deployment (Week 5)
- [ ] Load testing
- [ ] Performance benchmarking
- [ ] Security audit
- [ ] Production deployment

## 🎯 Success Criteria

1. **Performance**: 150+ tokens/second on M1
2. **Memory**: <50% RAM usage under load
3. **Latency**: <100ms for single embedding
4. **Reliability**: 99.9% uptime
5. **Scalability**: Handle 100+ concurrent requests

## 📚 Additional Resources

- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [Ops-MM Model Card](https://huggingface.co/OpenSearch-AI/Ops-MM-embedding-v1-7B)
- [FastAPI Best Practices](https://fastapi.tiangolo.com/deployment/)
- [Apple Metal Performance](https://developer.apple.com/metal/)

---

This architecture leverages MLX's native Apple Silicon optimization to deliver production-ready multimodal embeddings with excellent performance and minimal resource usage.