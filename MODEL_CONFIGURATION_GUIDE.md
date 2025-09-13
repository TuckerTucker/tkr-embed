# GPT-OSS-20B Model Configuration Guide
## Agent A: Performance & Testing - Model Integration Requirements

This guide provides the exact configuration changes needed to integrate a proper GPT-OSS-20B text generation model with the tkr-embed system.

---

## Current Issue Analysis

### ❌ Current Configuration Problem
The system is currently configured for a **multimodal embedding model** (`OpenSearch-AI/Ops-MM-embedding-v1-7B`) instead of a **text generation model**:

```yaml
# Current problematic config
model:
  model_path: "OpenSearch-AI/Ops-MM-embedding-v1-7B"  # ❌ Wrong model type
  # This is a Qwen2VL multimodal embedding model, not a causal language model
```

**Error Result:**
```
ValueError: Unrecognized configuration class <class 'transformers.models.qwen2_vl.configuration_qwen2_vl.Qwen2VLConfig'>
for this kind of AutoModel: AutoModelForCausalLM
```

---

## ✅ Required Model Specifications

### GPT-OSS-20B Compatible Models
The system needs a model compatible with `AutoModelForCausalLM`. Options include:

#### Option 1: Official GPT-OSS Models (Recommended)
```yaml
model:
  model_path: "microsoft/gpt-oss-20b"        # If available
  # OR
  model_path: "openai/gpt-oss-20b"           # Alternative path
```

#### Option 2: Compatible Large Language Models
```yaml
# High-quality alternatives that work with the existing infrastructure
model:
  model_path: "microsoft/DialoGPT-large"     # Smaller, good for testing
  # OR
  model_path: "EleutherAI/gpt-neox-20b"      # Similar size to GPT-OSS-20B
  # OR
  model_path: "bigscience/bloom-20b"         # 20B parameter alternative
  # OR
  model_path: "meta-llama/Llama-2-13b-hf"    # Proven performance
```

#### Option 3: Local Model Path
```yaml
model:
  model_path: "/path/to/local/gpt-oss-20b"   # If model is downloaded locally
```

---

## 🔧 Complete Configuration Fix

### 1. Update config.dev.yaml
```yaml
# /Volumes/tkr-riffic/@tkr-projects/tkr-embed/config.dev.yaml
environment: development
debug: true

server:
  host: "0.0.0.0"
  port: 8000
  reload: true

model:
  # ✅ CORRECTED MODEL CONFIGURATION
  model_path: "microsoft/DialoGPT-large"     # Start with smaller model for testing
  quantization: "q8"                         # Optimal for 32GB system
  cache_dir: "./models"
  device: "mps"                             # Apple Silicon optimization
  max_sequence_length: 2048                 # Appropriate for DialoGPT
  trust_remote_code: true
  max_tokens: 1024                          # Conservative for testing

generation:
  max_tokens: 1024
  temperature: 0.7
  top_p: 0.9
  top_k: 50
  streaming_enabled: true
  generation_timeout_seconds: 30

# Security settings
security:
  require_api_key: false                    # For development
  cors_origins: ["*"]                       # Development only

# Rate limiting (lenient for testing)
rate_limit:
  enabled: true
  requests_per_minute: 100
  requests_per_hour: 1000
```

### 2. Production Configuration
```yaml
# config.prod.yaml - For production deployment
environment: production
debug: false

model:
  # Production-ready large model
  model_path: "EleutherAI/gpt-neox-20b"     # Full 20B model
  quantization: "q8"                        # Performance/quality balance
  cache_dir: "/opt/models"                  # Production model cache
  device: "mps"
  max_sequence_length: 8192                 # Full context window
  max_tokens: 4096

generation:
  max_tokens: 4096
  generation_timeout_seconds: 60            # Longer timeout for large model

security:
  require_api_key: true                     # Required for production
  cors_origins: ["https://your-domain.com"]

rate_limit:
  enabled: true
  requests_per_minute: 20                   # Conservative for 20B model
  requests_per_hour: 500
```

---

## 🧪 Testing Configuration

### Step-by-Step Validation Process

#### 1. Test with Small Model First
```yaml
# Start with a small, fast model to validate infrastructure
model:
  model_path: "microsoft/DialoGPT-medium"   # ~355M parameters
  quantization: "none"                      # No quantization needed
  max_tokens: 256                          # Quick testing
```

#### 2. Validate Model Compatibility
```python
# Test script to validate model before full integration
import os
os.environ['GENERATION_ENV'] = 'testing'

from transformers import AutoConfig, AutoModelForCausalLM

def test_model_compatibility(model_path):
    print(f"Testing model: {model_path}")

    try:
        # Test config loading
        config = AutoConfig.from_pretrained(model_path)
        print(f"✅ Config loaded: {type(config)}")

        # Test model architecture compatibility
        print(f"Architecture: {config.architectures}")

        # Verify it's a causal LM
        if hasattr(config, 'vocab_size'):
            print(f"✅ Vocab size: {config.vocab_size}")
        else:
            print("❌ No vocab_size - not a language model")
            return False

        # Test model loading (without weights)
        model = AutoModelForCausalLM.from_config(config)
        print(f"✅ Model architecture compatible")

        return True

    except Exception as e:
        print(f"❌ Model incompatible: {e}")
        return False

# Test models
test_models = [
    "microsoft/DialoGPT-medium",
    "microsoft/DialoGPT-large",
    "EleutherAI/gpt-neox-20b"
]

for model in test_models:
    if test_model_compatibility(model):
        print(f"✅ {model} is compatible")
        break
```

#### 3. Server Startup Test
```bash
# Test server with corrected configuration
source start_env
GENERATION_ENV=testing python -c "
from tkr_embed.config import get_config
config = get_config()
print(f'Model path: {config.model.model_path}')
print(f'Environment: {config.environment}')
"

# Start server for validation
python -m tkr_embed.api.server
```

---

## 🚀 Performance Optimization for 20B Models

### Memory Optimization
```yaml
model:
  quantization: "q8"                        # Best quality/memory trade-off

# Memory manager settings for 32GB system
memory:
  model_memory_limit: 18                    # GB reserved for model
  kv_cache_limit: 4                         # GB for attention cache
  system_reserve: 8                         # GB for OS and other processes
  batch_size: 2                            # Conservative for large model
```

### Apple Silicon Optimizations
```yaml
model:
  device: "mps"                            # Metal Performance Shaders
  torch_compile: false                     # Disable for compatibility

hardware:
  metal_memory_limit: 0.8                  # 80% of GPU memory
  cpu_threads: 8                           # Optimal for M1
  memory_efficiency: true                  # Enable memory optimizations
```

---

## 📊 Expected Performance After Fix

### Small Model (DialoGPT-medium) - Testing
- **Startup Time:** 30-60 seconds
- **Memory Usage:** 3-5GB total
- **Generation Speed:** 100-200 tokens/second
- **Latency:** 50-100ms

### Large Model (20B) - Production
- **Startup Time:** 2-5 minutes
- **Memory Usage:** 15-20GB total
- **Generation Speed:** 50-150 tokens/second
- **Latency:** 100-300ms

---

## 🔧 Implementation Checklist

### [ ] 1. Model Configuration
- [ ] Choose appropriate model from compatible list
- [ ] Update config.dev.yaml with corrected model path
- [ ] Verify model architecture compatibility
- [ ] Test model loading without server

### [ ] 2. Server Validation
- [ ] Test server startup with new configuration
- [ ] Validate health endpoint returns model_loaded: true
- [ ] Test basic text generation functionality
- [ ] Verify all reasoning levels work

### [ ] 3. Performance Testing
- [ ] Run performance validation suite
- [ ] Execute load testing with real model
- [ ] Measure actual tokens/second generation
- [ ] Validate memory usage under load

### [ ] 4. Production Readiness
- [ ] Configure production model (20B)
- [ ] Set up proper quantization
- [ ] Configure rate limiting for large model
- [ ] Set production security settings

---

## 🚨 Critical Notes

1. **Model Size:** 20B models require significant download time (30-60GB)
2. **Memory:** Ensure 18GB+ available RAM for quantized 20B model
3. **Storage:** Models cache to disk - ensure sufficient space
4. **Network:** Initial model download requires stable high-speed connection
5. **Compatibility:** Always validate model architecture before production deployment

---

## 📞 Support & Troubleshooting

### Common Issues & Solutions

#### Issue: Model Download Fails
```bash
# Increase timeout and retry
HF_HUB_DOWNLOAD_TIMEOUT=300 python -m tkr_embed.api.server
```

#### Issue: Out of Memory
```yaml
# Reduce model size or increase quantization
model:
  quantization: "q4"    # More aggressive quantization
  max_tokens: 512       # Reduce generation length
```

#### Issue: Slow Generation
```yaml
# Optimize for speed over quality
generation:
  temperature: 1.0      # Faster sampling
  top_k: 20            # Reduce search space
  max_tokens: 256      # Shorter responses
```

---

**Guide prepared by:** Agent A - Performance & Testing Specialist
**Status:** Ready for implementation
**Priority:** Critical - Required for system functionality