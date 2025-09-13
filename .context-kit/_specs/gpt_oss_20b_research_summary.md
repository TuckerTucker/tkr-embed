# GPT-OSS-20B Model Research Summary

## Model Overview

**GPT-OSS-20B** is OpenAI's open-weight Mixture of Experts (MoE) language model designed for reasoning and agentic tasks.

### Key Specifications
- **Total Parameters**: 21 billion
- **Active Parameters**: 3.6 billion per token (sparse activation)
- **Architecture**: Mixture of Experts Transformer
- **License**: Apache 2.0 (commercial use allowed)
- **Memory Optimized**: MXFP4 quantization for 16GB operation

## Model Architecture

### Core Features
- **Alternating attention patterns**: Dense and locally banded sparse attention (similar to GPT-3)
- **Grouped multi-query attention**: Group size of 8 for inference efficiency
- **MoE structure**: ~8 experts with 2 active per token (estimated)
- **Post-training quantization**: MXFP4 quantization of MoE weights

### Reasoning Capabilities
The model supports three adjustable reasoning levels:
- **Low**: Fast responses for general dialogue
- **Medium**: Balanced speed and detail
- **High**: Deep and detailed analysis

Reasoning level is controlled via system prompts (e.g., "Reasoning: high").

## MLX Framework Compatibility

### Memory Requirements (32GB Apple Silicon M1)
| Quantization | Model Memory | Total Memory | Compatible |
|--------------|-------------|--------------|------------|
| None (FP32)  | 84.0 GB     | 92.0 GB      | ❌ No      |
| Q8 (8-bit)   | 21.0 GB     | 29.0 GB      | ✅ Yes     |
| Q4 (4-bit)   | 10.5 GB     | 18.5 GB      | ✅ Yes     |
| MXFP4        | 16.0 GB     | 24.0 GB      | ✅ Yes     |

**Recommended Configuration**: Q8 quantization for optimal performance/memory balance

### Apple Silicon Optimizations
- **Unified Memory Architecture**: MLX eliminates CPU-GPU data transfer overhead
- **Metal Performance Shaders**: Native GPU acceleration
- **MoE Quantization**: Specialized techniques for sparse expert models
- **Memory Efficiency**: Only active experts (3.6B) loaded during inference

## Implementation Feasibility

### ✅ Advantages
- **Open License**: Apache 2.0 allows commercial deployment
- **Memory Efficient**: Fits in 32GB with quantization
- **MLX Native**: Optimized for Apple Silicon architecture
- **Fine-tunable**: Can be customized with LoRA/QLoRA
- **Chain of Thought**: Full reasoning process access

### ⚠️ Challenges
- **Harmony Format**: Requires specific response format for proper functioning
- **MoE Complexity**: All experts must be loaded in memory
- **Quantization Research**: MoE quantization is still an active research area
- **Metal Support**: Current MLX Metal detection shows limitations

## Technical Implementation Path

1. **Model Loading**: Use transformers library with MLX backend
2. **Quantization**: Apply Q8 or Q4 quantization for memory efficiency
3. **Memory Management**: Leverage unified memory architecture
4. **Reasoning Integration**: Implement system prompt formatting
5. **Performance Optimization**: Batch processing and caching

## Integration with tkr-embed

The GPT-OSS-20B model can be integrated into the existing MLX embedding server as an additional reasoning-capable model alongside the current multimodal embedding pipeline. This would provide:

- **Reasoning Embeddings**: Context-aware embeddings with reasoning capabilities
- **Multimodal Reasoning**: Text + image reasoning with embedding extraction
- **Performance Scaling**: MoE architecture for efficient large-scale inference

## Next Steps

1. **Model Download**: Pull model from HuggingFace (`openai/gpt-oss-20b`)
2. **MLX Conversion**: Convert to MLX format with Q8 quantization
3. **Interface Implementation**: Implement the specification in `gpt_oss_20b_model_spec.py`
4. **Integration Testing**: Test with existing tkr-embed infrastructure
5. **Performance Benchmarking**: Measure throughput and memory usage

---

**Created**: September 13, 2025
**Specification File**: `/Volumes/tkr-riffic/@tkr-projects/tkr-embed/.context-kit/_specs/gpt_oss_20b_model_spec.py`