# tkr-embed API Documentation

## Overview

The tkr-embed server provides a production-ready text generation API powered by the GPT-OSS-20B model, optimized for Apple Silicon and built on the MLX framework. This API supports single text generation, chat completion, and streaming generation with sophisticated reasoning capabilities.

## Base URL

```
http://localhost:8008
```

## Authentication

All generation endpoints require API key authentication (configurable). Include your API key in requests using one of these methods:

### Header Authentication (Recommended)
```bash
curl -H "X-API-Key: your-api-key-here" \
     -H "Content-Type: application/json" \
     http://localhost:8008/generate
```

### Query Parameter Authentication
```bash
curl "http://localhost:8008/generate?api_key=your-api-key-here"
```

## Core Endpoints

### 1. Health Check

**GET** `/health`

Check server status and model readiness.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "framework": "MLX",
  "device": "Apple Silicon GPU",
  "memory_usage_gb": 18.5,
  "uptime_seconds": 3600,
  "generation_ready": true,
  "active_conversations": 2
}
```

### 2. Model Information

**GET** `/info`

Get detailed model information and capabilities.

**Response:**
```json
{
  "model_path": "microsoft/gpt-oss-20b",
  "framework": "MLX",
  "mlx_version": "0.29.0",
  "quantization": "Q8_0",
  "context_length": 8192,
  "vocab_size": 50257,
  "supported_tasks": ["text_generation", "chat_completion"],
  "load_time": 15.2,
  "memory_usage_gb": 18.5,
  "reasoning_capabilities": ["low", "medium", "high"]
}
```

## Generation Endpoints

### 3. Text Generation

**POST** `/generate`

Generate text completion for a single prompt.

**Request Body:**
```json
{
  "text": "Explain the concept of machine learning in simple terms.",
  "max_tokens": 256,
  "temperature": 0.7,
  "reasoning_level": "medium",
  "top_p": 0.9,
  "top_k": 50,
  "repetition_penalty": 1.1
}
```

**Response:**
```json
{
  "generated_text": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every task...",
  "tokens_used": 156,
  "reasoning_level": "medium",
  "processing_time": 2.34,
  "finish_reason": "stop",
  "model": "gpt-oss-20b",
  "prompt_tokens": 12,
  "completion_tokens": 144
}
```

### 4. Chat Completion

**POST** `/chat`

Generate responses for conversational interactions.

**Request Body:**
```json
{
  "messages": [
    {"role": "user", "content": "What is machine learning?"},
    {"role": "assistant", "content": "Machine learning is..."},
    {"role": "user", "content": "Can you give me an example?"}
  ],
  "system_prompt": "You are a helpful AI assistant specialized in explaining technical concepts.",
  "reasoning_level": "medium",
  "max_tokens": 256,
  "temperature": 0.7,
  "top_p": 0.9
}
```

**Response:**
```json
{
  "response": "Certainly! A great example of machine learning is email spam detection...",
  "conversation_id": "conv_1694876543210",
  "tokens_used": 178,
  "reasoning_level": "medium",
  "processing_time": 1.85,
  "finish_reason": "stop",
  "model": "gpt-oss-20b",
  "prompt_tokens": 34,
  "completion_tokens": 144
}
```

### 5. Streaming Generation

**POST** `/stream`

Stream text generation with Server-Sent Events for real-time response.

**Request Body:** (Same as `/generate`)

**Response Format:** Server-Sent Events (SSE)
```
data: {"chunk": {"delta": "Machine", "finish_reason": null, "tokens_generated": 1}, "conversation_id": null, "reasoning_level": "medium", "model": "gpt-oss-20b"}

data: {"chunk": {"delta": " learning", "finish_reason": null, "tokens_generated": 2}, "conversation_id": null, "reasoning_level": "medium", "model": "gpt-oss-20b"}

data: {"chunk": {"delta": "", "finish_reason": "stop", "tokens_generated": 144}, "conversation_id": null, "reasoning_level": "medium", "model": "gpt-oss-20b"}

data: [DONE]
```

## Request Parameters

### Common Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_tokens` | integer | 512 | Maximum tokens to generate (1-4096) |
| `temperature` | float | 0.7 | Sampling temperature (0.0-2.0) |
| `reasoning_level` | string | "medium" | Reasoning complexity: "low", "medium", "high" |
| `top_p` | float | 0.9 | Nucleus sampling probability (0.0-1.0) |
| `top_k` | integer | 50 | Top-k sampling parameter (1-100) |
| `repetition_penalty` | float | 1.1 | Repetition penalty (1.0-2.0) |

### Text Generation Specific

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `text` | string | Yes | Input prompt (1-10000 characters) |
| `stream` | boolean | No | Enable streaming response |

### Chat Completion Specific

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `messages` | array | Yes | Conversation messages (1-50 messages) |
| `system_prompt` | string | No | System prompt (max 2000 characters) |
| `stream` | boolean | No | Enable streaming response |

### Message Format

```json
{
  "role": "user|assistant|system",
  "content": "Message content"
}
```

## Reasoning Levels

The API supports three reasoning levels that affect the model's response style and depth:

### Low Reasoning (`"low"`)
- **Use case:** Quick responses, simple queries, basic information
- **Characteristics:** Fast, concise, direct answers
- **Performance:** ~50ms response time, lower token usage
- **Example:** Simple factual questions, basic definitions

### Medium Reasoning (`"medium"`)
- **Use case:** Balanced responses, explanations, moderate complexity
- **Characteristics:** Thoughtful responses with context
- **Performance:** ~100ms response time, moderate token usage
- **Example:** Explaining concepts, providing detailed answers

### High Reasoning (`"high"`)
- **Use case:** Complex analysis, detailed explanations, problem-solving
- **Characteristics:** Deep analysis, comprehensive responses
- **Performance:** ~200ms response time, higher token usage
- **Example:** Multi-step reasoning, complex problem solving

## Error Handling

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request - Invalid parameters |
| 401 | Unauthorized - Invalid or missing API key |
| 403 | Forbidden - Insufficient permissions |
| 429 | Too Many Requests - Rate limit exceeded |
| 500 | Internal Server Error |
| 503 | Service Unavailable - Model not ready |

### Error Response Format

```json
{
  "error": "Invalid request parameters",
  "detail": "Temperature must be between 0.0 and 2.0",
  "request_id": "req_12345",
  "error_code": "VALIDATION_ERROR"
}
```

### Common Error Codes

- `VALIDATION_ERROR`: Invalid request parameters
- `MODEL_NOT_READY`: Model is still loading
- `RATE_LIMIT_EXCEEDED`: Too many requests
- `AUTHENTICATION_FAILED`: Invalid API key
- `GENERATION_FAILED`: Model generation error

## Rate Limiting

Rate limits are applied per API key:

- **Requests per minute:** 60
- **Requests per hour:** 1000

Rate limit headers are included in responses:
```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 59
X-RateLimit-Reset: 1694876543
```

## Model Specifications

### GPT-OSS-20B Model Details

- **Parameters:** 20 billion
- **Context Length:** 8,192 tokens
- **Vocabulary Size:** 50,257 tokens
- **Quantization:** Q8_0 (8-bit quantization)
- **Memory Usage:** ~18.5 GB
- **Framework:** MLX 0.29.0
- **Hardware:** Optimized for Apple Silicon

### Performance Characteristics

- **Throughput:** 150+ tokens/second
- **Latency:** <100ms (medium reasoning)
- **Concurrent Requests:** 100+ supported
- **Memory Efficiency:** <50% system RAM utilization

## Best Practices

### 1. Choosing Reasoning Levels
- Use `"low"` for simple, factual queries
- Use `"medium"` for balanced explanations (recommended default)
- Use `"high"` for complex analysis and reasoning tasks

### 2. Temperature Settings
- **0.0-0.3:** Deterministic, factual responses
- **0.4-0.8:** Balanced creativity and consistency
- **0.9-1.5:** Creative, varied responses
- **1.6-2.0:** Highly creative, less predictable

### 3. Token Management
- Set appropriate `max_tokens` based on expected response length
- Monitor `tokens_used` in responses for cost tracking
- Use shorter prompts for better context utilization

### 4. Streaming for Real-time Applications
- Use `/stream` endpoint for chat interfaces
- Handle SSE properly in your client application
- Parse JSON chunks incrementally

### 5. Error Handling
- Always check HTTP status codes
- Implement retry logic for transient errors (429, 503)
- Handle rate limiting gracefully
- Log error details for debugging

## Administration

### Admin Endpoints

Admin endpoints require `admin` permission and are available under `/admin`:

- `POST /admin/api-keys` - Create new API keys
- `GET /admin/api-keys` - List all API keys
- `DELETE /admin/api-keys` - Revoke API keys
- `GET /admin/stats` - Server statistics
- `GET /admin/config` - Server configuration

See the [Admin API Documentation](ADMIN_API.md) for detailed information.

## SDKs and Examples

Comprehensive usage examples are available in the `/docs/examples/` directory:

- [Python Examples](examples/python/)
- [JavaScript Examples](examples/javascript/)
- [cURL Examples](examples/curl/)
- [Streaming Examples](examples/streaming/)

## Support

For issues, questions, or feature requests:

1. Check the [troubleshooting guide](TROUBLESHOOTING.md)
2. Review [deployment guides](deployment/)
3. Submit issues via the project repository

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and updates.