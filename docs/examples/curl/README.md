# cURL Examples for tkr-embed API

This directory contains practical cURL examples for testing and integrating with the tkr-embed GPT-OSS-20B generation API.

## Setup

Before running these examples, ensure:

1. The tkr-embed server is running on `http://localhost:8008`
2. You have a valid API key (replace `your-api-key-here` in examples)
3. cURL is installed on your system

## Basic Usage

### 1. Health Check

```bash
curl -X GET http://localhost:8008/health
```

### 2. Model Information

```bash
curl -X GET http://localhost:8008/info
```

## Text Generation Examples

### Basic Text Generation

```bash
curl -X POST http://localhost:8008/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -d '{
    "text": "Explain quantum computing in simple terms.",
    "max_tokens": 256,
    "temperature": 0.7,
    "reasoning_level": "medium"
  }'
```

### Low Reasoning (Fast Response)

```bash
curl -X POST http://localhost:8008/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -d '{
    "text": "What is the capital of France?",
    "max_tokens": 50,
    "temperature": 0.1,
    "reasoning_level": "low"
  }'
```

### High Reasoning (Detailed Analysis)

```bash
curl -X POST http://localhost:8008/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -d '{
    "text": "Analyze the potential impact of artificial intelligence on the job market over the next decade.",
    "max_tokens": 500,
    "temperature": 0.8,
    "reasoning_level": "high",
    "top_p": 0.9,
    "repetition_penalty": 1.1
  }'
```

### Creative Writing (High Temperature)

```bash
curl -X POST http://localhost:8008/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -d '{
    "text": "Write a creative opening line for a science fiction story about time travel.",
    "max_tokens": 150,
    "temperature": 1.2,
    "reasoning_level": "medium",
    "top_p": 0.95
  }'
```

### Technical Documentation (Low Temperature)

```bash
curl -X POST http://localhost:8008/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -d '{
    "text": "Write API documentation for a REST endpoint that creates a new user account.",
    "max_tokens": 300,
    "temperature": 0.3,
    "reasoning_level": "medium",
    "repetition_penalty": 1.05
  }'
```

## Chat Completion Examples

### Basic Chat

```bash
curl -X POST http://localhost:8008/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hello! Can you help me learn Python?"}
    ],
    "max_tokens": 256,
    "temperature": 0.7,
    "reasoning_level": "medium"
  }'
```

### Chat with System Prompt

```bash
curl -X POST http://localhost:8008/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -d '{
    "messages": [
      {"role": "user", "content": "How do I optimize a slow database query?"}
    ],
    "system_prompt": "You are a senior database administrator with expertise in query optimization and performance tuning. Provide detailed, actionable advice.",
    "max_tokens": 400,
    "temperature": 0.6,
    "reasoning_level": "high"
  }'
```

### Multi-turn Conversation

```bash
curl -X POST http://localhost:8008/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is machine learning?"},
      {"role": "assistant", "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every task."},
      {"role": "user", "content": "Can you give me a practical example?"}
    ],
    "system_prompt": "You are a patient teacher explaining technical concepts with clear examples.",
    "max_tokens": 300,
    "temperature": 0.7,
    "reasoning_level": "medium"
  }'
```

## Streaming Generation Examples

### Basic Streaming

```bash
curl -X POST http://localhost:8008/stream \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -H "Accept: text/event-stream" \
  -N \
  -d '{
    "text": "Write a short story about a robot learning to paint.",
    "max_tokens": 300,
    "temperature": 0.8,
    "reasoning_level": "medium"
  }'
```

### Streaming with Processing

Save the streaming output to a file and process it:

```bash
curl -X POST http://localhost:8008/stream \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -H "Accept: text/event-stream" \
  -N \
  -d '{
    "text": "Explain how neural networks work.",
    "max_tokens": 400,
    "temperature": 0.7,
    "reasoning_level": "high"
  }' > stream_output.txt
```

Then process the Server-Sent Events:

```bash
# Extract just the content deltas
grep "data: " stream_output.txt | \
sed 's/data: //' | \
jq -r 'select(.chunk.delta != null) | .chunk.delta' | \
tr -d '\n'
```

## Testing Different Parameters

### Temperature Comparison

Test different creativity levels:

```bash
# Conservative (temperature 0.1)
curl -X POST http://localhost:8008/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -d '{"text": "Complete this sentence: The future of AI is", "max_tokens": 50, "temperature": 0.1}' \
  | jq -r '.generated_text'

# Balanced (temperature 0.7)
curl -X POST http://localhost:8008/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -d '{"text": "Complete this sentence: The future of AI is", "max_tokens": 50, "temperature": 0.7}' \
  | jq -r '.generated_text'

# Creative (temperature 1.5)
curl -X POST http://localhost:8008/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -d '{"text": "Complete this sentence: The future of AI is", "max_tokens": 50, "temperature": 1.5}' \
  | jq -r '.generated_text'
```

### Reasoning Level Comparison

```bash
# Quick response
curl -X POST http://localhost:8008/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -d '{"text": "Explain blockchain", "reasoning_level": "low", "max_tokens": 100}' \
  | jq -r '.generated_text'

# Detailed explanation
curl -X POST http://localhost:8008/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -d '{"text": "Explain blockchain", "reasoning_level": "high", "max_tokens": 300}' \
  | jq -r '.generated_text'
```

## Error Testing

### Invalid API Key

```bash
curl -X POST http://localhost:8008/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: invalid-key" \
  -d '{"text": "Hello world"}' \
  -w "Status: %{http_code}\n"
```

### Invalid Parameters

```bash
# Temperature out of range
curl -X POST http://localhost:8008/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -d '{"text": "Hello", "temperature": 3.0}' \
  -w "Status: %{http_code}\n"

# Empty prompt
curl -X POST http://localhost:8008/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -d '{"text": "", "max_tokens": 100}' \
  -w "Status: %{http_code}\n"
```

## Performance Testing

### Measure Response Time

```bash
curl -X POST http://localhost:8008/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -d '{
    "text": "Write a summary of machine learning concepts.",
    "max_tokens": 200,
    "temperature": 0.7
  }' \
  -w "Time: %{time_total}s\n" \
  -s | jq '.processing_time'
```

### Concurrent Requests

Test multiple requests simultaneously:

```bash
# Run 5 concurrent requests
for i in {1..5}; do
  curl -X POST http://localhost:8008/generate \
    -H "Content-Type: application/json" \
    -H "X-API-Key: your-api-key-here" \
    -d "{\"text\":\"Request $i: Explain AI\",\"max_tokens\":100}" \
    -w "Request $i time: %{time_total}s\n" &
done
wait
```

## Useful cURL Options

- `-s, --silent`: Don't show progress meter
- `-S, --show-error`: Show error even with -s
- `-w, --write-out`: Display information after completion
- `-N, --no-buffer`: Disable buffering (important for streaming)
- `-H, --header`: Add headers
- `-d, --data`: Send POST data
- `-X, --request`: Specify request method
- `| jq`: Pretty-print JSON responses (requires jq installation)

## Automation Scripts

### Simple Performance Test

```bash
#!/bin/bash
# performance_test.sh

API_KEY="your-api-key-here"
BASE_URL="http://localhost:8008"

echo "Testing tkr-embed performance..."

# Test health
curl -s "$BASE_URL/health" | jq -r '.status'

# Test different reasoning levels
for level in low medium high; do
  echo "Testing $level reasoning..."

  response=$(curl -s -X POST "$BASE_URL/generate" \
    -H "Content-Type: application/json" \
    -H "X-API-Key: $API_KEY" \
    -d "{
      \"text\": \"Explain quantum computing\",
      \"reasoning_level\": \"$level\",
      \"max_tokens\": 150
    }")

  processing_time=$(echo "$response" | jq -r '.processing_time')
  tokens=$(echo "$response" | jq -r '.tokens_used')

  echo "$level: ${processing_time}s, $tokens tokens"
done
```

Make it executable and run:

```bash
chmod +x performance_test.sh
./performance_test.sh
```

## Troubleshooting

### Common Issues

1. **Connection refused**: Server not running
   ```bash
   curl -I http://localhost:8008/health
   ```

2. **401 Unauthorized**: Invalid API key
   ```bash
   curl -H "X-API-Key: your-api-key" http://localhost:8008/health
   ```

3. **429 Rate Limited**: Too many requests
   - Wait for rate limit reset
   - Check rate limit headers

4. **503 Service Unavailable**: Model not ready
   - Check server logs
   - Wait for model loading to complete

### Debug Mode

Add verbose output to see full request/response:

```bash
curl -v -X POST http://localhost:8008/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -d '{"text": "Hello", "max_tokens": 50}'
```

This will show all HTTP headers and connection details for debugging.