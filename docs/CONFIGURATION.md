# Configuration Guide for tkr-embed

This comprehensive guide covers all configuration options for the tkr-embed GPT-OSS-20B text generation server.

## Configuration Overview

The tkr-embed server uses a hierarchical configuration system with the following priority:

1. **Environment variables** (highest priority)
2. **YAML configuration files**
3. **Default values** (lowest priority)

## Configuration Files

### Auto-Detection Order

The server automatically detects configuration files in this order:

1. `config.dev.yaml` - Development configuration (no authentication required)
2. `config.yaml` - Default/production configuration
3. Built-in defaults - Fallback configuration

### config.dev.yaml (Development Mode)

```yaml
environment: "development"
debug: true

server:
  host: "127.0.0.1"
  port: 8000

model:
  model_path: "microsoft/gpt-oss-20b"
  quantization: "auto"
  context_length: 8192
  cache_dir: "./models"

security:
  require_api_key: false  # No authentication required
  require_https: false
  cors_origins: ["*"]

rate_limit:
  enabled: false  # No rate limiting in development

cache:
  enabled: true
  max_size: 500
  ttl_seconds: 1800

logging:
  level: "DEBUG"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

### config.yaml (Production Mode)

```yaml
environment: "production"
debug: false

server:
  host: "0.0.0.0"
  port: 8000
  workers: 1

model:
  model_path: "microsoft/gpt-oss-20b"
  quantization: "auto"
  context_length: 8192
  cache_dir: "./models"
  load_timeout: 300

security:
  require_api_key: true
  require_https: false  # Set to true in production with SSL
  cors_origins: ["https://yourapp.com"]
  cors_methods: ["GET", "POST"]
  cors_headers: ["*"]

rate_limit:
  enabled: true
  requests_per_minute: 60
  requests_per_hour: 1000
  requests_per_day: 10000
  burst_size: 10
  cleanup_interval: 3600

cache:
  enabled: true
  max_size: 1000
  ttl_seconds: 3600

batch_processing:
  enabled: true
  max_batch_size: 8
  timeout_seconds: 30

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "/var/log/tkr-embed.log"
```

## Environment Variables

### Server Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `GENERATION_ENV` | string | `"production"` | Environment mode: development, production, testing |
| `GENERATION_HOST` | string | `"0.0.0.0"` | Server bind address |
| `GENERATION_PORT` | integer | `8000` | Server port |
| `GENERATION_DEBUG` | boolean | `false` | Enable debug mode |
| `GENERATION_WORKERS` | integer | `1` | Number of worker processes |

### Model Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `GENERATION_MODEL_PATH` | string | `"microsoft/gpt-oss-20b"` | Hugging Face model path |
| `GENERATION_QUANTIZATION` | string | `"auto"` | Quantization: auto, q4, q8, none |
| `GENERATION_DEVICE` | string | `"auto"` | Device: auto, cpu, gpu |
| `GENERATION_CONTEXT_LENGTH` | integer | `8192` | Maximum context length in tokens |
| `GENERATION_CACHE_DIR` | string | `"./models"` | Model cache directory |
| `GENERATION_LOAD_TIMEOUT` | integer | `300` | Model loading timeout (seconds) |

### Security Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `GENERATION_API_KEY` | string | `None` | Master API key for admin operations |
| `GENERATION_REQUIRE_API_KEY` | boolean | `true` | Require API key for generation endpoints |
| `GENERATION_REQUIRE_HTTPS` | boolean | `false` | Require HTTPS connections |
| `GENERATION_CORS_ORIGINS` | string | `["*"]` | Allowed CORS origins (JSON array) |

### Rate Limiting Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `GENERATION_RATE_LIMIT_ENABLED` | boolean | `true` | Enable rate limiting |
| `GENERATION_RATE_LIMIT_RPM` | integer | `60` | Requests per minute |
| `GENERATION_RATE_LIMIT_RPH` | integer | `1000` | Requests per hour |
| `GENERATION_RATE_LIMIT_RPD` | integer | `10000` | Requests per day |
| `GENERATION_RATE_LIMIT_BURST` | integer | `10` | Burst request allowance |

### Caching Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `GENERATION_CACHE_ENABLED` | boolean | `true` | Enable LRU caching |
| `GENERATION_CACHE_SIZE` | integer | `1000` | Maximum cache entries |
| `GENERATION_CACHE_TTL` | integer | `3600` | Cache TTL in seconds |

### Logging Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `GENERATION_LOG_LEVEL` | string | `"INFO"` | Log level: DEBUG, INFO, WARNING, ERROR |
| `GENERATION_LOG_FILE` | string | `None` | Log file path (stdout if not set) |
| `GENERATION_LOG_FORMAT` | string | See below | Log format string |

Default log format:
```
"%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

## Detailed Configuration Sections

### Model Configuration

#### Quantization Options

The `quantization` setting controls memory usage and model quality:

```yaml
model:
  quantization: "auto"  # Recommended - auto-selects based on available RAM
  # quantization: "q8"   # High quality, higher memory usage (~18GB)
  # quantization: "q4"   # Lower quality, lower memory usage (~12GB)
  # quantization: "none" # Full precision, highest memory usage (~35GB)
```

**Quantization Guidelines:**
- **32GB+ RAM**: Use `"auto"` or `"q8"` for best quality
- **16GB RAM**: Use `"q4"` to fit in memory
- **64GB+ RAM**: Consider `"none"` for maximum quality

#### Context Length

```yaml
model:
  context_length: 8192  # Maximum tokens per request (model maximum)
```

**Context Length Guidelines:**
- **Default**: 8192 tokens (model maximum)
- **Lower values**: Reduce memory usage but limit input size
- **Cannot exceed model maximum**: 8192 for GPT-OSS-20B

#### Model Cache Directory

```yaml
model:
  cache_dir: "./models"  # Local model cache directory
  # cache_dir: "/data/models"  # Production path
```

**Cache Directory Notes:**
- Requires ~25GB free space for model files
- Should be on fast storage (SSD recommended)
- Persistent across server restarts
- Automatically created if doesn't exist

### Security Configuration

#### Authentication Modes

**Development Mode (No Authentication):**
```yaml
security:
  require_api_key: false
  cors_origins: ["*"]
```

**Production Mode (API Key Required):**
```yaml
security:
  require_api_key: true
  cors_origins: ["https://yourapp.com", "https://api.yourapp.com"]
```

#### HTTPS Configuration

```yaml
security:
  require_https: true  # Redirect HTTP to HTTPS
  cors_methods: ["GET", "POST", "OPTIONS"]
  cors_headers: ["Authorization", "Content-Type", "X-API-Key"]
```

**HTTPS Setup Notes:**
- Use reverse proxy (nginx, Apache) for SSL termination
- Set `require_https: true` in production
- Configure proper CORS origins (not `["*"]` in production)

### Rate Limiting Configuration

#### Basic Rate Limiting

```yaml
rate_limit:
  enabled: true
  requests_per_minute: 60
  requests_per_hour: 1000
  requests_per_day: 10000
  burst_size: 10
```

#### Advanced Rate Limiting

```yaml
rate_limit:
  enabled: true
  requests_per_minute: 120    # Higher limit for production
  requests_per_hour: 5000     # Hourly limit
  requests_per_day: 50000     # Daily limit
  burst_size: 20              # Allow bursts of 20 requests
  cleanup_interval: 1800      # Clean expired entries every 30 minutes
  algorithm: "token_bucket"   # Algorithm: token_bucket, sliding_window
```

**Rate Limiting Guidelines:**
- **Development**: Disable or set high limits
- **Production API**: 60-120 RPM depending on use case
- **Internal Services**: Higher limits (500+ RPM)
- **Public APIs**: Lower limits (10-60 RPM)

### Caching Configuration

#### Basic Caching

```yaml
cache:
  enabled: true
  max_size: 1000      # Cache up to 1000 prompt-response pairs
  ttl_seconds: 3600   # Cache for 1 hour
```

#### Advanced Caching

```yaml
cache:
  enabled: true
  max_size: 5000      # Larger cache for high-traffic scenarios
  ttl_seconds: 7200   # 2-hour cache duration
  cleanup_interval: 300  # Clean expired entries every 5 minutes
  hash_algorithm: "sha256"  # Hashing algorithm for cache keys
```

**Caching Guidelines:**
- **High Traffic**: Increase `max_size` to 5000+
- **Development**: Smaller cache (500-1000 entries)
- **Memory Constrained**: Reduce `max_size` and `ttl_seconds`
- **Frequent Updates**: Lower `ttl_seconds` (1800-3600)

### Batch Processing Configuration

```yaml
batch_processing:
  enabled: true
  max_batch_size: 8      # Process up to 8 requests together
  timeout_seconds: 30    # Batch timeout
  queue_size: 100        # Maximum queue size
```

**Batch Processing Guidelines:**
- **High Throughput**: Increase `max_batch_size` to 16-32
- **Low Latency**: Decrease `max_batch_size` to 2-4
- **Memory Constrained**: Reduce `max_batch_size`

### Logging Configuration

#### Development Logging

```yaml
logging:
  level: "DEBUG"
  format: "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
  handlers:
    - console
```

#### Production Logging

```yaml
logging:
  level: "INFO"
  format: "%(asctime)s [%(levelname)s] %(process)d %(name)s: %(message)s"
  file: "/var/log/tkr-embed/server.log"
  max_bytes: 10485760    # 10MB log rotation
  backup_count: 5        # Keep 5 backup files
  handlers:
    - file
    - console
```

#### Structured Logging (JSON)

```yaml
logging:
  level: "INFO"
  format: "json"
  fields:
    - timestamp
    - level
    - logger
    - message
    - request_id
    - processing_time
```

## Hardware-Specific Configurations

### Apple Silicon M1 (16GB RAM)

```yaml
model:
  quantization: "q4"     # Reduce memory usage
  context_length: 4096   # Smaller context if needed

cache:
  max_size: 500          # Smaller cache

batch_processing:
  max_batch_size: 4      # Smaller batches
```

### Apple Silicon M1 (32GB RAM) - Recommended

```yaml
model:
  quantization: "auto"   # Automatic Q8_0 selection
  context_length: 8192   # Full context length

cache:
  max_size: 1000         # Standard cache

batch_processing:
  max_batch_size: 8      # Standard batch size
```

### Apple Silicon M3 Max (64GB+ RAM)

```yaml
model:
  quantization: "q8"     # High quality quantization
  context_length: 8192   # Full context length

cache:
  max_size: 2000         # Larger cache

batch_processing:
  max_batch_size: 16     # Larger batches

rate_limit:
  requests_per_minute: 200  # Higher rate limits
```

## Environment-Specific Configurations

### Development Environment

```yaml
environment: "development"
debug: true

security:
  require_api_key: false
  require_https: false

rate_limit:
  enabled: false

logging:
  level: "DEBUG"
```

### Testing Environment

```yaml
environment: "testing"
debug: false

model:
  quantization: "q4"     # Faster loading for tests

security:
  require_api_key: true

cache:
  enabled: false         # Disable cache for consistent tests

logging:
  level: "WARNING"       # Reduce log noise
```

### Production Environment

```yaml
environment: "production"
debug: false

security:
  require_api_key: true
  require_https: true
  cors_origins: ["https://yourdomain.com"]

rate_limit:
  enabled: true
  requests_per_minute: 100

cache:
  enabled: true
  max_size: 2000

logging:
  level: "INFO"
  file: "/var/log/tkr-embed.log"
```

## Configuration Validation

### Built-in Validation

The server automatically validates configuration on startup:

- **Model path**: Checks if model exists or can be downloaded
- **Directories**: Creates cache directories if they don't exist
- **Port availability**: Validates that the port is available
- **Memory requirements**: Warns if insufficient RAM for quantization level

### Manual Validation

Check configuration without starting the server:

```bash
python -c "
from tkr_embed.config import get_config
config = get_config()
print('Configuration loaded successfully!')
print(f'Environment: {config.environment.value}')
print(f'Model: {config.model.model_path}')
print(f'Quantization: {config.model.quantization}')
"
```

## Troubleshooting Configuration Issues

### Common Issues

**1. Model fails to load:**
```yaml
# Increase timeout for slow internet connections
model:
  load_timeout: 600  # 10 minutes
```

**2. Memory errors:**
```yaml
# Reduce memory usage
model:
  quantization: "q4"

cache:
  max_size: 250

batch_processing:
  max_batch_size: 2
```

**3. Rate limiting too strict:**
```yaml
# Increase limits for internal APIs
rate_limit:
  requests_per_minute: 300
  burst_size: 50
```

**4. CORS errors in browser:**
```yaml
security:
  cors_origins: ["http://localhost:3000", "https://yourapp.com"]
  cors_methods: ["GET", "POST", "OPTIONS"]
  cors_headers: ["*"]
```

### Configuration Debugging

Enable debug mode to see detailed configuration information:

```bash
export GENERATION_DEBUG=true
python -m tkr_embed.api.server
```

This will log:
- Configuration file used
- All configuration values
- Environment variable overrides
- Model loading progress
- Memory usage statistics

### Performance Tuning

**For Maximum Throughput:**
```yaml
model:
  quantization: "q8"

cache:
  max_size: 5000
  ttl_seconds: 7200

batch_processing:
  max_batch_size: 16

rate_limit:
  requests_per_minute: 300
```

**For Minimum Latency:**
```yaml
model:
  quantization: "q4"

cache:
  max_size: 2000
  ttl_seconds: 1800

batch_processing:
  max_batch_size: 1  # Process requests immediately

rate_limit:
  burst_size: 20
```

**For Memory Efficiency:**
```yaml
model:
  quantization: "q4"
  context_length: 4096

cache:
  max_size: 250
  ttl_seconds: 1800

batch_processing:
  max_batch_size: 4
```

## Configuration Best Practices

1. **Use environment-specific configs**: Separate dev, test, and prod configurations
2. **Override with environment variables**: For sensitive values and deployment-specific settings
3. **Monitor memory usage**: Adjust quantization and cache sizes based on available RAM
4. **Set appropriate rate limits**: Based on your use case and infrastructure
5. **Enable HTTPS in production**: Always use SSL/TLS for production deployments
6. **Configure proper logging**: Use structured logging for production monitoring
7. **Test configurations**: Validate settings in staging before production deployment
8. **Document customizations**: Keep track of configuration changes for your deployment

This configuration guide provides the foundation for optimizing tkr-embed for your specific use case and infrastructure requirements.