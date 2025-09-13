# Deployment Guide for tkr-embed

This comprehensive guide covers deploying the tkr-embed GPT-OSS-20B text generation server across different environments and hardware configurations.

## Deployment Overview

tkr-embed supports multiple deployment scenarios:

- **Development**: Local development with minimal configuration
- **Staging**: Testing environment with production-like settings
- **Production**: High-availability production deployment
- **Edge**: Single-instance edge deployments

## Hardware Requirements

### Minimum Requirements

**For Development/Testing:**
- Apple Silicon M1 with 16GB RAM
- 25GB free disk space
- Python 3.9+
- Internet connection for model download

**Performance Characteristics:**
- Model loading: ~15-20 seconds
- Memory usage: ~12GB (Q4_0 quantization)
- Throughput: ~100 tokens/second
- Concurrent requests: 10-20

### Recommended Requirements

**For Production Deployment:**
- Apple Silicon M2/M3 with 32GB RAM
- 50GB free SSD space (NVMe preferred)
- Python 3.10+
- Stable internet connection

**Performance Characteristics:**
- Model loading: ~10-15 seconds
- Memory usage: ~18GB (Q8_0 quantization)
- Throughput: 150+ tokens/second
- Concurrent requests: 50-100

### High-Performance Requirements

**For High-Throughput Production:**
- Apple Silicon M3 Max/Ultra with 64GB+ RAM
- 100GB+ NVMe SSD storage
- Python 3.11+
- Dedicated network interface

**Performance Characteristics:**
- Model loading: ~8-12 seconds
- Memory usage: ~25GB (optimized settings)
- Throughput: 200+ tokens/second
- Concurrent requests: 100-200

## Quick Deployment

### 1. Development Deployment

```bash
# Clone and setup
git clone <repository-url> tkr-embed
cd tkr-embed

# Setup environment
source start_env
pip install -r requirements.txt

# Start in development mode (no authentication)
python -m tkr_embed.api.server
```

**Access:**
- API: http://localhost:8008
- Documentation: http://localhost:8008/docs
- Health Check: http://localhost:8008/health

### 2. Production Quick Start

```bash
# Clone and setup
git clone <repository-url> tkr-embed
cd tkr-embed

# Setup environment
source start_env
pip install -r requirements.txt

# Set production API key
export GENERATION_API_KEY="your-secure-api-key-here"

# Start in production mode
GENERATION_ENV=production python -m tkr_embed.api.server
```

## Detailed Deployment Scenarios

### Development Environment

**Purpose**: Local development and testing

**Configuration (config.dev.yaml):**
```yaml
environment: "development"
debug: true

server:
  host: "127.0.0.1"
  port: 8008

model:
  quantization: "auto"
  cache_dir: "./models"

security:
  require_api_key: false
  cors_origins: ["*"]

rate_limit:
  enabled: false

logging:
  level: "DEBUG"
```

**Deployment Steps:**
```bash
# 1. Setup environment
source start_env
pip install -r requirements.txt

# 2. Verify configuration
ls config.dev.yaml  # Should exist

# 3. Start server
python -m tkr_embed.api.server

# 4. Verify deployment
curl http://localhost:8008/health
curl -X POST http://localhost:8008/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "max_tokens": 50}'
```

### Staging Environment

**Purpose**: Production-like testing environment

**Configuration (config.staging.yaml):**
```yaml
environment: "production"
debug: false

server:
  host: "0.0.0.0"
  port: 8008

model:
  quantization: "q8"
  cache_dir: "/data/models"

security:
  require_api_key: true
  cors_origins: ["https://staging.yourapp.com"]

rate_limit:
  enabled: true
  requests_per_minute: 120

cache:
  max_size: 1000

logging:
  level: "INFO"
  file: "/var/log/tkr-embed-staging.log"
```

**Deployment Steps:**
```bash
# 1. Create required directories
sudo mkdir -p /data/models /var/log
sudo chown $(whoami) /data/models /var/log

# 2. Setup environment
source start_env
pip install -r requirements.txt

# 3. Set API key
export GENERATION_API_KEY="staging-api-key-here"

# 4. Start with staging config
GENERATION_CONFIG_FILE="config.staging.yaml" \
  python -m tkr_embed.api.server

# 5. Verify deployment
curl http://localhost:8008/health
curl -H "X-API-Key: staging-api-key-here" \
  -X POST http://localhost:8008/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "Test deployment", "max_tokens": 50}'
```

### Production Environment

**Purpose**: High-availability production deployment

**Configuration (config.production.yaml):**
```yaml
environment: "production"
debug: false

server:
  host: "0.0.0.0"
  port: 8008

model:
  model_path: "microsoft/gpt-oss-20b"
  quantization: "auto"
  cache_dir: "/opt/tkr-embed/models"
  load_timeout: 300

security:
  require_api_key: true
  require_https: true
  cors_origins: ["https://yourapp.com"]
  cors_methods: ["GET", "POST"]

rate_limit:
  enabled: true
  requests_per_minute: 100
  requests_per_hour: 5000
  requests_per_day: 50000

cache:
  enabled: true
  max_size: 2000
  ttl_seconds: 3600

batch_processing:
  enabled: true
  max_batch_size: 8

logging:
  level: "INFO"
  file: "/var/log/tkr-embed/server.log"
  max_bytes: 10485760
  backup_count: 5
```

**Deployment Steps:**
```bash
# 1. Create required directories and user
sudo useradd -r -s /bin/false tkr-embed
sudo mkdir -p /opt/tkr-embed/{models,logs}
sudo mkdir -p /var/log/tkr-embed
sudo chown -R tkr-embed:tkr-embed /opt/tkr-embed /var/log/tkr-embed

# 2. Deploy application
sudo -u tkr-embed git clone <repository-url> /opt/tkr-embed/app
cd /opt/tkr-embed/app

# 3. Setup Python environment
sudo -u tkr-embed python -m venv /opt/tkr-embed/venv
sudo -u tkr-embed /opt/tkr-embed/venv/bin/pip install -r requirements.txt

# 4. Configure environment
sudo tee /opt/tkr-embed/env > /dev/null <<EOF
GENERATION_ENV=production
GENERATION_API_KEY=your-production-api-key-here
GENERATION_CONFIG_FILE=/opt/tkr-embed/app/config.production.yaml
EOF

# 5. Create systemd service
sudo tee /etc/systemd/system/tkr-embed.service > /dev/null <<EOF
[Unit]
Description=tkr-embed GPT-OSS-20B Text Generation Server
After=network.target

[Service]
Type=simple
User=tkr-embed
Group=tkr-embed
WorkingDirectory=/opt/tkr-embed/app
EnvironmentFile=/opt/tkr-embed/env
ExecStart=/opt/tkr-embed/venv/bin/python -m tkr_embed.api.server
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# 6. Start and enable service
sudo systemctl daemon-reload
sudo systemctl enable tkr-embed
sudo systemctl start tkr-embed

# 7. Verify deployment
sudo systemctl status tkr-embed
curl http://localhost:8008/health
```

## Reverse Proxy Configuration

### nginx Configuration

**For SSL termination and load balancing:**

```nginx
upstream tkr_embed_backend {
    server 127.0.0.1:8008;
    # Add more servers for load balancing
    # server 127.0.0.1:8001;
    # server 127.0.0.1:8002;
}

server {
    listen 80;
    server_name api.yourapp.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.yourapp.com;

    # SSL configuration
    ssl_certificate /path/to/ssl/certificate.crt;
    ssl_certificate_key /path/to/ssl/private.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;

    # Logging
    access_log /var/log/nginx/tkr-embed-access.log;
    error_log /var/log/nginx/tkr-embed-error.log;

    location / {
        proxy_pass http://tkr_embed_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts for generation requests
        proxy_connect_timeout 5s;
        proxy_send_timeout 60s;
        proxy_read_timeout 120s;

        # Buffer sizes
        proxy_buffering off;
        proxy_request_buffering off;
    }

    # Special handling for streaming endpoints
    location /stream {
        proxy_pass http://tkr_embed_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # SSE-specific configuration
        proxy_buffering off;
        proxy_cache off;
        proxy_set_header Connection '';
        proxy_http_version 1.1;
        chunked_transfer_encoding off;

        # Long timeout for streaming
        proxy_read_timeout 300s;
    }

    # Health check endpoint
    location /health {
        proxy_pass http://tkr_embed_backend;
        access_log off;
    }
}
```

### Apache Configuration

```apache
<VirtualHost *:80>
    ServerName api.yourapp.com
    Redirect permanent / https://api.yourapp.com/
</VirtualHost>

<VirtualHost *:443>
    ServerName api.yourapp.com

    # SSL Configuration
    SSLEngine on
    SSLCertificateFile /path/to/ssl/certificate.crt
    SSLCertificateKeyFile /path/to/ssl/private.key

    # Logging
    CustomLog /var/log/apache2/tkr-embed-access.log combined
    ErrorLog /var/log/apache2/tkr-embed-error.log

    # Proxy configuration
    ProxyPreserveHost On
    ProxyRequests Off

    # Main proxy
    ProxyPass / http://127.0.0.1:8008/
    ProxyPassReverse / http://127.0.0.1:8008/

    # Streaming endpoint configuration
    ProxyPass /stream http://127.0.0.1:8008/stream
    ProxyPassReverse /stream http://127.0.0.1:8008/stream

    # Headers for streaming
    <Location "/stream">
        ProxyPassReverse /
        SetEnv proxy-nokeepalive 1
        SetEnv proxy-initial-not-pooled 1
    </Location>
</VirtualHost>
```

## Load Balancing Strategies

### Single Instance Deployment

**Best for**: Development, small-scale production

```yaml
# config.yaml
server:
  host: "0.0.0.0"
  port: 8008

# nginx upstream
upstream tkr_embed_backend {
    server 127.0.0.1:8008;
}
```

### Multi-Instance Deployment

**Best for**: High-availability production

```bash
# Start multiple instances on different ports
GENERATION_PORT=8008 python -m tkr_embed.api.server &
GENERATION_PORT=8001 python -m tkr_embed.api.server &
GENERATION_PORT=8002 python -m tkr_embed.api.server &
```

```nginx
# nginx load balancing
upstream tkr_embed_backend {
    least_conn;  # Use least connections algorithm
    server 127.0.0.1:8008 weight=1 max_fails=3 fail_timeout=30s;
    server 127.0.0.1:8001 weight=1 max_fails=3 fail_timeout=30s;
    server 127.0.0.1:8002 weight=1 max_fails=3 fail_timeout=30s;
}
```

### Health Check Configuration

```nginx
# Health check location
location /health {
    proxy_pass http://tkr_embed_backend;
    proxy_connect_timeout 2s;
    proxy_send_timeout 2s;
    proxy_read_timeout 2s;
    access_log off;
}

# Upstream health checks (nginx plus)
upstream tkr_embed_backend {
    server 127.0.0.1:8008;
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;

    health_check uri=/health interval=30s
                 fails=3 passes=2;
}
```

## Monitoring and Logging

### System Monitoring

**CPU and Memory Monitoring:**
```bash
# Install monitoring tools
sudo apt install htop iotop nethogs

# Monitor model loading and memory usage
htop -p $(pgrep -f tkr_embed)

# Check memory usage
ps aux | grep tkr_embed
cat /proc/$(pgrep -f tkr_embed)/status | grep VmRSS
```

**Disk Space Monitoring:**
```bash
# Monitor model cache directory
df -h /opt/tkr-embed/models

# Monitor log files
du -sh /var/log/tkr-embed/
```

### Application Logging

**Centralized Logging with rsyslog:**
```bash
# Configure rsyslog
sudo tee /etc/rsyslog.d/tkr-embed.conf > /dev/null <<EOF
# tkr-embed logging
local0.*    /var/log/tkr-embed/server.log
& stop
EOF

sudo systemctl restart rsyslog
```

**Log Rotation:**
```bash
# Configure logrotate
sudo tee /etc/logrotate.d/tkr-embed > /dev/null <<EOF
/var/log/tkr-embed/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 tkr-embed tkr-embed
    postrotate
        systemctl reload tkr-embed
    endscript
}
EOF
```

### Performance Metrics

**Custom Metrics Collection:**
```python
# Custom metrics endpoint (add to server.py)
@app.get("/metrics")
async def get_metrics():
    """Prometheus-compatible metrics endpoint"""
    metrics = {
        "generation_requests_total": request_counter,
        "generation_duration_seconds": response_times,
        "model_memory_usage_bytes": get_memory_usage(),
        "cache_hit_rate": cache_hit_rate,
        "active_connections": active_connections
    }
    return metrics
```

**Integration with Prometheus:**
```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'tkr-embed'
    static_configs:
      - targets: ['localhost:8008']
    metrics_path: '/metrics'
    scrape_interval: 30s
```

## Security Hardening

### Network Security

**Firewall Configuration (UFW):**
```bash
# Basic firewall setup
sudo ufw enable
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow SSH
sudo ufw allow ssh

# Allow HTTP/HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Deny direct access to application port
sudo ufw deny 8008/tcp
```

**API Key Security:**
```bash
# Generate secure API keys
openssl rand -base64 32

# Store in secure location
sudo mkdir -p /etc/tkr-embed
sudo chmod 700 /etc/tkr-embed
echo "GENERATION_API_KEY=your-secure-key" | sudo tee /etc/tkr-embed/secrets.env
sudo chmod 600 /etc/tkr-embed/secrets.env
```

### Application Security

**Rate Limiting Configuration:**
```yaml
rate_limit:
  enabled: true
  requests_per_minute: 60
  requests_per_hour: 1000
  requests_per_day: 10000
  burst_size: 10

  # Advanced security
  block_on_exceed: true
  block_duration: 3600  # 1 hour block
```

**CORS Security:**
```yaml
security:
  cors_origins:
    - "https://yourapp.com"
    - "https://api.yourapp.com"
  cors_methods: ["GET", "POST"]
  cors_headers: ["Authorization", "Content-Type", "X-API-Key"]
  cors_credentials: false
```

## Backup and Recovery

### Model Cache Backup

```bash
# Backup model cache
sudo tar -czf /backup/tkr-embed-models-$(date +%Y%m%d).tar.gz \
  /opt/tkr-embed/models/

# Automated backup script
sudo tee /opt/tkr-embed/backup.sh > /dev/null <<'EOF'
#!/bin/bash
BACKUP_DIR="/backup/tkr-embed"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p "$BACKUP_DIR"

# Backup models
tar -czf "$BACKUP_DIR/models-$DATE.tar.gz" /opt/tkr-embed/models/

# Backup configuration
tar -czf "$BACKUP_DIR/config-$DATE.tar.gz" /opt/tkr-embed/app/*.yaml

# Clean old backups (keep 30 days)
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +30 -delete
EOF

chmod +x /opt/tkr-embed/backup.sh

# Schedule daily backup
echo "0 2 * * * /opt/tkr-embed/backup.sh" | sudo crontab -u tkr-embed -
```

### Configuration Backup

```bash
# Version control for configurations
cd /opt/tkr-embed
git init
git add *.yaml
git commit -m "Initial configuration"

# Automated config backup
echo "0 6 * * * cd /opt/tkr-embed && git add -A && git commit -m 'Auto backup $(date)'" | \
  sudo crontab -u tkr-embed -
```

### Disaster Recovery

**Recovery Procedure:**
```bash
# 1. Restore application
git clone <repository-url> /opt/tkr-embed/app-new
cd /opt/tkr-embed/app-new

# 2. Restore environment
source start_env
pip install -r requirements.txt

# 3. Restore models
tar -xzf /backup/tkr-embed-models-latest.tar.gz -C /

# 4. Restore configuration
tar -xzf /backup/tkr-embed-config-latest.tar.gz -C /

# 5. Test deployment
python -c "from tkr_embed.config import get_config; print('Config OK')"
python -c "import tkr_embed.api.server; print('Server OK')"

# 6. Start service
sudo systemctl stop tkr-embed
sudo mv /opt/tkr-embed/app /opt/tkr-embed/app-old
sudo mv /opt/tkr-embed/app-new /opt/tkr-embed/app
sudo systemctl start tkr-embed
```

## Troubleshooting Deployment

### Common Issues

**1. Model Download Fails:**
```bash
# Check disk space
df -h

# Check internet connectivity
curl -I https://huggingface.co

# Manual model download
python -c "
from transformers import AutoModel
model = AutoModel.from_pretrained('microsoft/gpt-oss-20b')
"
```

**2. Memory Issues:**
```bash
# Check available memory
free -h

# Reduce memory usage
export GENERATION_QUANTIZATION=q4
export GENERATION_CACHE_SIZE=250
```

**3. Permission Issues:**
```bash
# Fix file permissions
sudo chown -R tkr-embed:tkr-embed /opt/tkr-embed
sudo chmod 755 /opt/tkr-embed/app
sudo chmod 644 /opt/tkr-embed/app/*.yaml
```

**4. Port Conflicts:**
```bash
# Check port usage
sudo netstat -tlnp | grep :8008

# Change port
export GENERATION_PORT=8001
```

**5. SSL/TLS Issues:**
```bash
# Test SSL configuration
openssl s_client -connect api.yourapp.com:443

# Check certificate validity
openssl x509 -in /path/to/certificate.crt -text -noout
```

### Performance Troubleshooting

**Slow Response Times:**
1. Check model quantization level
2. Monitor memory usage and swap
3. Verify cache hit rates
4. Check disk I/O for model files
5. Monitor network latency

**High Memory Usage:**
1. Reduce quantization level (q8 → q4)
2. Decrease cache size
3. Reduce batch size
4. Monitor for memory leaks

**Rate Limiting Issues:**
1. Check rate limit configuration
2. Monitor API usage patterns
3. Implement client-side retry logic
4. Consider increasing limits

This deployment guide provides the foundation for running tkr-embed reliably in production environments across different hardware configurations and scaling requirements.