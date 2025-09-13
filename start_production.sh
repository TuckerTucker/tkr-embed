#!/bin/bash
# tkr-embed | MLX Multimodal Embedding Server - Production Startup Script

set -e  # Exit on any error

echo "🚀 Starting tkr-embed | MLX Multimodal Embedding Server (Production Mode)"
echo "============================================================="

# Check if virtual environment exists
if [ ! -d "tkr_env" ]; then
    echo "❌ Virtual environment not found. Run setup first."
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source start_env

# Set production environment variables
export EMBEDDING_ENV=production
export EMBEDDING_DEBUG=false
export EMBEDDING_LOG_LEVEL=INFO

# Set API key if not already set
if [ -z "$EMBEDDING_API_KEY" ]; then
    echo "⚠️  WARNING: EMBEDDING_API_KEY not set in environment"
    echo "   A development key will be generated automatically"
    echo "   For production, set EMBEDDING_API_KEY environment variable"
fi

# Check required dependencies
echo "🔍 Checking dependencies..."
python -c "
import sys
required_packages = ['fastapi', 'uvicorn', 'transformers', 'torch', 'mlx']
missing = []

for package in required_packages:
    try:
        __import__(package)
        print(f'   ✅ {package}')
    except ImportError:
        missing.append(package)
        print(f'   ❌ {package} (missing)')

if missing:
    print(f'\\n❌ Missing required packages: {missing}')
    print('   Run: pip install -r requirements.txt')
    sys.exit(1)
else:
    print('   ✅ All dependencies available')
"

# Check available space for model download
echo "💾 Checking disk space..."
python -c "
import shutil
import os

# Check space in home directory (for model cache)
home_free = shutil.disk_usage(os.path.expanduser('~')).free
free_gb = home_free / (1024**3)

print(f'   Free space: {free_gb:.1f} GB')

if free_gb < 20:
    print('   ⚠️  Warning: Low disk space. Model requires ~15GB')
    print('   Consider freeing up space before starting')
else:
    print('   ✅ Sufficient disk space available')
"

# Check configuration
echo "⚙️  Loading configuration..."
if [ -f "config.yaml" ]; then
    echo "   ✅ Using config.yaml"
else
    echo "   ℹ️  No config.yaml found, using defaults"
fi

# Start the server
echo ""
echo "🌟 Starting server..."
echo "   Server will be available at: http://localhost:8008"
echo "   API documentation: http://localhost:8008/docs"
echo "   Health check: http://localhost:8008/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo "============================================================="

# Start with uvicorn
exec uvicorn tkr_embed.api.server:app \
    --host 0.0.0.0 \
    --port 8008 \
    --workers 1 \
    --access-log \
    --log-level info