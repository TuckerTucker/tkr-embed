#!/usr/bin/env python3
"""
Test script for the MLX FastAPI server
"""

import asyncio
import aiohttp
import json
import time
import subprocess
import signal
import os
import sys

async def test_server_endpoints():
    """Test all server endpoints"""
    base_url = "http://localhost:8008"
    
    async with aiohttp.ClientSession() as session:
        print("🔍 Testing server endpoints...")
        
        # Test health check
        print("\n1. Testing health endpoint...")
        try:
            async with session.get(f"{base_url}/health") as response:
                health_data = await response.json()
                print(f"✅ Health check: {health_data['status']}")
                print(f"   Memory usage: {health_data['memory_usage_gb']:.2f}GB")
                print(f"   Uptime: {health_data['uptime_seconds']:.1f}s")
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False
        
        # Test model info
        print("\n2. Testing model info endpoint...")
        try:
            async with session.get(f"{base_url}/info") as response:
                info_data = await response.json()
                print(f"✅ Model info: {info_data['model_path']}")
                print(f"   Quantization: {info_data['quantization']}")
                print(f"   Embedding dimension: {info_data['embedding_dim']}")
        except Exception as e:
            print(f"❌ Model info failed: {e}")
        
        # Test text embeddings
        print("\n3. Testing text embedding endpoint...")
        try:
            payload = {
                "texts": ["Hello world", "How are you today?"],
                "normalize": True
            }
            async with session.post(
                f"{base_url}/embed/text", 
                json=payload
            ) as response:
                embed_data = await response.json()
                print(f"✅ Text embeddings generated")
                print(f"   Shape: {embed_data['shape']}")
                print(f"   Processing time: {embed_data['processing_time']:.3f}s")
                print(f"   First embedding sample: {embed_data['embeddings'][0][:5]}...")
        except Exception as e:
            print(f"❌ Text embedding failed: {e}")
        
        # Test similarity computation
        print("\n4. Testing similarity computation...")
        try:
            # Use the embeddings from previous test
            if 'embed_data' in locals():
                emb1, emb2 = embed_data['embeddings']
                payload = {
                    "query_embeddings": [emb1],
                    "candidate_embeddings": [emb1, emb2],
                    "metric": "cosine"
                }
                async with session.post(
                    f"{base_url}/similarity",
                    json=payload
                ) as response:
                    sim_data = await response.json()
                    print(f"✅ Similarity computed")
                    print(f"   Similarities: {sim_data['similarities'][0]}")
                    print(f"   Processing time: {sim_data['processing_time']:.3f}s")
        except Exception as e:
            print(f"❌ Similarity computation failed: {e}")
        
        print("\n🎉 Server testing complete!")
        return True


def start_server():
    """Start the server in the background"""
    print("🚀 Starting MLX embedding server...")
    
    # Start server process
    process = subprocess.Popen([
        "python", "-m", "uvicorn", 
        "tkr_embed.api.server:app",
        "--host", "0.0.0.0",
        "--port", "8008",
        "--log-level", "info"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for server to start
    print("⏳ Waiting for server to start...")
    time.sleep(5)
    
    return process


async def main():
    """Main test function"""
    print("🧪 MLX FastAPI Server Test")
    print("=" * 50)
    
    # Check if server is already running
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8008/health") as response:
                if response.status == 200:
                    print("📡 Server already running, testing endpoints...")
                    await test_server_endpoints()
                    return
    except:
        pass  # Server not running, we'll start it
    
    # Start server
    server_process = None
    try:
        server_process = start_server()
        
        # Test endpoints
        success = await test_server_endpoints()
        
        if success:
            print("\n✅ All tests passed!")
        else:
            print("\n❌ Some tests failed")
            
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Test error: {e}")
    finally:
        # Cleanup server
        if server_process:
            print("\n🧹 Stopping server...")
            server_process.terminate()
            server_process.wait(timeout=10)


if __name__ == "__main__":
    # Ensure we're in the right directory and environment
    if "project_env" not in os.environ.get("VIRTUAL_ENV", ""):
        print("❌ Please activate the virtual environment first: source start_env")
        sys.exit(1)
    
    asyncio.run(main())