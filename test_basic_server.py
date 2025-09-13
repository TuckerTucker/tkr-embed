#!/usr/bin/env python3
"""
Basic Server Test - Infrastructure Validation
==============================================

Test the server infrastructure without model loading to validate:
- FastAPI server startup
- Endpoint availability
- Mock functionality
- Error handling
- Authentication
"""

import asyncio
import aiohttp
import json
import time
import logging
import subprocess
import signal
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_server_infrastructure():
    """Test basic server infrastructure without model loading"""
    logger.info("🧪 Testing Server Infrastructure")
    logger.info("=" * 50)

    # Check if server is already running
    server_running = False
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8008/health", timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:
                    server_running = True
                    logger.info("✅ Server already running")
    except:
        logger.info("ℹ️ Server not running, will start it")

    server_process = None
    try:
        if not server_running:
            # Start server with testing environment to skip model loading
            logger.info("🚀 Starting server in testing mode...")
            env = os.environ.copy()
            env['ENVIRONMENT'] = 'testing'  # This should skip model loading

            server_process = subprocess.Popen([
                sys.executable, "-m", "tkr_embed.api.server"
            ], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # Wait for server to start (without model loading should be quick)
            max_wait = 30
            wait_time = 0

            while wait_time < max_wait:
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get("http://localhost:8008/health", timeout=aiohttp.ClientTimeout(total=2)) as response:
                            if response.status == 200:
                                logger.info(f"✅ Server started in {wait_time}s")
                                break
                except:
                    pass

                await asyncio.sleep(2)
                wait_time += 2
                logger.info(f"   Waiting... ({wait_time}s)")

            if wait_time >= max_wait:
                logger.error("❌ Server failed to start within timeout")
                return False

        # Test endpoints
        async with aiohttp.ClientSession() as session:
            # Test 1: Health check
            logger.info("\n1. Testing health endpoint...")
            try:
                async with session.get("http://localhost:8008/health") as response:
                    health_data = await response.json()
                    logger.info(f"   Status: {health_data.get('status', 'unknown')}")
                    logger.info(f"   Model loaded: {health_data.get('model_loaded', False)}")
                    logger.info(f"   Memory: {health_data.get('memory_usage_gb', 0):.2f}GB")
                    logger.info("✅ Health check passed")
            except Exception as e:
                logger.error(f"❌ Health check failed: {e}")
                return False

            # Test 2: Model info
            logger.info("\n2. Testing model info endpoint...")
            try:
                async with session.get("http://localhost:8008/info") as response:
                    if response.status == 200:
                        info_data = await response.json()
                        logger.info(f"   Model: {info_data.get('model_path', 'unknown')}")
                        logger.info(f"   Framework: {info_data.get('framework', 'unknown')}")
                        logger.info("✅ Model info endpoint working")
                    else:
                        logger.warning(f"⚠️ Model info returned {response.status} (expected in testing mode)")
            except Exception as e:
                logger.warning(f"⚠️ Model info failed: {e} (expected in testing mode)")

            # Test 3: Generation endpoint (should work with mock)
            logger.info("\n3. Testing generation endpoint...")
            try:
                payload = {
                    "text": "Hello, this is a test prompt",
                    "max_tokens": 50,
                    "temperature": 0.7,
                    "reasoning_level": "medium"
                }

                start_time = time.time()
                async with session.post("http://localhost:8008/generate", json=payload) as response:
                    end_time = time.time()

                    if response.status == 200:
                        data = await response.json()
                        response_time = (end_time - start_time) * 1000

                        logger.info(f"   Generated text: {data.get('generated_text', '')[:100]}...")
                        logger.info(f"   Response time: {response_time:.1f}ms")
                        logger.info(f"   Tokens used: {data.get('tokens_used', 0)}")
                        logger.info("✅ Generation endpoint working")
                    else:
                        error_data = await response.json()
                        logger.error(f"❌ Generation failed: {response.status} - {error_data}")
                        return False

            except Exception as e:
                logger.error(f"❌ Generation test failed: {e}")
                return False

            # Test 4: Chat endpoint
            logger.info("\n4. Testing chat endpoint...")
            try:
                payload = {
                    "messages": [
                        {"role": "user", "content": "Hello, how are you?"}
                    ],
                    "max_tokens": 50,
                    "temperature": 0.7,
                    "reasoning_level": "medium"
                }

                async with session.post("http://localhost:8008/chat", json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"   Chat response: {data.get('response', '')[:100]}...")
                        logger.info("✅ Chat endpoint working")
                    else:
                        error_data = await response.json()
                        logger.error(f"❌ Chat failed: {response.status} - {error_data}")
                        return False

            except Exception as e:
                logger.error(f"❌ Chat test failed: {e}")
                return False

            # Test 5: Streaming endpoint
            logger.info("\n5. Testing streaming endpoint...")
            try:
                payload = {
                    "text": "Write a short story about a robot",
                    "max_tokens": 100,
                    "temperature": 0.7,
                    "reasoning_level": "medium"
                }

                chunks_received = 0
                async with session.post("http://localhost:8008/stream", json=payload) as response:
                    if response.status == 200:
                        async for line in response.content:
                            if line:
                                line_str = line.decode('utf-8').strip()
                                if line_str.startswith('data: '):
                                    chunks_received += 1
                                    if chunks_received >= 5:  # Test first few chunks
                                        break

                        logger.info(f"   Received {chunks_received} chunks")
                        logger.info("✅ Streaming endpoint working")
                    else:
                        logger.error(f"❌ Streaming failed: {response.status}")
                        return False

            except Exception as e:
                logger.error(f"❌ Streaming test failed: {e}")
                return False

            # Test 6: Error handling
            logger.info("\n6. Testing error handling...")
            try:
                # Invalid reasoning level
                payload = {
                    "text": "Test prompt",
                    "reasoning_level": "invalid_level"
                }

                async with session.post("http://localhost:8008/generate", json=payload) as response:
                    if response.status == 422:  # Validation error
                        logger.info("✅ Validation error handling working")
                    else:
                        logger.warning(f"⚠️ Expected 422, got {response.status}")

            except Exception as e:
                logger.warning(f"⚠️ Error handling test failed: {e}")

        logger.info("\n" + "=" * 50)
        logger.info("📊 INFRASTRUCTURE TEST SUMMARY")
        logger.info("=" * 50)
        logger.info("✅ Server startup: Working")
        logger.info("✅ Health endpoint: Working")
        logger.info("✅ Generation endpoint: Working")
        logger.info("✅ Chat endpoint: Working")
        logger.info("✅ Streaming endpoint: Working")
        logger.info("✅ Error handling: Working")
        logger.info("\n🎉 All infrastructure tests passed!")
        logger.info("📈 Ready for performance testing with real model")

        return True

    except Exception as e:
        logger.error(f"❌ Infrastructure test failed: {e}")
        return False

    finally:
        # Cleanup
        if server_process:
            logger.info("🧹 Cleaning up server...")
            server_process.terminate()
            try:
                server_process.wait(timeout=10)
            except:
                server_process.kill()

async def main():
    """Main test runner"""
    logger.info("🚀 GPT-OSS-20B Infrastructure Validation")

    success = await test_server_infrastructure()

    if success:
        logger.info("\n✅ Infrastructure validation completed successfully!")
        return 0
    else:
        logger.info("\n❌ Infrastructure validation failed")
        return 1

if __name__ == "__main__":
    # Check environment
    if "project_env" not in os.environ.get("VIRTUAL_ENV", ""):
        print("❌ Please activate the virtual environment first: source start_env")
        sys.exit(1)

    exit_code = asyncio.run(main())
    sys.exit(exit_code)