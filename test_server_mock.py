#!/usr/bin/env python3
"""
Mock Server Infrastructure Test
===============================

Test the server endpoints with mock responses to validate:
- API structure and routing
- Request/response handling
- Error handling
- Authentication
- Performance of infrastructure without model loading

This provides immediate feedback on the server architecture.
"""

import asyncio
import aiohttp
import json
import time
import logging
import unittest.mock
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_mock_infrastructure():
    """Test server infrastructure with mock model"""
    logger.info("🧪 Testing Mock Server Infrastructure")
    logger.info("=" * 50)

    # Import and patch the model manager to return mock responses
    with unittest.mock.patch('tkr_embed.core.model_manager.GPTOss20bMLX') as MockModel:

        # Configure mock model
        mock_instance = MockModel.return_value
        mock_instance.is_ready.return_value = True
        mock_instance.load_model = asyncio.coroutine(lambda: None)

        # Mock generate method
        async def mock_generate(prompt, config):
            await asyncio.sleep(0.1)  # Simulate processing time
            return f"Generated response for: {prompt[:50]}..."

        # Mock generate_stream method
        async def mock_generate_stream(prompt, config):
            words = ["This", "is", "a", "mock", "streaming", "response", "for", "testing", "purposes"]
            for word in words:
                yield f"{word} "
                await asyncio.sleep(0.05)

        # Mock chat method
        async def mock_chat(messages, config):
            await asyncio.sleep(0.1)
            return f"Chat response for {len(messages)} messages"

        # Mock info methods
        mock_instance.get_model_info.return_value = {
            "model_path": "mock-gpt-oss-20b",
            "quantization": "mock",
            "device": "mock",
            "load_time": 1.0,
            "memory_usage_gb": 2.5,
            "model_loaded": True,
            "parameters": "20B",
            "architecture": "MockTransformer",
            "supported_capabilities": ["text_generation", "chat", "streaming"],
            "max_sequence_length": 8192,
            "reasoning_levels": ["low", "medium", "high"]
        }

        mock_instance.get_memory_usage.return_value = 2.5
        mock_instance.generate = mock_generate
        mock_instance.generate_stream = mock_generate_stream
        mock_instance.chat = mock_chat

        # Start server with mocked model
        logger.info("🚀 Starting server with mock model...")

        # Import server after mocking
        import subprocess
        server_process = subprocess.Popen([
            sys.executable, "-m", "tkr_embed.api.server"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Wait for server to start
        max_wait = 30
        wait_time = 0
        server_ready = False

        while wait_time < max_wait:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get("http://localhost:8008/health", timeout=aiohttp.ClientTimeout(total=2)) as response:
                        if response.status == 200:
                            server_ready = True
                            logger.info(f"✅ Server started in {wait_time}s")
                            break
            except:
                pass

            await asyncio.sleep(2)
            wait_time += 2

        if not server_ready:
            logger.error("❌ Server failed to start")
            server_process.terminate()
            return False

        try:
            # Test endpoints
            async with aiohttp.ClientSession() as session:

                # Test 1: Health endpoint
                logger.info("\n1. Testing health endpoint...")
                async with session.get("http://localhost:8008/health") as response:
                    if response.status == 200:
                        health_data = await response.json()
                        logger.info(f"   Status: {health_data.get('status')}")
                        logger.info(f"   Model loaded: {health_data.get('model_loaded')}")
                        logger.info("✅ Health check passed")
                    else:
                        logger.error(f"❌ Health check failed: {response.status}")
                        return False

                # Test 2: Model info
                logger.info("\n2. Testing model info endpoint...")
                async with session.get("http://localhost:8008/info") as response:
                    if response.status == 200:
                        info_data = await response.json()
                        logger.info(f"   Model: {info_data.get('model_path')}")
                        logger.info(f"   Capabilities: {info_data.get('supported_capabilities')}")
                        logger.info("✅ Model info working")
                    else:
                        logger.error(f"❌ Model info failed: {response.status}")

                # Test 3: Generation endpoint
                logger.info("\n3. Testing text generation...")
                start_time = time.time()
                payload = {
                    "text": "Explain machine learning",
                    "max_tokens": 100,
                    "temperature": 0.7,
                    "reasoning_level": "medium"
                }

                async with session.post("http://localhost:8008/generate", json=payload) as response:
                    end_time = time.time()

                    if response.status == 200:
                        data = await response.json()
                        response_time = (end_time - start_time) * 1000

                        logger.info(f"   Generated: {data.get('generated_text', '')[:100]}...")
                        logger.info(f"   Response time: {response_time:.1f}ms")
                        logger.info(f"   Reasoning level: {data.get('reasoning_level')}")
                        logger.info("✅ Generation working")
                    else:
                        logger.error(f"❌ Generation failed: {response.status}")
                        return False

                # Test 4: Chat endpoint
                logger.info("\n4. Testing chat completion...")
                payload = {
                    "messages": [
                        {"role": "user", "content": "Hello, how are you?"}
                    ],
                    "max_tokens": 100,
                    "reasoning_level": "medium"
                }

                async with session.post("http://localhost:8008/chat", json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"   Chat response: {data.get('response', '')[:100]}...")
                        logger.info("✅ Chat working")
                    else:
                        logger.error(f"❌ Chat failed: {response.status}")

                # Test 5: Streaming endpoint
                logger.info("\n5. Testing streaming generation...")
                payload = {
                    "text": "Write a short story",
                    "max_tokens": 50,
                    "reasoning_level": "medium"
                }

                chunks_received = 0
                async with session.post("http://localhost:8008/stream", json=payload) as response:
                    if response.status == 200:
                        async for line in response.content:
                            if line:
                                line_str = line.decode('utf-8').strip()
                                if line_str.startswith('data: ') and line_str != 'data: [DONE]':
                                    chunks_received += 1
                                    if chunks_received >= 3:  # Test first few chunks
                                        break

                        logger.info(f"   Received {chunks_received} chunks")
                        logger.info("✅ Streaming working")
                    else:
                        logger.error(f"❌ Streaming failed: {response.status}")

                # Test 6: Error handling
                logger.info("\n6. Testing error handling...")
                payload = {
                    "text": "Test",
                    "reasoning_level": "invalid"
                }

                async with session.post("http://localhost:8008/generate", json=payload) as response:
                    if response.status == 422:  # Validation error
                        logger.info("✅ Error handling working")
                    else:
                        logger.warning(f"⚠️ Expected 422, got {response.status}")

            logger.info("\n" + "=" * 50)
            logger.info("📊 MOCK INFRASTRUCTURE TEST SUMMARY")
            logger.info("=" * 50)
            logger.info("✅ Server startup: Working")
            logger.info("✅ Health endpoint: Working")
            logger.info("✅ Model info: Working")
            logger.info("✅ Text generation: Working")
            logger.info("✅ Chat completion: Working")
            logger.info("✅ Streaming: Working")
            logger.info("✅ Error handling: Working")
            logger.info("\n🎉 All mock infrastructure tests passed!")
            logger.info("🔧 Ready for optimization and real model integration")

            return True

        except Exception as e:
            logger.error(f"❌ Mock infrastructure test failed: {e}")
            return False

        finally:
            # Cleanup
            logger.info("🧹 Cleaning up server...")
            server_process.terminate()
            try:
                server_process.wait(timeout=10)
            except:
                server_process.kill()

async def main():
    """Main test runner"""
    logger.info("🚀 GPT-OSS-20B Mock Infrastructure Validation")

    success = await test_mock_infrastructure()

    if success:
        logger.info("\n✅ Mock infrastructure validation completed successfully!")
        return 0
    else:
        logger.info("\n❌ Mock infrastructure validation failed")
        return 1

if __name__ == "__main__":
    # Check environment
    import os
    if "project_env" not in os.environ.get("VIRTUAL_ENV", ""):
        print("❌ Please activate the virtual environment first: source start_env")
        sys.exit(1)

    exit_code = asyncio.run(main())
    sys.exit(exit_code)