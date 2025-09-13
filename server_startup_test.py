#!/usr/bin/env python3
"""
Server Startup Test for Integration Testing
Tests server initialization and basic endpoint availability
"""

import asyncio
import aiohttp
import time
import logging
import subprocess
import signal
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ServerStartupTest")

class ServerTester:
    def __init__(self):
        self.server_process = None
        self.base_url = "http://localhost:8008"

    async def start_server(self):
        """Start the server in background"""
        logger.info("Starting server...")

        env = os.environ.copy()
        env["GENERATION_ENV"] = "testing"  # Use testing environment
        env["GENERATION_DEBUG"] = "true"

        self.server_process = subprocess.Popen(
            ["python", "-m", "tkr_embed.api.server"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Give server time to start
        await asyncio.sleep(3)

        if self.server_process.poll() is not None:
            # Server exited
            stdout, stderr = self.server_process.communicate()
            logger.error(f"Server failed to start: {stderr.decode()}")
            return False

        logger.info("Server started successfully")
        return True

    async def test_health_endpoint(self):
        """Test the health endpoint"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/health") as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"Health check passed: {data}")
                        return True
                    else:
                        logger.error(f"Health check failed with status {response.status}")
                        return False
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def test_info_endpoint(self):
        """Test the info endpoint"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/info") as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"Info endpoint passed: {data}")
                        return True
                    else:
                        logger.error(f"Info endpoint failed with status {response.status}")
                        return False
        except Exception as e:
            logger.error(f"Info endpoint failed: {e}")
            return False

    async def test_docs_endpoint(self):
        """Test the docs endpoint"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/docs") as response:
                    if response.status == 200:
                        logger.info("Docs endpoint accessible")
                        return True
                    else:
                        logger.error(f"Docs endpoint failed with status {response.status}")
                        return False
        except Exception as e:
            logger.error(f"Docs endpoint failed: {e}")
            return False

    def stop_server(self):
        """Stop the server"""
        if self.server_process:
            logger.info("Stopping server...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
                self.server_process.wait()
            logger.info("Server stopped")

    async def run_server_tests(self):
        """Run all server tests"""
        logger.info("=" * 60)
        logger.info("STARTING SERVER INTEGRATION TESTS")
        logger.info("=" * 60)

        results = {}

        try:
            # Start server
            server_started = await self.start_server()
            results["server_startup"] = server_started

            if not server_started:
                logger.error("Server failed to start - aborting tests")
                return results

            # Give server more time to initialize
            await asyncio.sleep(2)

            # Test endpoints
            results["health_endpoint"] = await self.test_health_endpoint()
            results["info_endpoint"] = await self.test_info_endpoint()
            results["docs_endpoint"] = await self.test_docs_endpoint()

        except Exception as e:
            logger.error(f"Server test failed: {e}")
            results["error"] = str(e)

        finally:
            self.stop_server()

        # Report results
        logger.info("=" * 60)
        logger.info("SERVER INTEGRATION TEST RESULTS")
        logger.info("=" * 60)

        passed = 0
        failed = 0

        for test_name, result in results.items():
            if test_name != "error":
                status = "✅ PASS" if result else "❌ FAIL"
                logger.info(f"{test_name:<20}: {status}")
                if result:
                    passed += 1
                else:
                    failed += 1

        if "error" in results:
            logger.error(f"Test suite error: {results['error']}")

        success_rate = (passed / (passed + failed) * 100) if (passed + failed) > 0 else 0
        logger.info(f"\nPassed: {passed}, Failed: {failed}")
        logger.info(f"Success Rate: {success_rate:.1f}%")

        overall_status = "✅ SERVER READY" if success_rate >= 75 else "❌ SERVER ISSUES"
        logger.info(f"Overall Status: {overall_status}")

        return results

async def main():
    tester = ServerTester()
    await tester.run_server_tests()

if __name__ == "__main__":
    asyncio.run(main())