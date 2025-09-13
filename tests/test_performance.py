#!/usr/bin/env python3
"""
Comprehensive Performance Testing Suite for GPT-OSS-20B Generation System
=========================================================================

Tests all aspects of the text generation system performance including:
- Server startup and model loading times
- Generation throughput and latency
- Reasoning level performance differences
- Memory usage and optimization
- Concurrent request handling
- Streaming performance

Performance Targets:
- Generation Speed: 150+ tokens/second
- Latency: <100ms per request
- Memory Usage: <50% of 32GB system
- Concurrency: 100+ simultaneous requests
"""

import asyncio
import aiohttp
import json
import time
import statistics
import psutil
import logging
import subprocess
import signal
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import pytest

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PerformanceTest")

@dataclass
class PerformanceMetrics:
    """Container for performance test results"""
    test_name: str
    tokens_per_second: float
    latency_ms: float
    memory_usage_gb: float
    concurrent_requests: int
    success_rate: float
    reasoning_level: Optional[str] = None
    error_details: Optional[str] = None

class PerformanceTestSuite:
    """Comprehensive performance testing suite"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.test_results: List[PerformanceMetrics] = []
        self.server_process: Optional[subprocess.Popen] = None
        self.api_key = None

        # Performance targets
        self.targets = {
            "tokens_per_second": 150,
            "latency_ms": 100,
            "memory_usage_percent": 50,
            "concurrent_requests": 100,
            "success_rate": 95.0
        }

    async def setup_test_environment(self):
        """Setup test environment and start server if needed"""
        logger.info("Setting up performance test environment...")

        # Check if server is already running
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/health") as response:
                    if response.status == 200:
                        logger.info("Server already running")
                        return True
        except:
            pass

        # Start server
        logger.info("Starting GPT-OSS-20B server...")
        self.server_process = subprocess.Popen([
            "python", "-m", "tkr_embed.api.server",
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Wait for server to be ready (with model loading)
        max_wait = 300  # 5 minutes for model loading
        wait_time = 0

        while wait_time < max_wait:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.base_url}/health") as response:
                        if response.status == 200:
                            health_data = await response.json()
                            if health_data.get("model_loaded", False):
                                logger.info(f"Server ready after {wait_time}s")
                                return True
            except:
                pass

            await asyncio.sleep(5)
            wait_time += 5
            logger.info(f"Waiting for server... ({wait_time}s)")

        raise TimeoutError("Server failed to start within timeout")

    async def test_server_startup_performance(self) -> PerformanceMetrics:
        """Test server startup and model loading performance"""
        logger.info("Testing server startup performance...")

        startup_start = time.time()

        # If server is already running, restart it to measure startup
        if self.server_process:
            self.server_process.terminate()
            self.server_process.wait()

        # Start fresh server
        start_time = time.time()
        self.server_process = subprocess.Popen([
            "python", "-m", "tkr_embed.api.server",
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Wait for model to be loaded
        model_loaded = False
        while time.time() - start_time < 300:  # 5 minute timeout
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.base_url}/health") as response:
                        if response.status == 200:
                            health_data = await response.json()
                            if health_data.get("model_loaded", False):
                                model_loaded = True
                                break
            except:
                pass
            await asyncio.sleep(2)

        startup_time = time.time() - start_time

        # Get memory usage after startup
        memory_usage = self._get_memory_usage()

        result = PerformanceMetrics(
            test_name="server_startup",
            tokens_per_second=0,  # Not applicable
            latency_ms=startup_time * 1000,
            memory_usage_gb=memory_usage,
            concurrent_requests=0,
            success_rate=100.0 if model_loaded else 0.0,
            error_details=None if model_loaded else "Model failed to load"
        )

        self.test_results.append(result)
        logger.info(f"Startup test: {startup_time:.1f}s, Memory: {memory_usage:.1f}GB")
        return result

    async def test_generation_performance(self, reasoning_level: str = "medium") -> PerformanceMetrics:
        """Test text generation performance for a specific reasoning level"""
        logger.info(f"Testing generation performance (reasoning: {reasoning_level})")

        test_prompts = [
            "Explain the concept of machine learning in simple terms.",
            "Write a short story about a robot learning to paint.",
            "Describe the benefits and challenges of renewable energy.",
            "Compare different programming languages for web development.",
            "Explain quantum computing to a high school student."
        ]

        latencies = []
        token_rates = []
        memory_usages = []
        successful_requests = 0

        async with aiohttp.ClientSession() as session:
            for prompt in test_prompts:
                try:
                    start_time = time.time()

                    payload = {
                        "text": prompt,
                        "max_tokens": 512,
                        "temperature": 0.7,
                        "reasoning_level": reasoning_level
                    }

                    async with session.post(f"{self.base_url}/generate", json=payload) as response:
                        if response.status == 200:
                            data = await response.json()

                            end_time = time.time()
                            latency = (end_time - start_time) * 1000

                            # Calculate tokens per second
                            tokens_generated = data.get("completion_tokens", 0)
                            generation_time = data.get("processing_time", end_time - start_time)
                            tokens_per_sec = tokens_generated / generation_time if generation_time > 0 else 0

                            latencies.append(latency)
                            token_rates.append(tokens_per_sec)
                            successful_requests += 1

                            logger.debug(f"Generated {tokens_generated} tokens in {generation_time:.3f}s ({tokens_per_sec:.1f} tok/s)")
                        else:
                            logger.error(f"Request failed with status {response.status}")

                except Exception as e:
                    logger.error(f"Generation request failed: {e}")

                # Get memory usage
                memory_usages.append(self._get_memory_usage())

                # Small delay between requests
                await asyncio.sleep(0.5)

        # Calculate metrics
        avg_latency = statistics.mean(latencies) if latencies else 0
        avg_tokens_per_sec = statistics.mean(token_rates) if token_rates else 0
        avg_memory = statistics.mean(memory_usages) if memory_usages else 0
        success_rate = (successful_requests / len(test_prompts)) * 100

        result = PerformanceMetrics(
            test_name=f"generation_performance_{reasoning_level}",
            tokens_per_second=avg_tokens_per_sec,
            latency_ms=avg_latency,
            memory_usage_gb=avg_memory,
            concurrent_requests=1,
            success_rate=success_rate,
            reasoning_level=reasoning_level
        )

        self.test_results.append(result)
        logger.info(f"Generation ({reasoning_level}): {avg_tokens_per_sec:.1f} tok/s, {avg_latency:.1f}ms, {success_rate:.1f}% success")
        return result

    async def test_reasoning_level_performance(self) -> List[PerformanceMetrics]:
        """Test performance differences across reasoning levels"""
        logger.info("Testing reasoning level performance differences...")

        results = []
        for level in ["low", "medium", "high"]:
            result = await self.test_generation_performance(level)
            results.append(result)

            # Allow system to stabilize between tests
            await asyncio.sleep(2)

        return results

    async def test_chat_performance(self) -> PerformanceMetrics:
        """Test chat completion performance"""
        logger.info("Testing chat completion performance...")

        test_conversations = [
            [
                {"role": "user", "content": "Hello, can you help me understand neural networks?"},
                {"role": "assistant", "content": "Of course! Neural networks are computational models inspired by biological neural networks."},
                {"role": "user", "content": "Can you explain backpropagation?"}
            ],
            [
                {"role": "user", "content": "What's the difference between supervised and unsupervised learning?"}
            ],
            [
                {"role": "system", "content": "You are a helpful programming assistant."},
                {"role": "user", "content": "How do I implement a binary search in Python?"}
            ]
        ]

        latencies = []
        token_rates = []
        successful_requests = 0

        async with aiohttp.ClientSession() as session:
            for messages in test_conversations:
                try:
                    start_time = time.time()

                    payload = {
                        "messages": messages,
                        "max_tokens": 256,
                        "temperature": 0.7,
                        "reasoning_level": "medium"
                    }

                    async with session.post(f"{self.base_url}/chat", json=payload) as response:
                        if response.status == 200:
                            data = await response.json()

                            end_time = time.time()
                            latency = (end_time - start_time) * 1000

                            tokens_generated = data.get("completion_tokens", 0)
                            generation_time = data.get("processing_time", end_time - start_time)
                            tokens_per_sec = tokens_generated / generation_time if generation_time > 0 else 0

                            latencies.append(latency)
                            token_rates.append(tokens_per_sec)
                            successful_requests += 1

                        else:
                            logger.error(f"Chat request failed with status {response.status}")

                except Exception as e:
                    logger.error(f"Chat request failed: {e}")

                await asyncio.sleep(0.5)

        avg_latency = statistics.mean(latencies) if latencies else 0
        avg_tokens_per_sec = statistics.mean(token_rates) if token_rates else 0
        success_rate = (successful_requests / len(test_conversations)) * 100

        result = PerformanceMetrics(
            test_name="chat_performance",
            tokens_per_second=avg_tokens_per_sec,
            latency_ms=avg_latency,
            memory_usage_gb=self._get_memory_usage(),
            concurrent_requests=1,
            success_rate=success_rate
        )

        self.test_results.append(result)
        logger.info(f"Chat: {avg_tokens_per_sec:.1f} tok/s, {avg_latency:.1f}ms, {success_rate:.1f}% success")
        return result

    async def test_streaming_performance(self) -> PerformanceMetrics:
        """Test streaming generation performance"""
        logger.info("Testing streaming generation performance...")

        test_prompt = "Write a detailed explanation of how artificial intelligence has evolved over the past decade, including major breakthroughs and current challenges."

        try:
            start_time = time.time()
            first_token_time = None
            total_tokens = 0
            chunks_received = 0

            payload = {
                "text": test_prompt,
                "max_tokens": 1024,
                "temperature": 0.7,
                "reasoning_level": "medium"
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.base_url}/stream", json=payload) as response:
                    if response.status == 200:
                        async for line in response.content:
                            if line:
                                line_str = line.decode('utf-8').strip()
                                if line_str.startswith('data: '):
                                    data_str = line_str[6:]  # Remove 'data: ' prefix
                                    if data_str != '[DONE]':
                                        try:
                                            chunk_data = json.loads(data_str)
                                            if 'chunk' in chunk_data:
                                                chunks_received += 1
                                                if first_token_time is None:
                                                    first_token_time = time.time()

                                                # Count tokens in delta
                                                delta = chunk_data['chunk'].get('delta', '')
                                                if delta:
                                                    total_tokens += len(delta.split())
                                        except json.JSONDecodeError:
                                            pass

            end_time = time.time()
            total_time = end_time - start_time
            first_token_latency = (first_token_time - start_time) * 1000 if first_token_time else 0

            tokens_per_sec = total_tokens / total_time if total_time > 0 else 0

            result = PerformanceMetrics(
                test_name="streaming_performance",
                tokens_per_second=tokens_per_sec,
                latency_ms=first_token_latency,
                memory_usage_gb=self._get_memory_usage(),
                concurrent_requests=1,
                success_rate=100.0 if chunks_received > 0 else 0.0
            )

            logger.info(f"Streaming: {tokens_per_sec:.1f} tok/s, first token: {first_token_latency:.1f}ms, {chunks_received} chunks")

        except Exception as e:
            result = PerformanceMetrics(
                test_name="streaming_performance",
                tokens_per_second=0,
                latency_ms=0,
                memory_usage_gb=self._get_memory_usage(),
                concurrent_requests=1,
                success_rate=0.0,
                error_details=str(e)
            )
            logger.error(f"Streaming test failed: {e}")

        self.test_results.append(result)
        return result

    async def test_concurrent_requests(self, num_concurrent: int = 10) -> PerformanceMetrics:
        """Test concurrent request handling performance"""
        logger.info(f"Testing concurrent requests (n={num_concurrent})...")

        test_prompt = "Explain the concept of distributed computing."

        async def single_request(session, request_id):
            try:
                start_time = time.time()
                payload = {
                    "text": f"{test_prompt} (Request {request_id})",
                    "max_tokens": 256,
                    "temperature": 0.7,
                    "reasoning_level": "medium"
                }

                async with session.post(f"{self.base_url}/generate", json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        end_time = time.time()

                        tokens_generated = data.get("completion_tokens", 0)
                        generation_time = data.get("processing_time", end_time - start_time)

                        return {
                            "success": True,
                            "latency": (end_time - start_time) * 1000,
                            "tokens": tokens_generated,
                            "generation_time": generation_time
                        }
                    else:
                        return {"success": False, "error": f"HTTP {response.status}"}

            except Exception as e:
                return {"success": False, "error": str(e)}

        # Execute concurrent requests
        start_time = time.time()

        async with aiohttp.ClientSession() as session:
            tasks = [single_request(session, i) for i in range(num_concurrent)]
            results = await asyncio.gather(*tasks)

        end_time = time.time()
        total_time = end_time - start_time

        # Analyze results
        successful_requests = [r for r in results if r.get("success", False)]
        failed_requests = [r for r in results if not r.get("success", False)]

        if successful_requests:
            avg_latency = statistics.mean([r["latency"] for r in successful_requests])
            total_tokens = sum([r["tokens"] for r in successful_requests])
            avg_tokens_per_sec = total_tokens / total_time if total_time > 0 else 0
        else:
            avg_latency = 0
            avg_tokens_per_sec = 0

        success_rate = (len(successful_requests) / num_concurrent) * 100

        result = PerformanceMetrics(
            test_name=f"concurrent_requests_{num_concurrent}",
            tokens_per_second=avg_tokens_per_sec,
            latency_ms=avg_latency,
            memory_usage_gb=self._get_memory_usage(),
            concurrent_requests=num_concurrent,
            success_rate=success_rate,
            error_details=f"{len(failed_requests)} failed requests" if failed_requests else None
        )

        self.test_results.append(result)
        logger.info(f"Concurrent ({num_concurrent}): {avg_tokens_per_sec:.1f} tok/s, {avg_latency:.1f}ms, {success_rate:.1f}% success")
        return result

    async def test_load_performance(self) -> List[PerformanceMetrics]:
        """Test system performance under increasing load"""
        logger.info("Testing load performance...")

        load_levels = [1, 5, 10, 20, 50]
        results = []

        for load_level in load_levels:
            logger.info(f"Testing load level: {load_level} concurrent requests")
            result = await self.test_concurrent_requests(load_level)
            results.append(result)

            # Allow system to stabilize between load tests
            await asyncio.sleep(5)

        return results

    async def test_memory_usage(self) -> PerformanceMetrics:
        """Test memory usage patterns during generation"""
        logger.info("Testing memory usage patterns...")

        initial_memory = self._get_memory_usage()
        memory_samples = [initial_memory]

        # Generate several requests while monitoring memory
        test_prompts = [
            "Write a long technical explanation of machine learning algorithms.",
            "Create a detailed story about space exploration.",
            "Explain complex mathematical concepts in detail.",
            "Describe the history and future of artificial intelligence.",
            "Write a comprehensive guide to software engineering best practices."
        ]

        async with aiohttp.ClientSession() as session:
            for prompt in test_prompts:
                payload = {
                    "text": prompt,
                    "max_tokens": 1024,
                    "temperature": 0.7,
                    "reasoning_level": "high"
                }

                try:
                    async with session.post(f"{self.base_url}/generate", json=payload) as response:
                        await response.json()  # Consume response

                    # Sample memory after each request
                    memory_samples.append(self._get_memory_usage())

                except Exception as e:
                    logger.error(f"Memory test request failed: {e}")

                await asyncio.sleep(1)

        max_memory = max(memory_samples)
        avg_memory = statistics.mean(memory_samples)
        memory_growth = max_memory - initial_memory

        # Calculate memory usage as percentage of system memory
        total_system_memory = psutil.virtual_memory().total / (1024**3)
        memory_percentage = (max_memory / total_system_memory) * 100

        result = PerformanceMetrics(
            test_name="memory_usage",
            tokens_per_second=0,  # Not applicable
            latency_ms=0,  # Not applicable
            memory_usage_gb=max_memory,
            concurrent_requests=0,
            success_rate=100.0,
            error_details=f"Growth: {memory_growth:.1f}GB, System: {memory_percentage:.1f}%"
        )

        self.test_results.append(result)
        logger.info(f"Memory: {max_memory:.1f}GB peak, {memory_percentage:.1f}% of system, {memory_growth:.1f}GB growth")
        return result

    def _get_memory_usage(self) -> float:
        """Get current system memory usage in GB"""
        return psutil.virtual_memory().used / (1024**3)

    async def test_error_scenarios(self) -> PerformanceMetrics:
        """Test error handling and recovery"""
        logger.info("Testing error scenarios...")

        error_tests = [
            # Invalid reasoning level
            {
                "name": "invalid_reasoning_level",
                "payload": {"text": "Test prompt", "reasoning_level": "invalid"},
                "expected_status": 422
            },
            # Token limit exceeded
            {
                "name": "token_limit_exceeded",
                "payload": {"text": "Test prompt", "max_tokens": 99999},
                "expected_status": 422
            },
            # Invalid temperature
            {
                "name": "invalid_temperature",
                "payload": {"text": "Test prompt", "temperature": 2.5},
                "expected_status": 422
            },
            # Empty prompt
            {
                "name": "empty_prompt",
                "payload": {"text": ""},
                "expected_status": 422
            }
        ]

        passed_tests = 0

        async with aiohttp.ClientSession() as session:
            for test in error_tests:
                try:
                    async with session.post(f"{self.base_url}/generate", json=test["payload"]) as response:
                        if response.status == test["expected_status"]:
                            passed_tests += 1
                            logger.debug(f"Error test '{test['name']}' passed")
                        else:
                            logger.warning(f"Error test '{test['name']}' failed: expected {test['expected_status']}, got {response.status}")

                except Exception as e:
                    logger.error(f"Error test '{test['name']}' failed with exception: {e}")

        success_rate = (passed_tests / len(error_tests)) * 100

        result = PerformanceMetrics(
            test_name="error_scenarios",
            tokens_per_second=0,
            latency_ms=0,
            memory_usage_gb=self._get_memory_usage(),
            concurrent_requests=0,
            success_rate=success_rate
        )

        self.test_results.append(result)
        logger.info(f"Error handling: {passed_tests}/{len(error_tests)} tests passed ({success_rate:.1f}%)")
        return result

    async def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run all performance tests and generate comprehensive report"""
        logger.info("Starting comprehensive performance testing...")

        start_time = time.time()

        try:
            # Setup test environment
            await self.setup_test_environment()

            # Run all tests
            await self.test_server_startup_performance()
            await self.test_reasoning_level_performance()
            await self.test_chat_performance()
            await self.test_streaming_performance()
            await self.test_load_performance()
            await self.test_memory_usage()
            await self.test_error_scenarios()

        except Exception as e:
            logger.error(f"Comprehensive testing failed: {e}")

        finally:
            # Cleanup
            if self.server_process:
                self.server_process.terminate()

        total_time = time.time() - start_time

        # Generate report
        report = self.generate_performance_report(total_time)
        return report

    def generate_performance_report(self, total_test_time: float) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        logger.info("Generating performance report...")

        # Aggregate metrics
        generation_tests = [r for r in self.test_results if "generation" in r.test_name]
        concurrent_tests = [r for r in self.test_results if "concurrent" in r.test_name]

        # Calculate summary statistics
        avg_tokens_per_sec = statistics.mean([r.tokens_per_second for r in generation_tests if r.tokens_per_second > 0]) if generation_tests else 0
        avg_latency = statistics.mean([r.latency_ms for r in generation_tests if r.latency_ms > 0]) if generation_tests else 0
        max_memory = max([r.memory_usage_gb for r in self.test_results], default=0)
        max_concurrent = max([r.concurrent_requests for r in concurrent_tests], default=0)
        avg_success_rate = statistics.mean([r.success_rate for r in self.test_results])

        # Performance assessment
        performance_assessment = {
            "tokens_per_second": {
                "value": avg_tokens_per_sec,
                "target": self.targets["tokens_per_second"],
                "passed": avg_tokens_per_sec >= self.targets["tokens_per_second"]
            },
            "latency_ms": {
                "value": avg_latency,
                "target": self.targets["latency_ms"],
                "passed": avg_latency <= self.targets["latency_ms"]
            },
            "memory_usage_percent": {
                "value": (max_memory / (psutil.virtual_memory().total / (1024**3))) * 100,
                "target": self.targets["memory_usage_percent"],
                "passed": (max_memory / (psutil.virtual_memory().total / (1024**3))) * 100 <= self.targets["memory_usage_percent"]
            },
            "concurrent_requests": {
                "value": max_concurrent,
                "target": self.targets["concurrent_requests"],
                "passed": max_concurrent >= self.targets["concurrent_requests"]
            },
            "success_rate": {
                "value": avg_success_rate,
                "target": self.targets["success_rate"],
                "passed": avg_success_rate >= self.targets["success_rate"]
            }
        }

        # Count passed/failed metrics
        passed_metrics = sum(1 for metric in performance_assessment.values() if metric["passed"])
        total_metrics = len(performance_assessment)
        overall_pass_rate = (passed_metrics / total_metrics) * 100

        # Generate detailed report
        report = {
            "test_summary": {
                "total_tests": len(self.test_results),
                "total_time_seconds": total_test_time,
                "overall_pass_rate": overall_pass_rate,
                "system_ready_for_production": overall_pass_rate >= 80
            },
            "performance_metrics": {
                "average_tokens_per_second": avg_tokens_per_sec,
                "average_latency_ms": avg_latency,
                "peak_memory_usage_gb": max_memory,
                "max_concurrent_requests": max_concurrent,
                "average_success_rate": avg_success_rate
            },
            "target_compliance": performance_assessment,
            "detailed_results": [asdict(result) for result in self.test_results],
            "recommendations": self._generate_recommendations(performance_assessment)
        }

        # Log summary
        logger.info("=" * 80)
        logger.info("PERFORMANCE TEST REPORT")
        logger.info("=" * 80)
        logger.info(f"Overall Pass Rate: {overall_pass_rate:.1f}% ({passed_metrics}/{total_metrics} metrics)")
        logger.info(f"Average Tokens/Second: {avg_tokens_per_sec:.1f} (target: {self.targets['tokens_per_second']})")
        logger.info(f"Average Latency: {avg_latency:.1f}ms (target: <{self.targets['latency_ms']}ms)")
        logger.info(f"Peak Memory: {max_memory:.1f}GB ({(max_memory / (psutil.virtual_memory().total / (1024**3))) * 100:.1f}% of system)")
        logger.info(f"Max Concurrent: {max_concurrent} requests")
        logger.info(f"Average Success Rate: {avg_success_rate:.1f}%")

        production_ready = "✅ READY" if report["test_summary"]["system_ready_for_production"] else "❌ NOT READY"
        logger.info(f"Production Readiness: {production_ready}")
        logger.info("=" * 80)

        return report

    def _generate_recommendations(self, assessment: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations based on test results"""
        recommendations = []

        if not assessment["tokens_per_second"]["passed"]:
            recommendations.append(f"Optimize generation speed - current: {assessment['tokens_per_second']['value']:.1f} tok/s, target: {assessment['tokens_per_second']['target']} tok/s")

        if not assessment["latency_ms"]["passed"]:
            recommendations.append(f"Reduce response latency - current: {assessment['latency_ms']['value']:.1f}ms, target: <{assessment['latency_ms']['target']}ms")

        if not assessment["memory_usage_percent"]["passed"]:
            recommendations.append(f"Optimize memory usage - current: {assessment['memory_usage_percent']['value']:.1f}%, target: <{assessment['memory_usage_percent']['target']}%")

        if not assessment["concurrent_requests"]["passed"]:
            recommendations.append(f"Improve concurrent request handling - current: {assessment['concurrent_requests']['value']}, target: {assessment['concurrent_requests']['target']}+")

        if not assessment["success_rate"]["passed"]:
            recommendations.append(f"Improve error handling and reliability - current: {assessment['success_rate']['value']:.1f}%, target: {assessment['success_rate']['target']}%+")

        # General recommendations
        if len(recommendations) == 0:
            recommendations.append("System performance meets all targets - consider stress testing with higher loads")

        return recommendations

# CLI interface for running tests
async def main():
    """Main test runner"""
    print("🧪 GPT-OSS-20B Performance Testing Suite")
    print("=" * 60)

    # Create test suite
    test_suite = PerformanceTestSuite()

    try:
        # Run comprehensive tests
        report = await test_suite.run_comprehensive_tests()

        # Save report to file
        import json
        report_file = Path("performance_test_report.json")
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, default=str)

        print(f"\n📊 Performance report saved to: {report_file}")

        # Return appropriate exit code
        if report["test_summary"]["system_ready_for_production"]:
            print("✅ System ready for production")
            return 0
        else:
            print("❌ System needs optimization before production")
            return 1

    except Exception as e:
        print(f"❌ Performance testing failed: {e}")
        return 1

if __name__ == "__main__":
    # Check environment
    if "project_env" not in os.environ.get("VIRTUAL_ENV", ""):
        print("❌ Please activate the virtual environment first: source start_env")
        sys.exit(1)

    exit_code = asyncio.run(main())
    sys.exit(exit_code)