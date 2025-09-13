#!/usr/bin/env python3
"""
Performance Test Execution Script
=================================

Quick performance validation of the GPT-OSS-20B system with mock model.
This script will test the current system capabilities and identify optimization opportunities.
"""

import asyncio
import sys
import os
import time
import logging
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

# Import test suites
from tests.test_performance import PerformanceTestSuite

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def quick_performance_validation():
    """Run a quick performance validation of the system"""
    logger.info("🧪 Starting Quick Performance Validation")
    logger.info("=" * 60)

    test_suite = PerformanceTestSuite()

    try:
        # Test 1: Server startup (if not already running)
        logger.info("1. Testing server availability...")
        await test_suite.setup_test_environment()
        logger.info("✅ Server is available")

        # Test 2: Basic generation performance
        logger.info("2. Testing basic text generation...")
        gen_result = await test_suite.test_generation_performance("medium")
        logger.info(f"   Tokens/sec: {gen_result.tokens_per_second:.1f}")
        logger.info(f"   Latency: {gen_result.latency_ms:.1f}ms")
        logger.info(f"   Success rate: {gen_result.success_rate:.1f}%")

        # Test 3: Reasoning levels comparison
        logger.info("3. Testing reasoning levels...")
        reasoning_results = await test_suite.test_reasoning_level_performance()
        for result in reasoning_results:
            logger.info(f"   {result.reasoning_level}: {result.tokens_per_second:.1f} tok/s, {result.latency_ms:.1f}ms")

        # Test 4: Chat functionality
        logger.info("4. Testing chat completion...")
        chat_result = await test_suite.test_chat_performance()
        logger.info(f"   Chat tokens/sec: {chat_result.tokens_per_second:.1f}")
        logger.info(f"   Chat latency: {chat_result.latency_ms:.1f}ms")

        # Test 5: Streaming performance
        logger.info("5. Testing streaming generation...")
        stream_result = await test_suite.test_streaming_performance()
        logger.info(f"   Streaming tokens/sec: {stream_result.tokens_per_second:.1f}")
        logger.info(f"   First token latency: {stream_result.latency_ms:.1f}ms")

        # Test 6: Concurrent requests (light load)
        logger.info("6. Testing concurrent requests (10 users)...")
        concurrent_result = await test_suite.test_concurrent_requests(10)
        logger.info(f"   Concurrent tokens/sec: {concurrent_result.tokens_per_second:.1f}")
        logger.info(f"   Concurrent success rate: {concurrent_result.success_rate:.1f}%")

        # Test 7: Memory usage
        logger.info("7. Testing memory usage...")
        memory_result = await test_suite.test_memory_usage()
        logger.info(f"   Peak memory: {memory_result.memory_usage_gb:.1f}GB")

        # Test 8: Error handling
        logger.info("8. Testing error scenarios...")
        error_result = await test_suite.test_error_scenarios()
        logger.info(f"   Error handling: {error_result.success_rate:.1f}% tests passed")

        # Generate summary report
        logger.info("\n" + "=" * 60)
        logger.info("📊 PERFORMANCE VALIDATION SUMMARY")
        logger.info("=" * 60)

        # Collect all results
        all_results = [gen_result, chat_result, stream_result, concurrent_result, memory_result, error_result]
        all_results.extend(reasoning_results)

        # Calculate averages
        gen_results = [r for r in all_results if r.tokens_per_second > 0]
        avg_tokens_per_sec = sum(r.tokens_per_second for r in gen_results) / len(gen_results) if gen_results else 0
        avg_latency = sum(r.latency_ms for r in gen_results) / len(gen_results) if gen_results else 0
        avg_success_rate = sum(r.success_rate for r in all_results) / len(all_results)

        logger.info(f"Average Generation Speed: {avg_tokens_per_sec:.1f} tokens/second")
        logger.info(f"Average Response Latency: {avg_latency:.1f}ms")
        logger.info(f"Peak Memory Usage: {memory_result.memory_usage_gb:.1f}GB")
        logger.info(f"Overall Success Rate: {avg_success_rate:.1f}%")

        # Performance assessment against targets
        targets = test_suite.targets
        logger.info("\n📈 TARGET COMPLIANCE:")

        speed_status = "✅ PASS" if avg_tokens_per_sec >= targets["tokens_per_second"] else "❌ FAIL"
        logger.info(f"Generation Speed: {speed_status} ({avg_tokens_per_sec:.1f}/{targets['tokens_per_second']} tok/s)")

        latency_status = "✅ PASS" if avg_latency <= targets["latency_ms"] else "❌ FAIL"
        logger.info(f"Response Latency: {latency_status} ({avg_latency:.1f}/{targets['latency_ms']}ms)")

        # Memory check (assuming 32GB system)
        memory_percent = (memory_result.memory_usage_gb / 32) * 100
        memory_status = "✅ PASS" if memory_percent <= targets["memory_usage_percent"] else "❌ FAIL"
        logger.info(f"Memory Usage: {memory_status} ({memory_percent:.1f}/{targets['memory_usage_percent']}%)")

        success_status = "✅ PASS" if avg_success_rate >= targets["success_rate"] else "❌ FAIL"
        logger.info(f"Success Rate: {success_status} ({avg_success_rate:.1f}/{targets['success_rate']}%)")

        # Overall assessment
        passed_targets = sum([
            avg_tokens_per_sec >= targets["tokens_per_second"],
            avg_latency <= targets["latency_ms"],
            memory_percent <= targets["memory_usage_percent"],
            avg_success_rate >= targets["success_rate"]
        ])

        overall_status = "✅ PRODUCTION READY" if passed_targets >= 3 else "⚠️ NEEDS OPTIMIZATION"
        logger.info(f"\nOVERALL STATUS: {overall_status} ({passed_targets}/4 targets met)")

        # Optimization recommendations
        logger.info("\n🔧 OPTIMIZATION RECOMMENDATIONS:")
        if avg_tokens_per_sec < targets["tokens_per_second"]:
            logger.info("- Optimize text generation speed (consider batch processing, model quantization)")
        if avg_latency > targets["latency_ms"]:
            logger.info("- Reduce response latency (optimize inference pipeline, reduce model load)")
        if memory_percent > targets["memory_usage_percent"]:
            logger.info("- Optimize memory usage (implement better caching, memory cleanup)")
        if avg_success_rate < targets["success_rate"]:
            logger.info("- Improve error handling and system reliability")

        if passed_targets >= 3:
            logger.info("- System meets production requirements")
            logger.info("- Consider load testing with real model for final validation")

        return passed_targets >= 3

    except Exception as e:
        logger.error(f"Performance validation failed: {e}")
        return False

    finally:
        # Cleanup
        if hasattr(test_suite, 'server_process') and test_suite.server_process:
            test_suite.server_process.terminate()

async def main():
    """Main execution function"""
    logger.info("🚀 GPT-OSS-20B Performance Validation")

    success = await quick_performance_validation()

    if success:
        logger.info("\n🎉 Performance validation completed successfully!")
        return 0
    else:
        logger.info("\n❌ Performance validation identified issues that need resolution")
        return 1

if __name__ == "__main__":
    # Check environment
    if "project_env" not in os.environ.get("VIRTUAL_ENV", ""):
        print("❌ Please activate the virtual environment first: source start_env")
        sys.exit(1)

    exit_code = asyncio.run(main())
    sys.exit(exit_code)