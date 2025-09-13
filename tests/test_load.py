#!/usr/bin/env python3
"""
Load Testing Framework for GPT-OSS-20B Generation System
========================================================

Specialized load testing for concurrent generation requests with:
- Gradual load ramping
- Sustained load testing
- Stress testing beyond normal capacity
- Resource monitoring during load
- Detailed performance degradation analysis

Load Test Scenarios:
1. Ramp-up test: Gradually increase concurrent users
2. Sustained load: Maintain steady concurrent requests
3. Stress test: Push beyond target capacity
4. Spike test: Sudden load increases
5. Volume test: Large number of requests over time
"""

import asyncio
import aiohttp
import json
import time
import statistics
import psutil
import logging
import random
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("LoadTest")

@dataclass
class LoadTestResult:
    """Result from a load test scenario"""
    scenario_name: str
    timestamp: float
    concurrent_users: int
    requests_per_second: float
    average_response_time_ms: float
    percentile_95_response_time_ms: float
    success_rate: float
    error_count: int
    memory_usage_gb: float
    cpu_usage_percent: float
    tokens_per_second: float

@dataclass
class LoadTestConfig:
    """Configuration for load testing"""
    base_url: str = "http://localhost:8008"
    max_concurrent_users: int = 100
    test_duration_seconds: int = 300
    ramp_up_seconds: int = 60
    ramp_down_seconds: int = 60
    request_timeout_seconds: int = 30
    think_time_seconds: float = 1.0

class LoadTestRunner:
    """Advanced load testing framework"""

    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.results: List[LoadTestResult] = []
        self.active_requests = 0
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0

        # Test prompts of varying complexity
        self.test_prompts = [
            # Short prompts (low load)
            "What is machine learning?",
            "Explain quantum computing briefly.",
            "Define artificial intelligence.",

            # Medium prompts (medium load)
            "Explain the differences between supervised and unsupervised learning in detail.",
            "Describe the history and future potential of artificial intelligence.",
            "Write a comprehensive guide to neural network architectures.",

            # Long prompts (high load)
            "Write a detailed technical analysis of transformer models, including their architecture, training process, attention mechanisms, and applications in natural language processing. Include code examples and mathematical formulations.",
            "Create a comprehensive business plan for a startup developing AI-powered healthcare solutions, including market analysis, technical architecture, regulatory considerations, and financial projections.",
            "Explain the complete process of training a large language model from data collection to deployment, including preprocessing, model architecture selection, training techniques, evaluation metrics, and optimization strategies."
        ]

    async def simulate_user_session(self, user_id: int, session_duration: float) -> Dict[str, Any]:
        """Simulate a single user session with multiple requests"""
        session_start = time.time()
        session_requests = 0
        session_successful = 0
        session_response_times = []
        session_tokens = 0

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.request_timeout_seconds)
        ) as session:

            while time.time() - session_start < session_duration:
                try:
                    self.active_requests += 1
                    self.total_requests += 1

                    # Select random prompt weighted by complexity
                    prompt_weights = [3, 3, 3, 2, 2, 2, 1, 1, 1]  # Favor simpler prompts
                    prompt = random.choices(self.test_prompts, weights=prompt_weights)[0]

                    # Random generation parameters
                    max_tokens = random.choice([256, 512, 1024])
                    temperature = random.uniform(0.5, 1.0)
                    reasoning_level = random.choice(["low", "medium", "high"])

                    payload = {
                        "text": prompt,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "reasoning_level": reasoning_level
                    }

                    request_start = time.time()

                    async with session.post(f"{self.config.base_url}/generate", json=payload) as response:
                        if response.status == 200:
                            data = await response.json()
                            request_end = time.time()

                            response_time = (request_end - request_start) * 1000
                            session_response_times.append(response_time)
                            session_successful += 1
                            session_tokens += data.get("completion_tokens", 0)

                        else:
                            logger.warning(f"User {user_id} request failed: HTTP {response.status}")
                            self.failed_requests += 1

                    session_requests += 1

                except asyncio.TimeoutError:
                    logger.warning(f"User {user_id} request timeout")
                    self.failed_requests += 1
                except Exception as e:
                    logger.error(f"User {user_id} request error: {e}")
                    self.failed_requests += 1
                finally:
                    self.active_requests = max(0, self.active_requests - 1)

                # Think time between requests
                if self.config.think_time_seconds > 0:
                    await asyncio.sleep(random.uniform(0.5, self.config.think_time_seconds * 1.5))

        return {
            "user_id": user_id,
            "requests": session_requests,
            "successful": session_successful,
            "response_times": session_response_times,
            "tokens_generated": session_tokens
        }

    async def ramp_up_load_test(self) -> List[LoadTestResult]:
        """Gradually increase load to test system behavior under increasing stress"""
        logger.info("Starting ramp-up load test...")

        results = []
        test_start = time.time()

        # Define ramp-up stages
        stages = [
            (5, 30),    # 5 users for 30 seconds
            (10, 30),   # 10 users for 30 seconds
            (20, 30),   # 20 users for 30 seconds
            (50, 30),   # 50 users for 30 seconds
            (100, 60),  # 100 users for 60 seconds
        ]

        for users, duration in stages:
            stage_start = time.time()
            logger.info(f"Ramping up to {users} concurrent users for {duration}s")

            # Start user sessions
            tasks = []
            for i in range(users):
                task = asyncio.create_task(self.simulate_user_session(i, duration))
                tasks.append(task)

            # Monitor performance during this stage
            stage_results = []

            while time.time() - stage_start < duration:
                # Collect metrics every 5 seconds
                await asyncio.sleep(5)

                # Calculate current metrics
                current_time = time.time()
                elapsed = current_time - stage_start

                if elapsed > 10:  # Allow time for requests to start
                    rps = self.total_requests / (current_time - test_start) if current_time > test_start else 0
                    success_rate = (self.successful_requests / max(self.total_requests, 1)) * 100

                    # System resource usage
                    memory_usage = psutil.virtual_memory().used / (1024**3)
                    cpu_usage = psutil.cpu_percent(interval=1)

                    result = LoadTestResult(
                        scenario_name="ramp_up",
                        timestamp=current_time,
                        concurrent_users=users,
                        requests_per_second=rps,
                        average_response_time_ms=0,  # Will calculate from user sessions
                        percentile_95_response_time_ms=0,
                        success_rate=success_rate,
                        error_count=self.failed_requests,
                        memory_usage_gb=memory_usage,
                        cpu_usage_percent=cpu_usage,
                        tokens_per_second=0  # Will calculate from user sessions
                    )

                    stage_results.append(result)

            # Wait for all user sessions to complete
            user_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Calculate final metrics for this stage
            all_response_times = []
            total_tokens = 0

            for user_result in user_results:
                if isinstance(user_result, dict) and "response_times" in user_result:
                    all_response_times.extend(user_result["response_times"])
                    total_tokens += user_result.get("tokens_generated", 0)

            if all_response_times:
                avg_response_time = statistics.mean(all_response_times)
                p95_response_time = np.percentile(all_response_times, 95)
                tokens_per_second = total_tokens / duration if duration > 0 else 0

                # Update stage results with calculated metrics
                for result in stage_results:
                    result.average_response_time_ms = avg_response_time
                    result.percentile_95_response_time_ms = p95_response_time
                    result.tokens_per_second = tokens_per_second

            results.extend(stage_results)

            # Reset counters for next stage
            self.total_requests = 0
            self.successful_requests = 0
            self.failed_requests = 0

            logger.info(f"Stage complete: {users} users, avg response: {avg_response_time:.1f}ms")

        return results

    async def sustained_load_test(self, concurrent_users: int = 50, duration: int = 300) -> List[LoadTestResult]:
        """Test system under sustained load"""
        logger.info(f"Starting sustained load test: {concurrent_users} users for {duration}s")

        results = []
        test_start = time.time()

        # Start all user sessions
        tasks = []
        for i in range(concurrent_users):
            task = asyncio.create_task(self.simulate_user_session(i, duration))
            tasks.append(task)

        # Monitor performance throughout the test
        while time.time() - test_start < duration:
            await asyncio.sleep(10)  # Sample every 10 seconds

            current_time = time.time()
            elapsed = current_time - test_start

            if elapsed > 30:  # Allow warm-up time
                rps = self.total_requests / elapsed if elapsed > 0 else 0
                success_rate = (self.successful_requests / max(self.total_requests, 1)) * 100

                memory_usage = psutil.virtual_memory().used / (1024**3)
                cpu_usage = psutil.cpu_percent(interval=1)

                result = LoadTestResult(
                    scenario_name="sustained_load",
                    timestamp=current_time,
                    concurrent_users=concurrent_users,
                    requests_per_second=rps,
                    average_response_time_ms=0,  # Will calculate after completion
                    percentile_95_response_time_ms=0,
                    success_rate=success_rate,
                    error_count=self.failed_requests,
                    memory_usage_gb=memory_usage,
                    cpu_usage_percent=cpu_usage,
                    tokens_per_second=0
                )

                results.append(result)
                logger.info(f"Sustained load: {rps:.1f} RPS, {success_rate:.1f}% success, {memory_usage:.1f}GB memory")

        # Wait for completion and calculate final metrics
        user_results = await asyncio.gather(*tasks, return_exceptions=True)

        all_response_times = []
        total_tokens = 0

        for user_result in user_results:
            if isinstance(user_result, dict) and "response_times" in user_result:
                all_response_times.extend(user_result["response_times"])
                total_tokens += user_result.get("tokens_generated", 0)

        if all_response_times:
            avg_response_time = statistics.mean(all_response_times)
            p95_response_time = np.percentile(all_response_times, 95)
            tokens_per_second = total_tokens / duration if duration > 0 else 0

            # Update results with calculated metrics
            for result in results:
                result.average_response_time_ms = avg_response_time
                result.percentile_95_response_time_ms = p95_response_time
                result.tokens_per_second = tokens_per_second

        return results

    async def stress_test(self) -> List[LoadTestResult]:
        """Push system beyond normal capacity to find breaking point"""
        logger.info("Starting stress test...")

        results = []
        test_start = time.time()

        # Aggressive stress test stages
        stress_stages = [
            (50, 60),    # 50 users for 60 seconds
            (100, 60),   # 100 users for 60 seconds
            (200, 60),   # 200 users for 60 seconds (stress)
            (500, 30),   # 500 users for 30 seconds (extreme stress)
        ]

        for users, duration in stress_stages:
            stage_start = time.time()
            logger.info(f"Stress testing with {users} concurrent users for {duration}s")

            # Shorter think time for stress test
            original_think_time = self.config.think_time_seconds
            self.config.think_time_seconds = 0.1  # Very short think time

            try:
                tasks = []
                for i in range(users):
                    task = asyncio.create_task(self.simulate_user_session(i, duration))
                    tasks.append(task)

                # Monitor during stress
                stage_results = []

                while time.time() - stage_start < duration:
                    await asyncio.sleep(5)

                    current_time = time.time()
                    elapsed = current_time - stage_start

                    if elapsed > 10:
                        rps = self.total_requests / (current_time - test_start) if current_time > test_start else 0
                        success_rate = (self.successful_requests / max(self.total_requests, 1)) * 100

                        memory_usage = psutil.virtual_memory().used / (1024**3)
                        cpu_usage = psutil.cpu_percent(interval=1)

                        result = LoadTestResult(
                            scenario_name="stress_test",
                            timestamp=current_time,
                            concurrent_users=users,
                            requests_per_second=rps,
                            average_response_time_ms=0,
                            percentile_95_response_time_ms=0,
                            success_rate=success_rate,
                            error_count=self.failed_requests,
                            memory_usage_gb=memory_usage,
                            cpu_usage_percent=cpu_usage,
                            tokens_per_second=0
                        )

                        stage_results.append(result)

                        # Log warning if system is struggling
                        if success_rate < 80:
                            logger.warning(f"System struggling: {success_rate:.1f}% success rate")
                        if memory_usage > 28:  # > 87.5% of 32GB
                            logger.warning(f"High memory usage: {memory_usage:.1f}GB")

                # Wait for completion
                user_results = await asyncio.gather(*tasks, return_exceptions=True)

                # Calculate metrics
                all_response_times = []
                total_tokens = 0

                for user_result in user_results:
                    if isinstance(user_result, dict) and "response_times" in user_result:
                        all_response_times.extend(user_result["response_times"])
                        total_tokens += user_result.get("tokens_generated", 0)

                if all_response_times:
                    avg_response_time = statistics.mean(all_response_times)
                    p95_response_time = np.percentile(all_response_times, 95)
                    tokens_per_second = total_tokens / duration if duration > 0 else 0

                    for result in stage_results:
                        result.average_response_time_ms = avg_response_time
                        result.percentile_95_response_time_ms = p95_response_time
                        result.tokens_per_second = tokens_per_second

                results.extend(stage_results)

                logger.info(f"Stress stage complete: {users} users, {avg_response_time:.1f}ms avg response")

            finally:
                # Restore original think time
                self.config.think_time_seconds = original_think_time

                # Reset counters
                self.total_requests = 0
                self.successful_requests = 0
                self.failed_requests = 0

                # Allow system to recover between stages
                await asyncio.sleep(30)

        return results

    async def spike_test(self) -> List[LoadTestResult]:
        """Test sudden load spikes"""
        logger.info("Starting spike test...")

        results = []
        test_start = time.time()

        # Baseline load: 10 users
        logger.info("Establishing baseline load (10 users)")
        baseline_tasks = []
        for i in range(10):
            task = asyncio.create_task(self.simulate_user_session(i, 180))  # 3 minutes
            baseline_tasks.append(task)

        await asyncio.sleep(30)  # Let baseline stabilize

        # Sudden spike: add 90 more users
        logger.info("Triggering spike: adding 90 users")
        spike_tasks = []
        for i in range(10, 100):
            task = asyncio.create_task(self.simulate_user_session(i, 60))  # 1 minute spike
            spike_tasks.append(task)

        # Monitor during spike
        spike_start = time.time()
        while time.time() - spike_start < 60:
            await asyncio.sleep(5)

            current_time = time.time()
            elapsed = current_time - test_start

            rps = self.total_requests / elapsed if elapsed > 0 else 0
            success_rate = (self.successful_requests / max(self.total_requests, 1)) * 100

            memory_usage = psutil.virtual_memory().used / (1024**3)
            cpu_usage = psutil.cpu_percent(interval=1)

            result = LoadTestResult(
                scenario_name="spike_test",
                timestamp=current_time,
                concurrent_users=100,
                requests_per_second=rps,
                average_response_time_ms=0,
                percentile_95_response_time_ms=0,
                success_rate=success_rate,
                error_count=self.failed_requests,
                memory_usage_gb=memory_usage,
                cpu_usage_percent=cpu_usage,
                tokens_per_second=0
            )

            results.append(result)
            logger.info(f"Spike test: {rps:.1f} RPS, {success_rate:.1f}% success")

        # Wait for spike tasks to complete
        await asyncio.gather(*spike_tasks, return_exceptions=True)

        # Continue monitoring baseline recovery
        logger.info("Monitoring post-spike recovery")
        recovery_start = time.time()
        while time.time() - recovery_start < 60:
            await asyncio.sleep(5)

            current_time = time.time()
            elapsed = current_time - test_start

            rps = self.total_requests / elapsed if elapsed > 0 else 0
            success_rate = (self.successful_requests / max(self.total_requests, 1)) * 100

            result = LoadTestResult(
                scenario_name="spike_recovery",
                timestamp=current_time,
                concurrent_users=10,  # Back to baseline
                requests_per_second=rps,
                average_response_time_ms=0,
                percentile_95_response_time_ms=0,
                success_rate=success_rate,
                error_count=self.failed_requests,
                memory_usage_gb=psutil.virtual_memory().used / (1024**3),
                cpu_usage_percent=psutil.cpu_percent(interval=1),
                tokens_per_second=0
            )

            results.append(result)

        # Wait for baseline tasks to complete
        await asyncio.gather(*baseline_tasks, return_exceptions=True)

        return results

    def generate_load_test_report(self, all_results: List[LoadTestResult]) -> Dict[str, Any]:
        """Generate comprehensive load test report with visualizations"""
        logger.info("Generating load test report...")

        # Group results by scenario
        scenarios = {}
        for result in all_results:
            if result.scenario_name not in scenarios:
                scenarios[result.scenario_name] = []
            scenarios[result.scenario_name].append(result)

        # Calculate summary statistics for each scenario
        scenario_summaries = {}
        for scenario_name, results in scenarios.items():
            if results:
                scenario_summaries[scenario_name] = {
                    "avg_response_time_ms": statistics.mean([r.average_response_time_ms for r in results if r.average_response_time_ms > 0]),
                    "max_response_time_ms": max([r.percentile_95_response_time_ms for r in results if r.percentile_95_response_time_ms > 0], default=0),
                    "avg_success_rate": statistics.mean([r.success_rate for r in results]),
                    "min_success_rate": min([r.success_rate for r in results]),
                    "max_rps": max([r.requests_per_second for r in results]),
                    "max_concurrent_users": max([r.concurrent_users for r in results]),
                    "max_memory_gb": max([r.memory_usage_gb for r in results]),
                    "max_cpu_percent": max([r.cpu_usage_percent for r in results]),
                    "avg_tokens_per_second": statistics.mean([r.tokens_per_second for r in results if r.tokens_per_second > 0])
                }

        # Overall system limits
        max_stable_users = 0
        breaking_point_users = 0

        for result in all_results:
            if result.success_rate >= 95:  # Stable performance
                max_stable_users = max(max_stable_users, result.concurrent_users)
            elif result.success_rate < 80:  # System breakdown
                if breaking_point_users == 0:
                    breaking_point_users = result.concurrent_users

        # Performance assessment
        assessment = {
            "max_stable_concurrent_users": max_stable_users,
            "breaking_point_users": breaking_point_users,
            "production_ready": max_stable_users >= 50,  # Target: handle 50+ concurrent users
            "scalability_rating": self._calculate_scalability_rating(all_results)
        }

        report = {
            "test_summary": {
                "total_scenarios": len(scenarios),
                "total_data_points": len(all_results),
                "test_duration": all_results[-1].timestamp - all_results[0].timestamp if all_results else 0
            },
            "scenario_summaries": scenario_summaries,
            "performance_assessment": assessment,
            "detailed_results": [asdict(result) for result in all_results],
            "recommendations": self._generate_load_test_recommendations(assessment, scenario_summaries)
        }

        # Create visualizations
        self._create_load_test_visualizations(all_results)

        return report

    def _calculate_scalability_rating(self, results: List[LoadTestResult]) -> str:
        """Calculate scalability rating based on performance degradation"""
        if not results:
            return "Unknown"

        # Find maximum users with good performance (>90% success rate)
        good_performance_users = [r.concurrent_users for r in results if r.success_rate >= 90]
        max_good_users = max(good_performance_users) if good_performance_users else 0

        if max_good_users >= 100:
            return "Excellent"
        elif max_good_users >= 50:
            return "Good"
        elif max_good_users >= 20:
            return "Fair"
        else:
            return "Poor"

    def _generate_load_test_recommendations(self, assessment: Dict[str, Any], scenarios: Dict[str, Any]) -> List[str]:
        """Generate load testing recommendations"""
        recommendations = []

        if assessment["max_stable_concurrent_users"] < 50:
            recommendations.append(f"Increase concurrent user capacity - currently stable up to {assessment['max_stable_concurrent_users']} users")

        if assessment["breaking_point_users"] > 0 and assessment["breaking_point_users"] < 100:
            recommendations.append(f"System breaks down at {assessment['breaking_point_users']} concurrent users - improve error handling and resource management")

        # Scenario-specific recommendations
        if "stress_test" in scenarios:
            stress_stats = scenarios["stress_test"]
            if stress_stats["min_success_rate"] < 80:
                recommendations.append("Poor performance under stress - implement graceful degradation and request throttling")

        if "spike_test" in scenarios:
            spike_stats = scenarios["spike_test"]
            if spike_stats["min_success_rate"] < 85:
                recommendations.append("System struggles with sudden load spikes - implement auto-scaling and load balancing")

        if not assessment["production_ready"]:
            recommendations.append("System not ready for production load - optimize performance and increase capacity")

        return recommendations

    def _create_load_test_visualizations(self, results: List[LoadTestResult]):
        """Create performance visualization charts"""
        try:
            if not results:
                return

            # Extract data for plotting
            timestamps = [r.timestamp for r in results]
            start_time = min(timestamps)
            relative_times = [(t - start_time) / 60 for t in timestamps]  # Convert to minutes

            concurrent_users = [r.concurrent_users for r in results]
            response_times = [r.average_response_time_ms for r in results]
            success_rates = [r.success_rate for r in results]
            memory_usage = [r.memory_usage_gb for r in results]
            rps = [r.requests_per_second for r in results]

            # Create subplots
            fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 12))

            # Concurrent Users over Time
            ax1.plot(relative_times, concurrent_users, 'b-', linewidth=2)
            ax1.set_title('Concurrent Users Over Time')
            ax1.set_xlabel('Time (minutes)')
            ax1.set_ylabel('Concurrent Users')
            ax1.grid(True)

            # Response Time over Time
            ax2.plot(relative_times, response_times, 'r-', linewidth=2)
            ax2.set_title('Average Response Time')
            ax2.set_xlabel('Time (minutes)')
            ax2.set_ylabel('Response Time (ms)')
            ax2.grid(True)

            # Success Rate over Time
            ax3.plot(relative_times, success_rates, 'g-', linewidth=2)
            ax3.set_title('Success Rate Over Time')
            ax3.set_xlabel('Time (minutes)')
            ax3.set_ylabel('Success Rate (%)')
            ax3.grid(True)
            ax3.set_ylim(0, 100)

            # Memory Usage over Time
            ax4.plot(relative_times, memory_usage, 'm-', linewidth=2)
            ax4.set_title('Memory Usage Over Time')
            ax4.set_xlabel('Time (minutes)')
            ax4.set_ylabel('Memory (GB)')
            ax4.grid(True)

            # Requests per Second
            ax5.plot(relative_times, rps, 'c-', linewidth=2)
            ax5.set_title('Requests per Second')
            ax5.set_xlabel('Time (minutes)')
            ax5.set_ylabel('RPS')
            ax5.grid(True)

            # Users vs Response Time Correlation
            ax6.scatter(concurrent_users, response_times, alpha=0.6, c='orange')
            ax6.set_title('Users vs Response Time Correlation')
            ax6.set_xlabel('Concurrent Users')
            ax6.set_ylabel('Response Time (ms)')
            ax6.grid(True)

            plt.tight_layout()
            plt.savefig('load_test_results.png', dpi=300, bbox_inches='tight')
            logger.info("Load test visualizations saved to load_test_results.png")

        except Exception as e:
            logger.warning(f"Could not create visualizations: {e}")

    async def run_comprehensive_load_tests(self) -> Dict[str, Any]:
        """Run all load test scenarios"""
        logger.info("Starting comprehensive load testing...")

        all_results = []

        try:
            # Run all load test scenarios
            logger.info("1/4 Running ramp-up load test...")
            ramp_results = await self.ramp_up_load_test()
            all_results.extend(ramp_results)

            await asyncio.sleep(60)  # Recovery time

            logger.info("2/4 Running sustained load test...")
            sustained_results = await self.sustained_load_test(50, 300)
            all_results.extend(sustained_results)

            await asyncio.sleep(60)  # Recovery time

            logger.info("3/4 Running stress test...")
            stress_results = await self.stress_test()
            all_results.extend(stress_results)

            await asyncio.sleep(120)  # Longer recovery time

            logger.info("4/4 Running spike test...")
            spike_results = await self.spike_test()
            all_results.extend(spike_results)

        except Exception as e:
            logger.error(f"Load testing failed: {e}")

        # Generate comprehensive report
        report = self.generate_load_test_report(all_results)

        return report

# CLI interface
async def main():
    """Main load test runner"""
    print("🔥 GPT-OSS-20B Load Testing Framework")
    print("=" * 60)

    config = LoadTestConfig(
        max_concurrent_users=500,
        test_duration_seconds=300,
        request_timeout_seconds=30
    )

    runner = LoadTestRunner(config)

    try:
        report = await runner.run_comprehensive_load_tests()

        # Save report
        import json
        report_file = Path("load_test_report.json")
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, default=str)

        print(f"\n📊 Load test report saved to: {report_file}")

        # Summary
        assessment = report["performance_assessment"]
        print(f"\n📈 Load Test Summary:")
        print(f"Max Stable Users: {assessment['max_stable_concurrent_users']}")
        print(f"Scalability Rating: {assessment['scalability_rating']}")
        print(f"Production Ready: {'✅ Yes' if assessment['production_ready'] else '❌ No'}")

        return 0 if assessment["production_ready"] else 1

    except Exception as e:
        print(f"❌ Load testing failed: {e}")
        return 1

if __name__ == "__main__":
    import sys
    import os

    # Check environment
    if "project_env" not in os.environ.get("VIRTUAL_ENV", ""):
        print("❌ Please activate the virtual environment first: source start_env")
        sys.exit(1)

    exit_code = asyncio.run(main())
    sys.exit(exit_code)