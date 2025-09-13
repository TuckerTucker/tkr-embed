#!/usr/bin/env python3
"""
Integration Testing for Synchronization Point 2
Tests all Phase 2 components and their integration points
"""

import sys
import os
import asyncio
import time
import traceback
from typing import Dict, Any, List, Optional, Tuple
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("IntegrationTest")

# Test results storage
test_results: Dict[str, Dict[str, Any]] = {}

class IntegrationTestResult:
    """Test result tracking"""
    def __init__(self, test_name: str, component: str):
        self.test_name = test_name
        self.component = component
        self.passed = False
        self.error_msg = ""
        self.details = {}
        self.start_time = time.time()
        self.end_time = None

    def pass_test(self, details: Dict[str, Any] = None):
        self.passed = True
        self.end_time = time.time()
        self.details = details or {}

    def fail_test(self, error_msg: str, details: Dict[str, Any] = None):
        self.passed = False
        self.error_msg = error_msg
        self.end_time = time.time()
        self.details = details or {}

def run_test(test_name: str, component: str):
    """Decorator for test functions"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            result = IntegrationTestResult(test_name, component)
            logger.info(f"Running {component} test: {test_name}")

            try:
                test_output = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
                if test_output is False:
                    result.fail_test("Test returned False")
                else:
                    result.pass_test(test_output if isinstance(test_output, dict) else {})
            except Exception as e:
                result.fail_test(f"{type(e).__name__}: {str(e)}", {"traceback": traceback.format_exc()})

            # Store result
            if component not in test_results:
                test_results[component] = {}
            test_results[component][test_name] = result

            status = "✅ PASS" if result.passed else "❌ FAIL"
            duration = result.end_time - result.start_time
            logger.info(f"{status} {component}: {test_name} ({duration:.2f}s)")
            if not result.passed:
                logger.error(f"Error: {result.error_msg}")

            return result.passed
        return wrapper
    return decorator

# =============================================================================
# 1. IMPORT VALIDATION TESTS
# =============================================================================

@run_test("import_config", "config")
def test_import_config():
    """Test config module imports"""
    try:
        from tkr_embed.config import get_config, Config, ModelConfig, GenerationConfig
        config = get_config()
        return {
            "config_type": str(type(config)),
            "model_path": config.model.model_path,
            "quantization": config.model.quantization,
            "generation_enabled": config.generation.streaming_enabled
        }
    except Exception as e:
        raise e

@run_test("import_model_manager", "model_manager")
def test_import_model_manager():
    """Test model manager imports"""
    try:
        from tkr_embed.core.model_manager import GPTOss20bMLX
        from tkr_embed.core.model_manager import ReasoningLevel, GenerationConfig, ModelNotReadyError

        # Test instantiation (don't load model)
        manager = GPTOss20bMLX(model_path="test/model", quantization="auto")

        return {
            "manager_class": str(type(manager)),
            "model_path": manager.model_path,
            "quantization": manager.quantization,
            "reasoning_levels": list(manager.reasoning_prompts.keys())
        }
    except Exception as e:
        raise e

@run_test("import_api_models", "api_models")
def test_import_api_models():
    """Test API models imports"""
    try:
        from tkr_embed.api.models import (
            GenerationRequest, GenerationResponse, ChatRequest, ChatResponse,
            ReasoningLevel, HealthResponse, ModelInfoResponse, ErrorResponse
        )

        # Test model instantiation
        gen_req = GenerationRequest(text="test prompt")

        return {
            "generation_request_fields": list(gen_req.model_fields.keys()),
            "reasoning_levels": [level.value for level in ReasoningLevel],
            "default_max_tokens": gen_req.max_tokens,
            "default_temperature": gen_req.temperature
        }
    except Exception as e:
        raise e

@run_test("import_server", "api_server")
def test_import_server():
    """Test server imports"""
    try:
        from tkr_embed.api.server import app
        from tkr_embed.api.error_handlers import setup_error_handlers, SafeModelOperation
        from tkr_embed.api.auth import authenticate, optional_auth
        from tkr_embed.api.rate_limiter import apply_rate_limit

        return {
            "app_type": str(type(app)),
            "app_title": app.title,
            "openapi_url": app.openapi_url,
            "docs_url": app.docs_url
        }
    except Exception as e:
        raise e

@run_test("import_infrastructure", "infrastructure")
def test_import_infrastructure():
    """Test infrastructure imports"""
    try:
        from tkr_embed.utils.memory_manager import MemoryManager
        from tkr_embed.utils.lru_cache import GenerationCache, CachedGenerationProcessor

        # Test instantiation
        mem_manager = MemoryManager()
        cache = GenerationCache(max_size=100, ttl=600)

        return {
            "memory_profile_tier": mem_manager.memory_profile["tier"],
            "recommended_quantization": mem_manager.memory_profile["quantization"],
            "cache_max_size": cache.max_size,
            "cache_ttl": cache.ttl
        }
    except Exception as e:
        raise e

# =============================================================================
# 2. INTERFACE COMPLIANCE TESTS
# =============================================================================

@run_test("model_interface_compliance", "model_manager")
async def test_model_interface_compliance():
    """Test that model manager implements the interface correctly"""
    try:
        from tkr_embed.core.model_manager import GPTOss20bMLX
        from tkr_embed.core.model_manager import GenerationConfig, ReasoningLevel

        manager = GPTOss20bMLX(model_path="test/model", quantization="auto")

        # Test interface methods exist
        interface_methods = [
            'load_model', 'generate', 'generate_stream', 'chat',
            'get_model_info', 'is_ready', 'get_memory_usage'
        ]

        missing_methods = []
        for method in interface_methods:
            if not hasattr(manager, method):
                missing_methods.append(method)

        if missing_methods:
            return False

        # Test method signatures (basic check)
        assert callable(manager.load_model)
        assert callable(manager.generate)
        assert callable(manager.generate_stream)
        assert callable(manager.chat)
        assert callable(manager.get_model_info)
        assert callable(manager.is_ready)
        assert callable(manager.get_memory_usage)

        # Test GenerationConfig instantiation
        config = GenerationConfig(
            max_tokens=1024,
            temperature=0.5,
            reasoning_level=ReasoningLevel.MEDIUM
        )

        return {
            "interface_methods": interface_methods,
            "missing_methods": missing_methods,
            "config_test": {
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
                "reasoning_level": config.reasoning_level
            }
        }
    except Exception as e:
        raise e

@run_test("api_interface_compliance", "api_server")
def test_api_interface_compliance():
    """Test API endpoint interface compliance"""
    try:
        from tkr_embed.api.server import app
        from tkr_embed.api.models import GenerationRequest, ChatRequest

        # Check that required endpoints exist
        required_endpoints = ['/generate', '/chat', '/stream', '/health', '/info']
        available_routes = [route.path for route in app.routes]

        missing_endpoints = []
        for endpoint in required_endpoints:
            if endpoint not in available_routes:
                missing_endpoints.append(endpoint)

        # Test model validation
        gen_req = GenerationRequest(text="test")
        chat_req = ChatRequest(messages=[{"role": "user", "content": "test"}])

        return {
            "required_endpoints": required_endpoints,
            "available_routes": available_routes,
            "missing_endpoints": missing_endpoints,
            "validation_test": {
                "generation_request_valid": gen_req.text == "test",
                "chat_request_valid": len(chat_req.messages) == 1
            }
        }
    except Exception as e:
        raise e

@run_test("infrastructure_interface_compliance", "infrastructure")
def test_infrastructure_interface_compliance():
    """Test infrastructure interface compliance"""
    try:
        from tkr_embed.utils.memory_manager import MemoryManager
        from tkr_embed.utils.lru_cache import GenerationCache

        mem_manager = MemoryManager()
        cache = GenerationCache()

        # Test MemoryManager interface
        memory_methods = ['optimize_for_generation', 'allocate_for_model', 'get_available_memory',
                         'get_memory_stats', 'check_memory_pressure', 'cleanup_memory']

        missing_memory_methods = []
        for method in memory_methods:
            if not hasattr(mem_manager, method) or not callable(getattr(mem_manager, method)):
                missing_memory_methods.append(method)

        # Test Cache interface
        cache_methods = ['get', 'put', 'clear', 'get_stats']
        missing_cache_methods = []
        for method in cache_methods:
            if not hasattr(cache, method) or not callable(getattr(cache, method)):
                missing_cache_methods.append(method)

        return {
            "memory_manager": {
                "required_methods": memory_methods,
                "missing_methods": missing_memory_methods,
                "memory_profile": mem_manager.memory_profile["tier"]
            },
            "cache": {
                "required_methods": cache_methods,
                "missing_methods": missing_cache_methods,
                "max_size": cache.max_size
            }
        }
    except Exception as e:
        raise e

# =============================================================================
# 3. COMPONENT INTEGRATION TESTS
# =============================================================================

@run_test("model_manager_initialization", "integration")
async def test_model_manager_initialization():
    """Test model manager initialization with real config"""
    try:
        from tkr_embed.config import get_config
        from tkr_embed.core.model_manager import GPTOss20bMLX

        config = get_config()
        manager = GPTOss20bMLX(
            model_path=config.model.model_path,
            quantization=config.model.quantization
        )

        # Test basic properties
        model_info = manager.get_model_info()
        memory_usage = manager.get_memory_usage()
        is_ready = manager.is_ready()

        return {
            "model_path": manager.model_path,
            "quantization": manager.quantization,
            "device": manager.device,
            "model_info": model_info,
            "memory_usage_gb": memory_usage,
            "is_ready": is_ready
        }
    except Exception as e:
        raise e

@run_test("server_app_configuration", "integration")
def test_server_app_configuration():
    """Test server configuration integration"""
    try:
        from tkr_embed.config import get_config
        from tkr_embed.api.server import app

        config = get_config()

        # Test app configuration
        app_config = {
            "title": app.title,
            "version": app.version,
            "debug": app.debug,
            "docs_url": app.docs_url
        }

        # Test middleware count
        middleware_count = len(app.user_middleware)

        return {
            "app_config": app_config,
            "middleware_count": middleware_count,
            "config_environment": config.environment.value,
            "config_debug": config.debug
        }
    except Exception as e:
        raise e

@run_test("memory_manager_integration", "integration")
def test_memory_manager_integration():
    """Test memory manager integration with config"""
    try:
        from tkr_embed.config import get_config
        from tkr_embed.utils.memory_manager import MemoryManager

        config = get_config()
        mem_manager = MemoryManager()

        # Test memory optimization
        mem_manager.optimize_for_generation()

        # Get memory stats
        stats = mem_manager.get_memory_stats()
        optimal_config = mem_manager.get_optimal_config()

        # Test batch size suggestion
        suggested_batch = mem_manager.suggest_batch_size(4)
        suggested_tokens = mem_manager.suggest_max_tokens(config.model.max_tokens)

        return {
            "memory_stats": stats,
            "optimal_config": optimal_config,
            "suggested_batch_size": suggested_batch,
            "suggested_max_tokens": suggested_tokens
        }
    except Exception as e:
        raise e

@run_test("cache_integration", "integration")
def test_cache_integration():
    """Test cache integration with generation config"""
    try:
        from tkr_embed.utils.lru_cache import GenerationCache
        from tkr_embed.config import get_config

        config = get_config()
        cache = GenerationCache(
            max_size=config.cache.max_size,
            ttl=config.cache.ttl_seconds
        )

        # Test cache operations
        test_config = {
            "max_tokens": 512,
            "temperature": 0.7,
            "reasoning_level": "medium"
        }

        # Test put/get
        cache.put("test prompt", test_config, "test result", 50, 1.5)
        cached_result = cache.get("test prompt", test_config)

        # Test stats
        stats = cache.get_stats()

        return {
            "cache_size": cache.max_size,
            "cache_ttl": cache.ttl,
            "put_get_test": cached_result == "test result",
            "stats": stats
        }
    except Exception as e:
        raise e

# =============================================================================
# 4. CONFIGURATION CONSISTENCY TESTS
# =============================================================================

@run_test("config_loading", "configuration")
def test_config_loading():
    """Test configuration loading consistency"""
    try:
        from tkr_embed.config import get_config, ConfigManager

        config = get_config()

        # Test config structure
        config_sections = [
            'model', 'generation', 'security', 'rate_limit',
            'cache', 'logging', 'monitoring'
        ]

        missing_sections = []
        for section in config_sections:
            if not hasattr(config, section):
                missing_sections.append(section)

        # Test critical values
        critical_config = {
            "environment": config.environment.value,
            "model_path": config.model.model_path,
            "quantization": config.model.quantization,
            "max_tokens": config.model.max_tokens,
            "streaming_enabled": config.generation.streaming_enabled,
            "cache_enabled": config.cache.enabled
        }

        return {
            "config_sections": config_sections,
            "missing_sections": missing_sections,
            "critical_config": critical_config
        }
    except Exception as e:
        raise e

@run_test("model_config_consistency", "configuration")
def test_model_config_consistency():
    """Test model configuration consistency across components"""
    try:
        from tkr_embed.config import get_config
        from tkr_embed.core.model_manager import GPTOss20bMLX
        from tkr_embed.utils.memory_manager import MemoryManager

        config = get_config()
        manager = GPTOss20bMLX()  # Uses config defaults
        mem_manager = MemoryManager()

        # Check consistency
        consistency_checks = {
            "manager_model_path": manager.model_path == config.model.model_path,
            "manager_quantization_matches": manager.quantization in ["auto", config.model.quantization],
            "memory_quantization_matches": mem_manager.memory_profile["quantization"] in ["q4", "q8", "mxfp4", "none"],
            "cache_dir_exists": Path(config.model.cache_dir).exists() or config.model.cache_dir == "./models"
        }

        return {
            "consistency_checks": consistency_checks,
            "config_model_path": config.model.model_path,
            "manager_model_path": manager.model_path,
            "memory_tier": mem_manager.memory_profile["tier"]
        }
    except Exception as e:
        raise e

# =============================================================================
# 5. ERROR HANDLING INTEGRATION TESTS
# =============================================================================

@run_test("error_handler_integration", "error_handling")
async def test_error_handler_integration():
    """Test error handling integration"""
    try:
        from tkr_embed.api.error_handlers import (
            ModelNotReadyError, TokenLimitExceededError, ReasoningLevelError,
            validate_reasoning_level, validate_token_limits, SafeGenerationOperation
        )

        # Test error creation
        errors_created = []

        try:
            raise ModelNotReadyError()
        except ModelNotReadyError as e:
            errors_created.append(("ModelNotReadyError", e.error_code, e.status_code))

        try:
            raise TokenLimitExceededError("Token limit test", 5000, 4096)
        except TokenLimitExceededError as e:
            errors_created.append(("TokenLimitExceededError", e.error_code, e.status_code))

        # Test validation functions
        valid_level = validate_reasoning_level("high")
        valid_tokens = validate_token_limits(2048, 4096)

        # Test SafeGenerationOperation
        try:
            with SafeGenerationOperation("test_operation"):
                pass
            safe_operation_works = True
        except Exception:
            safe_operation_works = False

        return {
            "errors_created": errors_created,
            "validation_tests": {
                "reasoning_level": valid_level == "high",
                "token_limits": valid_tokens == 2048
            },
            "safe_operation_works": safe_operation_works
        }
    except Exception as e:
        raise e

@run_test("generation_error_flow", "error_handling")
async def test_generation_error_flow():
    """Test generation-specific error flow"""
    try:
        from tkr_embed.core.model_manager import GPTOss20bMLX, GenerationConfig, ReasoningLevel
        from tkr_embed.api.error_handlers import SafeGenerationOperation

        manager = GPTOss20bMLX(model_path="test/model")

        # Test model not ready error
        model_ready_error_raised = False
        try:
            with SafeGenerationOperation("test_generation", manager):
                pass
        except Exception as e:
            model_ready_error_raised = "MODEL_NOT_READY" in str(e) or "not ready" in str(e).lower()

        # Test generation config creation
        config = GenerationConfig(
            max_tokens=512,
            temperature=0.8,
            reasoning_level=ReasoningLevel.HIGH
        )

        return {
            "model_ready_error_raised": model_ready_error_raised,
            "generation_config": {
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
                "reasoning_level": config.reasoning_level
            }
        }
    except Exception as e:
        raise e

# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

async def run_all_tests():
    """Run all integration tests"""
    logger.info("=" * 80)
    logger.info("STARTING INTEGRATION TESTING FOR SYNCHRONIZATION POINT 2")
    logger.info("=" * 80)

    start_time = time.time()

    # Run all tests
    test_functions = [
        # Import validation
        test_import_config,
        test_import_model_manager,
        test_import_api_models,
        test_import_server,
        test_import_infrastructure,

        # Interface compliance
        test_model_interface_compliance,
        test_api_interface_compliance,
        test_infrastructure_interface_compliance,

        # Component integration
        test_model_manager_initialization,
        test_server_app_configuration,
        test_memory_manager_integration,
        test_cache_integration,

        # Configuration consistency
        test_config_loading,
        test_model_config_consistency,

        # Error handling integration
        test_error_handler_integration,
        test_generation_error_flow
    ]

    passed_tests = 0
    failed_tests = 0

    for test_func in test_functions:
        try:
            if asyncio.iscoroutinefunction(test_func):
                success = await test_func()
            else:
                success = test_func()

            if success:
                passed_tests += 1
            else:
                failed_tests += 1

        except Exception as e:
            logger.error(f"Test {test_func.__name__} failed with exception: {e}")
            failed_tests += 1

    total_time = time.time() - start_time

    # Generate report
    generate_integration_report(passed_tests, failed_tests, total_time)

def generate_integration_report(passed_tests: int, failed_tests: int, total_time: float):
    """Generate integration test report"""

    logger.info("=" * 80)
    logger.info("INTEGRATION TEST REPORT - SYNCHRONIZATION POINT 2")
    logger.info("=" * 80)

    total_tests = passed_tests + failed_tests
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

    logger.info(f"Total Tests: {total_tests}")
    logger.info(f"Passed: {passed_tests} ✅")
    logger.info(f"Failed: {failed_tests} ❌")
    logger.info(f"Success Rate: {success_rate:.1f}%")
    logger.info(f"Total Time: {total_time:.2f}s")

    # Component breakdown
    logger.info("\n" + "=" * 40)
    logger.info("COMPONENT BREAKDOWN")
    logger.info("=" * 40)

    component_summary = {}
    for component, tests in test_results.items():
        passed = sum(1 for test in tests.values() if test.passed)
        failed = sum(1 for test in tests.values() if not test.passed)
        component_summary[component] = {"passed": passed, "failed": failed}

        status = "✅ READY" if failed == 0 else "❌ ISSUES"
        logger.info(f"{component:<20}: {passed}/{passed + failed} passed {status}")

    # Detailed failures
    if failed_tests > 0:
        logger.info("\n" + "=" * 40)
        logger.info("DETAILED FAILURE REPORT")
        logger.info("=" * 40)

        for component, tests in test_results.items():
            for test_name, result in tests.items():
                if not result.passed:
                    logger.error(f"\n{component}.{test_name}:")
                    logger.error(f"  Error: {result.error_msg}")
                    if "traceback" in result.details:
                        logger.error(f"  Traceback: {result.details['traceback'][:500]}...")

    # Overall readiness assessment
    logger.info("\n" + "=" * 40)
    logger.info("PHASE 3 READINESS ASSESSMENT")
    logger.info("=" * 40)

    critical_components = ["model_manager", "api_server", "infrastructure", "configuration"]
    all_critical_ready = True

    for component in critical_components:
        if component in component_summary:
            component_ready = component_summary[component]["failed"] == 0
            status = "✅ READY" if component_ready else "❌ NOT READY"
            logger.info(f"{component:<20}: {status}")
            if not component_ready:
                all_critical_ready = False
        else:
            logger.warning(f"{component:<20}: ⚠️  NO TESTS")
            all_critical_ready = False

    overall_status = "✅ SYSTEM READY FOR PHASE 3" if all_critical_ready and success_rate >= 90 else "❌ SYSTEM NOT READY"
    logger.info(f"\nOVERALL STATUS: {overall_status}")

    if success_rate >= 90 and all_critical_ready:
        logger.info("\n🎉 All critical systems operational. Phase 3 can proceed!")
    else:
        logger.warning("\n⚠️  Critical issues detected. Resolve before Phase 3.")

    logger.info("\n" + "=" * 80)

if __name__ == "__main__":
    asyncio.run(run_all_tests())