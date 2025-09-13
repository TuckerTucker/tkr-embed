#!/usr/bin/env python3
"""
Final Integration Test for Synchronization Point 2
Focus on core functionality without network dependencies
"""

import asyncio
import sys
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FinalIntegrationTest")

async def test_app_creation():
    """Test FastAPI app creation and configuration"""
    try:
        from tkr_embed.api.server import app
        from tkr_embed.config import get_config

        config = get_config()

        # Test app properties
        app_info = {
            "title": app.title,
            "version": app.version,
            "debug": app.debug,
            "middleware_count": len(app.user_middleware)
        }

        # Test routes exist
        routes = [route.path for route in app.routes if hasattr(route, 'path')]
        required_routes = ['/health', '/info', '/generate', '/chat', '/stream']

        missing_routes = [route for route in required_routes if route not in routes]

        logger.info(f"App created: {app_info}")
        logger.info(f"Available routes: {routes}")

        if missing_routes:
            logger.error(f"Missing routes: {missing_routes}")
            return False

        return True

    except Exception as e:
        logger.error(f"App creation test failed: {e}")
        return False

async def test_model_manager_creation():
    """Test model manager creation and basic functionality"""
    try:
        from tkr_embed.core.model_manager import GPTOss20bMLX
        from tkr_embed.core.model_manager import GenerationConfig, ReasoningLevel

        # Create model manager (don't load actual model)
        manager = GPTOss20bMLX(model_path="test/model", quantization="q8")

        # Test basic properties
        model_info = manager.get_model_info()
        memory_usage = manager.get_memory_usage()
        is_ready = manager.is_ready()

        # Test config creation
        config = GenerationConfig(
            max_tokens=512,
            temperature=0.7,
            reasoning_level=ReasoningLevel.MEDIUM
        )

        logger.info(f"Model manager created: {manager.model_path}")
        logger.info(f"Memory usage: {memory_usage:.2f}GB")
        logger.info(f"Is ready: {is_ready}")
        logger.info(f"Config created: {config.max_tokens} tokens, {config.reasoning_level}")

        return True

    except Exception as e:
        logger.error(f"Model manager test failed: {e}")
        return False

async def test_memory_manager():
    """Test memory manager functionality"""
    try:
        from tkr_embed.utils.memory_manager import MemoryManager

        mem_manager = MemoryManager()
        mem_manager.optimize_for_generation()

        stats = mem_manager.get_memory_stats()
        available = mem_manager.get_available_memory()
        pressure = mem_manager.check_memory_pressure()

        logger.info(f"Memory manager tier: {stats['profile_tier']}")
        logger.info(f"Available memory: {available:.2f}GB")
        logger.info(f"Memory pressure: {pressure}")

        return True

    except Exception as e:
        logger.error(f"Memory manager test failed: {e}")
        return False

async def test_cache_functionality():
    """Test cache functionality"""
    try:
        from tkr_embed.utils.lru_cache import GenerationCache

        cache = GenerationCache(max_size=50, ttl=300)

        # Test cache operations
        test_config = {
            "max_tokens": 512,
            "temperature": 0.7,
            "reasoning_level": "medium"
        }

        # Put and get
        cache.put("test prompt", test_config, "test result", 100, 1.5)
        result = cache.get("test prompt", test_config)

        # Test stats
        stats = cache.get_stats()

        logger.info(f"Cache test: stored and retrieved '{result}'")
        logger.info(f"Cache stats: {stats['hit_rate']:.2%} hit rate, {stats['size']} entries")

        return result == "test result"

    except Exception as e:
        logger.error(f"Cache test failed: {e}")
        return False

async def test_error_handling():
    """Test error handling functionality"""
    try:
        from tkr_embed.api.error_handlers import (
            ModelNotReadyError, TokenLimitExceededError,
            validate_reasoning_level, validate_token_limits
        )

        # Test error creation
        try:
            raise ModelNotReadyError("Test error")
        except ModelNotReadyError as e:
            logger.info(f"ModelNotReadyError created: {e.error_code}")

        # Test validation
        level = validate_reasoning_level("high")
        tokens = validate_token_limits(2048)

        logger.info(f"Validation tests: level={level}, tokens={tokens}")

        return level == "high" and tokens == 2048

    except Exception as e:
        logger.error(f"Error handling test failed: {e}")
        return False

async def test_config_integration():
    """Test configuration integration"""
    try:
        from tkr_embed.config import get_config

        config = get_config()

        # Test config sections
        sections = ['model', 'generation', 'security', 'cache']
        for section in sections:
            if not hasattr(config, section):
                logger.error(f"Missing config section: {section}")
                return False

        logger.info(f"Config environment: {config.environment.value}")
        logger.info(f"Model path: {config.model.model_path}")
        logger.info(f"Quantization: {config.model.quantization}")

        return True

    except Exception as e:
        logger.error(f"Config integration test failed: {e}")
        return False

async def run_final_integration_tests():
    """Run all final integration tests"""
    logger.info("=" * 80)
    logger.info("FINAL INTEGRATION TESTS - SYNCHRONIZATION POINT 2")
    logger.info("=" * 80)

    tests = [
        ("App Creation", test_app_creation),
        ("Model Manager", test_model_manager_creation),
        ("Memory Manager", test_memory_manager),
        ("Cache Functionality", test_cache_functionality),
        ("Error Handling", test_error_handling),
        ("Config Integration", test_config_integration)
    ]

    results = {}
    passed = 0
    failed = 0

    for test_name, test_func in tests:
        logger.info(f"\nRunning: {test_name}")
        try:
            result = await test_func()
            results[test_name] = result
            if result:
                passed += 1
                logger.info(f"✅ {test_name} PASSED")
            else:
                failed += 1
                logger.error(f"❌ {test_name} FAILED")
        except Exception as e:
            failed += 1
            results[test_name] = False
            logger.error(f"❌ {test_name} FAILED with exception: {e}")

    # Final report
    total = passed + failed
    success_rate = (passed / total * 100) if total > 0 else 0

    logger.info("\n" + "=" * 80)
    logger.info("FINAL INTEGRATION TEST RESULTS")
    logger.info("=" * 80)
    logger.info(f"Total Tests: {total}")
    logger.info(f"Passed: {passed} ✅")
    logger.info(f"Failed: {failed} ❌")
    logger.info(f"Success Rate: {success_rate:.1f}%")

    if success_rate >= 90:
        logger.info("\n🎉 EXCELLENT! All core systems operational.")
        logger.info("✅ SYSTEM READY FOR PHASE 3")
    elif success_rate >= 75:
        logger.info("\n👍 GOOD! Most systems operational.")
        logger.info("⚠️  Minor issues detected - review failed tests")
    else:
        logger.info("\n⚠️  CRITICAL ISSUES DETECTED")
        logger.info("❌ RESOLVE ISSUES BEFORE PHASE 3")

    logger.info("\n" + "=" * 80)

    return results

if __name__ == "__main__":
    asyncio.run(run_final_integration_tests())