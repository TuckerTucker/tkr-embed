#!/usr/bin/env python3
"""
Text Generation Examples for tkr-embed GPT-OSS-20B API

This module demonstrates various text generation patterns using different
reasoning levels and parameters.
"""

import asyncio
import httpx
import json
from typing import Optional, Dict, Any


class TkrEmbedClient:
    """Client for tkr-embed generation API"""

    def __init__(self, base_url: str = "http://localhost:8008", api_key: Optional[str] = None):
        self.base_url = base_url
        self.api_key = api_key
        self.client = httpx.AsyncClient()

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with API key if available"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers

    async def generate_text(
        self,
        text: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        reasoning_level: str = "medium",
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1
    ) -> Dict[str, Any]:
        """Generate text completion"""

        payload = {
            "text": text,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "reasoning_level": reasoning_level,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty
        }

        response = await self.client.post(
            f"{self.base_url}/generate",
            headers=self._get_headers(),
            json=payload
        )
        response.raise_for_status()
        return response.json()

    async def health_check(self) -> Dict[str, Any]:
        """Check server health"""
        response = await self.client.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()


async def demonstrate_reasoning_levels():
    """Demonstrate different reasoning levels"""

    client = TkrEmbedClient(api_key="your-api-key-here")

    try:
        # Check if server is ready
        health = await client.health_check()
        print(f"Server Status: {health['status']}")
        print(f"Model Ready: {health['generation_ready']}")
        print("-" * 50)

        prompt = "Explain quantum computing and its potential applications."

        # Low reasoning - quick, concise response
        print("🔸 LOW REASONING LEVEL")
        result = await client.generate_text(
            text=prompt,
            reasoning_level="low",
            max_tokens=150,
            temperature=0.5
        )
        print(f"Response: {result['generated_text']}")
        print(f"Tokens: {result['tokens_used']}, Time: {result['processing_time']:.2f}s")
        print()

        # Medium reasoning - balanced explanation
        print("🔹 MEDIUM REASONING LEVEL")
        result = await client.generate_text(
            text=prompt,
            reasoning_level="medium",
            max_tokens=300,
            temperature=0.7
        )
        print(f"Response: {result['generated_text']}")
        print(f"Tokens: {result['tokens_used']}, Time: {result['processing_time']:.2f}s")
        print()

        # High reasoning - detailed analysis
        print("🔺 HIGH REASONING LEVEL")
        result = await client.generate_text(
            text=prompt,
            reasoning_level="high",
            max_tokens=500,
            temperature=0.8
        )
        print(f"Response: {result['generated_text']}")
        print(f"Tokens: {result['tokens_used']}, Time: {result['processing_time']:.2f}s")

    finally:
        await client.close()


async def demonstrate_temperature_effects():
    """Demonstrate how temperature affects generation"""

    client = TkrEmbedClient(api_key="your-api-key-here")

    try:
        prompt = "Write a creative opening line for a science fiction story."
        temperatures = [0.1, 0.7, 1.2, 1.8]

        print("🌡️  TEMPERATURE EFFECTS DEMONSTRATION")
        print("-" * 50)

        for temp in temperatures:
            print(f"\nTemperature: {temp}")
            result = await client.generate_text(
                text=prompt,
                temperature=temp,
                max_tokens=100,
                reasoning_level="medium"
            )
            print(f"Output: {result['generated_text']}")

    finally:
        await client.close()


async def demonstrate_use_cases():
    """Demonstrate various practical use cases"""

    client = TkrEmbedClient(api_key="your-api-key-here")

    use_cases = [
        {
            "name": "Technical Documentation",
            "prompt": "Write documentation for a REST API endpoint that creates user accounts.",
            "params": {"reasoning_level": "medium", "temperature": 0.3, "max_tokens": 300}
        },
        {
            "name": "Creative Writing",
            "prompt": "Write a short poem about artificial intelligence and human creativity.",
            "params": {"reasoning_level": "high", "temperature": 1.1, "max_tokens": 200}
        },
        {
            "name": "Code Explanation",
            "prompt": "Explain this Python code: def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
            "params": {"reasoning_level": "medium", "temperature": 0.4, "max_tokens": 250}
        },
        {
            "name": "Business Analysis",
            "prompt": "Analyze the potential risks and benefits of implementing AI in customer service.",
            "params": {"reasoning_level": "high", "temperature": 0.6, "max_tokens": 400}
        },
        {
            "name": "Quick Facts",
            "prompt": "What is the capital of Japan?",
            "params": {"reasoning_level": "low", "temperature": 0.1, "max_tokens": 50}
        }
    ]

    try:
        print("🎯 PRACTICAL USE CASES")
        print("-" * 50)

        for case in use_cases:
            print(f"\n📋 {case['name'].upper()}")
            print(f"Prompt: {case['prompt']}")

            result = await client.generate_text(
                text=case['prompt'],
                **case['params']
            )

            print(f"Response: {result['generated_text']}")
            print(f"Stats: {result['tokens_used']} tokens, {result['processing_time']:.2f}s")
            print(f"Settings: {case['params']}")

    finally:
        await client.close()


async def demonstrate_error_handling():
    """Demonstrate proper error handling"""

    client = TkrEmbedClient(api_key="invalid-key")

    try:
        print("🚨 ERROR HANDLING DEMONSTRATION")
        print("-" * 50)

        # Test various error conditions
        error_tests = [
            {
                "name": "Invalid API Key",
                "params": {"text": "Hello world", "temperature": 0.7}
            },
            {
                "name": "Invalid Temperature",
                "params": {"text": "Hello world", "temperature": 3.0}
            },
            {
                "name": "Empty Prompt",
                "params": {"text": "", "temperature": 0.7}
            },
            {
                "name": "Too Many Tokens",
                "params": {"text": "Hello world", "max_tokens": 10000}
            }
        ]

        for test in error_tests:
            print(f"\nTesting: {test['name']}")
            try:
                result = await client.generate_text(**test['params'])
                print(f"Unexpected success: {result}")
            except httpx.HTTPError as e:
                print(f"Expected error: {e.response.status_code if hasattr(e, 'response') else 'Network error'}")
                if hasattr(e, 'response') and e.response.status_code != 500:
                    error_detail = e.response.json()
                    print(f"Error details: {error_detail}")
            except Exception as e:
                print(f"Other error: {type(e).__name__}: {e}")

    finally:
        await client.close()


if __name__ == "__main__":
    print("🚀 tkr-embed Text Generation Examples")
    print("=" * 50)

    # Run demonstrations
    asyncio.run(demonstrate_reasoning_levels())
    print("\n" + "=" * 50)

    asyncio.run(demonstrate_temperature_effects())
    print("\n" + "=" * 50)

    asyncio.run(demonstrate_use_cases())
    print("\n" + "=" * 50)

    asyncio.run(demonstrate_error_handling())

    print("\n✅ All examples completed!")
    print("\nNext steps:")
    print("1. Set your actual API key in the examples")
    print("2. Ensure the server is running on http://localhost:8008")
    print("3. Modify prompts and parameters for your use case")