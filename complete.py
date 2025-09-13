#!/usr/bin/env python3
"""
Text completion client for GPT-OSS-20B server.
Sends prompts for text completion (not chat).
"""

import argparse
import asyncio
import aiohttp
import json
import sys
from typing import Optional


class TextCompletionClient:
    def __init__(self, base_url: str = "http://localhost:8008", api_key: Optional[str] = None):
        self.base_url = base_url
        self.api_key = api_key or "test-api-key"

    async def complete(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7,
                       reasoning_level: str = "medium", repetition_penalty: float = 1.1,
                       top_p: float = 0.9, top_k: int = 50, stream: bool = False):
        """Send text completion request to server."""

        headers = {"X-API-Key": self.api_key}

        payload = {
            "text": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "reasoning_level": reasoning_level,
            "repetition_penalty": repetition_penalty,
            "top_p": top_p,
            "top_k": top_k
        }

        endpoint = f"{self.base_url}/{'stream' if stream else 'generate'}"

        async with aiohttp.ClientSession() as session:
            if stream:
                await self._stream_completion(session, endpoint, headers, payload)
            else:
                await self._single_completion(session, endpoint, headers, payload)

    async def _single_completion(self, session, endpoint, headers, payload):
        """Handle single text completion."""
        try:
            async with session.post(endpoint, json=payload, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    print(data["generated_text"])

                    if args.verbose:
                        print(f"\n---")
                        print(f"Tokens used: {data.get('tokens_used', 'N/A')}")
                        print(f"Time: {data.get('processing_time', 'N/A')}s")
                        print(f"Model: {data.get('model', 'N/A')}")
                else:
                    error = await response.text()
                    print(f"Error: {error}", file=sys.stderr)
                    sys.exit(1)
        except Exception as e:
            print(f"Connection error: {e}", file=sys.stderr)
            sys.exit(1)

    async def _stream_completion(self, session, endpoint, headers, payload):
        """Handle streaming text completion."""
        try:
            async with session.post(endpoint, json=payload, headers=headers) as response:
                if response.status == 200:
                    async for line in response.content:
                        if line:
                            line = line.decode('utf-8').strip()
                            if line.startswith("data: "):
                                data = line[6:]
                                if data == "[DONE]":
                                    print()  # Final newline
                                    break
                                try:
                                    chunk = json.loads(data)
                                    if "text" in chunk:
                                        print(chunk["text"], end="", flush=True)
                                except json.JSONDecodeError:
                                    continue
                else:
                    error = await response.text()
                    print(f"Error: {error}", file=sys.stderr)
                    sys.exit(1)
        except Exception as e:
            print(f"Connection error: {e}", file=sys.stderr)
            sys.exit(1)


async def main():
    parser = argparse.ArgumentParser(description="Text completion client for GPT-OSS-20B")
    parser.add_argument("prompt", help="Text prompt for completion")
    parser.add_argument("--max-tokens", "-m", type=int, default=512,
                       help="Maximum tokens to generate (default: 512)")
    parser.add_argument("--temperature", "-t", type=float, default=0.7,
                       help="Temperature for generation (default: 0.7)")
    parser.add_argument("--repetition-penalty", "-rp", type=float, default=1.1,
                       help="Repetition penalty 1.0-2.0 (default: 1.1, higher = less repetition)")
    parser.add_argument("--top-p", "-p", type=float, default=0.9,
                       help="Nucleus sampling probability (default: 0.9)")
    parser.add_argument("--top-k", "-k", type=int, default=50,
                       help="Top-k sampling parameter (default: 50)")
    parser.add_argument("--reasoning", "-r", choices=["low", "medium", "high"],
                       default="medium", help="Reasoning complexity level")
    parser.add_argument("--stream", "-s", action="store_true",
                       help="Enable streaming output")
    parser.add_argument("--api-key", help="API key for authentication")
    parser.add_argument("--url", "-u", default="http://localhost:8008",
                       help="Server URL (default: http://localhost:8008)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show generation statistics")

    global args
    args = parser.parse_args()

    client = TextCompletionClient(base_url=args.url, api_key=args.api_key)

    await client.complete(
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        reasoning_level=args.reasoning,
        repetition_penalty=args.repetition_penalty,
        top_p=args.top_p,
        top_k=args.top_k,
        stream=args.stream
    )


if __name__ == "__main__":
    asyncio.run(main())