#!/usr/bin/env python3
"""
Streaming Generation Examples for tkr-embed GPT-OSS-20B API

This module demonstrates real-time text generation using Server-Sent Events (SSE)
for building responsive chat interfaces and real-time applications.
"""

import asyncio
import httpx
import json
import time
from typing import Optional, Dict, Any, AsyncIterator
import sseclient


class TkrEmbedStreamingClient:
    """Client for tkr-embed streaming generation API"""

    def __init__(self, base_url: str = "http://localhost:8008", api_key: Optional[str] = None):
        self.base_url = base_url
        self.api_key = api_key

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with API key if available"""
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Cache-Control": "no-cache"
        }
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers

    async def stream_generation(
        self,
        text: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        reasoning_level: str = "medium",
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream text generation with Server-Sent Events"""

        payload = {
            "text": text,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "reasoning_level": reasoning_level,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty
        }

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/stream",
                headers=self._get_headers(),
                json=payload,
                timeout=30.0
            ) as response:
                response.raise_for_status()

                buffer = ""
                async for chunk in response.aiter_text():
                    buffer += chunk

                    while "\n\n" in buffer:
                        line, buffer = buffer.split("\n\n", 1)
                        line = line.strip()

                        if line.startswith("data: "):
                            data = line[6:]  # Remove "data: " prefix

                            if data == "[DONE]":
                                return

                            try:
                                chunk_data = json.loads(data)
                                yield chunk_data
                            except json.JSONDecodeError:
                                # Skip malformed JSON
                                continue


class StreamingChat:
    """Real-time chat interface using streaming generation"""

    def __init__(self, client: TkrEmbedStreamingClient):
        self.client = client
        self.conversation_history = []

    async def send_message_streaming(
        self,
        message: str,
        reasoning_level: str = "medium",
        temperature: float = 0.7,
        callback=None
    ) -> str:
        """Send message and stream the response"""

        print(f"User: {message}")
        print("Assistant: ", end="", flush=True)

        # For streaming, we'll use single text generation
        # In a real chat system, you'd maintain conversation context
        full_response = ""
        start_time = time.time()
        token_count = 0

        try:
            async for chunk in self.client.stream_generation(
                text=message,
                reasoning_level=reasoning_level,
                temperature=temperature,
                max_tokens=300
            ):
                if "chunk" in chunk:
                    delta = chunk["chunk"].get("delta", "")
                    if delta:
                        print(delta, end="", flush=True)
                        full_response += delta
                        token_count += 1

                        # Call callback if provided (for UI updates)
                        if callback:
                            await callback(delta, chunk)

                    # Check if generation is complete
                    finish_reason = chunk["chunk"].get("finish_reason")
                    if finish_reason:
                        break

        except Exception as e:
            print(f"\nStreaming error: {e}")
            return ""

        end_time = time.time()
        processing_time = end_time - start_time

        print()  # New line after response
        print(f"[Tokens: {token_count}, Time: {processing_time:.2f}s, Speed: {token_count/processing_time:.1f} tokens/s]")
        print()

        return full_response


async def demonstrate_basic_streaming():
    """Demonstrate basic streaming functionality"""

    print("🌊 BASIC STREAMING DEMONSTRATION")
    print("-" * 50)

    client = TkrEmbedStreamingClient(api_key="your-api-key-here")

    prompts = [
        "Write a short story about a robot learning to paint.",
        "Explain how neural networks work in simple terms.",
        "Create a Python function to calculate fibonacci numbers with comments."
    ]

    for i, prompt in enumerate(prompts, 1):
        print(f"\n--- Example {i} ---")
        print(f"Prompt: {prompt}")
        print("Response: ", end="", flush=True)

        full_text = ""
        start_time = time.time()

        try:
            async for chunk in client.stream_generation(
                text=prompt,
                reasoning_level="medium",
                temperature=0.7,
                max_tokens=200
            ):
                if "chunk" in chunk:
                    delta = chunk["chunk"].get("delta", "")
                    if delta:
                        print(delta, end="", flush=True)
                        full_text += delta

        except Exception as e:
            print(f"\nError: {e}")

        end_time = time.time()
        print(f"\n[Completed in {end_time - start_time:.2f}s]")
        print()


async def demonstrate_streaming_chat():
    """Demonstrate streaming chat interface"""

    print("💬 STREAMING CHAT DEMONSTRATION")
    print("-" * 50)
    print("Starting interactive chat session...")
    print("(In a real application, this would be a web interface)")
    print()

    client = TkrEmbedStreamingClient(api_key="your-api-key-here")
    chat = StreamingChat(client)

    # Simulated conversation
    messages = [
        "Hello! Can you help me understand machine learning?",
        "What's the difference between supervised and unsupervised learning?",
        "Can you give me a practical example of each?",
        "Thanks! How would I get started learning ML programming?"
    ]

    for message in messages:
        response = await chat.send_message_streaming(
            message,
            reasoning_level="medium",
            temperature=0.7
        )
        await asyncio.sleep(1)  # Pause between messages


async def demonstrate_streaming_with_callbacks():
    """Demonstrate streaming with UI update callbacks"""

    print("📡 STREAMING WITH CALLBACKS DEMONSTRATION")
    print("-" * 50)

    class StreamingStats:
        def __init__(self):
            self.total_tokens = 0
            self.start_time = None
            self.chunks_received = 0
            self.response_buffer = ""

        async def update_callback(self, delta: str, chunk_data: Dict[str, Any]):
            """Callback function for processing stream updates"""

            if self.start_time is None:
                self.start_time = time.time()

            self.chunks_received += 1
            self.response_buffer += delta

            # Calculate streaming stats
            elapsed = time.time() - self.start_time
            self.total_tokens = chunk_data["chunk"].get("tokens_generated", 0)

            # Simulate UI updates (in a real app, this would update the interface)
            if self.chunks_received % 5 == 0:  # Update every 5 chunks
                speed = self.total_tokens / elapsed if elapsed > 0 else 0
                print(f"\n[Live stats: {self.total_tokens} tokens, {speed:.1f} tokens/s]", end="")

    client = TkrEmbedStreamingClient(api_key="your-api-key-here")
    stats = StreamingStats()

    prompt = "Write a detailed explanation of how blockchain technology works, including its key components and benefits."

    print(f"Prompt: {prompt}")
    print("Response: ", end="", flush=True)

    try:
        async for chunk in client.stream_generation(
            text=prompt,
            reasoning_level="high",
            temperature=0.6,
            max_tokens=500
        ):
            if "chunk" in chunk:
                delta = chunk["chunk"].get("delta", "")
                if delta:
                    print(delta, end="", flush=True)
                    await stats.update_callback(delta, chunk)

    except Exception as e:
        print(f"\nError: {e}")

    # Final stats
    final_elapsed = time.time() - stats.start_time
    final_speed = stats.total_tokens / final_elapsed if final_elapsed > 0 else 0

    print(f"\n\nFinal Statistics:")
    print(f"  Total tokens: {stats.total_tokens}")
    print(f"  Total time: {final_elapsed:.2f}s")
    print(f"  Average speed: {final_speed:.1f} tokens/s")
    print(f"  Chunks received: {stats.chunks_received}")


async def demonstrate_streaming_comparison():
    """Compare streaming vs non-streaming performance"""

    print("⚡ STREAMING VS NON-STREAMING COMPARISON")
    print("-" * 50)

    # Regular generation client for comparison
    async def non_streaming_generation(prompt: str) -> Dict[str, Any]:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:8008/generate",
                headers={"Content-Type": "application/json", "X-API-Key": "your-api-key-here"},
                json={
                    "text": prompt,
                    "max_tokens": 300,
                    "temperature": 0.7,
                    "reasoning_level": "medium"
                }
            )
            response.raise_for_status()
            return response.json()

    prompt = "Explain the benefits of microservices architecture and when to use it."

    # Test non-streaming
    print("🐌 Non-streaming generation:")
    start_time = time.time()
    result = await non_streaming_generation(prompt)
    end_time = time.time()

    print(f"Result: {result['generated_text'][:100]}...")
    print(f"Time to first token: {end_time - start_time:.2f}s")
    print(f"Total tokens: {result['tokens_used']}")
    print()

    # Test streaming
    print("🚀 Streaming generation:")
    client = TkrEmbedStreamingClient(api_key="your-api-key-here")

    start_time = time.time()
    first_token_time = None
    token_count = 0

    print("Response: ", end="", flush=True)

    try:
        async for chunk in client.stream_generation(
            text=prompt,
            reasoning_level="medium",
            temperature=0.7,
            max_tokens=300
        ):
            if "chunk" in chunk:
                delta = chunk["chunk"].get("delta", "")
                if delta and first_token_time is None:
                    first_token_time = time.time()

                if delta:
                    print(delta, end="", flush=True)
                    token_count += 1

    except Exception as e:
        print(f"\nError: {e}")

    end_time = time.time()

    print()
    print(f"Time to first token: {first_token_time - start_time:.2f}s" if first_token_time else "N/A")
    print(f"Total time: {end_time - start_time:.2f}s")
    print(f"Total tokens: {token_count}")

    if first_token_time:
        ttft_improvement = ((end_time - start_time) - (first_token_time - start_time)) / (end_time - start_time) * 100
        print(f"Perceived speed improvement: ~{ttft_improvement:.0f}% faster user experience")


async def demonstrate_error_handling_streaming():
    """Demonstrate error handling in streaming"""

    print("🚨 STREAMING ERROR HANDLING")
    print("-" * 50)

    client = TkrEmbedStreamingClient(api_key="invalid-key")

    try:
        print("Testing with invalid API key...")
        async for chunk in client.stream_generation(
            text="Hello world",
            reasoning_level="medium"
        ):
            print(f"Chunk: {chunk}")

    except httpx.HTTPStatusError as e:
        print(f"HTTP Error: {e.response.status_code}")
        print(f"Error details: {e.response.text}")
    except Exception as e:
        print(f"Other error: {type(e).__name__}: {e}")


if __name__ == "__main__":
    print("🚀 tkr-embed Streaming Generation Examples")
    print("=" * 50)

    # Run demonstrations
    asyncio.run(demonstrate_basic_streaming())
    print("\n" + "=" * 50)

    asyncio.run(demonstrate_streaming_chat())
    print("\n" + "=" * 50)

    asyncio.run(demonstrate_streaming_with_callbacks())
    print("\n" + "=" * 50)

    asyncio.run(demonstrate_streaming_comparison())
    print("\n" + "=" * 50)

    asyncio.run(demonstrate_error_handling_streaming())

    print("\n✅ All streaming examples completed!")
    print("\nImplementation Notes:")
    print("1. Streaming provides better user experience with immediate feedback")
    print("2. Use callbacks to update UI in real-time")
    print("3. Handle network errors gracefully")
    print("4. Buffer incomplete JSON chunks properly")
    print("5. Consider implementing retry logic for production applications")
    print("\nNext steps:")
    print("1. Integrate streaming into your web frontend")
    print("2. Add reconnection logic for robust production use")
    print("3. Implement proper cancellation mechanisms")
    print("4. Add progress indicators and loading states")