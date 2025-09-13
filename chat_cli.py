#!/usr/bin/env python3
"""
Simple Chat CLI for GPT-OSS-20B Text Generation Service
=======================================================

A command-line interface for interacting with the tkr-embed GPT-OSS-20B
text generation service. Supports both single generation and interactive chat.

Usage:
    python chat_cli.py                          # Interactive chat mode
    python chat_cli.py "Your question here"     # Single generation
    python chat_cli.py --stream "Your prompt"   # Streaming generation
    python chat_cli.py --reasoning high "Complex question"  # High reasoning
"""

import asyncio
import json
import sys
import argparse
from typing import List, Dict, Optional
import aiohttp
import time

# Configuration
DEFAULT_BASE_URL = "http://localhost:8008"
DEFAULT_REASONING_LEVEL = "medium"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 2048


class ChatCLI:
    """Simple CLI client for GPT-OSS-20B text generation service."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        api_key: Optional[str] = None
    ):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.conversation_history: List[Dict[str, str]] = []

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with optional API key."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def check_server_health(self) -> bool:
        """Check if the server is running and healthy."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/health") as response:
                    if response.status == 200:
                        data = await response.json()
                        print(f"🟢 Server is healthy - Status: {data.get('status', 'unknown')}")
                        if not data.get('model_loaded', False):
                            print("⚠️  Model not loaded - responses will be mock/placeholder")
                        return True
                    else:
                        print(f"🔴 Server returned status {response.status}")
                        return False
        except Exception as e:
            print(f"🔴 Cannot connect to server: {e}")
            print(f"   Make sure the server is running at {self.base_url}")
            return False

    async def generate_text(
        self,
        prompt: str,
        reasoning_level: str = DEFAULT_REASONING_LEVEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS
    ) -> Optional[str]:
        """Generate text using the /generate endpoint."""
        payload = {
            "text": prompt,
            "reasoning_level": reasoning_level,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/generate",
                    headers=self._get_headers(),
                    json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('generated_text', '')
                    else:
                        error_text = await response.text()
                        print(f"🔴 Generation failed ({response.status}): {error_text}")
                        return None
        except Exception as e:
            print(f"🔴 Request failed: {e}")
            return None

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        reasoning_level: str = DEFAULT_REASONING_LEVEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS
    ) -> Optional[str]:
        """Generate chat response using the /chat endpoint."""
        payload = {
            "messages": messages,
            "reasoning_level": reasoning_level,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat",
                    headers=self._get_headers(),
                    json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('response', data.get('generated_text', ''))
                    else:
                        error_text = await response.text()
                        print(f"🔴 Chat failed ({response.status}): {error_text}")
                        return None
        except Exception as e:
            print(f"🔴 Chat request failed: {e}")
            return None

    async def stream_generation(
        self,
        prompt: str,
        reasoning_level: str = DEFAULT_REASONING_LEVEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS
    ) -> None:
        """Stream text generation using the /stream endpoint."""
        payload = {
            "text": prompt,
            "reasoning_level": reasoning_level,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "streaming": True
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/stream",
                    headers=self._get_headers(),
                    json=payload
                ) as response:
                    if response.status == 200:
                        print("🤖 Assistant (streaming): ", end="", flush=True)
                        async for line in response.content:
                            line = line.decode('utf-8').strip()
                            if line.startswith('data: '):
                                data_str = line[6:]  # Remove 'data: ' prefix
                                if data_str == '[DONE]':
                                    break
                                try:
                                    data = json.loads(data_str)
                                    if 'chunk' in data:
                                        print(data['chunk'], end="", flush=True)
                                except json.JSONDecodeError:
                                    continue
                        print()  # New line after streaming
                    else:
                        error_text = await response.text()
                        print(f"🔴 Streaming failed ({response.status}): {error_text}")
        except Exception as e:
            print(f"🔴 Streaming request failed: {e}")

    async def interactive_chat(self):
        """Run interactive chat mode."""
        print("🚀 GPT-OSS-20B Chat CLI")
        print("=" * 50)
        print("Commands:")
        print("  /quit, /exit, /q     - Exit chat")
        print("  /clear               - Clear conversation history")
        print("  /reasoning <level>   - Set reasoning level (low/medium/high)")
        print("  /temp <value>        - Set temperature (0.0-1.0)")
        print("  /stream <on|off>     - Toggle streaming mode")
        print("  /help                - Show this help")
        print()

        # Check server health
        if not await self.check_server_health():
            print("Cannot proceed without server connection.")
            return

        print("Type your message and press Enter. Use /quit to exit.")
        print("-" * 50)

        # Chat settings
        reasoning_level = DEFAULT_REASONING_LEVEL
        temperature = DEFAULT_TEMPERATURE
        streaming = False

        while True:
            try:
                # Get user input
                user_input = input("👤 You: ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith('/'):
                    command_parts = user_input.split()
                    command = command_parts[0].lower()

                    if command in ['/quit', '/exit', '/q']:
                        print("👋 Goodbye!")
                        break
                    elif command == '/clear':
                        self.conversation_history.clear()
                        print("🗑️  Conversation history cleared.")
                        continue
                    elif command == '/reasoning':
                        if len(command_parts) > 1:
                            new_level = command_parts[1].lower()
                            if new_level in ['low', 'medium', 'high']:
                                reasoning_level = new_level
                                print(f"🧠 Reasoning level set to: {reasoning_level}")
                            else:
                                print("❌ Invalid reasoning level. Use: low, medium, or high")
                        else:
                            print(f"🧠 Current reasoning level: {reasoning_level}")
                        continue
                    elif command == '/temp':
                        if len(command_parts) > 1:
                            try:
                                new_temp = float(command_parts[1])
                                if 0.0 <= new_temp <= 1.0:
                                    temperature = new_temp
                                    print(f"🌡️  Temperature set to: {temperature}")
                                else:
                                    print("❌ Temperature must be between 0.0 and 1.0")
                            except ValueError:
                                print("❌ Invalid temperature value")
                        else:
                            print(f"🌡️  Current temperature: {temperature}")
                        continue
                    elif command == '/stream':
                        if len(command_parts) > 1:
                            setting = command_parts[1].lower()
                            if setting in ['on', 'true', 'yes']:
                                streaming = True
                                print("🌊 Streaming mode enabled")
                            elif setting in ['off', 'false', 'no']:
                                streaming = False
                                print("📝 Streaming mode disabled")
                            else:
                                print("❌ Use /stream on or /stream off")
                        else:
                            print(f"🌊 Streaming mode: {'on' if streaming else 'off'}")
                        continue
                    elif command == '/help':
                        print("\nCommands:")
                        print("  /quit, /exit, /q     - Exit chat")
                        print("  /clear               - Clear conversation history")
                        print("  /reasoning <level>   - Set reasoning level (low/medium/high)")
                        print("  /temp <value>        - Set temperature (0.0-1.0)")
                        print("  /stream <on|off>     - Toggle streaming mode")
                        print("  /help                - Show this help")
                        print()
                        continue
                    else:
                        print(f"❌ Unknown command: {command}")
                        continue

                # Add user message to conversation history
                self.conversation_history.append({"role": "user", "content": user_input})

                # Generate response
                start_time = time.time()

                if streaming:
                    await self.stream_generation(
                        user_input,
                        reasoning_level=reasoning_level,
                        temperature=temperature
                    )
                    response = "Generated via streaming"  # Placeholder for history
                else:
                    if len(self.conversation_history) > 1:
                        # Use chat endpoint for conversation context
                        response = await self.chat_completion(
                            self.conversation_history,
                            reasoning_level=reasoning_level,
                            temperature=temperature
                        )
                    else:
                        # Use generate endpoint for single prompt
                        response = await self.generate_text(
                            user_input,
                            reasoning_level=reasoning_level,
                            temperature=temperature
                        )

                    if response:
                        end_time = time.time()
                        response_time = end_time - start_time
                        print(f"🤖 Assistant: {response}")
                        print(f"⏱️  Response time: {response_time:.2f}s | Reasoning: {reasoning_level}")
                    else:
                        print("❌ Failed to generate response")
                        continue

                # Add assistant response to conversation history
                if response and not streaming:
                    self.conversation_history.append({"role": "assistant", "content": response})

                # Keep conversation history manageable (last 10 exchanges)
                if len(self.conversation_history) > 20:
                    self.conversation_history = self.conversation_history[-20:]

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except EOFError:
                print("\n👋 Goodbye!")
                break


async def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Simple Chat CLI for GPT-OSS-20B Text Generation Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python chat_cli.py                              # Interactive chat
  python chat_cli.py "What is machine learning?"  # Single generation
  python chat_cli.py --stream "Tell me a story"   # Streaming generation
  python chat_cli.py --reasoning high "Explain quantum computing"
        """
    )

    parser.add_argument(
        'prompt',
        nargs='?',
        help='Single prompt for text generation (omit for interactive mode)'
    )
    parser.add_argument(
        '--url',
        default=DEFAULT_BASE_URL,
        help=f'Base URL of the generation service (default: {DEFAULT_BASE_URL})'
    )
    parser.add_argument(
        '--api-key',
        help='API key for authentication (if required)'
    )
    parser.add_argument(
        '--reasoning',
        choices=['low', 'medium', 'high'],
        default=DEFAULT_REASONING_LEVEL,
        help=f'Reasoning level (default: {DEFAULT_REASONING_LEVEL})'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f'Generation temperature 0.0-1.0 (default: {DEFAULT_TEMPERATURE})'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f'Maximum tokens to generate (default: {DEFAULT_MAX_TOKENS})'
    )
    parser.add_argument(
        '--stream',
        action='store_true',
        help='Use streaming generation'
    )

    args = parser.parse_args()

    # Validate temperature
    if not 0.0 <= args.temperature <= 1.0:
        print("❌ Temperature must be between 0.0 and 1.0")
        sys.exit(1)

    # Create CLI client
    cli = ChatCLI(base_url=args.url, api_key=args.api_key)

    # Check server health first
    if not await cli.check_server_health():
        sys.exit(1)

    if args.prompt:
        # Single generation mode
        print(f"🧠 Reasoning: {args.reasoning} | 🌡️  Temperature: {args.temperature}")
        print(f"👤 Prompt: {args.prompt}")
        print()

        start_time = time.time()

        if args.stream:
            await cli.stream_generation(
                args.prompt,
                reasoning_level=args.reasoning,
                temperature=args.temperature,
                max_tokens=args.max_tokens
            )
        else:
            response = await cli.generate_text(
                args.prompt,
                reasoning_level=args.reasoning,
                temperature=args.temperature,
                max_tokens=args.max_tokens
            )

            if response:
                end_time = time.time()
                response_time = end_time - start_time
                print(f"🤖 Assistant: {response}")
                print(f"\n⏱️  Response time: {response_time:.2f}s")
            else:
                print("❌ Failed to generate response")
                sys.exit(1)
    else:
        # Interactive chat mode
        await cli.interactive_chat()


if __name__ == "__main__":
    asyncio.run(main())