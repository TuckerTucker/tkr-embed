#!/usr/bin/env python3
"""
Chat Completion Examples for tkr-embed GPT-OSS-20B API

This module demonstrates conversational AI patterns including multi-turn
conversations, system prompts, and context management.
"""

import asyncio
import httpx
import json
from typing import Optional, Dict, Any, List


class TkrEmbedChatClient:
    """Client for tkr-embed chat completion API"""

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

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        reasoning_level: str = "medium",
        top_p: float = 0.9
    ) -> Dict[str, Any]:
        """Generate chat completion"""

        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "reasoning_level": reasoning_level,
            "top_p": top_p
        }

        if system_prompt:
            payload["system_prompt"] = system_prompt

        response = await self.client.post(
            f"{self.base_url}/chat",
            headers=self._get_headers(),
            json=payload
        )
        response.raise_for_status()
        return response.json()

    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()


class ConversationManager:
    """Manages multi-turn conversations with the API"""

    def __init__(self, client: TkrEmbedChatClient, system_prompt: Optional[str] = None):
        self.client = client
        self.system_prompt = system_prompt
        self.messages: List[Dict[str, str]] = []

    def add_message(self, role: str, content: str):
        """Add a message to the conversation"""
        self.messages.append({"role": role, "content": content})

    async def send_message(
        self,
        content: str,
        reasoning_level: str = "medium",
        temperature: float = 0.7
    ) -> str:
        """Send a user message and get assistant response"""

        # Add user message
        self.add_message("user", content)

        # Get response
        result = await self.client.chat_completion(
            messages=self.messages,
            system_prompt=self.system_prompt,
            reasoning_level=reasoning_level,
            temperature=temperature
        )

        # Add assistant response to conversation
        assistant_response = result["response"]
        self.add_message("assistant", assistant_response)

        return assistant_response

    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get conversation statistics"""
        user_messages = [msg for msg in self.messages if msg["role"] == "user"]
        assistant_messages = [msg for msg in self.messages if msg["role"] == "assistant"]

        return {
            "total_messages": len(self.messages),
            "user_messages": len(user_messages),
            "assistant_messages": len(assistant_messages),
            "conversation_length": sum(len(msg["content"]) for msg in self.messages)
        }


async def demonstrate_basic_chat():
    """Demonstrate basic chat functionality"""

    client = TkrEmbedChatClient(api_key="your-api-key-here")

    try:
        print("💬 BASIC CHAT DEMONSTRATION")
        print("-" * 50)

        messages = [
            {"role": "user", "content": "Hello! Can you help me understand machine learning?"}
        ]

        result = await client.chat_completion(messages=messages)

        print("User: Hello! Can you help me understand machine learning?")
        print(f"Assistant: {result['response']}")
        print(f"Conversation ID: {result['conversation_id']}")
        print(f"Processing time: {result['processing_time']:.2f}s")

    finally:
        await client.close()


async def demonstrate_system_prompts():
    """Demonstrate different system prompts"""

    client = TkrEmbedChatClient(api_key="your-api-key-here")

    system_prompts = [
        {
            "name": "Technical Expert",
            "prompt": "You are a senior software engineer with expertise in distributed systems, machine learning, and cloud architecture. Provide detailed, technical responses with code examples when appropriate.",
            "user_message": "How would you design a scalable API for real-time chat?"
        },
        {
            "name": "Creative Writer",
            "prompt": "You are a creative writing assistant. Help users with storytelling, character development, and narrative techniques. Be imaginative and inspiring.",
            "user_message": "Help me create a compelling character for my sci-fi novel."
        },
        {
            "name": "Business Consultant",
            "prompt": "You are a business strategy consultant with expertise in startup growth, market analysis, and operational efficiency. Provide actionable insights and data-driven recommendations.",
            "user_message": "What metrics should a SaaS startup track in its first year?"
        },
        {
            "name": "Educational Tutor",
            "prompt": "You are a patient and encouraging tutor. Break down complex concepts into simple, understandable steps. Use analogies and examples to make learning enjoyable.",
            "user_message": "Explain quantum entanglement in simple terms."
        }
    ]

    try:
        print("🎭 SYSTEM PROMPT DEMONSTRATIONS")
        print("-" * 50)

        for config in system_prompts:
            print(f"\n📋 {config['name'].upper()}")
            print(f"System Prompt: {config['prompt'][:100]}...")
            print(f"User: {config['user_message']}")

            messages = [{"role": "user", "content": config['user_message']}]

            result = await client.chat_completion(
                messages=messages,
                system_prompt=config['prompt'],
                reasoning_level="medium",
                temperature=0.7
            )

            print(f"Assistant: {result['response'][:200]}...")
            print(f"Full response length: {len(result['response'])} characters")
            print()

    finally:
        await client.close()


async def demonstrate_multi_turn_conversation():
    """Demonstrate multi-turn conversation management"""

    client = TkrEmbedChatClient(api_key="your-api-key-here")

    system_prompt = "You are a helpful programming mentor. Guide users through coding problems step by step."
    conversation = ConversationManager(client, system_prompt)

    try:
        print("🔄 MULTI-TURN CONVERSATION DEMONSTRATION")
        print("-" * 50)

        # Conversation flow
        conversation_flow = [
            "I'm learning Python. Can you help me understand functions?",
            "Can you show me an example of a function that calculates the area of a circle?",
            "Great! Now how would I modify it to handle invalid input?",
            "What about using type hints to make it more professional?",
            "Perfect! Can you explain why type hints are important?"
        ]

        for i, user_message in enumerate(conversation_flow, 1):
            print(f"\n--- Turn {i} ---")
            print(f"User: {user_message}")

            response = await conversation.send_message(
                user_message,
                reasoning_level="medium" if i <= 3 else "high",
                temperature=0.6
            )

            print(f"Assistant: {response}")

        # Show conversation summary
        print("\n--- CONVERSATION SUMMARY ---")
        summary = conversation.get_conversation_summary()
        print(json.dumps(summary, indent=2))

    finally:
        await client.close()


async def demonstrate_reasoning_in_chat():
    """Demonstrate how reasoning levels affect chat responses"""

    client = TkrEmbedChatClient(api_key="your-api-key-here")

    try:
        print("🧠 REASONING LEVELS IN CHAT")
        print("-" * 50)

        base_conversation = [
            {"role": "user", "content": "I'm building a web application and need to choose between React and Vue.js. What should I consider?"}
        ]

        reasoning_levels = ["low", "medium", "high"]

        for level in reasoning_levels:
            print(f"\n🔸 {level.upper()} REASONING")

            result = await client.chat_completion(
                messages=base_conversation.copy(),
                reasoning_level=level,
                temperature=0.7,
                max_tokens=300
            )

            print(f"Response: {result['response']}")
            print(f"Processing time: {result['processing_time']:.2f}s")
            print(f"Tokens used: {result['tokens_used']}")

    finally:
        await client.close()


async def demonstrate_conversation_patterns():
    """Demonstrate different conversation patterns"""

    client = TkrEmbedChatClient(api_key="your-api-key-here")

    patterns = [
        {
            "name": "Q&A Session",
            "system_prompt": "You are a knowledgeable assistant providing concise, accurate answers.",
            "messages": [
                {"role": "user", "content": "What is Docker?"},
                {"role": "assistant", "content": "Docker is a containerization platform..."},
                {"role": "user", "content": "How is it different from virtual machines?"}
            ]
        },
        {
            "name": "Brainstorming",
            "system_prompt": "You are a creative brainstorming partner. Generate diverse, innovative ideas.",
            "messages": [
                {"role": "user", "content": "I need ideas for a mobile app that helps people learn languages."},
                {"role": "assistant", "content": "Here are some creative language learning app ideas..."},
                {"role": "user", "content": "I like the gamification approach. What specific game mechanics could work?"}
            ]
        },
        {
            "name": "Problem Solving",
            "system_prompt": "You are a systematic problem solver. Break down complex issues into manageable steps.",
            "messages": [
                {"role": "user", "content": "My website is loading slowly. How do I diagnose the issue?"},
                {"role": "assistant", "content": "Let's approach this systematically..."},
                {"role": "user", "content": "I checked the network tab and see some large image files. What's the best way to optimize them?"}
            ]
        }
    ]

    try:
        print("🎪 CONVERSATION PATTERNS")
        print("-" * 50)

        for pattern in patterns:
            print(f"\n📌 {pattern['name'].upper()}")
            print(f"Context: {pattern['system_prompt'][:50]}...")

            result = await client.chat_completion(
                messages=pattern['messages'],
                system_prompt=pattern['system_prompt'],
                reasoning_level="medium",
                temperature=0.8
            )

            print("Conversation:")
            for msg in pattern['messages']:
                speaker = "User" if msg['role'] == 'user' else "Assistant"
                print(f"  {speaker}: {msg['content'][:60]}...")

            print(f"Final Response: {result['response'][:150]}...")
            print()

    finally:
        await client.close()


if __name__ == "__main__":
    print("🚀 tkr-embed Chat Completion Examples")
    print("=" * 50)

    # Run demonstrations
    asyncio.run(demonstrate_basic_chat())
    print("\n" + "=" * 50)

    asyncio.run(demonstrate_system_prompts())
    print("\n" + "=" * 50)

    asyncio.run(demonstrate_multi_turn_conversation())
    print("\n" + "=" * 50)

    asyncio.run(demonstrate_reasoning_in_chat())
    print("\n" + "=" * 50)

    asyncio.run(demonstrate_conversation_patterns())

    print("\n✅ All chat examples completed!")
    print("\nNext steps:")
    print("1. Set your actual API key in the examples")
    print("2. Ensure the server is running on http://localhost:8008")
    print("3. Experiment with different system prompts and conversation flows")
    print("4. Try building a complete chat application using ConversationManager")