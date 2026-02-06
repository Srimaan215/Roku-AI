"""
Quick test: DeepSeek-R1 14B with LoRA adapters
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from core.personalized_roku import PersonalizedRoku

print("=" * 60)
print("Testing DeepSeek-R1 14B with Roku")
print("=" * 60)
print()

print("Loading DeepSeek-R1 14B with personality adapter...")
roku = PersonalizedRoku(username="Srimaan", verbose=True)

print("\n" + "-" * 60)
print("Test 1: Simple factual question")
print("-" * 60)

response = roku.quick_ask("What's my name?", max_tokens=100)
print(f"\nQ: What's my name?")
print(f"A: {response}")

print("\n" + "-" * 60)
print("Test 2: Context-aware question")
print("-" * 60)

response = roku.quick_ask("What am I studying?", max_tokens=150)
print(f"\nQ: What am I studying?")
print(f"A: {response}")

print("\n" + "-" * 60)
print("Test 3: Personality test")
print("-" * 60)

response = roku.quick_ask("Tell me about yourself.", max_tokens=150)
print(f"\nQ: Tell me about yourself.")
print(f"A: {response}")

print("\n" + "=" * 60)
print("✅ DeepSeek-R1 14B loaded successfully!")
print(f"Active adapters: {roku.llm.active_adapters}")
print(f"Adapter scales: {roku.llm.adapter_info}")
print("=" * 60)
