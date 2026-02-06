"""
Simple DeepSeek-R1 14B test (no adapters)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from core.multi_lora import MultiLoRALlama

print("="*60)
print("Loading DeepSeek-R1 14B (this may take a minute...)")
print("="*60)

llm = MultiLoRALlama(verbose=True)

print("\n" + "="*60)
print("Testing base model (no adapters)")
print("="*60)

messages = [
    {"role": "system", "content": "You are Roku, a helpful AI assistant."},
    {"role": "user", "content": "Hello! What can you help me with?"}
]

print("\nGenerating response...")
response = llm.chat(messages, max_tokens=100)

print(f"\nResponse: {response}")
print("\n" + "="*60)
print("✅ DeepSeek-R1 14B working!")
print("="*60)
