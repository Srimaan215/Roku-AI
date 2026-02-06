"""Test CoT prompting with Instruct model"""
from llama_cpp import Llama

print('Loading original Instruct model...')
llm = Llama(
    model_path='models/base/Llama-3.2-3B-Instruct-Q4_K_M.gguf',
    n_ctx=2048,
    n_gpu_layers=-1,
    verbose=False
)

# More explicit context - make reasoning path clearer
prompt = """<|start_header_id|>system<|end_header_id|>

You are a helpful assistant. Think step by step before answering.

CURRENT CONTEXT:
- Today is Saturday, January 31, 2026
- Current time: 4:30 PM
- "Tonight" means THIS evening (Saturday evening)
- Calendar for TODAY: No remaining events

When asked about availability, check if there are events in the calendar for that time period.<|eot_id|><|start_header_id|>user<|end_header_id|>

Am I free tonight?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Let me check your calendar for tonight (Saturday evening):
"""

print('Testing refined CoT...')
response = llm(prompt, max_tokens=100, temperature=0.3, stop=['<|eot_id|>'])
print(response['choices'][0]['text'].strip())

