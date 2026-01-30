"""
Merge LoRA adapter with base model and convert to GGUF for llama.cpp

This script:
1. Loads the base Llama 3.2 3B model
2. Merges the trained LoRA adapter
3. Saves the merged model
4. Converts to GGUF format (requires llama.cpp)
"""
import argparse
import os
import shutil
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


# Paths
BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
ADAPTERS_DIR = Path.home() / "Roku/roku-ai/models/adapters"
MERGED_DIR = Path.home() / "Roku/roku-ai/models/merged"
GGUF_DIR = Path.home() / "Roku/roku-ai/models/gguf"


def merge_adapter(adapter_name: str) -> Path:
    """
    Merge LoRA adapter with base model
    
    Args:
        adapter_name: Name of adapter (e.g., 'personality', 'work')
        
    Returns:
        Path to merged model
    """
    adapter_path = ADAPTERS_DIR / f"{adapter_name}_lora"
    
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_path}")
    
    print(f"\n{'='*60}")
    print(f"Merging {adapter_name} adapter with base model")
    print(f"{'='*60}")
    
    # Load base model without device_map to avoid offloading issues
    print(f"\nLoading base model: {BASE_MODEL}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    
    # Load and merge LoRA adapter
    print(f"Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=False)
    
    print("Merging weights...")
    merged_model = model.merge_and_unload()
    
    # Save merged model
    output_path = MERGED_DIR / f"roku-{adapter_name}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving merged model to: {output_path}")
    merged_model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)
    
    print(f"✅ Merged model saved!")
    
    return output_path


def convert_to_gguf(model_path: Path, quantization: str = "q4_k_m") -> Path:
    """
    Convert merged model to GGUF format
    
    Note: Requires llama.cpp to be installed
    
    Args:
        model_path: Path to merged HuggingFace model
        quantization: Quantization type (q4_k_m, q5_k_m, q8_0, f16)
        
    Returns:
        Path to GGUF file
    """
    print(f"\n{'='*60}")
    print(f"Converting to GGUF ({quantization})")
    print(f"{'='*60}")
    
    # Check for llama.cpp convert script
    llama_cpp_path = Path.home() / "llama.cpp"
    convert_script = llama_cpp_path / "convert_hf_to_gguf.py"
    
    if not convert_script.exists():
        print("\n⚠️  llama.cpp not found at ~/llama.cpp")
        print("To convert to GGUF, you need to:")
        print("  1. git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp")
        print("  2. cd ~/llama.cpp && pip install -r requirements.txt")
        print("  3. Re-run this script")
        print(f"\nAlternatively, use the merged model directly with HuggingFace transformers:")
        print(f"  model_path = '{model_path}'")
        return None
    
    # Create output directory
    GGUF_DIR.mkdir(parents=True, exist_ok=True)
    
    model_name = model_path.name
    output_file = GGUF_DIR / f"{model_name}-{quantization}.gguf"
    
    # Run conversion
    import subprocess
    
    print(f"Running conversion...")
    cmd = [
        "python", str(convert_script),
        str(model_path),
        "--outfile", str(output_file),
        "--outtype", quantization,
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Conversion failed: {result.stderr}")
        return None
    
    print(f"✅ GGUF file created: {output_file}")
    
    return output_file


def test_merged_model(model_path: Path):
    """Test the merged model with HuggingFace transformers"""
    print(f"\n{'='*60}")
    print(f"Testing merged model")
    print(f"{'='*60}")
    
    from transformers import pipeline
    
    print(f"Loading model from: {model_path}")
    
    pipe = pipeline(
        "text-generation",
        model=str(model_path),
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    # Test conversation
    messages = [
        {"role": "system", "content": "You are Roku, a sophisticated personal AI assistant inspired by J.A.R.V.I.S. Be warm, witty, and conversational."},
        {"role": "user", "content": "Hello! What's your name and how can you help me?"},
    ]
    
    print("\n--- Test Response ---")
    output = pipe(messages, max_new_tokens=150, do_sample=True, temperature=0.7)
    response = output[0]["generated_text"][-1]["content"]
    print(f"Roku: {response}")
    
    # Second test
    messages.append({"role": "assistant", "content": response})
    messages.append({"role": "user", "content": "I'm feeling a bit stressed about work. Any suggestions?"})
    
    print("\n--- Follow-up Response ---")
    output = pipe(messages, max_new_tokens=200, do_sample=True, temperature=0.7)
    response = output[0]["generated_text"][-1]["content"]
    print(f"Roku: {response}")


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter and convert to GGUF")
    parser.add_argument(
        "--adapter",
        type=str,
        default="personality",
        help="Adapter name to merge (e.g., personality, work)",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="q4_k_m",
        choices=["q4_k_m", "q5_k_m", "q8_0", "f16"],
        help="GGUF quantization type",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test the merged model after merging",
    )
    parser.add_argument(
        "--skip-gguf",
        action="store_true",
        help="Skip GGUF conversion",
    )
    
    args = parser.parse_args()
    
    # Step 1: Merge adapter
    merged_path = merge_adapter(args.adapter)
    
    # Step 2: Convert to GGUF (optional)
    if not args.skip_gguf:
        gguf_path = convert_to_gguf(merged_path, args.quantization)
        if gguf_path:
            print(f"\nTo use with llama.cpp, update your model path to:")
            print(f"  {gguf_path}")
    
    # Step 3: Test (optional)
    if args.test:
        test_merged_model(merged_path)
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    main()
