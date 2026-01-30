#!/usr/bin/env python
"""
Train the Personal LoRA adapter for Roku.
Uses curated personal data from onboarding interview.
"""
import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# Import curated training data
from data.personal_srimaan_curated import get_training_data


def format_for_llama(example):
    """Format example for Llama 3.2 chat format"""
    return {
        "text": f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are Roku, a personal AI assistant for Srimaan. You know him well - he's a Junior at UMass Amherst studying Computational Neuroscience and Biochemistry, works in research labs at UMass and Brown, and is working towards a PhD. Be helpful, casual but detailed, and personalized. You run locally on his device with full privacy.<|eot_id|><|start_header_id|>user<|end_header_id|>

{example['instruction']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{example['output']}<|eot_id|>"""
    }


def train_personal_adapter():
    """Train the personal LoRA adapter"""
    print("=" * 60)
    print("🧑 Training Personal Adapter for Srimaan")
    print("=" * 60)
    
    # Paths
    base_model_id = "meta-llama/Llama-3.2-3B-Instruct"
    output_dir = Path.home() / "Roku/roku-ai/models/adapters/personal"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load training data
    print("\n📚 Loading training data...")
    training_data = get_training_data()
    print(f"   {len(training_data)} training examples")
    
    # Format for training
    formatted_data = [format_for_llama(ex) for ex in training_data]
    dataset = Dataset.from_list(formatted_data)
    
    # Device setup
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\n🖥️  Device: {device}")
    
    # Load tokenizer
    print("\n🔤 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load model (no quantization for MPS)
    print("\n🧠 Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map={"": device},
        trust_remote_code=True,
    )
    
    # LoRA config - same as personality adapter
    print("\n⚙️  Configuring LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Training arguments
    print("\n🏋️ Setting up training...")
    training_args = SFTConfig(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=False,  # Use fp32 for MPS
        save_strategy="epoch",
        logging_steps=10,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none",
        dataset_text_field="text",
    )
    
    # Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        processing_class=tokenizer,
    )
    
    # Train!
    print("\n🚀 Starting training...")
    start_time = datetime.now()
    trainer.train()
    elapsed = datetime.now() - start_time
    print(f"\n⏱️  Training completed in {elapsed}")
    
    # Save the adapter
    print("\n💾 Saving adapter...")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    
    print(f"\n✅ Personal adapter saved to: {output_dir}")
    print("\nNext: Convert to GGUF format for llama.cpp")
    print(f"   python tools/llama.cpp/convert_lora_to_gguf.py {output_dir}")
    
    return output_dir


if __name__ == "__main__":
    train_personal_adapter()
