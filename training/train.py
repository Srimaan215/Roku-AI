"""
Train LoRA adapters for Roku

This script fine-tunes Llama 3.2 3B with LoRA to give Roku
its personality and domain-specific knowledge.

Usage:
    python train.py --domain personality  # Train base personality
    python train.py --domain work          # Train work adapter
    python train.py --domain all           # Train all adapters
"""
import argparse
import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


# Paths
BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
DATA_DIR = Path.home() / "Roku/roku-ai/training/data"
OUTPUT_DIR = Path.home() / "Roku/roku-ai/models/adapters"


def format_prompt(example):
    """Format training example for Llama 3.2 Instruct"""
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{example['instruction']}<|eot_id|><|start_header_id|>user<|end_header_id|>

{example['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{example['output']}<|eot_id|>"""


def train_adapter(
    domain: str,
    epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    use_4bit: bool = True,
):
    """
    Train a LoRA adapter for a specific domain
    
    Args:
        domain: Domain name (personality, work, home, health, personal)
        epochs: Number of training epochs
        batch_size: Batch size (reduce if OOM)
        learning_rate: Learning rate
        lora_r: LoRA rank (higher = more expressive, larger adapter)
        lora_alpha: LoRA alpha scaling
        use_4bit: Use 4-bit quantization (saves memory)
    """
    print(f"\n{'='*60}")
    print(f"Training {domain.upper()} adapter")
    print(f"{'='*60}")
    
    # Check for MPS (Apple Silicon) or CUDA
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using Apple Silicon (MPS)")
    elif torch.cuda.is_available():
        device = "cuda"
        print(f"Using CUDA: {torch.cuda.get_device_name()}")
    else:
        device = "cpu"
        print("Using CPU (this will be slow)")
    
    # Load training data
    data_file = DATA_DIR / f"{domain}_training.jsonl"
    if not data_file.exists():
        raise FileNotFoundError(f"Training data not found: {data_file}")
    
    dataset = load_dataset("json", data_files=str(data_file))["train"]
    print(f"Loaded {len(dataset)} training examples")
    
    # Load tokenizer
    print(f"\nLoading tokenizer from {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load model with optional quantization
    print(f"Loading model...")
    
    model_kwargs = {
        "torch_dtype": torch.float16 if device != "cpu" else torch.float32,
        "device_map": "auto" if device == "cuda" else None,
    }
    
    # Note: 4-bit quantization requires CUDA, not supported on MPS yet
    if use_4bit and device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_config
        print("Using 4-bit quantization")
    
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **model_kwargs)
    
    if device == "mps":
        model = model.to(device)
    
    # Prepare for training
    if use_4bit and device == "cuda":
        model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Output path
    output_path = OUTPUT_DIR / f"{domain}_lora"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Training config
    training_args = SFTConfig(
        output_dir=str(output_path),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=(device == "cuda"),
        bf16=False,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none",  # Disable wandb/tensorboard
        dataset_text_field="text",
    )
    
    # Format dataset
    def format_dataset(example):
        return {"text": format_prompt(example)}
    
    formatted_dataset = dataset.map(format_dataset)
    
    # Train
    trainer = SFTTrainer(
        model=model,
        train_dataset=formatted_dataset,
        args=training_args,
        processing_class=tokenizer,
    )
    
    print("\nStarting training...")
    trainer.train()
    
    # Save adapter
    print(f"\nSaving adapter to {output_path}")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    print(f"\n✅ {domain.upper()} adapter trained and saved!")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Train Roku LoRA adapters")
    parser.add_argument(
        "--domain",
        type=str,
        default="personality",
        choices=["personality", "work", "home", "health", "personal", "combined", "all"],
        help="Domain to train adapter for",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    
    args = parser.parse_args()
    
    domains = ["personality", "work", "home", "health", "personal"] if args.domain == "all" else [args.domain]
    
    for domain in domains:
        train_adapter(
            domain=domain,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            lora_r=args.lora_r,
            use_4bit=not args.no_4bit,
        )
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    print(f"\nAdapters saved to: {OUTPUT_DIR}")
    print("\nTo use with llama.cpp, you'll need to merge and convert to GGUF.")
    print("Or use with Hugging Face transformers directly.")


if __name__ == "__main__":
    main()
