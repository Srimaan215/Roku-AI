"""
LoRA Adapter Training Pipeline for Roku

This module handles:
1. Training data preparation
2. LoRA fine-tuning using PEFT
3. Exporting adapters to GGUF format for llama.cpp
"""
import json
import os
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class TrainingExample:
    """Single training example for LoRA fine-tuning"""
    instruction: str  # System context/instruction
    input: str        # User input
    output: str       # Expected assistant response
    domain: str       # Domain label (work, home, health, personal)


@dataclass
class DomainDataset:
    """Dataset for a specific domain adapter"""
    domain: str
    examples: List[TrainingExample] = field(default_factory=list)
    
    def add_example(self, instruction: str, input: str, output: str):
        """Add a training example"""
        self.examples.append(TrainingExample(
            instruction=instruction,
            input=input,
            output=output,
            domain=self.domain
        ))
    
    def to_jsonl(self, filepath: Path):
        """Export to JSONL format for training"""
        with open(filepath, 'w') as f:
            for ex in self.examples:
                json.dump({
                    "instruction": ex.instruction,
                    "input": ex.input,
                    "output": ex.output,
                }, f)
                f.write('\n')
        print(f"Exported {len(self.examples)} examples to {filepath}")
    
    @classmethod
    def from_jsonl(cls, filepath: Path, domain: str) -> 'DomainDataset':
        """Load from JSONL file"""
        dataset = cls(domain=domain)
        with open(filepath, 'r') as f:
            for line in f:
                data = json.loads(line)
                dataset.add_example(
                    instruction=data.get("instruction", ""),
                    input=data["input"],
                    output=data["output"]
                )
        return dataset


class AdapterTrainer:
    """
    Train LoRA adapters for domain-specific personalization
    
    Training flow:
    1. Collect conversation data per domain
    2. Format as instruction-input-output triples
    3. Fine-tune LoRA adapter using PEFT
    4. Convert to GGUF for llama.cpp
    """
    
    DOMAINS = ["work", "home", "health", "personal"]
    
    def __init__(
        self,
        base_model: str = "meta-llama/Llama-3.2-3B-Instruct",
        output_dir: Path = None,
        lora_r: int = 8,         # LoRA rank
        lora_alpha: int = 16,    # LoRA alpha (scaling)
        lora_dropout: float = 0.05,
    ):
        """
        Initialize trainer
        
        Args:
            base_model: HuggingFace model ID for base model
            output_dir: Directory to save trained adapters
            lora_r: LoRA rank (lower = smaller adapter, less expressive)
            lora_alpha: LoRA scaling factor
            lora_dropout: Dropout for regularization
        """
        self.base_model = base_model
        self.output_dir = output_dir or Path.home() / "Roku/roku-ai/models/adapters"
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def prepare_training_data(self, domain: str, conversations: List[Dict]) -> DomainDataset:
        """
        Convert raw conversations to training format
        
        Args:
            domain: Domain name (work, home, health, personal)
            conversations: List of {"user": "...", "assistant": "...", "context": "..."}
            
        Returns:
            DomainDataset ready for training
        """
        dataset = DomainDataset(domain=domain)
        
        # Domain-specific system prompts
        domain_prompts = {
            "work": "You are Roku, helping with work tasks. You know the user's colleagues, projects, and professional communication style.",
            "home": "You are Roku, helping with home and family matters. You know the user's smart home setup, family members, and household routines.",
            "health": "You are Roku, helping with health and wellness. You know the user's fitness goals, medications, and health patterns.",
            "personal": "You are Roku, the user's personal assistant. You know their hobbies, friends, preferences, and personal routines.",
        }
        
        instruction = domain_prompts.get(domain, "You are Roku, a helpful personal AI assistant.")
        
        for conv in conversations:
            dataset.add_example(
                instruction=instruction,
                input=conv["user"],
                output=conv["assistant"]
            )
        
        return dataset
    
    def train_adapter(
        self,
        domain: str,
        dataset: DomainDataset,
        epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-4,
    ) -> Path:
        """
        Train LoRA adapter for a domain
        
        Args:
            domain: Domain name
            dataset: Training dataset
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            
        Returns:
            Path to saved adapter
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
            from peft import LoraConfig, get_peft_model, TaskType
            from trl import SFTTrainer
        except ImportError:
            raise ImportError(
                "Training requires: pip install transformers peft trl datasets"
            )
        
        print(f"\n{'='*50}")
        print(f"Training {domain} adapter")
        print(f"Examples: {len(dataset.examples)}")
        print(f"LoRA rank: {self.lora_r}, alpha: {self.lora_alpha}")
        print(f"{'='*50}\n")
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype="auto",
            device_map="auto",
        )
        
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Attention layers
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        # Prepare data
        data_path = self.output_dir / f"{domain}_train.jsonl"
        dataset.to_jsonl(data_path)
        
        from datasets import load_dataset
        train_data = load_dataset("json", data_files=str(data_path))["train"]
        
        # Format for training
        def format_example(example):
            return f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
        
        # Training arguments
        output_path = self.output_dir / f"{domain}_lora"
        training_args = TrainingArguments(
            output_dir=str(output_path),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            logging_steps=10,
            save_strategy="epoch",
            fp16=False,  # Use bf16 on newer GPUs
            optim="adamw_torch",
        )
        
        # Train
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_data,
            formatting_func=format_example,
            args=training_args,
            tokenizer=tokenizer,
            max_seq_length=512,
        )
        
        trainer.train()
        
        # Save adapter
        model.save_pretrained(output_path)
        print(f"\nAdapter saved to: {output_path}")
        
        return output_path
    
    def convert_to_gguf(self, adapter_path: Path, output_name: str) -> Path:
        """
        Convert PEFT adapter to GGUF format for llama.cpp
        
        Note: This requires llama.cpp's convert scripts
        """
        # This is a placeholder - actual conversion requires:
        # 1. Merge LoRA weights with base model
        # 2. Convert merged model to GGUF
        # 
        # For now, we'll use the PEFT adapter directly with HF inference
        # or wait for llama.cpp's native LoRA-GGUF support
        
        print(f"Note: GGUF conversion requires merging weights.")
        print(f"Adapter at: {adapter_path}")
        
        output_path = self.output_dir / f"{output_name}.gguf"
        return output_path


def create_sample_training_data():
    """Create sample training data for each domain"""
    
    samples = {
        "work": [
            {"user": "What meetings do I have today?", "assistant": "You have a standup at 9am, 1:1 with Sarah at 2pm, and the sprint review at 4pm."},
            {"user": "Draft an email to the team about the deadline", "assistant": "Subject: Project Deadline Update\n\nHi team,\n\nJust a reminder that our deadline is this Friday. Please ensure all tasks are completed by EOD Thursday for final review.\n\nBest,\n[Your name]"},
            {"user": "What's the status on the API project?", "assistant": "The API project is at 75% completion. Backend is done, we're finishing the documentation. On track for next week's release."},
        ],
        "home": [
            {"user": "Turn off the living room lights", "assistant": "Living room lights are now off."},
            {"user": "What's the temperature at home?", "assistant": "It's currently 72°F inside. The thermostat is set to 70°F."},
            {"user": "When does Emma get home from school?", "assistant": "Emma's school bus usually arrives at 3:30pm. She should be home by 3:45pm."},
        ],
        "health": [
            {"user": "Did I take my medication today?", "assistant": "Yes, you took your morning vitamins at 8:15am. Your evening medication is due at 8pm."},
            {"user": "How many steps today?", "assistant": "You've walked 6,432 steps today. That's about 65% of your 10,000 step goal."},
            {"user": "How did I sleep last night?", "assistant": "You slept 7 hours 23 minutes with 1 hour 45 minutes of deep sleep. Your sleep score is 82."},
        ],
        "personal": [
            {"user": "Remind me about John's birthday", "assistant": "John's birthday is March 15th. Last year you got him a book about woodworking."},
            {"user": "What movie should I watch?", "assistant": "Based on your history, you might enjoy 'Arrival' - it's sci-fi with a thoughtful plot, similar to Interstellar which you loved."},
            {"user": "What's my favorite coffee order?", "assistant": "Your usual is a large oat milk latte with an extra shot. At Starbucks, that's a Venti Oatmilk Latte with an extra shot."},
        ],
    }
    
    return samples


# Example usage
if __name__ == "__main__":
    print("LoRA Adapter Training Pipeline")
    print("="*50)
    
    # Create sample data
    samples = create_sample_training_data()
    
    trainer = AdapterTrainer()
    
    for domain, conversations in samples.items():
        print(f"\n{domain.upper()} Domain:")
        dataset = trainer.prepare_training_data(domain, [
            {"user": c["user"], "assistant": c["assistant"]}
            for c in conversations
        ])
        print(f"  - {len(dataset.examples)} training examples")
        
        # Save sample data
        data_path = Path.home() / f"Roku/roku-ai/training/data/{domain}_sample.jsonl"
        dataset.to_jsonl(data_path)
    
    print("\n" + "="*50)
    print("Sample data created! To train adapters, run:")
    print("  trainer.train_adapter('work', dataset)")
    print("\nNote: Training requires transformers, peft, trl packages")
    print("  pip install transformers peft trl datasets")
