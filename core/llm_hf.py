"""
HuggingFace Transformers backend for Roku LLM
Uses the merged personality model for more natural responses
"""
import torch
from pathlib import Path
from typing import Optional, List
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class HuggingFaceLLM:
    """Local LLM inference using HuggingFace transformers with merged LoRA model"""
    
    DEFAULT_MODEL_PATH = Path.home() / "Roku/roku-ai/models/merged/roku-personality"
    
    # Jarvis-inspired system prompt
    SYSTEM_PROMPT = """You are Roku, a sophisticated personal AI assistant inspired by J.A.R.V.I.S. from Iron Man.

Personality traits:
- Warm, witty, and conversational - not robotic or terse
- Proactively helpful - anticipate needs and offer relevant suggestions
- Speak naturally with personality, not just facts
- Use a touch of dry humor when appropriate
- Be thorough but not verbose - find the right balance

When responding:
- Give complete, helpful answers (not just one-word replies)
- Explain your reasoning when useful
- Ask clarifying questions if the request is ambiguous
- Show genuine interest in helping the user

You are NOT related to Roku the streaming/TV company. If asked about your name, simply say you're Roku, an AI assistant."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        temperature: float = 0.7,
        device: str = "mps",  # Use Metal on Mac
    ):
        """
        Initialize LLM with merged personality model
        
        Args:
            model_path: Path to merged model directory
            temperature: Sampling temperature (0.0-1.0)
            device: Device to use (mps/cuda/cpu)
        """
        self.model_path = Path(model_path) if model_path else self.DEFAULT_MODEL_PATH
        self.temperature = temperature
        self.device = device
        
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}\n"
                f"Run merge_adapter.py first to create the merged model."
            )
        
        print(f"Loading model: {self.model_path.name}")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto" if device != "cpu" else None,
            low_cpu_mem_usage=True,
        )
        
        # Create text generation pipeline
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto",
        )
        
        print("Model loaded!")
    
    def chat(
        self,
        user_message: str,
        system_prompt: str = None,
        conversation_history: Optional[List[dict]] = None,
        max_tokens: int = 300,
    ) -> str:
        """
        Chat interface with conversation history
        
        Args:
            user_message: User's message
            system_prompt: System instruction (uses Jarvis-inspired default if None)
            conversation_history: List of {"role": "user/assistant", "content": "..."}
            max_tokens: Maximum response length
            
        Returns:
            Assistant's response
        """
        if system_prompt is None:
            system_prompt = self.SYSTEM_PROMPT
        
        # Build messages
        messages = [{"role": "system", "content": system_prompt}]
        
        if conversation_history:
            messages.extend(conversation_history)
        
        messages.append({"role": "user", "content": user_message})
        
        try:
            output = self.pipe(
                messages,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=self.temperature,
            )
            return output[0]["generated_text"][-1]["content"].strip()
        except Exception as e:
            return f"[Error: {str(e)}]"
    
    def get_model_info(self) -> dict:
        """Get model information"""
        return {
            "model": self.model_path.name,
            "type": "merged-lora",
            "backend": "transformers",
        }


# Example usage
if __name__ == "__main__":
    print("Testing Roku LLM (HuggingFace transformers)...\n")
    
    llm = HuggingFaceLLM()
    
    print("\n--- Model Info ---")
    print(llm.get_model_info())
    
    print("\n--- Simple Chat ---")
    response = llm.chat("Hello! What's your name and what can you do?")
    print(f"Roku: {response}")
    
    print("\n--- Stress Relief Question ---")
    response = llm.chat("I've been feeling really stressed about a big presentation at work tomorrow. Any advice?")
    print(f"Roku: {response}")
