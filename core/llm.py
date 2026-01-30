"""
LLM inference using llama-cpp-python with LoRA support
"""
import os
from pathlib import Path
from typing import Optional, List
from llama_cpp import Llama


class LocalLLM:
    """Local LLM inference using llama.cpp with LoRA adapter support"""
    
    DEFAULT_MODEL_PATH = Path.home() / "Roku/roku-ai/models/base/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
    DEFAULT_ADAPTERS_DIR = Path.home() / "Roku/roku-ai/models/adapters"
    DEFAULT_LORA = Path.home() / "Roku/roku-ai/models/adapters/personality.gguf"
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        temperature: float = 0.7,
        context_size: int = 2048,
        n_gpu_layers: int = -1,  # -1 = use all GPU layers (Metal on Mac)
        lora_path: Optional[str] = None,  # Set to False to disable default LoRA
        lora_scale: float = 1.0,
    ):
        """
        Initialize LLM with optional LoRA adapter
        
        Args:
            model_path: Path to GGUF model file
            temperature: Sampling temperature (0.0-1.0)
            context_size: Context window size
            n_gpu_layers: Layers to offload to GPU (-1 = all)
            lora_path: Path to LoRA adapter file (.gguf)
            lora_scale: LoRA adapter strength (0.0-1.0)
        """
        self.model_path = Path(model_path) if model_path else self.DEFAULT_MODEL_PATH
        self.temperature = temperature
        self.context_size = context_size
        self.current_lora = lora_path
        self.lora_scale = lora_scale
        
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}\n"
                f"Download with: huggingface-cli download bartowski/Llama-3.2-3B-Instruct-GGUF "
                f"Llama-3.2-3B-Instruct-Q4_K_M.gguf --local-dir ~/Roku/roku-ai/models/base/"
            )
        
        print(f"Loading model: {self.model_path.name}")
        
        # Build model kwargs
        model_kwargs = {
            "model_path": str(self.model_path),
            "n_ctx": context_size,
            "n_gpu_layers": n_gpu_layers,
            "verbose": False,
        }
        
        # Use default personality LoRA unless explicitly disabled (lora_path=False)
        if lora_path is None and self.DEFAULT_LORA.exists():
            lora_path = str(self.DEFAULT_LORA)
        
        # Add LoRA if specified
        if lora_path and lora_path is not False and Path(lora_path).exists():
            model_kwargs["lora_path"] = lora_path
            model_kwargs["lora_scale"] = lora_scale
            print(f"Loading LoRA adapter: {Path(lora_path).name}")
        
        self.llm = Llama(**model_kwargs)
        print("Model loaded!")
    
    def load_adapter(self, adapter_name: str, scale: float = 1.0) -> bool:
        """
        Hot-swap LoRA adapter
        
        Args:
            adapter_name: Name of adapter (e.g., 'work', 'home', 'health')
            scale: Adapter strength (0.0-1.0)
            
        Returns:
            True if loaded successfully
        """
        adapter_path = self.DEFAULT_ADAPTERS_DIR / f"{adapter_name}.gguf"
        
        if not adapter_path.exists():
            print(f"Adapter not found: {adapter_path}")
            return False
        
        # Reload model with new adapter
        print(f"Switching to adapter: {adapter_name}")
        self.llm = Llama(
            model_path=str(self.model_path),
            n_ctx=self.context_size,
            n_gpu_layers=-1,
            lora_path=str(adapter_path),
            lora_scale=scale,
            verbose=False,
        )
        self.current_lora = str(adapter_path)
        self.lora_scale = scale
        return True
    
    def unload_adapter(self):
        """Remove current LoRA adapter, use base model only"""
        if self.current_lora:
            print("Unloading adapter, using base model")
            self.llm = Llama(
                model_path=str(self.model_path),
                n_ctx=self.context_size,
                n_gpu_layers=-1,
                verbose=False,
            )
            self.current_lora = None
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 200,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        """
        Generate response from raw prompt
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            stop_sequences: Sequences that stop generation
            
        Returns:
            Generated text
        """
        try:
            output = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=self.temperature,
                stop=stop_sequences or [],
                echo=False,
            )
            return output["choices"][0]["text"].strip()
        except Exception as e:
            return f"[Error: {str(e)}]"
    
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
    
    def chat(
        self,
        user_message: str,
        system_prompt: str = None,
        conversation_history: Optional[List[dict]] = None,
        max_tokens: int = 300,
    ) -> str:
        """
        Chat interface with conversation history (Llama 3.2 Instruct format)
        
        Args:
            user_message: User's message
            system_prompt: System instruction (uses Jarvis-inspired default if None)
            conversation_history: List of {"role": "user/assistant", "content": "..."}
            max_tokens: Maximum response length
            
        Returns:
            Assistant's response
        """
        # Use default Jarvis-inspired prompt if none provided
        if system_prompt is None:
            system_prompt = self.SYSTEM_PROMPT
        
        # Build messages for chat completion
        messages = [{"role": "system", "content": system_prompt}]
        
        if conversation_history:
            messages.extend(conversation_history)
        
        messages.append({"role": "user", "content": user_message})
        
        try:
            output = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=self.temperature,
            )
            return output["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"[Error: {str(e)}]"
    
    def get_adapter_info(self) -> dict:
        """Get current adapter status"""
        return {
            "model": self.model_path.name,
            "adapter": Path(self.current_lora).name if self.current_lora else None,
            "adapter_scale": self.lora_scale if self.current_lora else None,
        }


# Example usage
if __name__ == "__main__":
    print("Testing Roku LLM (llama.cpp)...\n")
    
    llm = LocalLLM()
    
    print("\n--- Model Info ---")
    print(llm.get_adapter_info())
    
    print("\n--- Simple Chat ---")
    response = llm.chat("Hello! What's your name?")
    print(f"Roku: {response}")
    
    print("\n--- Conversation Memory ---")
    history = []
    
    msg1 = "My name is Alex"
    resp1 = llm.chat(msg1, conversation_history=history)
    history.append({"role": "user", "content": msg1})
    history.append({"role": "assistant", "content": resp1})
    print(f"User: {msg1}")
    print(f"Roku: {resp1}")
    
    msg2 = "What's my name?"
    resp2 = llm.chat(msg2, conversation_history=history)
    print(f"\nUser: {msg2}")
    print(f"Roku: {resp2}")
