"""
Multi-LoRA Support for Roku AI
Enables stacking multiple LoRA adapters simultaneously for cross-domain intelligence.

Example: personality + health + personal adapters active together,
allowing the model to correlate sleep data with work schedules.
"""
import os
import ctypes
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from llama_cpp import Llama
import llama_cpp.llama_cpp as llama_cpp_low  # Low-level C API bindings


@dataclass
class LoadedAdapter:
    """Represents a loaded LoRA adapter"""
    name: str
    path: Path
    scale: float
    handle: Any  # ctypes pointer (llama_adapter_lora_p)


class MultiLoRALlama:
    """
    Extended Llama wrapper with multi-adapter support.
    
    Uses the low-level llama.cpp API to stack multiple LoRA adapters:
    - llama_adapter_lora_init() - Load adapter from file
    - llama_set_adapter_lora() - Add adapter to context (can call multiple times!)
    - llama_rm_adapter_lora() - Remove specific adapter
    - llama_clear_adapter_lora() - Clear all adapters
    """
    
    # DeepSeek-R1 14B for better reasoning
    DEFAULT_MODEL_PATH = Path.home() / "Roku/roku-ai/models/base/DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf"
    DEFAULT_ADAPTERS_DIR = Path.home() / "Roku/roku-ai/models/adapters"
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        temperature: float = 0.7,
        context_size: int = 2048,
        n_gpu_layers: int = -1,
        verbose: bool = False,
    ):
        """
        Initialize Multi-LoRA Llama.
        
        Args:
            model_path: Path to base GGUF model
            temperature: Sampling temperature
            context_size: Context window size
            n_gpu_layers: GPU layers (-1 = all)
            verbose: Print debug info
        """
        self.model_path = Path(model_path) if model_path else self.DEFAULT_MODEL_PATH
        self.temperature = temperature
        self.context_size = context_size
        self.verbose = verbose
        
        # Track loaded adapters
        self._adapters: Dict[str, LoadedAdapter] = {}
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        if self.verbose:
            print(f"Loading base model: {self.model_path.name}")
        
        # Load base model WITHOUT any LoRA (we'll add them via low-level API)
        self.llm = Llama(
            model_path=str(self.model_path),
            n_ctx=context_size,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
        )
        
        if self.verbose:
            print("Base model loaded! Multi-LoRA ready.")
    
    @property
    def active_adapters(self) -> List[str]:
        """Get list of currently active adapter names"""
        return list(self._adapters.keys())
    
    @property
    def adapter_info(self) -> Dict[str, float]:
        """Get dict of adapter names to their scales"""
        return {name: adapter.scale for name, adapter in self._adapters.items()}
    
    def add_adapter(
        self,
        name: str,
        path: Optional[str] = None,
        scale: float = 1.0,
    ) -> bool:
        """
        Add a LoRA adapter to the active stack.
        
        Multiple adapters can be active simultaneously!
        
        Args:
            name: Adapter name (e.g., 'personality', 'health', 'personal')
            path: Path to .gguf adapter file (or auto-detect from name)
            scale: Adapter strength (0.0-1.0)
            
        Returns:
            True if added successfully
        """
        # If adapter already loaded, update scale instead
        if name in self._adapters:
            return self.set_adapter_scale(name, scale)
        
        # Resolve adapter path
        if path is None:
            adapter_path = self.DEFAULT_ADAPTERS_DIR / f"{name}.gguf"
        else:
            adapter_path = Path(path)
        
        if not adapter_path.exists():
            print(f"❌ Adapter not found: {adapter_path}")
            return False
        
        if self.verbose:
            print(f"Loading adapter: {name} (scale={scale})")
        
        # Load adapter using low-level API
        # Note: In 0.2.90, the function is llama_lora_adapter_* not llama_adapter_lora_*
        adapter_handle = llama_cpp_low.llama_lora_adapter_init(
            self.llm._model.model,
            str(adapter_path).encode("utf-8"),
        )
        
        if adapter_handle is None:
            print(f"❌ Failed to load adapter: {name}")
            return False
        
        # Add adapter to context
        result = llama_cpp_low.llama_lora_adapter_set(
            self.llm._ctx.ctx,
            adapter_handle,
            scale,
        )
        
        if result != 0:
            print(f"❌ Failed to set adapter: {name}")
            llama_cpp_low.llama_lora_adapter_free(adapter_handle)
            return False
        
        # Track the adapter
        self._adapters[name] = LoadedAdapter(
            name=name,
            path=adapter_path,
            scale=scale,
            handle=adapter_handle,
        )
        
        if self.verbose:
            print(f"✓ Adapter '{name}' active (scale={scale})")
        
        return True
    
    def remove_adapter(self, name: str) -> bool:
        """
        Remove a specific adapter from the active stack.
        
        Args:
            name: Adapter name to remove
            
        Returns:
            True if removed successfully
        """
        if name not in self._adapters:
            print(f"Adapter not active: {name}")
            return False
        
        adapter = self._adapters[name]
        
        # Remove from context
        result = llama_cpp_low.llama_lora_adapter_remove(
            self.llm._ctx.ctx,
            adapter.handle,
        )
        
        if result == -1:
            print(f"Warning: Adapter '{name}' may not have been in context")
        
        # Free the adapter
        llama_cpp_low.llama_lora_adapter_free(adapter.handle)
        
        del self._adapters[name]
        
        if self.verbose:
            print(f"✓ Removed adapter: {name}")
        
        return True
    
    def set_adapter_scale(self, name: str, scale: float) -> bool:
        """
        Update the scale of an active adapter.
        
        Note: llama.cpp doesn't have a direct "update scale" function,
        so we remove and re-add the adapter with the new scale.
        
        Args:
            name: Adapter name
            scale: New scale value (0.0-1.0)
            
        Returns:
            True if updated successfully
        """
        if name not in self._adapters:
            print(f"Adapter not active: {name}")
            return False
        
        adapter = self._adapters[name]
        old_scale = adapter.scale
        adapter_path = adapter.path
        
        # Remove and re-add with new scale
        self.remove_adapter(name)
        success = self.add_adapter(name, str(adapter_path), scale)
        
        if success and self.verbose:
            print(f"✓ Updated '{name}' scale: {old_scale} → {scale}")
        
        return success
    
    def clear_adapters(self) -> None:
        """Remove all active adapters"""
        llama_cpp_low.llama_lora_adapter_clear(self.llm._ctx.ctx)
        
        # Free all adapter handles
        for adapter in self._adapters.values():
            llama_cpp_low.llama_lora_adapter_free(adapter.handle)
        
        self._adapters.clear()
        
        if self.verbose:
            print("✓ Cleared all adapters")
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        stop: Optional[List[str]] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Generate text with all active adapters.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            stop: Stop sequences
            temperature: Override temperature
            
        Returns:
            Generated text
        """
        response = self.llm(
            prompt,
            max_tokens=max_tokens,
            stop=stop or ["<|eot_id|>", "<|end_of_text|>"],
            temperature=temperature or self.temperature,
            echo=False,
        )
        
        return response["choices"][0]["text"].strip()
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 256,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Chat completion with all active adapters.
        
        Args:
            messages: List of {"role": "user/assistant/system", "content": "..."}
            max_tokens: Maximum tokens
            temperature: Override temperature
            
        Returns:
            Assistant response
        """
        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature or self.temperature,
        )
        
        return response["choices"][0]["message"]["content"].strip()
    
    def __del__(self):
        """Cleanup adapters on deletion"""
        try:
            self.clear_adapters()
        except:
            pass


# Convenience function for common Roku configurations
def create_roku_llm(
    adapters: List[Tuple[str, float]] = None,
    verbose: bool = True,
) -> MultiLoRALlama:
    """
    Create a MultiLoRALlama with common Roku adapter configurations.
    
    Args:
        adapters: List of (adapter_name, scale) tuples
                  Default: [("personality", 1.0)]
        verbose: Print loading info
        
    Returns:
        Configured MultiLoRALlama instance
        
    Example:
        # Cross-domain intelligence: personality + health + personal
        llm = create_roku_llm([
            ("personality", 1.0),  # Friendly, helpful responses
            ("health", 0.8),       # Sleep/wellness knowledge
            ("personal", 0.8),     # Work schedule/preferences
        ])
    """
    if adapters is None:
        adapters = [("personality", 1.0)]
    
    llm = MultiLoRALlama(verbose=verbose)
    
    for name, scale in adapters:
        success = llm.add_adapter(name, scale=scale)
        if not success:
            print(f"⚠️ Skipped adapter: {name}")
    
    print(f"\n🎯 Active adapters: {llm.active_adapters}")
    return llm


if __name__ == "__main__":
    print("=" * 60)
    print("Multi-LoRA Test for Roku AI")
    print("=" * 60)
    
    # Test with personality adapter
    llm = create_roku_llm([("personality", 1.0)])
    
    # Test generation
    response = llm.chat([
        {"role": "system", "content": "You are Roku, a helpful AI assistant."},
        {"role": "user", "content": "Who are you?"}
    ])
    
    print(f"\n💬 Response:\n{response}")
    print(f"\n📊 Active adapters: {llm.adapter_info}")
