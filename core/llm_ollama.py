"""
LLM inference wrapper using Ollama
"""
import requests
import json
from typing import Optional, List


class LocalLLM:
    """Local LLM inference using Ollama"""
    
    def __init__(
        self,
        model: str = "llama3.2:3b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        context_size: int = 2048,
    ):
        """
        Initialize LLM
        
        Args:
            model: Ollama model name
            base_url: Ollama API URL
            temperature: Sampling temperature (0.0-1.0)
            context_size: Context window size
        """
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.context_size = context_size
        
        # Verify Ollama is running
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                raise ConnectionError("Ollama not responding")
            
            # Check if model is available
            models = response.json().get("models", [])
            model_names = [m["name"] for m in models]
            if model not in model_names and f"{model}:latest" not in model_names:
                print(f"Model '{model}' not found. Available: {model_names}")
                print(f"Run: ollama pull {model}")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                "Ollama not running. Start it with: ollama serve"
            )
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 200,
        stop_sequences: Optional[List[str]] = None,
        stream: bool = False,
    ) -> str:
        """
        Generate response from prompt
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            stop_sequences: Sequences that stop generation
            stream: Whether to stream response
            
        Returns:
            Generated text
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": self.temperature,
                "num_predict": max_tokens,
                "num_ctx": self.context_size,
            }
        }
        
        if stop_sequences:
            payload["options"]["stop"] = stop_sequences
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "").strip()
            
        except requests.exceptions.Timeout:
            return "[Error: Generation timed out]"
        except Exception as e:
            return f"[Error: {str(e)}]"
    
    def chat(
        self,
        user_message: str,
        system_prompt: str = "You are Roku, a personal AI assistant. When asked your name, just say 'Roku' - nothing more about its meaning or origin. Be concise. Never mention streaming, TV, or entertainment devices.",
        conversation_history: Optional[List[dict]] = None,
        max_tokens: int = 200,
    ) -> str:
        """
        Chat interface with conversation history
        
        Args:
            user_message: User's message
            system_prompt: System instruction
            conversation_history: List of {"role": "user/assistant", "content": "..."}
            max_tokens: Maximum response length
            
        Returns:
            Assistant's response
        """
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history
        if conversation_history:
            for msg in conversation_history:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        # Add current message
        messages.append({"role": "user", "content": user_message})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": max_tokens,
                "num_ctx": self.context_size,
            }
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("message", {}).get("content", "").strip()
            
        except requests.exceptions.Timeout:
            return "[Error: Generation timed out]"
        except Exception as e:
            return f"[Error: {str(e)}]"


# Example usage
if __name__ == "__main__":
    print("Testing Roku LLM...")
    
    llm = LocalLLM(model="llama3.2:3b")
    
    print("\n--- Simple Chat ---")
    response = llm.chat("Hello! What's your name and where does your name come from?")
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
