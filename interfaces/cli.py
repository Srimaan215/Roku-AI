"""
Command-line interface for Roku
"""
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.llm import LocalLLM
from core.context import ContextManager
from core.router import QueryRouter


class RokuCLI:
    """Interactive CLI for Roku"""
    
    def __init__(self, use_voice: bool = False):
        """
        Initialize CLI
        
        Args:
            use_voice: Enable voice input/output
        """
        print("🤖 Initializing Roku...")
        
        # Initialize components
        self.context = ContextManager()
        self.router = QueryRouter()
        
        # Try to initialize LLM via llama.cpp
        try:
            self.llm = LocalLLM(
                temperature=0.7,
            )
            self.llm_available = True
        except FileNotFoundError as e:
            print(f"⚠️  LLM not available: {e}")
            self.llm_available = False
        
        # Initialize voice if requested
        self.use_voice = use_voice
        self.voice = None
        if use_voice:
            try:
                from core.voice import VoiceInterface
                self.voice = VoiceInterface(whisper_model="base")
            except Exception as e:
                print(f"⚠️  Voice not available: {e}")
                self.use_voice = False
        
        print("✅ Roku ready!\n")
    
    def process_input(self, user_input: str) -> str:
        """Process user input and generate response"""
        # Route the query
        routing = self.router.route(user_input)
        
        # Check if we should use cloud
        if routing["use_cloud"]:
            return f"[Cloud API needed: {routing['cloud_reason']}]\n(Not implemented yet - would route to Claude)"
        
        # Use local LLM
        if not self.llm_available:
            return "[LLM not available - please run setup to download model]"
        
        # Get context
        context_prompt = self.context.build_context_prompt(user_input)
        
        # Get conversation history
        history = self.context.get_recent_history(n_messages=6)
        
        # Generate response
        response = self.llm.chat(
            user_message=user_input,
            conversation_history=history,
            max_tokens=300,
        )
        
        # Store in context
        self.context.add_message("user", user_input)
        self.context.add_message("assistant", response)
        
        return response
    
    def run(self):
        """Main interaction loop"""
        print("Commands: 'quit' to exit, 'clear' to reset, 'voice' to toggle voice\n")
        
        while True:
            try:
                # Get user input
                if self.use_voice and self.voice:
                    print("\n🎤 ", end="", flush=True)
                    user_input = self.voice.listen(timeout=10)
                    if not user_input:
                        continue
                else:
                    user_input = input("\n👤 You: ").strip()
                
                # Handle commands
                if not user_input:
                    continue
                
                if user_input.lower() in ["quit", "exit", "bye"]:
                    # Save conversation before exiting
                    self.context.save_conversation()
                    print("\n👋 Goodbye!")
                    break
                
                if user_input.lower() == "clear":
                    self.context.clear_history()
                    print("🗑️  Conversation cleared")
                    continue
                
                if user_input.lower() == "voice":
                    if self.voice is None:
                        print("⚠️  Voice not initialized")
                    else:
                        self.use_voice = not self.use_voice
                        status = "enabled" if self.use_voice else "disabled"
                        print(f"🎤 Voice input {status}")
                    continue
                
                if user_input.lower() == "save":
                    self.context.save_conversation()
                    continue
                
                # Get response
                print("🤖 Roku: ", end="", flush=True)
                
                response = self.process_input(user_input)
                print(response)
                
                # Speak response if voice enabled
                if self.use_voice and self.voice:
                    self.voice.speak_async(response)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")


def main():
    """Entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Roku CLI")
    parser.add_argument("--voice", action="store_true", help="Enable voice input/output")
    args = parser.parse_args()
    
    cli = RokuCLI(use_voice=args.voice)
    cli.run()


if __name__ == "__main__":
    main()
