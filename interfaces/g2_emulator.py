"""
Even G2 AR Glasses Display Emulator for Project Roku

Simulates the Even G2 display experience:
- 488px width micro OLED display
- Green text on black background
- Compact text formatting
- Voice input/output integration
- Status indicators (listening, thinking, responding)
"""
import tkinter as tk
from tkinter import font
import threading
import time
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class G2Display:
    """
    Even G2 Display Specifications:
    - Micro OLED display
    - 488px effective width
    - Green text for visibility
    - Limited vertical space (~5 lines)
    """
    WIDTH = 488
    HEIGHT = 160
    MAX_CHARS = 45
    MAX_LINES = 5
    
    # Colors matching G2 aesthetic
    BG_COLOR = "#000000"
    TEXT_COLOR = "#00FF41"  # Matrix green
    DIM_COLOR = "#006622"
    ACCENT_COLOR = "#00AAFF"


class G2Emulator:
    """Full-featured Even G2 AR display emulator with Roku integration"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Even G2 - Roku")
        self.root.configure(bg='#0d0d0d')
        self.root.resizable(False, False)
        
        # State
        self.is_listening = False
        self.is_thinking = False
        self.roku = None  # PersonalizedRokuCoT instance
        self.llm = None  # Legacy LLM (for backward compatibility)
        self.voice = None
        self.conversation_history = []
        
        self._setup_ui()
        self._setup_bindings()
        
    def _setup_ui(self):
        """Create the emulator UI"""
        # Main frame with padding
        main_frame = tk.Frame(self.root, bg='#0d0d0d')
        main_frame.pack(padx=15, pady=15)
        
        # Title bar
        title_frame = tk.Frame(main_frame, bg='#0d0d0d')
        title_frame.pack(fill='x', pady=(0, 10))
        
        tk.Label(
            title_frame,
            text="EVEN G2",
            font=('Helvetica', 10, 'bold'),
            fg='#444',
            bg='#0d0d0d'
        ).pack(side='left')
        
        self.status_indicator = tk.Label(
            title_frame,
            text="● READY",
            font=('Helvetica', 9),
            fg=G2Display.TEXT_COLOR,
            bg='#0d0d0d'
        )
        self.status_indicator.pack(side='right')
        
        # G2 Display frame (simulating glasses lens)
        lens_frame = tk.Frame(
            main_frame,
            bg='#111',
            highlightbackground='#333',
            highlightthickness=2,
        )
        lens_frame.pack()
        
        # The actual display canvas
        self.display = tk.Canvas(
            lens_frame,
            width=G2Display.WIDTH,
            height=G2Display.HEIGHT,
            bg=G2Display.BG_COLOR,
            highlightthickness=0,
        )
        self.display.pack(padx=3, pady=3)
        
        # Fonts
        self.main_font = font.Font(family='SF Mono', size=13, weight='normal')
        self.small_font = font.Font(family='SF Mono', size=10)
        self.icon_font = font.Font(family='SF Mono', size=16, weight='bold')
        
        # Fallback fonts if SF Mono not available
        try:
            self.main_font.actual()
        except:
            self.main_font = font.Font(family='Courier', size=13)
            self.small_font = font.Font(family='Courier', size=10)
            self.icon_font = font.Font(family='Courier', size=16, weight='bold')
        
        # Input area (simulates voice/text input)
        input_frame = tk.Frame(main_frame, bg='#0d0d0d')
        input_frame.pack(fill='x', pady=(10, 0))
        
        self.input_entry = tk.Entry(
            input_frame,
            font=('Helvetica', 11),
            bg='#1a1a1a',
            fg='white',
            insertbackground='white',
            relief='flat',
            width=50,
        )
        self.input_entry.pack(side='left', fill='x', expand=True, ipady=8, padx=(0, 5))
        self.input_entry.insert(0, "Type or press M for mic...")
        self.input_entry.bind('<FocusIn>', self._on_input_focus)
        self.input_entry.bind('<Return>', self._on_input_submit)
        
        # Voice button
        self.mic_btn = tk.Button(
            input_frame,
            text="🎤",
            font=('Helvetica', 14),
            bg='#1a1a1a',
            fg=G2Display.TEXT_COLOR,
            relief='flat',
            width=3,
            command=self._toggle_voice,
        )
        self.mic_btn.pack(side='right')
        
        # Help text
        tk.Label(
            main_frame,
            text="Enter: Send | M: Voice | Esc: Clear | Q: Quit",
            font=('Helvetica', 9),
            fg='#444',
            bg='#0d0d0d'
        ).pack(pady=(8, 0))
        
        # Show welcome message
        self._show_welcome()
    
    def _setup_bindings(self):
        """Set up keyboard bindings"""
        self.root.bind('<Escape>', lambda e: self.clear_display())
        self.root.bind('q', lambda e: self._quit_if_not_typing())
        self.root.bind('m', lambda e: self._voice_if_not_typing())
        self.root.bind('M', lambda e: self._voice_if_not_typing())
    
    def _quit_if_not_typing(self):
        """Only quit if not focused on input"""
        if self.root.focus_get() != self.input_entry:
            self.root.quit()
    
    def _voice_if_not_typing(self):
        """Only trigger voice if not focused on input"""
        if self.root.focus_get() != self.input_entry:
            self._toggle_voice()
    
    def _on_input_focus(self, event):
        """Clear placeholder on focus"""
        if self.input_entry.get() == "Type or press M for mic...":
            self.input_entry.delete(0, 'end')
    
    def _on_input_submit(self, event):
        """Handle text input submission"""
        text = self.input_entry.get().strip()
        if text and text != "Type or press M for mic...":
            self.input_entry.delete(0, 'end')
            self._process_input(text)
    
    def _toggle_voice(self):
        """Toggle voice input"""
        if self.is_listening:
            return
        
        if self.voice is None:
            self._show_message("Voice not available", sub="Enable with --voice flag")
            return
        
        self.is_listening = True
        self._update_status("LISTENING", G2Display.ACCENT_COLOR)
        self._show_listening()
        
        # Run voice input in thread
        threading.Thread(target=self._voice_input_thread, daemon=True).start()
    
    def _voice_input_thread(self):
        """Handle voice input in background"""
        try:
            text = self.voice.listen(timeout=5)
            self.root.after(0, lambda: self._on_voice_result(text))
        except Exception as e:
            self.root.after(0, lambda: self._on_voice_error(str(e)))
    
    def _on_voice_result(self, text):
        """Handle voice transcription result"""
        self.is_listening = False
        if text:
            self._process_input(text)
        else:
            self._show_message("Didn't catch that", sub="Try again")
            self._update_status("READY", G2Display.TEXT_COLOR)
    
    def _on_voice_error(self, error):
        """Handle voice input error"""
        self.is_listening = False
        self._show_message("Voice error", sub=error[:30])
        self._update_status("READY", G2Display.TEXT_COLOR)
    
    def _process_input(self, text: str):
        """Process user input and get Roku's response"""
        # Show user message briefly
        self._show_user_message(text)
        self._update_status("THINKING", "#FFAA00")
        self.is_thinking = True
        
        # Process in thread to keep UI responsive
        threading.Thread(
            target=self._get_response_thread,
            args=(text,),
            daemon=True
        ).start()
    
    def _get_response_thread(self, text: str):
        """Get Roku response in background"""
        try:
            if self.roku:
                # Use PersonalizedRokuCoT (RAG-CoT reasoning)
                response = self.roku.ask(
                    text,
                    max_tokens=100,  # Shorter for G2 display
                    show_reasoning=False
                )
            elif self.llm:
                # Legacy LLM support
                response = self.llm.chat(
                    text,
                    conversation_history=self.conversation_history,
                    max_tokens=100,
                )
                # Update history
                self.conversation_history.append({"role": "user", "content": text})
                self.conversation_history.append({"role": "assistant", "content": response})
                # Keep history manageable
                if len(self.conversation_history) > 10:
                    self.conversation_history = self.conversation_history[-10:]
            else:
                response = "Roku not connected. Run with --roku flag."
            
            self.root.after(0, lambda: self._show_response(response))
        except Exception as e:
            self.root.after(0, lambda: self._show_message("Error", sub=str(e)[:40]))
    
    def _show_response(self, response: str):
        """Display Roku's response"""
        self.is_thinking = False
        self._update_status("READY", G2Display.TEXT_COLOR)
        self._show_message(response)
        
        # Text-to-speech if available
        if self.voice:
            threading.Thread(
                target=lambda: self.voice.speak(response),
                daemon=True
            ).start()
    
    def _update_status(self, text: str, color: str):
        """Update status indicator"""
        self.status_indicator.config(text=f"● {text}", fg=color)
    
    def clear_display(self):
        """Clear the display"""
        self.display.delete("all")
        self._show_welcome()
    
    def _show_welcome(self):
        """Show welcome screen"""
        self.display.delete("all")
        
        # Roku logo/name
        self.display.create_text(
            G2Display.WIDTH // 2,
            50,
            text="ROKU",
            font=font.Font(family='Helvetica', size=28, weight='bold'),
            fill=G2Display.TEXT_COLOR,
        )
        
        # Tagline
        self.display.create_text(
            G2Display.WIDTH // 2,
            90,
            text="Your Personal AI",
            font=self.small_font,
            fill=G2Display.DIM_COLOR,
        )
        
        # Hint
        self.display.create_text(
            G2Display.WIDTH // 2,
            130,
            text="Type or speak to begin",
            font=self.small_font,
            fill='#333',
        )
    
    def _show_listening(self):
        """Show listening indicator"""
        self.display.delete("all")
        
        self.display.create_text(
            G2Display.WIDTH // 2,
            G2Display.HEIGHT // 2 - 15,
            text="🎤",
            font=font.Font(size=32),
            fill=G2Display.ACCENT_COLOR,
        )
        
        self.display.create_text(
            G2Display.WIDTH // 2,
            G2Display.HEIGHT // 2 + 25,
            text="Listening...",
            font=self.main_font,
            fill=G2Display.ACCENT_COLOR,
        )
    
    def _show_user_message(self, text: str):
        """Briefly show user's message"""
        self.display.delete("all")
        
        # Truncate if needed
        if len(text) > 60:
            text = text[:57] + "..."
        
        self.display.create_text(
            G2Display.WIDTH // 2,
            G2Display.HEIGHT // 2,
            text=f'"{text}"',
            font=self.small_font,
            fill=G2Display.DIM_COLOR,
            width=G2Display.WIDTH - 40,
        )
    
    def _show_message(self, text: str, sub: str = None):
        """Show a message on display with word wrapping"""
        self.display.delete("all")
        
        # Word wrap for G2 display
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = f"{current_line} {word}".strip()
            if len(test_line) <= G2Display.MAX_CHARS:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
        
        # Limit to max lines
        if len(lines) > G2Display.MAX_LINES:
            lines = lines[:G2Display.MAX_LINES - 1]
            lines.append("...")
        
        # Calculate vertical positioning
        line_height = 24
        total_height = len(lines) * line_height
        start_y = (G2Display.HEIGHT - total_height) // 2
        
        if sub:
            start_y -= 10
        
        # Draw lines
        for i, line in enumerate(lines):
            self.display.create_text(
                G2Display.WIDTH // 2,
                start_y + (i * line_height),
                text=line,
                font=self.main_font,
                fill=G2Display.TEXT_COLOR,
                anchor='center',
            )
        
        # Subtitle if provided
        if sub:
            self.display.create_text(
                G2Display.WIDTH // 2,
                start_y + (len(lines) * line_height) + 15,
                text=sub,
                font=self.small_font,
                fill=G2Display.DIM_COLOR,
            )
    
    def connect_llm(self, llm):
        """Connect LLM for responses (legacy)"""
        self.llm = llm
        self._update_status("READY", G2Display.TEXT_COLOR)
    
    def connect_roku(self, roku):
        """Connect PersonalizedRokuCoT instance"""
        self.roku = roku
        self._update_status("READY", G2Display.TEXT_COLOR)
    
    def connect_voice(self, voice):
        """Connect voice interface"""
        self.voice = voice
        self.mic_btn.config(fg=G2Display.TEXT_COLOR)
    
    def run(self):
        """Start the emulator"""
        self.root.mainloop()


def main():
    """Run G2 Emulator with optional Roku, LLM, and voice"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Even G2 AR Display Emulator")
    parser.add_argument("--roku", action="store_true", help="Connect PersonalizedRokuCoT (RAG-CoT)")
    parser.add_argument("--llm", action="store_true", help="Connect legacy LLM (for backward compatibility)")
    parser.add_argument("--voice", action="store_true", help="Enable voice input")
    parser.add_argument("--username", type=str, default="Srimaan", help="Username for Roku profile")
    args = parser.parse_args()
    
    print("🕶️  Starting Even G2 Emulator...")
    
    emulator = G2Emulator()
    
    if args.roku:
        try:
            from core.personalized_roku_cot import PersonalizedRokuCoT
            print(f"Loading PersonalizedRokuCoT for {args.username}...")
            roku = PersonalizedRokuCoT(
                username=args.username,
                enable_calendar=True,
                enable_weather=True,
                enable_smart_home=True,
                verbose=False
            )
            emulator.connect_roku(roku)
            print("✅ PersonalizedRokuCoT connected (RAG-CoT enabled)")
        except Exception as e:
            print(f"⚠️  PersonalizedRokuCoT not available: {e}")
            import traceback
            traceback.print_exc()
    
    elif args.llm:
        try:
            from core.llm import LocalLLM
            print("Loading legacy Roku LLM...")
            llm = LocalLLM()
            emulator.connect_llm(llm)
            print("✅ LLM connected")
        except Exception as e:
            print(f"⚠️  LLM not available: {e}")
    
    if args.voice:
        try:
            from core.voice import VoiceInterface
            print("Loading voice interface...")
            voice = VoiceInterface(whisper_model="base")
            emulator.connect_voice(voice)
            print("✅ Voice connected")
        except Exception as e:
            print(f"⚠️  Voice not available: {e}")
    
    print("\n🕶️  G2 Emulator running. Close window or press Q to quit.\n")
    emulator.run()


if __name__ == "__main__":
    main()
