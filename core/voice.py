"""
Voice input/output handling
"""
import speech_recognition as sr
import subprocess
from typing import Optional

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False


class VoiceInterface:
    """Handle voice input and output"""
    
    def __init__(self, whisper_model: str = "base"):
        """
        Initialize voice interface
        
        Args:
            whisper_model: Whisper model size (tiny, base, small, medium, large)
        """
        if not WHISPER_AVAILABLE:
            raise ImportError("Whisper not installed. Run: pip install openai-whisper")
        
        # Load Whisper for speech-to-text
        print(f"Loading Whisper ({whisper_model})...")
        self.whisper_model = whisper.load_model(whisper_model)
        
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        
        print("Voice interface ready!")
    
    def listen(self, timeout: int = 5) -> Optional[str]:
        """
        Listen for voice input
        
        Args:
            timeout: Seconds to wait for speech
            
        Returns:
            Transcribed text or None if failed
        """
        with sr.Microphone() as source:
            print("Listening... (speak now)")
            
            try:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                # Listen
                audio = self.recognizer.listen(source, timeout=timeout)
                
                print("Processing speech...")
                
                # Save audio to temp file for Whisper
                with open("/tmp/roku_audio.wav", "wb") as f:
                    f.write(audio.get_wav_data())
                
                # Transcribe with Whisper
                result = self.whisper_model.transcribe("/tmp/roku_audio.wav")
                text = result["text"].strip()
                
                print(f"You said: {text}")
                return text
                
            except sr.WaitTimeoutError:
                print("No speech detected (timeout)")
                return None
            except Exception as e:
                print(f"Error: {e}")
                return None
    
    def speak(self, text: str, voice: str = "Samantha"):
        """
        Speak text using macOS text-to-speech
        
        Args:
            text: Text to speak
            voice: macOS voice name (Samantha, Alex, etc.)
        """
        subprocess.run(["say", "-v", voice, text])
    
    def speak_async(self, text: str, voice: str = "Samantha"):
        """
        Speak without blocking (background process)
        
        Args:
            text: Text to speak
            voice: macOS voice name
        """
        subprocess.Popen(["say", "-v", voice, text])


# Example usage
if __name__ == "__main__":
    voice = VoiceInterface(whisper_model="tiny")
    
    # Test listening
    text = voice.listen(timeout=10)
    
    if text:
        voice.speak(f"You said: {text}")
