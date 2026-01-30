"""
Encryption utilities for data at rest
"""
from pathlib import Path
from typing import Any, Optional
import json
import os

try:
    from cryptography.fernet import Fernet
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False


class DataEncryption:
    """Handle data encryption/decryption"""
    
    def __init__(self, key_path: str = "~/.roku/encryption.key"):
        """
        Initialize encryption
        
        Args:
            key_path: Where to store encryption key
        """
        if not CRYPTO_AVAILABLE:
            raise ImportError("cryptography not installed. Run: pip install cryptography")
        
        self.key_path = Path(key_path).expanduser()
        self.key = self._load_or_create_key()
        self.cipher = Fernet(self.key)
    
    def _load_or_create_key(self) -> bytes:
        """Load existing key or create new one"""
        if self.key_path.exists():
            with open(self.key_path, 'rb') as f:
                return f.read()
        else:
            # Generate new key
            key = Fernet.generate_key()
            
            # Save with restrictive permissions
            self.key_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.key_path, 'wb') as f:
                f.write(key)
            os.chmod(self.key_path, 0o600)
            
            return key
    
    def encrypt(self, data: bytes) -> bytes:
        """Encrypt bytes data"""
        return self.cipher.encrypt(data)
    
    def decrypt(self, encrypted_data: bytes) -> bytes:
        """Decrypt bytes data"""
        return self.cipher.decrypt(encrypted_data)
    
    def encrypt_file(self, filepath: Path, output_path: Optional[Path] = None):
        """Encrypt a file"""
        with open(filepath, 'rb') as f:
            data = f.read()
        
        encrypted = self.encrypt(data)
        
        out_path = output_path or Path(str(filepath) + '.enc')
        with open(out_path, 'wb') as f:
            f.write(encrypted)
        
        return out_path
    
    def decrypt_file(self, filepath: Path) -> bytes:
        """Decrypt a file and return contents"""
        with open(filepath, 'rb') as f:
            encrypted_data = f.read()
        
        return self.decrypt(encrypted_data)
    
    def encrypt_json(self, data: Any, filepath: Path):
        """Encrypt and save JSON data"""
        json_str = json.dumps(data)
        encrypted = self.encrypt(json_str.encode())
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            f.write(encrypted)
    
    def decrypt_json(self, filepath: Path) -> Any:
        """Load and decrypt JSON data"""
        with open(filepath, 'rb') as f:
            encrypted = f.read()
        
        decrypted = self.decrypt(encrypted)
        return json.loads(decrypted.decode())


# Example usage
if __name__ == "__main__":
    encryptor = DataEncryption()
    
    # Test data
    test_data = {
        "conversations": [
            {"user": "Hello", "assistant": "Hi!"},
        ]
    }
    
    # Encrypt
    test_path = Path("/tmp/roku_test.enc")
    encryptor.encrypt_json(test_data, test_path)
    print("✅ Data encrypted")
    
    # Decrypt
    decrypted = encryptor.decrypt_json(test_path)
    print("✅ Data decrypted:", decrypted)
