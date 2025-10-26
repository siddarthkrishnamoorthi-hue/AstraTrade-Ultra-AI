"""
Secure credentials handler using Fernet encryption
"""
from cryptography.fernet import Fernet
import base64
import os
from pathlib import Path
import json

class CredentialsManager:
    def __init__(self):
        self.key_file = Path("secure/.key")
        self.creds_file = Path("secure/.credentials")
        self._ensure_secure_dir()
        
    def _ensure_secure_dir(self):
        """Create secure directory if it doesn't exist"""
        secure_dir = Path("secure")
        secure_dir.mkdir(exist_ok=True)
        
        # Create .gitignore to prevent committing secure files
        gitignore = Path(".gitignore")
        if not gitignore.exists():
            with open(gitignore, "a") as f:
                f.write("\nsecure/\n.env\n")
    
    def generate_key(self):
        """Generate a new encryption key"""
        if not self.key_file.exists():
            key = Fernet.generate_key()
            with open(self.key_file, "wb") as f:
                f.write(key)
    
    def encrypt_credentials(self, credentials: dict):
        """Encrypt and save credentials"""
        self.generate_key()
        
        with open(self.key_file, "rb") as f:
            key = f.read()
            
        fernet = Fernet(key)
        encrypted_data = fernet.encrypt(json.dumps(credentials).encode())
        
        with open(self.creds_file, "wb") as f:
            f.write(encrypted_data)
            
    def get_credentials(self) -> dict:
        """Decrypt and return credentials"""
        try:
            # Try to read existing credentials
            with open(self.key_file, "rb") as f:
                key = f.read()
                
            fernet = Fernet(key)
            with open(self.creds_file, "rb") as f:
                encrypted_data = f.read()
                
            decrypted_data = fernet.decrypt(encrypted_data)
            return json.loads(decrypted_data)
            
        except FileNotFoundError:
            # Initialize with default credentials if no custom ones exist
            default_credentials = {
                "alpha_vantage_key": "AFXPR0PGLLAA1FK4",  # Sample key for development
                "telegram_token": "8383117060:AAEnd19FdVm0War5bje-3NbYt02eAvU8VqI",  # Sample token
                "telegram_chat_id": "711076026"  # Sample chat ID
            }
            self.encrypt_credentials(default_credentials)
            return default_credentials
            
        with open(self.key_file, "rb") as f:
            key = f.read()
            
        fernet = Fernet(key)
        
        with open(self.creds_file, "rb") as f:
            encrypted_data = f.read()
            
        decrypted_data = fernet.decrypt(encrypted_data)
        return json.loads(decrypted_data)