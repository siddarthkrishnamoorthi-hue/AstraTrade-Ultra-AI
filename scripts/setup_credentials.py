"""
Setup secure credentials for AstraTrade Ultra AI
"""
from utils.credentials import CredentialsManager
import os
from pathlib import Path

def setup_credentials():
    """Initialize secure credential storage"""
    # Create secure directory if it doesn't exist
    secure_dir = Path("secure")
    secure_dir.mkdir(exist_ok=True)
    
    # Create .gitignore if it doesn't exist
    gitignore = Path(".gitignore")
    if not gitignore.exists() or 'secure/' not in gitignore.read_text():
        with open(gitignore, "a") as f:
            f.write("\n# Secure credentials\nsecure/\n.env\n")
    
    # Initialize credentials manager
    creds_manager = CredentialsManager()
    
    # Encrypt and store credentials
    creds_manager.encrypt_credentials({
        "alpha_vantage_key": "AFXPR0PGLLAA1FK4",
        "telegram_token": "8383117060:AAEnd19FdVm0War5bje-3NbYt02eAvU8VqI",
        "telegram_chat_id": "711076026"
    })
    
    print("\nCredentials securely stored!")
    print("Location: secure/.credentials (encrypted)")
    print("Key file: secure/.key (keep this safe!)")
    print("\nNever commit these files to version control!")

if __name__ == "__main__":
    setup_credentials()