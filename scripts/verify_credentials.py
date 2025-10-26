"""
Verify secure credentials for AstraTrade Ultra AI
"""
from utils.credentials import CredentialsManager
from pathlib import Path

def verify_credentials():
    """Verify that credentials are properly encrypted and accessible"""
    creds_manager = CredentialsManager()
    
    # Check if credential files exist
    creds_file = Path("secure/.credentials")
    key_file = Path("secure/.key")
    
    if not creds_file.exists() or not key_file.exists():
        print("Error: Credential files not found!")
        print("Please run setup_credentials.py first")
        return False
    
    try:
        # Try to decrypt credentials
        creds = creds_manager.get_credentials()
        
        # Verify all required credentials are present
        required_keys = [
            "alpha_vantage_key",
            "telegram_token",
            "telegram_chat_id"
        ]
        
        missing_keys = [key for key in required_keys if key not in creds]
        
        if missing_keys:
            print(f"Error: Missing credentials: {missing_keys}")
            return False
            
        print("\nCredentials verification successful!")
        print("All required credentials are securely stored")
        
        # Verify encryption
        with open(creds_file, 'rb') as f:
            encrypted_data = f.read()
            if any(cred.encode() in encrypted_data for cred in creds.values()):
                print("\nWARNING: Possible plaintext in encrypted file!")
                return False
                
        return True
        
    except Exception as e:
        print(f"Error verifying credentials: {str(e)}")
        return False

if __name__ == "__main__":
    verify_credentials()