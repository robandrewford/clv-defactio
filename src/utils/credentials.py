# Credential management for CLV 360
# src/utils/credentials.py
import os
from pathlib import Path

from google.auth import default
from google.oauth2 import service_account


class CredentialManager:
    """Manages GCP credentials"""

    def __init__(self):
        self.creds_dir = Path.home() / ".gcp" / "credentials"
        self.environment = os.getenv("ENVIRONMENT", "development")

    def get_credentials(self):
        """Get appropriate credentials for environment"""
        if self.environment == "local":
            # Use application default credentials
            return default()[0]

        # Get service account credentials
        creds_file = self.get_credentials_file()
        if creds_file.exists():
            return service_account.Credentials.from_service_account_file(
                str(creds_file)
            )
        else:
            raise FileNotFoundError(f"Credentials file not found: {creds_file}")

    def get_credentials_file(self) -> Path:
        """Get credentials file path"""
        return self.creds_dir / f"clv-{self.environment}-sa.json"

    @staticmethod
    def validate_credentials_file(file_path: Path) -> bool:
        """Validate credentials file"""
        if not file_path.exists():
            return False

        # Check permissions (600 or stricter)
        return oct(file_path.stat().st_mode)[-3:] in ["600", "400"]
