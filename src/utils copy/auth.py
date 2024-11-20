# src/utils/auth.py
import os

from google.oauth2 import service_account


def get_credentials():
    """Get appropriate credentials based on environment"""
    env = os.getenv("ENVIRONMENT", "development")

    if env == "development":
        # Use application default credentials (your personal account)
        return None  # ADC will be used automatically
    else:
        # Use service account
        return service_account.Credentials.from_service_account_file(
            os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        )
