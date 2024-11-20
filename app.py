# src/app.py
from src.utils.credentials import CredentialManager


def initialize_app():
    # Setup credentials
    cred_manager = CredentialManager()
    credentials = cred_manager.get_credentials()

    # Initialize clients with credentials
    storage_client = storage.Client(
        credentials=credentials, project=os.getenv("GCP_PROJECT_ID")
    )
