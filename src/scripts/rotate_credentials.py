# scripts/rotate_credentials.py
import datetime
import os
from google.cloud import iam_v1


def rotate_service_account_key(
    project_id: str, service_account_email: str, credentials_dir: str
):
    """Rotate service account key"""
    client = iam_v1.IAMClient()

    # Create new key
    key = client.create_service_account_key(
        name=f"projects/{project_id}/serviceAccounts/{service_account_email}"
    )

    # Save new key
    new_key_path = f"{credentials_dir}/key-{datetime.date.today()}.json"
    with open(new_key_path, "w") as f:
        f.write(key.private_key_data)

    # Set permissions
    os.chmod(new_key_path, 0o600)
