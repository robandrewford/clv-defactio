# scripts/check_credentials.py
import os
import stat
from pathlib import Path


def check_credentials_security():
    """Check credentials security settings"""
    gcp_dir = Path.home() / ".gcp"
    creds_dir = gcp_dir / "credentials"

    # Check directory permissions
    if gcp_dir.exists():
        mode = oct(gcp_dir.stat().st_mode)[-3:]
        if mode != "700":
            print(f"Warning: ~/.gcp permissions too loose: {mode}")

    # Check credentials files
    for cred_file in creds_dir.glob("*.json"):
        mode = oct(cred_file.stat().st_mode)[-3:]
        if mode not in ["600", "400"]:
            print(f"Warning: {cred_file.name} permissions too loose: {mode}")
