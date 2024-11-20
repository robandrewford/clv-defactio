# scripts/dev_setup.py
import os
import subprocess
from pathlib import Path


def setup_development_environment():
    """Setup local development environment"""
    try:
        # Create virtual environment
        subprocess.run(["python", "-m", "venv", ".venv"])

        # Activate virtual environment
        activate_script = ".venv/bin/activate"
        if os.name == "nt":  # Windows
            activate_script = ".venv\\Scripts\\activate"

        # Install dependencies
        subprocess.run(["pip", "install", "-r", "requirements-dev.txt"])

        # Setup pre-commit hooks
        subprocess.run(["pre-commit", "install"])

        # Setup GCP configuration
        setup_gcp_configuration()

        print("Development environment setup complete!")

    except Exception as e:
        print(f"Setup failed: {str(e)}")
        raise


def setup_gcp_configuration():
    """Setup GCP configuration for development"""
    try:
        # Check for gcloud SDK
        result = subprocess.run(["gcloud", "version"], capture_output=True)

        if result.returncode != 0:
            print("Please install Google Cloud SDK")
            return

        # Initialize gcloud
        subprocess.run(["gcloud", "init"])

        # Setup application default credentials
        subprocess.run(["gcloud", "auth", "application-default", "login"])

    except Exception as e:
        print(f"GCP configuration failed: {str(e)}")
        raise


if __name__ == "__main__":
    setup_development_environment()
