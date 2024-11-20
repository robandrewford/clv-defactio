# Validate environment variables for CLV 360
# scripts/validate_env.py
import os
import sys

from dotenv import load_dotenv

required_vars = [
    "GCP_PROJECT_ID",
    "GCP_REGION",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "ENVIRONMENT",
]

optional_vars = ["GCP_ZONE", "PIPELINE_BUCKET", "BIGQUERY_DATASET", "LOG_LEVEL"]


def validate_environment():
    """Validate environment variables"""
    load_dotenv()

    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        print("Error: Missing required environment variables:")
        for var in missing_vars:
            print(f"- {var}")
        sys.exit(1)

    print("Environment validation successful!")
    print("\nCurrent configuration:")
    for var in required_vars + optional_vars:
        value = os.getenv(var)
        if value:
            # Mask sensitive values
            if "CREDENTIALS" in var or "KEY" in var:
                value = "****"
            print(f"{var}: {value}")


if __name__ == "__main__":
    validate_environment()
