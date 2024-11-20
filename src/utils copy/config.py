# Configuration management for CLV 360
# src/utils/config.py
import os

from dotenv import load_dotenv


def load_environment():
    """Load the appropriate environment file"""
    env = os.getenv("ENVIRONMENT", "development")
    env_file = f".env.{env}"

    if os.path.exists(env_file):
        load_dotenv(env_file)
    else:
        load_dotenv()  # Fall back to .env
