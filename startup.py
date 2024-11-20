# src/startup.py
from scripts.validate_env import validate_environment
from src.utils.config import load_environment


def startup():
    """Application startup"""
    # Load environment
    load_environment()

    # Validate environment
    validate_environment()

    # Initialize services
    initialize_clients()
