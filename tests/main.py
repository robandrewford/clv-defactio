# src/main.py
import os
import warnings
import dotenv

from dotenv import load_dotenv
from google.cloud import bigquery, storage


def initialize_clients():
    """Initialize GCP clients"""
    load_dotenv()  # Load environment variables

    # Create clients
    storage_client = storage.Client(project=os.getenv("GCP_PROJECT_ID"))

    bq_client = bigquery.Client(project=os.getenv("GCP_PROJECT_ID"))

    return storage_client, bq_client
