import os
from google.cloud import aiplatform
from google.oauth2 import service_account
import googleapiclient.discovery

# Set the path to your service account key
credentials_path = os.path.expanduser("~/.gcp/credentials/clv-dev-sa-key.json")

# Load credentials from the service account key file
credentials = service_account.Credentials.from_service_account_file(
    credentials_path, 
    scopes=['https://www.googleapis.com/auth/cloud-platform']
)

# Project and location details
project_id = "logic-dna-240402"
location = "us-west1"
model_id = "tuned-gemini-pro-20241112-201706"
version_id = "1"

# Create the AI Platform service
aiplatform_service = googleapiclient.discovery.build('aiplatform', 'v1', credentials=credentials)

# Construct the full resource name
model_version_name = f"projects/{project_id}/locations/{location}/models/{model_id}/versions/{version_id}"

try:
    # Attempt to delete the model version
    request = aiplatform_service.projects().locations().models().versions().delete(name=model_version_name)
    response = request.execute()
    print(f"Model version {model_version_name} deletion initiated.")
except Exception as e:
    print(f"Error deleting model version: {e}")