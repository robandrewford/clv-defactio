import subprocess
import requests
import time
from typing import List, Optional
from dataclasses import dataclass
import yaml
from pathlib import Path
from google.cloud import aiplatform
from google.oauth2 import service_account
import os
from google.cloud import aiplatform_v1
from google.cloud.aiplatform_v1 import MetadataServiceClient

@dataclass
class CloudConfig:
    project_id: str
    region: str
    credentials_path: str
    service_account: str

    @classmethod
    def from_config(cls) -> 'CloudConfig':
        config_path = Path(__file__).parents[3] / "config" / "deployment_config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        credentials_path = os.path.expanduser(config['security']['service_account_key_path'])
        
        return cls(
            project_id=config['project_id'],
            region=config['region'],
            credentials_path=credentials_path,
            service_account=config['security']['service_account']
        )

class VertexAICleaner:
    def __init__(self):
        self.config = CloudConfig.from_config()
        
        # Check if credentials file exists
        if not os.path.exists(self.config.credentials_path):
            raise FileNotFoundError(
                f"Service account key file not found at {self.config.credentials_path}. "
                "Please ensure the file exists and the path is correct in deployment_config.yaml"
            )
            
        self.credentials = service_account.Credentials.from_service_account_file(
            self.config.credentials_path
        )
        self.headers = {
            'Authorization': f'Bearer {self.credentials.token}',
            'Content-Type': 'application/json'
        }

    def _get_access_token(self) -> str:
        """Get GCP access token."""
        return subprocess.check_output(
            ['gcloud', 'auth', 'print-access-token']
        ).decode('utf-8').strip()

    def _get_custom_jobs(self) -> List[str]:
        """Get list of all custom jobs."""
        try:
            result = subprocess.check_output([
                'gcloud', 'ai', 'custom-jobs', 'list',
                f'--region={self.config.region}',
                f'--project={self.config.project_id}',
                '--format=value(name)'
            ]).decode('utf-8').splitlines()
            return result
        except subprocess.CalledProcessError as e:
            print(f"Error getting custom jobs: {e}")
            return []

    def _get_models(self) -> List[str]:
        """Get list of all models."""
        try:
            result = subprocess.check_output([
                'gcloud', 'ai', 'models', 'list',
                f'--region={self.config.region}',
                f'--project={self.config.project_id}',
                '--format=value(name)'
            ]).decode('utf-8').splitlines()
            return result
        except subprocess.CalledProcessError as e:
            print(f"Error getting models: {e}")
            return []

    def _get_endpoints(self) -> List[str]:
        """Get list of all endpoints."""
        try:
            result = subprocess.check_output([
                'gcloud', 'ai', 'endpoints', 'list',
                f'--region={self.config.region}',
                f'--project={self.config.project_id}',
                '--format=value(name)'
            ]).decode('utf-8').splitlines()
            return result
        except subprocess.CalledProcessError as e:
            print(f"Error getting endpoints: {e}")
            return []

    def _get_experiments(self) -> List[str]:
        """Get list of all experiments."""
        try:
            result = subprocess.check_output([
                'gcloud', 'ai', 'experiments', 'list',
                f'--region={self.config.region}',
                f'--project={self.config.project_id}',
                '--format=value(name)'
            ]).decode('utf-8').splitlines()
            return result
        except subprocess.CalledProcessError as e:
            print(f"Error getting experiments: {e}")
            return []

    def _get_tensorboards(self) -> List[str]:
        """Get list of all tensorboards."""
        try:
            # Use credentials with gcloud command
            result = subprocess.check_output([
                'gcloud', 'ai', 'tensorboards', 'list',
                f'--region={self.config.region}',
                f'--project={self.config.project_id}',
                f'--impersonate-service-account={self.config.service_account}',
                '--format=value(name)'
            ]).decode('utf-8').splitlines()
            return result
        except subprocess.CalledProcessError as e:
            print(f"Error getting tensorboards: {e}")
            return []

    def _get_metadata_store(self) -> str:
        """Get the metadata store name."""
        try:
            client = MetadataServiceClient(credentials=self.credentials)
            parent = f"projects/{self.config.project_id}/locations/{self.config.region}"
            store_name = f"{parent}/metadataStores/default"
            return store_name
        except Exception as e:
            print(f"Error getting metadata store: {e}")
            return None

    def delete_resource(self, resource_id: str) -> bool:
        """Delete a specific resource by ID."""
        url = f'https://{self.config.region}-aiplatform.googleapis.com/v1/{resource_id}'
        try:
            response = requests.delete(url, headers=self.headers)
            return response.status_code in [200, 204]
        except requests.RequestException as e:
            print(f"Error deleting resource {resource_id}: {e}")
            return False

    def cleanup_custom_jobs(self):
        """Clean up all custom jobs."""
        print("Starting custom jobs cleanup...")
        jobs = self._get_custom_jobs()
        
        for job_id in jobs:
            print(f"Deleting custom job: {job_id}")
            if self.delete_resource(job_id):
                print(f"Successfully deleted job: {job_id}")
            else:
                print(f"Failed to delete job: {job_id}")
            time.sleep(1)  # Rate limiting

    def cleanup_models(self):
        """Clean up all models."""
        print("Starting models cleanup...")
        models = self._get_models()
        
        for model_id in models:
            print(f"Deleting model: {model_id}")
            if self.delete_resource(model_id):
                print(f"Successfully deleted model: {model_id}")
            else:
                print(f"Failed to delete model: {model_id}")
            time.sleep(1)

    def cleanup_endpoints(self):
        """Clean up all endpoints."""
        print("Starting endpoints cleanup...")
        endpoints = self._get_endpoints()
        
        for endpoint_id in endpoints:
            print(f"Deleting endpoint: {endpoint_id}")
            if self.delete_resource(endpoint_id):
                print(f"Successfully deleted endpoint: {endpoint_id}")
            else:
                print(f"Failed to delete endpoint: {endpoint_id}")
            time.sleep(1)

    def cleanup_tensorboards(self):
        """Clean up all tensorboards."""
        print("Starting tensorboards cleanup...")
        
        # Initialize Vertex AI with credentials
        aiplatform.init(
            project=self.config.project_id,
            location=self.config.region,
            credentials=self.credentials
        )
        
        try:
            # List all tensorboards
            tensorboards = self._get_tensorboards()
            
            for tensorboard_id in tensorboards:
                print(f"Deleting tensorboard: {tensorboard_id}")
                try:
                    # Create tensorboard instance and delete it
                    tensorboard = aiplatform.Tensorboard(
                        tensorboard_name=tensorboard_id
                    )
                    tensorboard.delete(force=True)  # Force deletion even if tensorboard has experiments
                    print(f"Successfully deleted tensorboard: {tensorboard_id}")
                except Exception as e:
                    print(f"Failed to delete tensorboard {tensorboard_id}: {e}")
                time.sleep(1)  # Rate limiting
                
        except Exception as e:
            print(f"Error during tensorboard cleanup: {e}")

    def cleanup_experiments(self, delete_backing_tensorboard_runs: bool = False):
        """Clean up all experiments using Vertex AI SDK."""
        print("Starting experiments cleanup...")
        
        # Initialize Vertex AI with credentials
        aiplatform.init(
            project=self.config.project_id,
            location=self.config.region,
            credentials=self.credentials
        )
        
        try:
            # Use the experiments() method instead of ExperimentResource
            experiments = aiplatform.Experiment.list(
                project=self.config.project_id,
                location=self.config.region,
                credentials=self.credentials
            )
            
            for experiment in experiments:
                print(f"Deleting experiment: {experiment.name}")
                try:
                    experiment.delete(
                        delete_backing_tensorboard_runs=delete_backing_tensorboard_runs,
                        force=True
                    )
                    print(f"Successfully deleted experiment: {experiment.name}")
                except Exception as e:
                    print(f"Failed to delete experiment {experiment.name}: {e}")
                time.sleep(1)
                
        except Exception as e:
            print(f"Error listing experiments: {e}")

    def _get_or_create_metadata_store(self, store_name='CLV'):
        """Get or create a metadata store with a specific name."""
        try:
            # Initialize Vertex AI
            aiplatform.init(
                project=self.config.project_id,
                location=self.config.region,
                credentials=self.credentials
            )
            parent = f"projects/{self.config.project_id}/locations/{self.config.region}"
            full_store_name = f"{parent}/metadataStores/{store_name}"
            
            try:
                # Try to get existing store
                metadata_store = aiplatform.MetadataStore.get(full_store_name)
                print(f"Found existing metadata store: {full_store_name}")
            except Exception:
                # If store doesn't exist, create it
                print(f"Creating new metadata store: {full_store_name}")
                metadata_store = aiplatform.MetadataStore.create(
                    display_name=store_name,
                    description=f"Metadata store for CLV project"
                    # project=self.config.project_id, # Not needed, already set in init
                    # location=self.config.region, # Not needed, already set in init
                    # credentials=self.credentials # Not needed, already set in init
                )
                print(f"Created metadata store: {metadata_store.resource_name}")
            
            return metadata_store.resource_name
            
        except Exception as e:
            print(f"Error with metadata store: {e}")
            return None

    def cleanup_metadata(self, store_name='CLV'):
        """Clean up ML metadata for a specific store."""
        print(f"Starting ML metadata cleanup for store: {store_name}")
        
        try:
            # Initialize the Metadata Service client
            client = MetadataServiceClient(credentials=self.credentials)
            
            # Set up parent and store name
            parent = f"projects/{self.config.project_id}/locations/{self.config.region}"
            full_store_name = f"{parent}/metadataStores/{store_name}"
            
            print(f"Getting metadata store: {full_store_name}")
            try:
                # Get the metadata store
                client.get_metadata_store(name=full_store_name)
                print("Metadata store exists")
            except Exception as e:
                print(f"Metadata store not found, skipping cleanup: {e}")
                return
            
            # List artifacts
            try:
                if not full_store_name:
                    print("Metadata store name is empty, skipping artifact cleanup.")
                    return

                print(f"Listing artifacts in metadata store: {full_store_name}")

                # Create list request
                list_request = aiplatform_v1.ListArtifactsRequest(
                    parent=full_store_name
                )
                
                # List artifacts
                artifacts_iterator = client.list_artifacts(request=list_request)
                
                artifact_count = 0
                # Delete each artifact
                for artifact in artifacts_iterator:
                    artifact_count += 1
                    print(f"Found artifact: {artifact.name}")
                    
                    try:
                        # Create delete request
                        delete_request = aiplatform_v1.DeleteArtifactRequest(
                            name=artifact.name,
                            force=True
                        )
                        # Delete the artifact
                        client.delete_artifact(request=delete_request)
                        print(f"Successfully deleted artifact: {artifact.name}")
                    except Exception as delete_error:
                        print(f"Failed to delete artifact {artifact.name}: {delete_error}")
                    
                    time.sleep(1)  # Rate limiting
                
                if artifact_count == 0:
                    print("No artifacts found in the metadata store")
            
            except Exception as list_error:
                print(f"Error listing/deleting artifacts: {list_error}")
            
            print("ML metadata cleanup completed")
        
        except Exception as e:
            print(f"Error during metadata cleanup: {e}")

    def delete_specific_artifact(self, artifact_id: str, store_name='CLV'):
        """Delete a specific artifact by ID from a specific store."""
        try:
            client = MetadataServiceClient(credentials=self.credentials)
            artifact_name = f"projects/{self.config.project_id}/locations/{self.config.region}/metadataStores/{store_name}/artifacts/{artifact_id}"
            
            print(f"Deleting artifact: {artifact_name}")
            delete_request = aiplatform_v1.DeleteArtifactRequest(
                name=artifact_name,
                force=True
            )
            client.delete_artifact(request=delete_request)
            print(f"Successfully deleted artifact: {artifact_name}")
        except Exception as e:
            print(f"Error deleting artifact {artifact_id}: {e}")

    def cleanup_all(self):
        """Clean up all resources."""
        self.cleanup_endpoints()      # Delete endpoints first
        self.cleanup_models()         # Then models
        self.cleanup_custom_jobs()    # Then jobs
        self.cleanup_tensorboards()   # Then tensorboards
        self.cleanup_experiments()    # Then experiments
        self.cleanup_metadata()       # Finally metadata


def main():
    cleaner = VertexAICleaner()
    
    import argparse
    parser = argparse.ArgumentParser(description='Clean up Vertex AI resources')
    parser.add_argument('--type', 
                       choices=['all', 'jobs', 'models', 'endpoints', 
                               'experiments', 'tensorboards', 'metadata', 'artifact'], 
                       default='all', 
                       help='Type of resources to clean up')
    parser.add_argument('--artifact-id',
                       help='Specific artifact ID to delete')
    parser.add_argument('--delete-tensorboard-runs',
                        action='store_true',
                        help='Delete backing tensorboard runs for experiments')
    parser.add_argument('--store-name',
                       default='CLV',
                       help='Name of the metadata store')
    args = parser.parse_args()
    
    if args.type == 'artifact' and args.artifact_id:
        cleaner.delete_specific_artifact(args.artifact_id, store_name=args.store_name)
    elif args.type == 'all':
        cleaner.cleanup_all()
    elif args.type == 'jobs':
        cleaner.cleanup_custom_jobs()
    elif args.type == 'models':
        cleaner.cleanup_models()
    elif args.type == 'endpoints':
        cleaner.cleanup_endpoints()
    elif args.type == 'tensorboards':
        cleaner.cleanup_tensorboards()
    elif args.type == 'experiments':
        cleaner.cleanup_tensorboards()  # Delete tensorboards first
        cleaner.cleanup_experiments(delete_backing_tensorboard_runs=args.delete_tensorboard_runs) 
    elif args.type == 'metadata' or args.type == 'artifact':
        cleaner.cleanup_metadata(store_name=args.store_name)


if __name__ == "__main__":
    main() 
