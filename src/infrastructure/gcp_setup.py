import os
from typing import Optional, Tuple
from google.cloud import storage, bigquery
from google.auth import default
from google.auth.exceptions import DefaultCredentialsError
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_gcp_project() -> str:
    """
    Retrieve the current GCP project ID
    
    Returns:
        str: Current GCP project ID
    """
    try:
        _, project = default()
        return project
    except DefaultCredentialsError:
        logger.error("GCP credentials not found. Please authenticate.")
        raise

def check_gcp_credentials() -> bool:
    """
    Verify GCP credentials are properly configured
    
    Returns:
        bool: True if credentials are valid, False otherwise
    """
    try:
        credentials, _ = default()
        return credentials.valid
    except DefaultCredentialsError as e:
        logger.error(f"GCP credentials not found: {str(e)}")
        return False

def setup_gcp_resources(
    project_id: Optional[str] = None,
    bucket_name: Optional[str] = None,
    dataset_id: Optional[str] = None,
    location: str = "US"
) -> Tuple[bool, Optional[str]]:
    """
    Setup required GCP resources for CLV pipeline
    
    Args:
        project_id (str, optional): GCP Project ID
        bucket_name (str, optional): GCS Bucket name
        dataset_id (str, optional): BigQuery Dataset ID
        location (str, optional): GCP resource location
    
    Returns:
        Tuple[bool, Optional[str]]: Success status and error message
    """
    try:
        # Use default project if not provided
        project_id = project_id or get_gcp_project()
        
        # Validate credentials
        if not check_gcp_credentials():
            raise ValueError("Invalid GCP credentials")
        
        # Setup Storage
        if bucket_name:
            storage_client = storage.Client()
            try:
                bucket = storage_client.get_bucket(bucket_name)
                logger.info(f"Using existing bucket: {bucket_name}")
            except Exception:
                bucket = storage_client.create_bucket(
                    bucket_name, 
                    location=location
                )
                logger.info(f"Created new bucket: {bucket_name}")
        
        # Setup BigQuery
        if dataset_id:
            bq_client = bigquery.Client()
            try:
                dataset_ref = f"{project_id}.{dataset_id}"
                dataset = bq_client.get_dataset(dataset_ref)
                logger.info(f"Using existing dataset: {dataset_id}")
            except Exception:
                dataset = bigquery.Dataset(f"{project_id}.{dataset_id}")
                dataset.location = location
                dataset = bq_client.create_dataset(dataset, exists_ok=True)
                logger.info(f"Created new dataset: {dataset_id}")
        
        return True, None
    
    except Exception as e:
        error_msg = f"Failed to setup GCP resources: {str(e)}"
        logger.error(error_msg)
        return False, error_msg

def create_service_account(
    project_id: Optional[str] = None,
    service_account_name: str = "vertex-ai-pipeline",
    display_name: str = "Vertex AI Pipeline Service Account"
) -> str:
    """
    Create a service account for Vertex AI pipeline
    
    Args:
        project_id (str, optional): GCP Project ID
        service_account_name (str): Service account ID
        display_name (str): Display name for service account
    
    Returns:
        str: Service account email
    """
    from google.cloud import iam_v1
    
    project_id = project_id or get_gcp_project()
    client = iam_v1.IAMClient()
    
    # Construct service account email
    service_account_email = f"{service_account_name}@{project_id}.iam.gserviceaccount.com"
    
    try:
        # Check if service account exists
        request = iam_v1.GetServiceAccountRequest(
            name=f"projects/{project_id}/serviceAccounts/{service_account_email}"
        )
        client.get_service_account(request=request)
        logger.info(f"Service account {service_account_email} already exists")
    except Exception:
        # Create service account
        request = iam_v1.CreateServiceAccountRequest(
            account={
                'project_id': project_id,
                'account_id': service_account_name,
                'display_name': display_name
            }
        )
        client.create_service_account(request=request)
        logger.info(f"Created service account: {service_account_email}")
    
    return service_account_email

def assign_service_account_roles(
    service_account_email: str,
    project_id: Optional[str] = None,
    roles: Optional[list] = None
) -> bool:
    """
    Assign roles to service account
    
    Args:
        service_account_email (str): Service account email
        project_id (str, optional): GCP Project ID
        roles (list, optional): List of roles to assign
    
    Returns:
        bool: Success status
    """
    from google.cloud import iam_v1
    
    project_id = project_id or get_gcp_project()
    
    # Default roles for Vertex AI pipeline
    default_roles = [
        "roles/aiplatform.user",
        "roles/storage.objectAdmin",
        "roles/bigquery.dataEditor",
        "roles/bigquery.jobUser"
    ]
    
    roles = roles or default_roles
    
    try:
        for role in roles:
            # Construct role resource name
            resource = f"//cloudresourcemanager.googleapis.com/projects/{project_id}"
            
            # Create policy binding
            policy_client = iam_v1.PolicyClient()
            policy = policy_client.get_iam_policy(request={'resource': resource})
            
            binding = iam_v1.Binding(
                role=role,
                members=[f"serviceAccount:{service_account_email}"]
            )
            policy.bindings.append(binding)
            
            # Set updated policy
            policy_client.set_iam_policy(
                request={
                    'resource': resource,
                    'policy': policy
                }
            )
            logger.info(f"Assigned role {role} to {service_account_email}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to assign roles: {str(e)}")
        return False

def generate_service_account_key(
    service_account_email: str,
    output_path: Optional[str] = None
) -> str:
    """
    Generate and save service account key
    
    Args:
        service_account_email (str): Service account email
        output_path (str, optional): Path to save key file
    
    Returns:
        str: Path to generated key file
    """
    from google.cloud import iam_v1
    
    client = iam_v1.IAMClient()
    
    # Generate key
    request = iam_v1.CreateServiceAccountKeyRequest(
        name=f"projects/-/serviceAccounts/{service_account_email}"
    )
    key = client.create_service_account_key(request=request)
    
    # Determine output path
    if not output_path:
        output_path = os.path.join(
            os.getcwd(), 
            f"{service_account_email.split('@')[0]}_key.json"
        )
    
    # Write key to file
    with open(output_path, 'wb') as f:
        f.write(key.private_key_data)
    
    logger.info(f"Service account key saved to {output_path}")
    return output_path

def main():
    """
    CLI-like setup for GCP resources
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="GCP Resource Setup")
    parser.add_argument("--project-id", help="GCP Project ID")
    parser.add_argument("--bucket-name", help="GCS Bucket Name")
    parser.add_argument("--dataset-id", help="BigQuery Dataset ID")
    
    args = parser.parse_args()
    
    # Setup resources
    success, error = setup_gcp_resources(
        project_id=args.project_id,
        bucket_name=args.bucket_name,
        dataset_id=args.dataset_id
    )
    
    if not success:
        print(f"Setup failed: {error}")
        exit(1)
    
    # Create service account
    service_account = create_service_account()
    assign_service_account_roles(service_account)
    generate_service_account_key(service_account)

if __name__ == "__main__":
    main()
