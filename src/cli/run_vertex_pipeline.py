import os
import click
import yaml
from typing import Dict, Any
from pathlib import Path

# Import your pipeline submission function
from src.pipeline.vertex_pipeline import submit_pipeline

def load_deployment_config(config_path: str = None) -> Dict[Any, Any]:
    """
    Load deployment configuration from YAML file
    
    Args:
        config_path (str, optional): Path to deployment config file
    
    Returns:
        Dict: Deployment configuration
    """
    # Default config path
    if not config_path:
        config_path = os.path.join(
            os.path.dirname(__file__), 
            '..', 
            'config', 
            'deployment_config.yaml'
        )
    
    # Expand user and resolve path
    config_path = os.path.expanduser(config_path)
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        click.echo(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        click.echo(f"Error parsing configuration file: {e}")
        raise

def set_gcp_environment(config: Dict[Any, Any]):
    """
    Set GCP-related environment variables from config
    
    Args:
        config (Dict): Deployment configuration
    """
    # Set project ID
    os.environ['GCP_PROJECT_ID'] = config.get('project_id', '')
    os.environ['GCP_REGION'] = config.get('region', 'us-west1')
    
    # Set service account credentials
    service_account_key = config.get('security', {}).get('service_account_key_path')
    if service_account_key:
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.expanduser(service_account_key)

@click.command()
@click.option('--config-path', 
              default=None, 
              help='Path to deployment configuration file')
@click.option('--input-path', 
              required=True, 
              help='GCS path to input data')
@click.option('--override-project', 
              default=None, 
              help='Override project ID from config')
@click.option('--override-region', 
              default=None, 
              help='Override region from config')
def main(config_path, input_path, override_project, override_region):
    """
    Run Vertex AI Pipeline for CLV Prediction
    
    Reads configuration from deployment_config.yaml and submits pipeline
    
    Args:
        config_path (str): Path to configuration file
        input_path (str): GCS path to input data
        override_project (str, optional): Override project ID
        override_region (str, optional): Override region
    """
    try:
        # Load configuration
        config = load_deployment_config(config_path)
        
        # Override project and region if provided
        if override_project:
            config['project_id'] = override_project
        if override_region:
            config['region'] = override_region
        
        # Set GCP environment
        set_gcp_environment(config)
        
        # Validate input path
        if not input_path.startswith('gs://'):
            click.echo("Error: Input path must be a GCS path (gs://)")
            return
        
        # Submit pipeline
        click.echo(f"Submitting pipeline with input: {input_path}")
        submit_pipeline(input_path)
        
        click.echo("Pipeline submitted successfully!")
    
    except Exception as e:
        click.echo(f"Pipeline submission failed: {str(e)}")
        raise click.ClickException(str(e))

if __name__ == '__main__':
    main() 