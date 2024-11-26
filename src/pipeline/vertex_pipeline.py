from src.pipeline.clv.vertex_components import hierarchical_clv_pipeline
from google.cloud import aiplatform
import os

def submit_pipeline(input_path: str):
    """
    Submit CLV pipeline to Vertex AI
    
    Args:
        input_path (str): GCS path to input data
    """
    # Get project and region from environment
    project_id = os.environ.get('GCP_PROJECT_ID')
    region = os.environ.get('GCP_REGION', 'us-west1')
    
    # Initialize Vertex AI
    aiplatform.init(project=project_id, location=region)
    
    # Load deployment config
    from src.cli.run_vertex_pipeline import load_deployment_config
    config = load_deployment_config()
    
    # Submit pipeline
    job = aiplatform.PipelineJob(
        display_name='CLV Prediction Pipeline',
        template_path=hierarchical_clv_pipeline,
        pipeline_root=f'gs://{project_id}-vertex-pipelines',
        parameter_values={
            'data_path': input_path,
            'config_dict': config
        }
    )
    
    job.submit() 