from src.pipeline.clv.runner import HierarchicalCLVRunner
from src.pipeline.clv.config import CLVPipelineConfig
from src.monitoring.alerts import send_alert
from src.infrastructure.gcp_setup import get_gcp_project

def trigger_pipeline(event, context):
    """
    Cloud Function to trigger CLV pipeline
    
    Expected event structure:
    {
        'input_table': Optional[str],  # Override default input table if needed
        'output_bucket': Optional[str]  # Override default output bucket if needed
    }
    """
    try:
        # Load configuration
        config = CLVPipelineConfig()
        
        # Get current GCP project
        project = get_gcp_project()
        
        # Determine input table (use event param or config default)
        input_table = event.get('input_table', 
            f"{project}.{config.input_dataset}.{config.input_table}")
        
        # Determine output bucket (use event param or config default)
        output_bucket = event.get('output_bucket', 
            f"gs://{config.output_bucket}/manual_trigger")
        
        # Initialize runner with configuration
        runner = HierarchicalCLVRunner(config)
        
        # Run pipeline
        job = runner.run_pipeline(
            input_table=input_table,
            output_bucket=output_bucket
        )
        
        # Return job details
        return {
            "job_id": job.job_id,
            "input_table": input_table,
            "output_bucket": output_bucket
        }
    
    except Exception as e:
        # Log and alert on any pipeline trigger failures
        error_message = f"CLV Pipeline Trigger Failed: {str(e)}"
        send_alert(error_message)
        raise RuntimeError(error_message) 