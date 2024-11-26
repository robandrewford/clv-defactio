import click
import logging
import time
from pathlib import Path
from typing import Optional
from datetime import datetime

# Import project-specific modules
from src.pipeline.clv.runner import HierarchicalCLVRunner
from src.pipeline.clv.config import CLVPipelineConfig
from src.pipeline.clv.vertex_components import VertexAIPipelineOrchestrator
from src.infrastructure.gcp_setup import get_gcp_project
from src.monitoring.alerts import send_alert
from src.pipeline.clv.data_validation import DataValidator
from src.pipeline.clv.model_registry import ModelRegistry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@click.group()
def cli():
    """Command line interface for Customer Lifetime Value (CLV) Pipeline"""
    pass

@cli.command()
@click.option('--input-table', help='BigQuery input table (overrides config)')
@click.option('--output-bucket', help='GCS output bucket (overrides config)')
@click.option('--config-path', default=None, help='Path to custom configuration')
@click.option('--validate/--no-validate', default=True, help='Validate input data before pipeline')
@click.option('--wait/--no-wait', default=False, help='Wait for pipeline completion')
@click.option('--register/--no-register', default=True, help='Register model after pipeline')
def run(input_table, output_bucket, config_path, validate, wait, register):
    """
    Run the full CLV pipeline with comprehensive options
    """
    try:
        # Load configuration
        config = CLVPipelineConfig(config_path) if config_path else CLVPipelineConfig()
        
        # Get current GCP project
        project = get_gcp_project()
        
        # Determine input table and output bucket
        input_table = input_table or f"{project}.{config.input_dataset}.{config.input_table}"
        output_bucket = output_bucket or f"gs://{config.output_bucket}/{datetime.now().strftime('%Y%m%d')}"
        
        # Optional: Validate input data
        if validate:
            validator = DataValidator(config)
            validation_result = validator.validate_input_data(input_table)
            if not validation_result['is_valid']:
                raise ValueError(f"Data validation failed: {validation_result['errors']}")
        
        # Initialize pipeline runner
        runner = HierarchicalCLVRunner(config)
        
        # Optional: Use Vertex AI Pipeline Orchestrator
        orchestrator = VertexAIPipelineOrchestrator(config)
        
        # Run pipeline
        click.echo(f"Starting CLV Pipeline for input: {input_table}")
        job = runner.run_pipeline(
            input_table=input_table,
            output_bucket=output_bucket
        )
        
        click.echo(f"Pipeline Job ID: {job.job_id}")
        
        # Wait for completion if requested
        if wait:
            while True:
                status = runner.get_pipeline_status(job.job_id)
                state = status.get('state', 'UNKNOWN')
                click.echo(f"Pipeline status: {state}")
                
                if state in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
                    break
                    
                time.sleep(60)  # Check every minute
            
            # Raise error if pipeline failed
            if state != 'SUCCEEDED':
                raise ValueError(f"Pipeline failed with status: {state}")
        
        # Optional: Register model
        if register:
            try:
                model_registry = ModelRegistry(config)
                model_version = model_registry.register_model(
                    job_id=job.job_id,
                    input_table=input_table,
                    output_bucket=output_bucket
                )
                click.echo(f"Model registered: {model_version}")
            except Exception as reg_error:
                logger.warning(f"Model registration failed: {str(reg_error)}")
        
        # Send success notification
        send_alert(f"CLV Pipeline Completed Successfully: {job.job_id}")
        
        return job.job_id
    
    except Exception as e:
        # Comprehensive error handling
        error_message = f"CLV Pipeline Execution Failed: {str(e)}"
        logger.error(error_message)
        send_alert(error_message)
        raise click.ClickException(error_message)

@cli.command()
@click.argument('job_id')
@click.option('--detailed/--simple', default=False, help='Show detailed pipeline status')
def status(job_id: str, detailed: bool):
    """Check status of a pipeline job"""
    try:
        # Initialize runner
        runner = HierarchicalCLVRunner()
        
        # Get pipeline status
        status = runner.get_pipeline_status(job_id)
        
        # Display status
        click.echo("\nPipeline Status:")
        click.echo(f"Job ID: {job_id}")
        click.echo(f"State: {status.get('state', 'UNKNOWN')}")
        click.echo(f"Start Time: {status.get('start_time', 'N/A')}")
        click.echo(f"End Time: {status.get('end_time', 'N/A')}")
        
        # Optional detailed output
        if detailed and 'details' in status:
            click.echo("\nDetailed Information:")
            for key, value in status.get('details', {}).items():
                click.echo(f"{key}: {value}")
        
        # Optional metrics
        if 'metrics' in status:
            click.echo("\nMetrics:")
            for metric, value in status.get('metrics', {}).items():
                click.echo(f"{metric}: {value}")
    
    except Exception as e:
        logger.error(f"Status check failed: {str(e)}")
        raise click.ClickException(str(e))

if __name__ == '__main__':
    cli() 