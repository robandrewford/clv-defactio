import click
import logging
import time
from pathlib import Path
from typing import Optional
from datetime import datetime
from src.pipeline.clv.runner import HierarchicalCLVRunner
from src.monitoring.alerts import send_alert

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PipelineConfig:
    def __init__(self, config_dir: str):
        self.config_dir = Path(config_dir)
        
    def validate(self) -> bool:
        """Validate configuration files exist"""
        required_configs = [
            "deployment_config.yaml",
            "model_config.yaml",
            "segment_config.yaml",
            "pipeline_config.yaml"
        ]
        
        for config in required_configs:
            if not (self.config_dir / config).exists():
                logger.error(f"Missing config file: {config}")
                return False
        return True

@click.group()
def cli():
    """Command line interface for CLV Pipeline"""
    pass

@cli.command()
@click.option('--input-table', required=True, help='BigQuery input table')
@click.option('--output-bucket', required=True, help='GCS output bucket')
@click.option('--config-dir', default='src/config', help='Configuration directory')
@click.option('--wait/--no-wait', default=False, help='Wait for pipeline completion')
def run(input_table: str, output_bucket: str, config_dir: str, wait: bool):
    """Run the CLV pipeline"""
    try:
        # Validate configuration
        config = PipelineConfig(config_dir)
        if not config.validate():
            raise ValueError("Invalid configuration")
            
        # Initialize and run pipeline
        runner = HierarchicalCLVRunner(config_dir)
        job = runner.run_pipeline(
            input_table=input_table,
            output_bucket=output_bucket
        )
        
        click.echo(f"Pipeline started: {job.job_id}")
        
        # Wait for completion if requested
        if wait:
            while True:
                status = runner.get_pipeline_status(job.job_id)
                state = status['state']
                click.echo(f"Pipeline status: {state}")
                
                if state in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
                    break
                    
                time.sleep(60)  # Check every minute
                
            if state != 'SUCCEEDED':
                raise ValueError(f"Pipeline failed with status: {state}")
                
            click.echo("Pipeline completed successfully")
            
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        send_alert(f"CLV Pipeline CLI Error: {str(e)}")
        raise click.ClickException(str(e))

@cli.command()
@click.argument('job-id')
def status(job_id: str):
    """Check status of a pipeline job"""
    try:
        runner = HierarchicalCLVRunner()
        status = runner.get_pipeline_status(job_id)
        
        click.echo("\nPipeline Status:")
        click.echo(f"State: {status['state']}")
        click.echo(f"Start Time: {status['start_time']}")
        click.echo(f"End Time: {status.get('end_time', 'N/A')}")
        
        if 'metrics' in status:
            click.echo("\nMetrics:")
            for metric, value in status['metrics'].items():
                click.echo(f"{metric}: {value}")
                
    except Exception as e:
        logger.error(f"Status check failed: {str(e)}")
        raise click.ClickException(str(e))

if __name__ == '__main__':
    cli() 