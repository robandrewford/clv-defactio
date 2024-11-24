from typing import Dict, Optional, Any
from datetime import datetime
from google.cloud import aiplatform
from .vertex_components import hierarchical_clv_pipeline as hierarchical_clv_pipeline
from .config import CLVConfigLoader
from .model import HierarchicalCLVModel
from .preprocessing import CLVDataPreprocessor
from .segmentation import CustomerSegmentation

class HierarchicalCLVRunner:
    """Runner for Hierarchical CLV Vertex AI Pipeline"""
    
    def __init__(self, config_dir: str = "src/config"):
        # Load configurations
        self.config_loader = CLVConfigLoader(config_dir)
        vertex_config = self.config_loader.get_vertex_config()
        
        # Set up runner configuration
        self.project_id = vertex_config.get("project_id")
        self.location = vertex_config.get("location", "us-central1")
        self.pipeline_root = vertex_config.get("pipeline_root")
        
        # Get machine configuration
        self.machine_config = vertex_config.get("resources", {})
        
        # Initialize Vertex AI
        aiplatform.init(
            project=self.project_id,
            location=self.location
        )
        
        self.processor = CLVDataPreprocessor(self.config_loader)
        self.segmenter = CustomerSegmentation(self.config_loader)
        self.model = HierarchicalCLVModel(self.config_loader)
    
    def run_pipeline(
        self,
        input_table: str,
        output_bucket: Optional[str] = None,
    ) -> aiplatform.PipelineJob:
        """Run the CLV pipeline on Vertex AI"""
        
        # Get storage configuration
        storage_config = self.config_loader.get_storage_config()
        output_bucket = output_bucket or storage_config.get("gcs", {}).get("bucket_name")
        
        if not output_bucket:
            raise ValueError("No output bucket specified in config or parameters")
            
        # Set up pipeline job
        job = aiplatform.PipelineJob(
            display_name=f"clv-pipeline-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            template_path="pipeline.json",
            pipeline_root=self.pipeline_root,
            parameter_values={
                'project_id': self.project_id,
                'input_table': input_table,
                'output_bucket': output_bucket,
                'config_dir': str(self.config_loader.config_dir)
            },
            machine_type=self.machine_config.get("machine_type", "n1-standard-4"),
            accelerator_type=self.machine_config.get("accelerator_type"),
            accelerator_count=self.machine_config.get("accelerator_count", 0)
        )
        
        # Set up monitoring if enabled
        monitoring_config = self.config_loader.get_monitoring_config()
        if monitoring_config.get("enable_monitoring", False):
            job.enable_monitoring(
                alert_email=monitoring_config.get("alert_email"),
                metrics=monitoring_config.get("metrics", []),
                metrics_frequency=monitoring_config.get("metrics_frequency", 60)
            )
        
        # Run the pipeline
        job.run(sync=False)
        return job

    def get_pipeline_status(self, job_id: str) -> Dict[str, Any]:
        """Get detailed pipeline status with monitoring metrics"""
        job = aiplatform.PipelineJob.get(job_id)
        
        status = {
            'state': job.state,
            'create_time': job.create_time,
            'start_time': job.start_time,
            'end_time': job.end_time,
            'error': job.error
        }
        
        # Add monitoring metrics if available
        monitoring_config = self.config_loader.get_monitoring_config()
        if monitoring_config.get("enable_monitoring", False):
            try:
                metrics = job.get_metrics()
                status['metrics'] = metrics
            except Exception as e:
                status['metrics_error'] = str(e)
                
        return status