from typing import Dict, Optional
from kfp import dsl
from kfp.v2 import compiler
from google.cloud import aiplatform
import logging

@dsl.component(
    base_image="python:3.9",
    packages_to_install=["pandas", "numpy", "scikit-learn"]
)
def preprocess_data(
    project: str,
    location: str,
    input_table: str,
    output_path: str
) -> str:
    """Preprocess data component for CLV pipeline"""
    import pandas as pd
    from google.cloud import bigquery

    # Read from BigQuery
    client = bigquery.Client(project=project)
    df = client.query(f"SELECT * FROM {input_table}").to_dataframe()
    
    # Preprocessing logic here
    
    # Save processed data
    df.to_parquet(output_path)
    return output_path

@dsl.component(
    base_image="python:3.9",
    packages_to_install=["pandas", "numpy", "scikit-learn"]
)
def train_model(
    data_path: str,
    model_dir: str,
    hyperparameters: Dict
) -> str:
    """Train CLV model component"""
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    import joblib
    
    # Load data
    df = pd.read_parquet(data_path)
    
    # Training logic here
    model = RandomForestRegressor(**hyperparameters)
    
    # Save model
    model_path = f"{model_dir}/model.joblib"
    joblib.dump(model, model_path)
    return model_path

@dsl.component(
    base_image="python:3.9",
    packages_to_install=["pandas", "numpy", "scikit-learn"]
)
def evaluate_model(
    model_path: str,
    test_data_path: str,
    metrics_path: str
) -> Dict:
    """Evaluate CLV model component"""
    import pandas as pd
    import joblib
    from sklearn.metrics import mean_squared_error, r2_score
    
    # Load model and data
    model = joblib.load(model_path)
    test_df = pd.read_parquet(test_data_path)
    
    # Evaluation logic here
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }
    
    # Save metrics
    pd.DataFrame([metrics]).to_json(metrics_path)
    return metrics

@dsl.pipeline(
    name='clv-prediction-pipeline',
    description='End-to-end CLV prediction pipeline'
)
def clv_pipeline(
    project_id: str,
    location: str,
    input_table: str,
    output_bucket: str,
    hyperparameters: Dict = None
):
    """Define the Vertex AI Pipeline"""
    
    # Set default hyperparameters
    if hyperparameters is None:
        hyperparameters = {
            'n_estimators': 100,
            'max_depth': 10
        }
    
    # Preprocess data
    preprocess_task = preprocess_data(
        project=project_id,
        location=location,
        input_table=input_table,
        output_path=f"{output_bucket}/processed_data.parquet"
    )
    
    # Train model
    train_task = train_model(
        data_path=preprocess_task.output,
        model_dir=f"{output_bucket}/model",
        hyperparameters=hyperparameters
    )
    
    # Evaluate model
    evaluate_task = evaluate_model(
        model_path=train_task.output,
        test_data_path=preprocess_task.output,
        metrics_path=f"{output_bucket}/metrics.json"
    )

class VertexPipelineRunner:
    """Runner for Vertex AI Pipelines"""
    
    def __init__(
        self,
        project_id: str,
        location: str = "us-central1",
        pipeline_root: str = None
    ):
        self.project_id = project_id
        self.location = location
        self.pipeline_root = pipeline_root or f"gs://{project_id}-pipeline-root"
        self.logger = logging.getLogger("vertex.pipeline")
        
        # Initialize Vertex AI
        aiplatform.init(
            project=project_id,
            location=location
        )
    
    def compile_pipeline(
        self,
        output_path: str = "pipeline.json"
    ):
        """Compile the pipeline to JSON format"""
        compiler.Compiler().compile(
            pipeline_func=clv_pipeline,
            package_path=output_path
        )
    
    def run_pipeline(
        self,
        input_table: str,
        output_bucket: str,
        pipeline_name: str = None,
        hyperparameters: Dict = None
    ) -> aiplatform.PipelineJob:
        """Run the pipeline on Vertex AI"""
        
        job = aiplatform.PipelineJob(
            display_name=pipeline_name or f"clv-pipeline-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            template_path="pipeline.json",
            pipeline_root=self.pipeline_root,
            parameter_values={
                'project_id': self.project_id,
                'location': self.location,
                'input_table': input_table,
                'output_bucket': output_bucket,
                'hyperparameters': hyperparameters
            }
        )
        
        job.run(sync=False)
        return job

    def get_pipeline_status(self, job_id: str) -> Dict:
        """Get status of a pipeline run"""
        job = aiplatform.PipelineJob.get(job_id)
        return {
            'state': job.state,
            'create_time': job.create_time,
            'start_time': job.start_time,
            'end_time': job.end_time,
            'error': job.error
        }
