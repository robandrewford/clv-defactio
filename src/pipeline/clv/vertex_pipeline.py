from google.cloud import aiplatform
from kfp import dsl
import os

# Configure GCP project and region
PROJECT_ID = os.environ.get('GCP_PROJECT_ID', 'your-project-id')
REGION = os.environ.get('GCP_REGION', 'us-central1')
BUCKET_URI = os.environ.get('GCP_BUCKET_URI', 'gs://your-bucket-name')

# Initialize Vertex AI
aiplatform.init(project=PROJECT_ID, location=REGION)

@dsl.component
def data_preprocessing_step(input_path: str) -> str:
    """Preprocessing step for CLV data"""
    from src.pipeline.clv.preprocessing import preprocess_data
    
    preprocessed_data = preprocess_data(input_path)
    return preprocessed_data

@dsl.component
def model_training_step(preprocessed_data: str) -> str:
    """Train CLV model"""
    from src.pipeline.clv.model import train_clv_model
    
    model_path = train_clv_model(preprocessed_data)
    return model_path

@dsl.component
def model_evaluation_step(model_path: str) -> dict:
    """Evaluate CLV model"""
    from src.pipeline.clv.evaluation import evaluate_model
    
    model_metrics = evaluate_model(model_path)
    return model_metrics

@dsl.pipeline(
    name='clv-prediction-pipeline',
    description='Customer Lifetime Value Prediction Pipeline'
)
def clv_pipeline(input_data_path: str):
    """Define the full CLV prediction pipeline"""
    preprocessing_task = data_preprocessing_step(input_data_path)
    training_task = model_training_step(preprocessing_task.output)
    evaluation_task = model_evaluation_step(training_task.output)

def submit_pipeline(input_data_path: str):
    """Submit pipeline to Vertex AI"""
    job = aiplatform.PipelineJob(
        display_name='CLV Prediction Pipeline',
        template_path=clv_pipeline,
        pipeline_root=BUCKET_URI,
        parameter_values={
            'input_data_path': input_data_path
        }
    )
    
    job.submit()

if __name__ == '__main__':
    # Example usage
    submit_pipeline('gs://your-bucket-name/input_data.csv') 