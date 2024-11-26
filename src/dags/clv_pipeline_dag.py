from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.google.cloud.operators.bigquery import BigQueryCheckOperator
from airflow.providers.google.cloud.sensors.gcs import GCSObjectExistenceSensor
from datetime import datetime, timedelta

# Import project-specific modules
from src.pipeline.clv.runner import HierarchicalCLVRunner
from src.pipeline.clv.config import CLVPipelineConfig
from src.monitoring.alerts import send_alert
from src.infrastructure.gcp_setup import get_gcp_project

default_args = {
    'owner': 'data-science',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2024, 1, 1)
}

def run_clv_pipeline(**context):
    """Execute CLV pipeline with enhanced error handling and logging"""
    try:
        # Use configuration from config module
        config = CLVPipelineConfig()
        
        # Dynamically get GCP project 
        project = get_gcp_project()
        
        # Construct input and output paths
        input_table = f"{project}.{config.input_dataset}.{config.input_table}"
        output_bucket = f"{config.output_bucket}/{context['ds']}"
        
        # Initialize runner with configuration
        runner = HierarchicalCLVRunner(config)
        
        # Run pipeline with dynamic parameters
        job = runner.run_pipeline(
            input_table=input_table,
            output_bucket=f"gs://{output_bucket}"
        )
        
        # Store job details for downstream tasks
        context['task_instance'].xcom_push(key='job_id', value=job.job_id)
        context['task_instance'].xcom_push(key='output_bucket', value=output_bucket)
        
        return job.job_id
        
    except Exception as e:
        # Enhanced error handling with detailed alert
        error_message = f"CLV Pipeline Failed: {str(e)}"
        send_alert(error_message)
        raise RuntimeError(error_message)

def check_pipeline_status(**context):
    """Comprehensive pipeline status check"""
    try:
        config = CLVPipelineConfig()
        runner = HierarchicalCLVRunner(config)
        
        # Retrieve job ID from previous task
        job_id = context['task_instance'].xcom_pull(key='job_id')
        
        # Get detailed pipeline status
        status = runner.get_pipeline_status(job_id)
        
        if status['state'] != 'SUCCEEDED':
            raise ValueError(f"Pipeline failed with status: {status}")
        
        return status
    
    except Exception as e:
        send_alert(f"Pipeline Status Check Failed: {str(e)}")
        raise

with DAG(
    'clv_pipeline',
    default_args=default_args,
    description='Customer Lifetime Value Prediction Pipeline',
    schedule_interval='@daily',
    catchup=False,
    tags=['clv', 'ml', 'customer_analytics']
) as dag:
    
    # Dynamic input table check
    check_input = BigQueryCheckOperator(
        task_id='check_input_data',
        sql="""
        SELECT COUNT(*) > 0 
        FROM `{{ var.value.gcp_project }}.{{ var.value.clv_input_dataset }}.{{ var.value.clv_input_table }}`
        """,
        use_legacy_sql=False
    )
    
    # Run CLV pipeline with dynamic configuration
    run_pipeline = PythonOperator(
        task_id='run_clv_pipeline',
        python_callable=run_clv_pipeline,
        provide_context=True
    )
    
    # Check pipeline execution status
    check_status = PythonOperator(
        task_id='check_pipeline_status',
        python_callable=check_pipeline_status,
        provide_context=True
    )
    
    # Verify output model existence
    check_output = GCSObjectExistenceSensor(
        task_id='check_output_exists',
        bucket="{{ var.value.clv_output_bucket }}",
        object="{{ ds }}/model/clv_model.pkl"
    )
    
    # Define workflow dependencies
    check_input >> run_pipeline >> check_status >> check_output