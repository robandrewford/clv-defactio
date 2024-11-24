from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.google.cloud.operators.bigquery import BigQueryCheckOperator
from airflow.providers.google.cloud.sensors.gcs import GCSObjectExistenceSensor
from datetime import datetime, timedelta
from src.pipeline.clv import HierarchicalCLVRunner
from src.monitoring.alerts import send_alert

default_args = {
    'owner': 'data-science',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2024, 1, 1)
}

def run_clv_pipeline(**context):
    """Execute CLV pipeline"""
    try:
        runner = HierarchicalCLVRunner()
        job = runner.run_pipeline(
            input_table=context['params']['input_table'],
            output_bucket=f"gs://{context['params']['output_bucket']}/{context['ds']}"
        )
        
        # Store job ID for downstream tasks
        context['task_instance'].xcom_push(key='job_id', value=job.job_id)
        return job.job_id
        
    except Exception as e:
        send_alert(f"CLV Pipeline Failed: {str(e)}")
        raise

def check_pipeline_status(**context):
    """Check pipeline completion status"""
    runner = HierarchicalCLVRunner()
    job_id = context['task_instance'].xcom_pull(key='job_id')
    status = runner.get_pipeline_status(job_id)
    
    if status['state'] != 'SUCCEEDED':
        raise ValueError(f"Pipeline failed with status: {status['state']}")
    
    return status

with DAG(
    'clv_pipeline',
    default_args=default_args,
    description='Customer Lifetime Value Prediction Pipeline',
    schedule_interval='@daily',
    catchup=False,
    tags=['clv', 'ml']
) as dag:
    
    # Check if input data exists
    check_input = BigQueryCheckOperator(
        task_id='check_input_data',
        sql=f"SELECT COUNT(*) > 0 FROM {{ params.input_table }}",
        use_legacy_sql=False
    )
    
    # Run CLV pipeline
    run_pipeline = PythonOperator(
        task_id='run_clv_pipeline',
        python_callable=run_clv_pipeline,
        provide_context=True,
        params={
            'input_table': 'your-project.dataset.table',
            'output_bucket': 'your-bucket'
        }
    )
    
    # Check pipeline status
    check_status = PythonOperator(
        task_id='check_pipeline_status',
        python_callable=check_pipeline_status,
        provide_context=True
    )
    
    # Check if output exists
    check_output = GCSObjectExistenceSensor(
        task_id='check_output_exists',
        bucket="{{ params.output_bucket }}",
        object="{{ ds }}/model/clv_model.pkl"
    )
    
    # Define workflow
    check_input >> run_pipeline >> check_status >> check_output