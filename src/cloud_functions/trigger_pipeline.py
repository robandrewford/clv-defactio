from src.pipeline.clv import HierarchicalCLVRunner

def trigger_pipeline(event, context):
    """Cloud Function to trigger CLV pipeline"""
    runner = HierarchicalCLVRunner()
    job = runner.run_pipeline(
        input_table=event['input_table'],
        output_bucket=event['output_bucket']
    )
    return {"job_id": job.job_id} 