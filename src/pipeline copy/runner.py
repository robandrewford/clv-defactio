from ..config import ConfigLoader

def run_vertex_pipeline():
    # Load configuration
    config = ConfigLoader()
    vertex_config = config.get_vertex_config()
    data_config = config.get_data_config()
    model_config = config.get_model_config()
    
    # Initialize the runner
    runner = VertexPipelineRunner(
        project_id=vertex_config['project_id'],
        location=vertex_config['location'],
        pipeline_root=vertex_config.get('pipeline_root')
    )
    
    # Compile the pipeline
    runner.compile_pipeline()
    
    # Run the pipeline
    job = runner.run_pipeline(
        input_table=data_config['input']['table'],
        output_bucket=data_config['output']['bucket'],
        hyperparameters=model_config['hyperparameters']
    )
    
    # Check status
    status = runner.get_pipeline_status(job.job_id)
    print(f"Pipeline status: {status}")
    
    return job

if __name__ == "__main__":
    run_vertex_pipeline() 