from kfp import dsl
from kfp import compiler
from kfp.dsl import (
    Input,
    Output,
    Dataset,
    Model,
    Metrics,
    Artifact,
    ClassificationMetrics
)
import joblib
import yaml
import pandas as pd
import numpy as np
from typing import Dict, Any

__all__ = ['hierarchical_clv_pipeline']

@dsl.component(
    base_image='python:3.10',
    packages_to_install=['pandas', 'numpy', 'scikit-learn', 'joblib']
)
def preprocess_data(
    data_path: str,
    output_data: Output[Dataset],
    config: dict
) -> None:
    """Preprocess data component for Vertex AI"""
    from src.pipeline.clv import CLVDataPreprocessor
    import pandas as pd
    
    # Load data from path
    df = pd.read_parquet(data_path)
    
    # Process data
    preprocessor = CLVDataPreprocessor(config)
    processed_data = preprocessor.process_data(df)
    
    # Save processed data
    processed_data.to_parquet(output_data.path)

@dsl.component(
    base_image='python:3.10',
    packages_to_install=['pandas', 'numpy', 'scikit-learn']
)
def create_segments(
    input_data: Input[Dataset],
    output_data: Output[Dataset],
    model_data: Output[Dataset],
    config: dict
) -> None:
    """Segment customers component for Vertex AI"""
    from src.pipeline.clv import CustomerSegmentation
    import pandas as pd
    
    # Load data
    df = pd.read_parquet(input_data.path)
    
    # Create segments
    segmenter = CustomerSegmentation(config)
    segmented_data, model_data_dict = segmenter.create_segments(df)
    
    # Save outputs
    segmented_data.to_parquet(output_data.path)
    pd.DataFrame(model_data_dict).to_parquet(model_data.path)

@dsl.component(
    base_image='python:3.10',
    packages_to_install=['pandas', 'numpy', 'scikit-learn', 'pymc']
)
def train_model(
    input_data: Input[Dataset],
    model: Output[Model],
    metrics: Output[Metrics],
    config: dict
) -> None:
    """Train CLV model component for Vertex AI"""
    from src.pipeline.clv import HierarchicalCLVModel
    import pandas as pd
    import json
    
    # Load data
    model_data = pd.read_parquet(input_data.path)
    
    # Train model
    clv_model = HierarchicalCLVModel(config)
    clv_model.build_model(model_data)
    trace = clv_model.sample(draws=1000, tune=500)
    
    # Save model
    with open(model.path, 'wb') as f:
        joblib.dump(clv_model, f)
    
    # Save metrics
    eval_metrics = clv_model.evaluate_model(model_data)
    with open(metrics.path, 'w') as f:
        json.dump(eval_metrics, f)

@dsl.pipeline(
    name='CLV Pipeline',
    description='End-to-end CLV prediction pipeline'
)
def hierarchical_clv_pipeline(
    data_path: str,
    config_dict: dict
) -> Output[Model]:
    """Define the CLV pipeline for Vertex AI"""
    # Define pipeline steps
    preprocess_op = preprocess_data(
        data_path=data_path,
        config=config_dict
    )
    
    segment_op = create_segments(
        input_data=preprocess_op.outputs['output_data'],
        config=config_dict
    )
    
    train_op = train_model(
        input_data=segment_op.outputs['model_data'],
        config=config_dict
    )
    
    return train_op.outputs['model']