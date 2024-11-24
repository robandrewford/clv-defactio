from kfp import dsl
import pandas as pd
from typing import Dict
from .config import CLVConfigLoader
from .registry import CLVModelRegistry

@dsl.component(
    base_image="python:3.9",
    packages_to_install=["pandas", "numpy", "pymc", "scikit-learn", "yaml"]
)
def segment_customers(
    data_path: str,
    config_dir: str,
    output_path: str
) -> str:
    """Vertex AI component for customer segmentation"""
    from .segmentation import CustomerSegmentation
    
    # Load configs
    config_loader = CLVConfigLoader(config_dir)
    
    # Load data
    df = pd.read_parquet(data_path)
    
    # Create segments
    segmenter = CustomerSegmentation(config_loader)
    segmented_df = segmenter.create_segments(df)
    
    # Save results
    segmented_df.to_parquet(output_path)
    return output_path

@dsl.component(
    base_image="python:3.9",
    packages_to_install=["pandas", "numpy", "pymc", "scikit-learn", "yaml"]
)
def train_clv_model(
    data_path: str,
    config_dir: str,
    model_dir: str
) -> str:
    """Vertex AI component for CLV model training"""
    from .model import HierarchicalCLVModel
    from .config import CLVConfigLoader
    from .registry import CLVModelRegistry
    
    # Load configs and data
    config_loader = CLVConfigLoader(config_dir)
    df = pd.read_parquet(data_path)
    
    # Train model
    model = HierarchicalCLVModel(config_loader)
    model.build_model(df)
    metrics = model.train_model()
    
    # Save to registry
    registry = CLVModelRegistry(config_loader)
    version = registry.save_model(model, metrics)
    
    return version

@dsl.component(
    base_image="python:3.9",
    packages_to_install=["pandas", "numpy", "scikit-learn", "yaml"]
)
def preprocess_data(
    data_path: str,
    config_dir: str,
    output_path: str
) -> str:
    """Vertex AI component for data preprocessing"""
    from .preprocessing import CLVDataPreprocessor
    from .config import CLVConfigLoader
    
    # Load configs and data
    config_loader = CLVConfigLoader(config_dir)
    df = pd.read_parquet(data_path)
    
    # Preprocess data
    preprocessor = CLVDataPreprocessor(config_loader)
    processed_df = preprocessor.process_data(df)
    
    # Save results
    processed_df.to_parquet(output_path)
    return output_path

@dsl.pipeline(
    name='hierarchical-clv-pipeline',
    description='End-to-end hierarchical CLV prediction pipeline'
)
def hierarchical_clv_pipeline(
    project_id: str,
    input_table: str,
    output_bucket: str,
    config_dir: str
):
    """Define the Vertex AI Pipeline for CLV"""
    
    # Preprocess data
    preprocess_task = preprocess_data(
        data_path=input_table,
        config_dir=config_dir,
        output_path=f"{output_bucket}/preprocessed_data.parquet"
    )
    
    # Segment customers
    segment_task = segment_customers(
        data_path=preprocess_task.output,
        config_dir=config_dir,
        output_path=f"{output_bucket}/segmented_data.parquet"
    )
    
    # Train CLV model
    train_task = train_clv_model(
        data_path=segment_task.output,
        config_dir=config_dir,
        model_dir=f"{output_bucket}/model"
    ) 