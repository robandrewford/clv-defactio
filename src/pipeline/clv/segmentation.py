from typing import Dict, Tuple, Any
import pandas as pd
import numpy as np
from .base import BaseProcessor
from datetime import datetime

class CustomerSegmentation(BaseProcessor):
    """Customer segmentation component"""
    
    def __init__(self, config):
        """Initialize segmentation with config"""
        super().__init__(config)
        self.config = config

    def process_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Process customer data and return segmented data with metadata"""
        processed_data = data.copy()
        
        # Calculate engagement score
        processed_data['engagement_score'] = self._calculate_engagement_score(processed_data)
        
        # Assign engagement levels
        processed_data['engagement_level'] = self._assign_engagement_levels(
            processed_data['engagement_score']
        )
        
        # Create metadata about the segmentation
        metadata = {
            'total_customers': len(processed_data),
            'segments': processed_data['engagement_level'].value_counts().to_dict(),
            'timestamp': datetime.now().isoformat()
        }
        
        return processed_data, metadata

    def _calculate_engagement_score(self, data: pd.DataFrame) -> pd.Series:
        """Calculate customer engagement score"""
        # Normalize numeric columns
        numeric_cols = ['frequency', 'recency', 'monetary']
        normalized = data[numeric_cols].apply(lambda x: (x - x.mean()) / x.std())
        
        # Weight and combine for engagement score
        weights = {
            'frequency': 0.4,
            'recency': -0.3,  # Negative because lower recency is better
            'monetary': 0.3
        }
        
        engagement_score = sum(normalized[col] * weight 
                             for col, weight in weights.items())
        
        return engagement_score

    def _assign_engagement_levels(self, scores: pd.Series) -> pd.Series:
        """Assign engagement levels based on score percentiles"""
        conditions = [
            scores > scores.quantile(0.75),
            scores > scores.quantile(0.5),
            scores > scores.quantile(0.25),
            scores <= scores.quantile(0.25)
        ]
        
        choices = ['Very High', 'High', 'Medium', 'Low']
        
        return pd.Series(np.select(conditions, choices), index=scores.index)