from typing import Dict, Tuple, Any
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from .base import BaseSegmentation
from datetime import datetime

class CustomerSegmentation(BaseSegmentation):
    """Customer segmentation component"""
    
    def __init__(self, config):
        """Initialize segmentation
        
        Args:
            config: Configuration object
        """
        super().__init__(config)
        self.segment_config = config.get('segment_config', {})
        self.n_segments = self.segment_config.get('n_segments', 3)
        self.features = self.segment_config.get('features', ['recency', 'frequency', 'monetary'])
        self.method = self.segment_config.get('method', 'kmeans')
        self.model = None

    def create_segments(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Create customer segments
        
        Args:
            df: DataFrame with customer data
            
        Returns:
            tuple: (segmented_df, model_data)
        """
        if df is None or len(df) == 0:
            raise ValueError("Input data cannot be empty")
            
        # Create copy to avoid modifying original
        processed_df = df.copy()
        
        # Ensure required features exist
        missing_features = [f for f in self.features if f not in processed_df.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
            
        # Prepare feature matrix
        X = processed_df[self.features].values
        
        # Perform segmentation
        if self.method == 'kmeans':
            if self.model is None:
                self.model = KMeans(n_clusters=self.n_segments, random_state=42)
            segments = self.model.fit_predict(X)
        else:
            raise ValueError(f"Unsupported segmentation method: {self.method}")
            
        # Add segment labels to dataframe
        processed_df['segment'] = segments
        
        # Prepare model data
        model_data = {
            'segment_labels': segments,
            'segment_centers': self.model.cluster_centers_ if self.method == 'kmeans' else None,
            'n_segments': self.n_segments,
            'features_used': self.features,
            'method': self.method,
            'timestamp': datetime.now().isoformat()
        }
        
        return processed_df, model_data

    def process_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Process data for segmentation
        
        Args:
            data: Input DataFrame
            
        Returns:
            Tuple containing segmented DataFrame and model data
        """
        return self.create_segments(data)

    def get_segment_profile(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate segment profiles
        
        Args:
            df: Segmented DataFrame
            
        Returns:
            DataFrame with segment profiles
        """
        if 'segment' not in df.columns:
            raise ValueError("DataFrame must contain segment labels")
            
        profiles = []
        for segment in range(self.n_segments):
            segment_data = df[df['segment'] == segment]
            profile = {
                'segment': segment,
                'size': len(segment_data),
                'percentage': len(segment_data) / len(df) * 100
            }
            
            # Add feature averages
            for feature in self.features:
                profile[f'avg_{feature}'] = segment_data[feature].mean()
                
            profiles.append(profile)
            
        return pd.DataFrame(profiles)

    def decode_segments(self, df: pd.DataFrame) -> pd.DataFrame:
        """Decode numeric segments to meaningful labels
        
        Args:
            df: DataFrame with segment labels
            
        Returns:
            DataFrame with decoded segments
        """
        segment_map = {
            0: 'Low Value',
            1: 'Medium Value',
            2: 'High Value'
        }
        df['segment_label'] = df['segment'].map(segment_map)
        return df

    def calculate_rfm_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate RFM scores
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with RFM scores
        """
        df = df.copy()
        
        # Calculate quintiles for each metric
        df['R_score'] = pd.qcut(df['recency'], q=5, labels=[5,4,3,2,1])
        df['F_score'] = pd.qcut(df['frequency'], q=5, labels=[1,2,3,4,5])
        df['M_score'] = pd.qcut(df['monetary'], q=5, labels=[1,2,3,4,5])
        
        return df

    def calculate_engagement_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate engagement score
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with engagement score
        """
        df = df.copy()
        
        # Simple engagement score based on available metrics
        engagement_factors = ['sms_active', 'email_active', 'is_loyalty_member']
        df['engagement_score'] = df[engagement_factors].sum(axis=1)
        
        return df