from typing import Dict, Tuple, Any
import pandas as pd
import numpy as np
from .base import BaseProcessor

class CustomerSegmentation(BaseProcessor):
    """Customer segmentation component"""
    
    def __init__(self, config_loader):
        self.config = config_loader
        self.segment_config = config_loader.model_config['segment_config']
        
    def process_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Process and segment customer data"""
        return self.create_segments(df)
        
    def create_segments(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Create customer segments based on RFM and other metrics"""
        df = df.copy()
        
        # Validate input data
        if (df['frequency'] < 0).any():
            raise ValueError("Frequency values cannot be negative")
            
        # Add validation for other metrics as needed
        if (df['transaction_amount'] < 0).any():
            raise ValueError("Transaction amounts cannot be negative")
            
        # Calculate RFM scores
        df = self._calculate_rfm_scores(df)
        
        # Add engagement metrics if configured
        if self.segment_config.get('use_engagement', False):
            df = self._add_engagement_scores(df)
            
        # Create final segments
        model_data = self._prepare_model_data(df)
        
        return df, model_data
        
    def _calculate_rfm_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate RFM scores for segmentation"""
        # Calculate R score
        df['R_score'] = pd.qcut(df['recency'], 4, labels=False, duplicates='drop')
        
        # Calculate F score
        df['F_score'] = pd.qcut(df['frequency'], 4, labels=False, duplicates='drop')
        
        # Calculate M score
        df['M_score'] = pd.qcut(df['monetary'], 4, labels=False, duplicates='drop')
        
        # Combined RFM score
        df['RFM_score'] = (df['R_score'].astype(str) + 
                          df['F_score'].astype(str) + 
                          df['M_score'].astype(str))
        
        return df
        
    def _add_engagement_scores(self, data):
        """
        Add engagement scores based on customer interactions
        
        Args:
            data (pd.DataFrame): Processed customer data
            
        Returns:
            pd.DataFrame: Data with engagement scores added
        """
        data['engagement_score'] = (
            data['sms_active'].astype(int) * 0.3 +
            data['email_active'].astype(int) * 0.3 +
            data['is_loyalty_member'].astype(int) * 0.4
        )
        return data
        
    def _prepare_model_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Prepare data for model training
        
        Args:
            df (pd.DataFrame): Processed and segmented customer data
            
        Returns:
            Dict[str, Any]: Dictionary containing model-ready data
        """
        model_data = {
            'customer_id': df['customer_id'].values,
            'frequency': df['frequency'].values,
            'recency': df['recency'].values,
            'monetary': df['monetary'].values,
            'rfm_score': df['RFM_score'].values
        }
        
        # Add engagement score if available
        if 'engagement_score' in df.columns:
            model_data['engagement_score'] = df['engagement_score'].values
            
        return model_data