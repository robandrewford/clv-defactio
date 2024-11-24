from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from .config import CLVConfigLoader

class CustomerSegmentation:
    """Handles customer segmentation and covariate preparation"""
    
    def __init__(self, config_loader: CLVConfigLoader):
        self.config = config_loader
        self.segment_config = config_loader.segment_config
        self.encoders = {}
        self.segment_stats = {}
        
    def create_segments(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
        """Create segments and prepare covariates for modeling"""
        df = df.copy()
        
        # Create base segments
        if self.segment_config['segment_config']['use_rfm']:
            df = self._create_rfm_segments(df)
            
        if self.segment_config['segment_config']['use_engagement']:
            df = self._create_engagement_segments(df)
            
        if self.segment_config['segment_config']['use_loyalty']:
            df = self._create_loyalty_segments(df)
            
        if self.segment_config['segment_config']['use_cohorts']:
            df = self._create_cohort_groups(df)
            
        # Prepare covariates for modeling
        model_data = self._prepare_model_covariates(df)
        
        return df, model_data
    
    def _create_rfm_segments(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create RFM segments with numeric scoring"""
        # Calculate RFM scores
        r_labels = range(1, self.segment_config['segment_config']['rfm_bins']['recency'] + 1)
        f_labels = range(1, self.segment_config['segment_config']['rfm_bins']['frequency'] + 1)
        m_labels = range(1, self.segment_config['segment_config']['rfm_bins']['monetary'] + 1)
        
        df['R_score'] = pd.qcut(df['recency'], q=len(r_labels), labels=r_labels)
        df['F_score'] = pd.qcut(df['frequency'], q=len(f_labels), labels=f_labels)
        df['M_score'] = pd.qcut(df['monetary'], q=len(m_labels), labels=m_labels)
        
        # Create combined RFM score
        df['RFM_score'] = (
            df['R_score'].astype(str) + 
            df['F_score'].astype(str) + 
            df['M_score'].astype(str)
        )
        
        # Store segment statistics
        self.segment_stats['rfm'] = {
            'r_quantiles': df.groupby('R_score')['recency'].agg(['mean', 'count']),
            'f_quantiles': df.groupby('F_score')['frequency'].agg(['mean', 'count']),
            'm_quantiles': df.groupby('M_score')['monetary'].agg(['mean', 'count'])
        }
        
        return df
    
    def _create_engagement_segments(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engagement segments with behavioral metrics"""
        metrics = self.segment_config['segment_config']['engagement_metrics']
        
        # Calculate engagement score
        df['engagement_score'] = 0
        for metric in metrics:
            if metric in df.columns:
                df['engagement_score'] += df[metric].astype(float)
                
        # Create engagement levels
        n_bins = self.segment_config['segment_config']['engagement_bins']
        df['engagement_level'] = pd.qcut(
            df['engagement_score'],
            q=n_bins,
            labels=[f'Level_{i+1}' for i in range(n_bins)]
        )
        
        # Store segment statistics
        self.segment_stats['engagement'] = {
            'level_stats': df.groupby('engagement_level')['engagement_score'].agg(['mean', 'count'])
        }
        
        return df
    
    def _prepare_model_covariates(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Prepare covariates for the hierarchical model"""
        # Initialize model data
        model_data = {
            'customer_id': df['customer_id'].values,
            'frequency': df['frequency'].values,
            'recency': df['recency'].values,
            'monetary_value': df['monetary'].values,
            'T': df['customer_age_days'].values
        }
        
        # Encode categorical segments for modeling
        segment_columns = []
        
        if self.segment_config['segment_config']['use_rfm']:
            segment_columns.extend(['R_score', 'F_score', 'M_score'])
            
        if self.segment_config['segment_config']['use_engagement']:
            segment_columns.append('engagement_level')
            
        if self.segment_config['segment_config']['use_loyalty']:
            segment_columns.append('loyalty_segment')
            
        if self.segment_config['segment_config']['use_cohorts']:
            segment_columns.append('cohort_segment')
            
        # Encode segments
        encoded_segments = pd.DataFrame()
        for col in segment_columns:
            if col in df.columns:
                encoder = LabelEncoder()
                encoded_segments[f'{col}_encoded'] = encoder.fit_transform(df[col])
                self.encoders[col] = encoder
                
        # Create segment IDs from combination of encoded segments
        if len(encoded_segments.columns) > 0:
            model_data['segment_ids'] = encoded_segments.apply(
                lambda x: '_'.join(x.astype(str)), axis=1
            ).astype('category').cat.codes
        else:
            model_data['segment_ids'] = np.zeros(len(df))
            
        # Add customer features as covariates
        feature_columns = [
            'category_diversity', 'brand_loyalty', 'price_sensitivity',
            'average_order_value', 'customer_lifetime'
        ]
        
        customer_features = []
        for col in feature_columns:
            if col in df.columns:
                customer_features.append(df[col].values.reshape(-1, 1))
                
        if customer_features:
            model_data['customer_features'] = np.hstack(customer_features)
        
        return model_data
    
    def get_segment_summary(self) -> Dict[str, pd.DataFrame]:
        """Get summary statistics for all segments"""
        return self.segment_stats
    
    def decode_segments(self, segment_ids: np.ndarray) -> pd.DataFrame:
        """Decode segment IDs back to original labels"""
        decoded = pd.DataFrame()
        for col, encoder in self.encoders.items():
            decoded[col] = encoder.inverse_transform(segment_ids)
        return decoded