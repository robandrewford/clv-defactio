from typing import Dict, Tuple, Any, Union
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from .base import BaseSegmentation
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class CustomerSegmentation(BaseSegmentation):
    """Customer segmentation component"""
    
    def __init__(self, config):
        """Initialize segmentation
        
        Args:
            config: Configuration object
        """
        super().__init__(config)
        self.segment_config = config.get('pipeline', {}).get('segmentation', {})
        self.n_clusters = self.segment_config.get('n_clusters', 3)
        self.features = self.segment_config.get('features', ['recency', 'frequency', 'monetary'])
        self.method = self.segment_config.get('method', 'kmeans')
        
        # Initialize the clustering model
        if self.method == 'kmeans':
            self.model = KMeans(
                n_clusters=self.n_clusters,
                random_state=42  # For reproducibility
            )
        else:
            raise ValueError(f"Unsupported clustering method: {self.method}")

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
            
        # Validate feature values
        if (processed_df['frequency'] < 0).any():
            raise ValueError("Frequency values must be non-negative")
        if (processed_df['recency'] < 0).any():
            raise ValueError("Recency values must be non-negative")
        if (processed_df['monetary'] < 0).any():
            raise ValueError("Monetary values must be non-negative")
            
        # Calculate RFM scores
        processed_df = self.calculate_rfm_scores(processed_df)
        
        # Calculate engagement score if required fields exist
        engagement_fields = ['sms_active', 'email_active', 'is_loyalty_member']
        if all(field in processed_df.columns for field in engagement_fields):
            processed_df = self.calculate_engagement_score(processed_df)
        
        # Calculate RFM combined score using weighted sum instead
        processed_df['RFM_score'] = (
            processed_df['R_score'].astype(int) * 100 + 
            processed_df['F_score'].astype(int) * 10 + 
            processed_df['M_score'].astype(int)
        )
        
        # Extract features for clustering
        X = processed_df[self.features].values
        
        # Adjust number of clusters if necessary
        n_unique_points = len(np.unique(X, axis=0))
        actual_n_clusters = min(self.n_clusters, n_unique_points)
        
        if actual_n_clusters < self.n_clusters:
            logger.warning(
                f"Reducing number of clusters from {self.n_clusters} to {actual_n_clusters} "
                "due to insufficient unique data points"
            )
            self.model.n_clusters = actual_n_clusters
        
        # Fit clustering model
        segments = self.model.fit_predict(X)
        
        # Add segment labels to dataframe
        processed_df['segment'] = segments
        
        # Create customer features dictionary
        customer_features = {
            'recency': processed_df['recency'].values,
            'frequency': processed_df['frequency'].values,
            'monetary': processed_df['monetary'].values,
            'R_score': processed_df['R_score'].values,
            'F_score': processed_df['F_score'].values,
            'M_score': processed_df['M_score'].values,
            'RFM_score': processed_df['RFM_score'].values,
            'segment': segments
        }
        
        if 'engagement_score' in processed_df.columns:
            customer_features['engagement_score'] = processed_df['engagement_score'].values
            customer_features['engagement_level'] = processed_df['engagement_level'].values
            customer_features['engagement_level_numeric'] = processed_df['engagement_level_numeric'].values
        
        # Prepare model data dictionary
        model_data = {
            'frequency': processed_df['frequency'].values,
            'recency': processed_df['recency'].values,
            'monetary_value': processed_df['monetary'].values,
            'T': processed_df['recency'].values,  # Using recency as proxy for customer age
            'segment_ids': segments,
            'customer_ids': processed_df['customer_id'].values,
            'features_used': self.features,
            'method': 'kmeans',
            'n_segments': actual_n_clusters,
            'segment_centers': self.model.cluster_centers_,
            'customer_features': customer_features,  # Add customer features
            'rfm_scores': {
                'R_score': processed_df['R_score'].values,
                'F_score': processed_df['F_score'].values,
                'M_score': processed_df['M_score'].values,
                'RFM_score': processed_df['RFM_score'].values
            }
        }
        
        return processed_df, model_data

    def process_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Process customer data and return segmented data with metadata
        
        Args:
            data: DataFrame containing customer transaction data
            
        Returns:
            Tuple containing:
                - Processed DataFrame with segmentation features
                - Metadata dictionary with processing information
        """
        try:
            # Create a copy to avoid modifying original data
            processed_data = data.copy()
            
            # Calculate basic metrics if not present
            if 'frequency' not in processed_data.columns:
                processed_data['frequency'] = processed_data.groupby('customer_id')['transaction_date'].transform('count')
            if 'recency' not in processed_data.columns:
                processed_data['recency'] = processed_data.groupby('customer_id')['transaction_date'].transform(
                    lambda x: (datetime.now() - pd.to_datetime(x).max()).days
                )
            if 'monetary' not in processed_data.columns:
                processed_data['monetary'] = processed_data.groupby('customer_id')['transaction_amount'].transform('mean')

            # Prepare metadata
            metadata = {
                'n_customers': len(processed_data['customer_id'].unique()),
                'processing_timestamp': datetime.now().isoformat(),
                'features_used': ['recency', 'frequency', 'monetary'],
                'customer_ids': processed_data['customer_id'].unique(),
                'T': processed_data['recency'].values,
                'frequency': processed_data['frequency'].values,
                'monetary': processed_data['monetary'].values
            }

            return processed_data, metadata

        except Exception as e:
            self.logger.error(f"Error in process_data: {str(e)}")
            raise

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
        for segment in range(self.n_clusters):
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

    def decode_segments(self, df: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """Decode numeric segments to meaningful labels
        
        Args:
            df: DataFrame or numpy array with segment labels
            
        Returns:
            DataFrame with decoded segments
        """
        segment_map = {
            0: 'Low Value',
            1: 'Medium Value',
            2: 'High Value'
        }
        
        # Convert numpy array to DataFrame if needed
        if isinstance(df, np.ndarray):
            df = pd.DataFrame({'segment': df})
        elif isinstance(df, pd.Series):
            df = pd.DataFrame({'segment': df})
        elif isinstance(df, pd.DataFrame) and 'segment' not in df.columns:
            df = pd.DataFrame({'segment': df.iloc[:, 0]})
        
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Map segments to labels
        result_df['segment_label'] = result_df['segment'].map(segment_map)
        
        return result_df

    def calculate_rfm_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate RFM scores
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with RFM scores
        """
        df = df.copy()
        
        try:
            # Calculate quintiles for each metric
            # Note: Recency is reversed (lower is better)
            
            # Helper function to create quintiles safely
            def create_quintiles(series, ascending=True):
                # Get unique values to determine actual number of bins needed
                unique_vals = series.nunique()
                n_bins = min(5, unique_vals)  # Use fewer bins if we have fewer unique values
                
                try:
                    if ascending:
                        labels = list(range(1, n_bins + 1))
                    else:
                        labels = list(range(n_bins, 0, -1))
                        
                    return pd.qcut(
                        series,
                        q=n_bins,
                        labels=labels,
                        duplicates='drop'
                    ).astype(float)
                except ValueError:
                    # If qcut fails, fall back to rank-based scoring
                    ranks = series.rank(method='dense', ascending=ascending)
                    normalized_ranks = ((ranks - 1) * (n_bins - 1) / (len(ranks) - 1) + 1).round()
                    return normalized_ranks
            
            # Calculate scores with proper error handling
            df['R_score'] = create_quintiles(df['recency'], ascending=False)  # Lower recency is better
            df['F_score'] = create_quintiles(df['frequency'], ascending=True)
            df['M_score'] = create_quintiles(df['monetary'], ascending=True)
            
            # Ensure all scores are numeric
            for col in ['R_score', 'F_score', 'M_score']:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(1)
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating RFM scores: {str(e)}")
            # Provide default scores if calculation fails
            for col in ['R_score', 'F_score', 'M_score']:
                df[col] = 1.0
            return df

    def calculate_engagement_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate engagement score and level
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with engagement score and level
        """
        df = df.copy()
        
        # Simple engagement score based on available metrics
        engagement_factors = ['sms_active', 'email_active', 'is_loyalty_member']
        df['engagement_score'] = df[engagement_factors].sum(axis=1)
        
        # Calculate engagement level based on score
        # Define thresholds for engagement levels
        df['engagement_level'] = pd.cut(
            df['engagement_score'],
            bins=[-np.inf, 0, 1, 2, np.inf],
            labels=['Inactive', 'Low', 'Medium', 'High']
        )
        
        # Add numeric engagement level for modeling
        engagement_level_map = {
            'Inactive': 0,
            'Low': 1,
            'Medium': 2,
            'High': 3
        }
        df['engagement_level_numeric'] = df['engagement_level'].map(engagement_level_map)
        
        return df