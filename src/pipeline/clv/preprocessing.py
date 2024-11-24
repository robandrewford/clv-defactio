from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from .config import CLVConfigLoader

class CLVDataPreprocessor:
    """Handles data preprocessing and feature engineering for CLV pipeline"""
    
    def __init__(self, config_loader: CLVConfigLoader):
        self.config = config_loader
        self.pipeline_config = config_loader.pipeline_config['data_processing']
        self.feature_config = config_loader.pipeline_config['feature_engineering']
        self.scalers = {}
        
    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main preprocessing pipeline"""
        df = df.copy()
        
        # Basic cleaning
        if self.pipeline_config['cleaning']['remove_duplicates']:
            df = self._remove_duplicates(df)
            
        if self.pipeline_config['cleaning']['handle_missing_values']:
            df = self._handle_missing_values(df)
            
        # Feature engineering
        df = self._engineer_time_features(df)
        df = self._engineer_customer_features(df)
        df = self._engineer_product_features(df)
        
        # Handle outliers after feature engineering
        if self.pipeline_config['cleaning']['remove_outliers']:
            df = self._handle_outliers(df)
            
        # Scale features
        df = self._scale_features(df)
            
        # Validate processed data
        self._validate_data(df)
        
        return df
    
    def _engineer_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer time-based features"""
        if not self.feature_config['time_features']['enable']:
            return df
            
        # Get transaction date column
        date_col = 'transaction_date'
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Add time features
        features = self.feature_config['time_features']['features']
        
        if 'day_of_week' in features:
            df['day_of_week'] = df[date_col].dt.dayofweek
            
        if 'month' in features:
            df['month'] = df[date_col].dt.month
            
        if 'quarter' in features:
            df['quarter'] = df[date_col].dt.quarter
            
        if 'is_weekend' in features:
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            
        if 'is_holiday' in features:
            # Add holiday detection logic here
            pass
            
        return df
    
    def _engineer_customer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer customer-level features"""
        if not self.feature_config['customer_features']['enable']:
            return df
            
        features = self.feature_config['customer_features']['features']
        
        # Group by customer
        customer_stats = df.groupby('customer_id').agg({
            'transaction_date': ['min', 'max', 'count'],
            'transaction_amount': ['mean', 'sum', 'std']
        })
        
        # Flatten column names
        customer_stats.columns = ['_'.join(col).strip() for col in customer_stats.columns]
        
        if 'purchase_frequency' in features:
            customer_stats['purchase_frequency'] = customer_stats['transaction_date_count']
            
        if 'average_order_value' in features:
            customer_stats['average_order_value'] = customer_stats['transaction_amount_mean']
            
        if 'customer_lifetime' in features:
            customer_stats['customer_lifetime'] = (
                customer_stats['transaction_date_max'] - 
                customer_stats['transaction_date_min']
            ).dt.days
            
        if 'days_since_last_purchase' in features:
            customer_stats['days_since_last_purchase'] = (
                datetime.now() - customer_stats['transaction_date_max']
            ).dt.days
            
        # Merge back to original dataframe
        df = df.merge(
            customer_stats,
            left_on='customer_id',
            right_index=True,
            how='left'
        )
        
        return df
    
    def _engineer_product_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer product-related features"""
        if not self.feature_config['product_features']['enable']:
            return df
            
        features = self.feature_config['product_features']['features']
        
        if 'category_diversity' in features:
            category_counts = df.groupby('customer_id')['category'].nunique()
            df = df.merge(
                category_counts.rename('category_diversity'),
                left_on='customer_id',
                right_index=True,
                how='left'
            )
            
        if 'brand_loyalty' in features:
            # Calculate brand loyalty as percentage of purchases from top brand
            brand_counts = df.groupby(['customer_id', 'brand'])['transaction_id'].count()
            top_brand_pct = (
                brand_counts.groupby('customer_id')
                .apply(lambda x: (x.max() / x.sum()) if len(x) > 0 else 0)
            )
            df = df.merge(
                top_brand_pct.rename('brand_loyalty'),
                left_on='customer_id',
                right_index=True,
                how='left'
            )
            
        if 'price_sensitivity' in features:
            # Calculate price sensitivity as std dev of purchase amounts
            price_sensitivity = df.groupby('customer_id')['transaction_amount'].std()
            df = df.merge(
                price_sensitivity.rename('price_sensitivity'),
                left_on='customer_id',
                right_index=True,
                how='left'
            )
            
        return df
    
    def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale numeric features"""
        features_to_scale = self.feature_config['scaling']['features_to_scale']
        
        for feature in features_to_scale:
            if feature in df.columns:
                scaler = StandardScaler()
                df[f"{feature}_scaled"] = scaler.fit_transform(
                    df[feature].values.reshape(-1, 1)
                )
                self.scalers[feature] = scaler
                
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate transactions"""
        return df.drop_duplicates()
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values according to config"""
        for col in df.columns:
            if df[col].isnull().any():
                if col in ['monetary', 'frequency']:
                    df[col] = df[col].fillna(0)
                else:
                    df[col] = df[col].fillna(df[col].median())
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers using configured method"""
        method = self.pipeline_config['cleaning']['outlier_method']
        threshold = self.pipeline_config['cleaning']['outlier_threshold']
        
        if method == 'iqr':
            return self._handle_outliers_iqr(df, threshold)
        elif method == 'zscore':
            return self._handle_outliers_zscore(df, threshold)
        
        return df
    
    def _handle_outliers_iqr(self, df: pd.DataFrame, threshold: float) -> pd.DataFrame:
        """Handle outliers using IQR method"""
        for col in ['monetary', 'frequency', 'recency']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            
        return df
    
    def _handle_outliers_zscore(self, df: pd.DataFrame, threshold: float) -> pd.DataFrame:
        """Handle outliers using Z-score method"""
        for col in ['monetary', 'frequency', 'recency']:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            df[col] = df[col].mask(z_scores > threshold, df[col].median())
        return df
    
    def _validate_data(self, df: pd.DataFrame) -> None:
        """Validate processed data meets requirements"""
        # Check required columns
        required_cols = self.pipeline_config['validation']['required_columns']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Check minimum rows
        min_rows = self.pipeline_config['validation']['min_rows']
        if len(df) < min_rows:
            raise ValueError(f"Dataset has fewer than {min_rows} rows")
            
        # Check maximum missing percentage
        max_missing = self.pipeline_config['validation']['max_missing_pct']
        missing_pct = df.isnull().sum() / len(df)
        if (missing_pct > max_missing).any():
            cols_over_threshold = missing_pct[missing_pct > max_missing].index
            raise ValueError(f"Columns exceed maximum missing threshold: {cols_over_threshold}") 