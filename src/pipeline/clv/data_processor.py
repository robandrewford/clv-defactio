from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from google.cloud import bigquery
from .base import BaseProcessor
from .queries import get_transaction_data_query

class CLVDataProcessor(BaseProcessor):
    """CLV data processing class with data quality checks"""
    
    def __init__(self, config=None):
        # ... previous init code ...
        self.data = None
        self.quality_flags = None
        self.customer_lookup = None

    def process_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process data and create customer lookup table
        Returns:
            Tuple of (processed_data, customer_lookup)
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        try:
            # ... previous processing steps ...

            # Create customer lookup table
            self.data.reset_index(drop=True, inplace=True)
            self.data['customer_index'] = self.data.index
            
            # Create lookup DataFrame
            self.customer_lookup = self.data[['customer_index', 'customer_id']].copy()
            
            # Remove customer identification from main DataFrame
            processed_data = self.data.drop(columns=['customer_index', 'customer_id'])
            
            print(f"Created lookup table with {len(self.customer_lookup):,} customers")
            print(f"Final processed data shape: {processed_data.shape}")
            
            return processed_data, self.customer_lookup

        except Exception as e:
            print(f"Error in data processing: {str(e)}")
            raise

    def get_processed_data(self) -> pd.DataFrame:
        """Return the processed DataFrame without customer identifiers"""
        if self.data is None:
            raise ValueError("No data processed. Call process_data() first.")
        return self.data.drop(columns=['customer_index', 'customer_id'])

    def get_customer_lookup(self) -> pd.DataFrame:
        """Return the customer lookup table"""
        if self.customer_lookup is None:
            raise ValueError("No lookup table created. Call process_data() first.")
        return self.customer_lookup.copy() 