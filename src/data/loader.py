from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from google.cloud import bigquery
from sklearn.preprocessing import StandardScaler


class CLVDataProcessor:
    """CLV data processing class with data quality checks"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {
            # Data processing parameters
            "MIN_FREQUENCY": 1,
            "MIN_REVENUE": 1,
            "MIN_TRANSACTION_VALUE": 1,
            "OUTLIER_THRESHOLD": 3,  # Number of IQRs for outlier detection
            # Query parameters
            "PROJECT_ID": "logic-dna-240402",
            "DATASET": "CLV",
            "TABLE": "T_CLV_360",
            "LIMIT": 10000000,
            "COHORT_MONTH": "2023-02-01",
            "MIN_PURCHASE_DATE": "2022-02-03",
            "MAX_PURCHASE_DATE": "CURRENT_DATE()",
            "INCLUDE_ONLINE": True,
            "INCLUDE_STORE": True,
            "MIN_LOYALTY_POINTS": 0,
        }
        self.data = None
        self.quality_flags = None

    def _build_query(self) -> str:
        """Build BigQuery query based on configuration parameters"""
        # Handle date parameters
        max_date = self.config["MAX_PURCHASE_DATE"] or "CURRENT_DATE()"
        min_date = f"DATE('{self.config['MIN_PURCHASE_DATE']}')"
        cohort_month = f"DATE('{self.config['COHORT_MONTH']}')"

        # Build channel condition
        channel_conditions = []
        if self.config["INCLUDE_ONLINE"]:
            channel_conditions.append("has_online_purchases = 1")
        if self.config["INCLUDE_STORE"]:
            channel_conditions.append("has_store_purchases = 1")
        channel_filter = (
            f"({' OR '.join(channel_conditions)})" if channel_conditions else "TRUE"
        )

        # ... rest of the query building code ...
        return query

    def load_data(
        self,
        query: Optional[str] = None,
        project_id: Optional[str] = None,
        csv_path: Optional[str] = None,
    ) -> "CLVDataProcessor":
        """Load data either from BigQuery or CSV file"""
        try:
            if csv_path:
                self.data = pd.read_csv(csv_path)
                print(f"Successfully loaded {len(self.data):,} records from CSV")
            else:
                client = bigquery.Client(
                    project=project_id or self.config["PROJECT_ID"]
                )
                default_query = self._build_query()
                self.data = client.query(query or default_query).to_dataframe()
                print(f"Successfully loaded {len(self.data):,} records from BigQuery")

            # Convert date columns
            date_columns = ["first_purchase_date", "last_purchase_date", "cohort_month"]
            for col in date_columns:
                if col in self.data.columns:
                    self.data[col] = pd.to_datetime(self.data[col])

            return self

        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise

    def process_data(self) -> "CLVDataProcessor":
        """Main data processing pipeline"""
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        try:
            print(f"Starting data processing. Initial shape: {self.data.shape}")

            # Basic cleaning
            self._clean_basic_data()

            # Calculate RFM metrics
            self._calculate_rfm_metrics()

            # Handle outliers and invalid values
            self._clean_monetary_values()
            self._clean_categorical_features()

            # Validate data quality
            self._validate_data_quality()

            # Generate final report
            self._generate_quality_report()

            return self

        except Exception as e:
            print(f"Error in data processing: {str(e)}")
            raise

    def _clean_basic_data(self) -> None:
        """Perform basic data cleaning operations"""
        # Remove duplicates
        self.data = self.data.drop_duplicates()

        # Remove rows with missing critical values
        critical_columns = ["customer_id", "total_revenue", "frequency"]
        self.data = self.data.dropna(subset=critical_columns)

        # Filter based on minimum thresholds
        self.data = self.data[
            (self.data["frequency"] >= self.config["MIN_FREQUENCY"])
            & (self.data["total_revenue"] >= self.config["MIN_REVENUE"])
        ]

        print(f"After basic cleaning, shape: {self.data.shape}")

    def _calculate_rfm_metrics(self) -> None:
        """Calculate Recency, Frequency, Monetary metrics"""
        current_date = pd.Timestamp.now()

        # Calculate recency in days
        self.data["recency"] = (current_date - self.data["last_purchase_date"]).dt.days

        # Calculate average transaction value
        self.data["avg_transaction_value"] = (
            self.data["total_revenue"] / self.data["frequency"]
        )

        # Calculate time between first and last purchase (customer age)
        self.data["customer_age"] = (
            self.data["last_purchase_date"] - self.data["first_purchase_date"]
        ).dt.days

        print("RFM metrics calculated successfully")

    def _clean_monetary_values(self) -> None:
        """Handle outliers in monetary values using IQR method"""
        monetary_columns = ["total_revenue", "avg_transaction_value"]

        for col in monetary_columns:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - self.config["OUTLIER_THRESHOLD"] * IQR
            upper_bound = Q3 + self.config["OUTLIER_THRESHOLD"] * IQR

            # Flag outliers
            outlier_mask = (self.data[col] < lower_bound) | (
                self.data[col] > upper_bound
            )
            self.data[f"{col}_is_outlier"] = outlier_mask

            # Cap outliers
            self.data[col] = self.data[col].clip(lower_bound, upper_bound)

        print("Monetary values cleaned and outliers handled")

    def _clean_categorical_features(self) -> None:
        """Clean and encode categorical features"""
        # Handle categorical columns
        cat_columns = ["customer_segment", "preferred_channel"]

        for col in cat_columns:
            if col in self.data.columns:
                # Fill missing values with 'Unknown'
                self.data[col] = self.data[col].fillna("Unknown")

                # Convert to category type
                self.data[col] = self.data[col].astype("category")

        print("Categorical features processed")

    def _validate_data_quality(self) -> None:
        """Perform data quality checks and set quality flags"""
        self.quality_flags = {
            "missing_values": self.data.isnull().sum().to_dict(),
            "negative_values": {
                col: (self.data[col] < 0).sum()
                for col in self.data.select_dtypes(include=[np.number]).columns
            },
            "zero_values": {
                col: (self.data[col] == 0).sum()
                for col in self.data.select_dtypes(include=[np.number]).columns
            },
            "unique_values": {
                col: self.data[col].nunique() for col in self.data.columns
            },
        }

        print("Data quality validation completed")

    def _generate_quality_report(self) -> None:
        """Generate a summary report of data quality"""
        if self.quality_flags is None:
            raise ValueError(
                "No quality flags available. Run _validate_data_quality first."
            )

        print("\nData Quality Report:")
        print("-" * 50)

        print("\nMissing Values:")
        for col, count in self.quality_flags["missing_values"].items():
            if count > 0:
                print(f"{col}: {count:,} missing values")

        print("\nNegative Values:")
        for col, count in self.quality_flags["negative_values"].items():
            if count > 0:
                print(f"{col}: {count:,} negative values")

        print("\nZero Values:")
        for col, count in self.quality_flags["zero_values"].items():
            if count > 0:
                print(f"{col}: {count:,} zero values")

        print("\nUnique Values:")
        for col, count in self.quality_flags["unique_values"].items():
            print(f"{col}: {count:,} unique values")

    def _scale_numerical_features(self, columns: List[str]) -> None:
        """Scale numerical features using StandardScaler"""
        scaler = StandardScaler()
        self.data[columns] = scaler.fit_transform(self.data[columns])
        print(f"Scaled numerical features: {columns}")

    def get_processed_data(self) -> pd.DataFrame:
        """Return the processed DataFrame"""
        return self.data.copy()
