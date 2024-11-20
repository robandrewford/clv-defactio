from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from google.cloud import bigquery
from sklearn.preprocessing import StandardScaler

from ..exceptions import (
    DataLoadError,
    DataProcessingError,
    DataValidationError
)
from ..utils.logging import get_logger

logger = get_logger(__name__)


class DataProcessor:
    """Base data processing class with core functionality"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DataProcessor

        Args:
            config: Configuration dictionary containing processing parameters
        """
        self.config = config
        self.data = None
        self.quality_flags = {}
        self.processing_history = []
        self.validation_schema = config.get("validation_schema", {})

    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate input data quality and schema

        Args:
            data: Input DataFrame to validate

        Returns:
            Tuple containing (is_valid: bool, validation_results: Dict)
        """
        validation_results = {
            "is_valid": True,
            "issues": [],
            "missing_columns": [],
            "invalid_values": {},
            "timestamp": datetime.now().isoformat(),
        }

        try:
            # Check required columns
            required_cols = self.validation_schema.get("required_columns", [])
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                validation_results["missing_columns"] = missing_cols
                validation_results["is_valid"] = False

            # Check data types
            for col, expected_type in self.validation_schema.get(
                "column_types", {}
            ).items():
                if col in data.columns and not data[col].dtype == expected_type:
                    validation_results["invalid_values"][
                        col
                    ] = f"Invalid type: {data[col].dtype}"
                    validation_results["is_valid"] = False

            # Check value ranges
            for col, ranges in self.validation_schema.get("value_ranges", {}).items():
                if col in data.columns:
                    invalid_mask = ~data[col].between(ranges["min"], ranges["max"])
                    if invalid_mask.any():
                        validation_results["invalid_values"][
                            col
                        ] = f"{invalid_mask.sum()} values out of range"
                        validation_results["is_valid"] = False

            return validation_results["is_valid"], validation_results

        except Exception as e:
            raise DataValidationError(f"Validation failed: {str(e)}")

    def _log_processing_step(self, step_name: str, details: Dict[str, Any]):
        """Log processing step with timestamp"""
        self.processing_history.append(
            {
                "step": step_name,
                "timestamp": datetime.now().isoformat(),
                "details": details,
            }
        )


class CLVDataProcessor(DataProcessor):
    """CLV-specific data processing with advanced features"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize CLV Data Processor

        Args:
            config: Configuration dictionary containing CLV-specific parameters
        """
        super().__init__(config)
        self.segment_bins = {"rfm": {}, "loyalty": {}, "engagement": {}}
        self.medians = {}
        self.scalers = {}
        self.feature_stats = {}

    def load_data(
        self,
        query: Optional[str] = None,
        project_id: Optional[str] = None,
        csv_path: Optional[str] = None,
    ) -> "CLVDataProcessor":
        """
        Load data from either BigQuery or CSV

        Args:
            query: Optional BigQuery SQL query
            project_id: Optional GCP project ID
            csv_path: Optional path to CSV file

        Returns:
            Self for method chaining
        """
        try:
            if csv_path:
                self.data = pd.read_csv(csv_path)
                logger.info(f"Loaded data from CSV: {csv_path}")
            else:
                client = bigquery.Client(
                    project=project_id or self.config["PROJECT_ID"]
                )
                query = query or self._build_default_query()
                self.data = client.query(query).to_dataframe()
                logger.info("Loaded data from BigQuery")

            self._convert_dates()
            self._log_processing_step(
                "data_load",
                {"source": "csv" if csv_path else "bigquery", "rows": len(self.data)},
            )
            return self

        except Exception as e:
            raise DataLoadError(f"Failed to load data: {str(e)}")

    def process_data(self) -> pd.DataFrame:
        """
        Execute main data processing pipeline

        Returns:
            Processed DataFrame
        """
        if self.data is None:
            raise ValueError("No data loaded")

        try:
            processing_steps = [
                ("clean_basic", self._clean_basic_data),
                ("rfm_metrics", self._calculate_rfm_metrics),
                ("monetary_clean", self._clean_monetary_values),
                ("categorical_clean", self._clean_categorical_features),
                ("feature_engineering", self._engineer_features),
                ("validation", self._validate_processed_data),
            ]

            for step_name, step_func in processing_steps:
                step_func()
                self._log_processing_step(
                    step_name,
                    {"rows": len(self.data), "columns": list(self.data.columns)},
                )

            self._generate_quality_report()
            return self.data

        except Exception as e:
            raise DataProcessingError(f"Processing failed: {str(e)}")

    def _clean_basic_data(self):
        """Perform basic data cleaning operations"""
        # Remove duplicates
        initial_rows = len(self.data)
        self.data = self.data.drop_duplicates()

        # Handle missing values
        for col, strategy in self.config["missing_value_strategy"].items():
            if col in self.data.columns:
                if strategy == "mean":
                    self.data[col].fillna(self.data[col].mean(), inplace=True)
                elif strategy == "median":
                    self.data[col].fillna(self.data[col].median(), inplace=True)
                elif strategy == "mode":
                    self.data[col].fillna(self.data[col].mode()[0], inplace=True)
                elif strategy == "zero":
                    self.data[col].fillna(0, inplace=True)

        # Remove outliers if configured
        if self.config.get("remove_outliers", False):
            self._handle_outliers()

        self.quality_flags["duplicates_removed"] = initial_rows - len(self.data)

    def _calculate_rfm_metrics(self):
        """Calculate Recency, Frequency, Monetary metrics"""
        current_date = pd.Timestamp.now()

        # Recency
        self.data["recency"] = (current_date - self.data["last_purchase_date"]).dt.days

        # Frequency (already exists in most cases, but verify)
        if "frequency" not in self.data.columns:
            self.data["frequency"] = self.data.groupby("customer_id")[
                "transaction_date"
            ].count()

        # Monetary
        if "monetary" not in self.data.columns:
            self.data["monetary"] = self.data.groupby("customer_id")[
                "transaction_amount"
            ].sum()

        # Calculate RFM scores
        self._calculate_rfm_scores()

    def _engineer_features(self):
        """Create advanced features for CLV analysis"""
        # Time-based features
        self.data["customer_age_days"] = (
            pd.Timestamp.now() - self.data["first_purchase_date"]
        ).dt.days
        self.data["avg_purchase_frequency"] = (
            self.data["frequency"] / self.data["customer_age_days"]
        )

        # Monetary features
        self.data["avg_transaction_value"] = (
            self.data["monetary"] / self.data["frequency"]
        )
        self.data["revenue_per_day"] = (
            self.data["monetary"] / self.data["customer_age_days"]
        )

        # Engagement features
        if all(col in self.data.columns for col in ["email_active", "sms_active"]):
            self.data["engagement_score"] = self.data["email_active"].astype(
                int
            ) + self.data["sms_active"].astype(int)

        # Seasonality features if transaction dates available
        if "transaction_date" in self.data.columns:
            self.data["purchase_month"] = self.data["transaction_date"].dt.month
            self.data["purchase_quarter"] = self.data["transaction_date"].dt.quarter

        # Scale numerical features
        self._scale_features()

    def _validate_processed_data(self) -> bool:
        """
        Validate processed data quality

        Returns:
            bool indicating if data meets quality standards
        """
        validation_checks = {
            "missing_values": self.data.isnull().sum().sum() == 0,
            "negative_values": not (
                self.data[["monetary", "frequency"]].lt(0).any().any()
            ),
            "sufficient_records": len(self.data)
            >= self.config.get("min_records", 1000),
        }

        self.quality_flags.update(validation_checks)
        return all(validation_checks.values())

    def _generate_quality_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive data quality report

        Returns:
            Dictionary containing quality metrics
        """
        report = {
            "record_count": len(self.data),
            "feature_count": len(self.data.columns),
            "missing_values": self.data.isnull().sum().to_dict(),
            "feature_stats": {
                col: {
                    "mean": self.data[col].mean(),
                    "median": self.data[col].median(),
                    "std": self.data[col].std(),
                }
                for col in self.data.select_dtypes(include=[np.number]).columns
            },
            "quality_flags": self.quality_flags,
            "processing_history": self.processing_history,
        }

        self.feature_stats = report
        return report

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Calculate feature importance scores

        Returns:
            Dictionary of feature importance scores
        """
        from sklearn.ensemble import RandomForestRegressor

        if "monetary" not in self.data.columns:
            raise ValueError("Monetary value required for feature importance")

        numeric_features = self.data.select_dtypes(include=[np.number]).columns
        features = [col for col in numeric_features if col != "monetary"]

        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(self.data[features], self.data["monetary"])

        return dict(zip(features, rf.feature_importances_))

    def save_processed_data(
        self, path: str, format: str = "csv", include_report: bool = True
    ):
        """
        Save processed data and quality report

        Args:
            path: Path to save the data
            format: Format to save ('csv' or 'parquet')
            include_report: Whether to save quality report
        """
        if format == "csv":
            self.data.to_csv(path, index=False)
        elif format == "parquet":
            self.data.to_parquet(path, index=False)
        else:
            raise ValueError("Unsupported format")

        if include_report:
            report_path = path.rsplit(".", 1)[0] + "_report.json"
            import json

            with open(report_path, "w") as f:
                json.dump(self._generate_quality_report(), f, indent=2)
