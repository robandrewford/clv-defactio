from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class ValidationResult:
    """Container for validation results"""

    is_valid: bool
    errors: List[str]
    warnings: List[str]
    metrics: Dict[str, Any]


class DataValidator:
    """Validates data quality and business rules for CLV analysis"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {
            "required_columns": [
                "customer_id",
                "total_revenue",
                "frequency",
                "first_purchase_date",
                "last_purchase_date",
            ],
            "numeric_columns": ["total_revenue", "frequency", "avg_transaction_value"],
            "date_columns": [
                "first_purchase_date",
                "last_purchase_date",
                "cohort_month",
            ],
            "min_frequency": 1,
            "min_revenue": 1.0,
            "min_transaction_value": 1.0,
            "outlier_threshold": 3.0,
        }
        self.reset()

    def reset(self) -> None:
        """Reset validation state"""
        self.errors = []
        self.warnings = []
        self.metrics = {}

    def validate_data(self, data: pd.DataFrame) -> ValidationResult:
        """
        Perform all validation checks on the dataset

        Args:
            data: DataFrame to validate

        Returns:
            ValidationResult object containing validation results
        """
        self.reset()

        # Basic data structure checks
        self._validate_structure(data)

        # Column presence and types
        self._validate_columns(data)

        # Data quality checks
        self._validate_missing_values(data)
        self._validate_duplicates(data)
        self._validate_numeric_ranges(data)
        self._validate_dates(data)

        # Business rule validation
        self._validate_business_rules(data)

        # Calculate quality metrics
        self._calculate_metrics(data)

        return ValidationResult(
            is_valid=len(self.errors) == 0,
            errors=self.errors,
            warnings=self.warnings,
            metrics=self.metrics,
        )

    def _validate_structure(self, data: pd.DataFrame) -> None:
        """Validate basic DataFrame structure"""
        if not isinstance(data, pd.DataFrame):
            self.errors.append("Input must be a pandas DataFrame")
            return

        if len(data) == 0:
            self.errors.append("DataFrame is empty")
            return

        self.metrics["row_count"] = len(data)
        self.metrics["column_count"] = len(data.columns)

    def _validate_columns(self, data: pd.DataFrame) -> None:
        """Validate presence and types of required columns"""
        # Check required columns
        missing_cols = set(self.config["required_columns"]) - set(data.columns)
        if missing_cols:
            self.errors.append(f"Missing required columns: {missing_cols}")

        # Validate numeric columns
        for col in self.config["numeric_columns"]:
            if col in data.columns and not pd.api.types.is_numeric_dtype(data[col]):
                self.errors.append(f"Column {col} must be numeric")

        # Validate date columns
        for col in self.config["date_columns"]:
            if col in data.columns and not pd.api.types.is_datetime64_any_dtype(
                data[col]
            ):
                self.errors.append(f"Column {col} must be datetime")

    def _validate_missing_values(self, data: pd.DataFrame) -> None:
        """Check for missing values"""
        missing_counts = data.isnull().sum()
        missing_cols = missing_counts[missing_counts > 0]

        if not missing_cols.empty:
            self.metrics["missing_values"] = missing_cols.to_dict()

            for col, count in missing_cols.items():
                if col in self.config["required_columns"]:
                    self.errors.append(
                        f"Required column {col} has {count} missing values"
                    )
                else:
                    self.warnings.append(f"Column {col} has {count} missing values")

    def _validate_duplicates(self, data: pd.DataFrame) -> None:
        """Check for duplicate records"""
        duplicate_count = data.duplicated().sum()
        self.metrics["duplicate_count"] = duplicate_count

        if duplicate_count > 0:
            self.warnings.append(f"Found {duplicate_count} duplicate rows")

    def _validate_numeric_ranges(self, data: pd.DataFrame) -> None:
        """Validate numeric value ranges"""
        for col in self.config["numeric_columns"]:
            if col not in data.columns:
                continue

            # Check for negative values
            neg_count = (data[col] < 0).sum()
            if neg_count > 0:
                self.errors.append(f"Found {neg_count} negative values in {col}")

            # Check for outliers using IQR method
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_count = (
                (data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR))
            ).sum()

            if outlier_count > 0:
                self.warnings.append(f"Found {outlier_count} outliers in {col}")
                self.metrics[f"{col}_outliers"] = outlier_count

    def _validate_dates(self, data: pd.DataFrame) -> None:
        """Validate date logic"""
        if (
            "first_purchase_date" in data.columns
            and "last_purchase_date" in data.columns
        ):
            # Check for dates in the future
            today = pd.Timestamp.now().date()
            future_first = (data["first_purchase_date"].dt.date > today).sum()
            future_last = (data["last_purchase_date"].dt.date > today).sum()

            if future_first > 0:
                self.errors.append(
                    f"Found {future_first} first purchase dates in the future"
                )
            if future_last > 0:
                self.errors.append(
                    f"Found {future_last} last purchase dates in the future"
                )

            # Check date order
            invalid_order = (
                data["first_purchase_date"] > data["last_purchase_date"]
            ).sum()

            if invalid_order > 0:
                self.errors.append(
                    f"Found {invalid_order} records where first purchase date is after last purchase date"
                )

    def _validate_business_rules(self, data: pd.DataFrame) -> None:
        """Validate business rules"""
        # Frequency validation
        if "frequency" in data.columns:
            invalid_freq = (data["frequency"] < self.config["min_frequency"]).sum()
            if invalid_freq > 0:
                self.errors.append(
                    f"Found {invalid_freq} records with frequency below minimum threshold"
                )

        # Revenue validation
        if "total_revenue" in data.columns:
            invalid_rev = (data["total_revenue"] < self.config["min_revenue"]).sum()
            if invalid_rev > 0:
                self.errors.append(
                    f"Found {invalid_rev} records with revenue below minimum threshold"
                )

        # Transaction value validation
        if "avg_transaction_value" in data.columns:
            invalid_trans = (
                data["avg_transaction_value"] < self.config["min_transaction_value"]
            ).sum()
            if invalid_trans > 0:
                self.errors.append(
                    f"Found {invalid_trans} records with transaction value below minimum threshold"
                )

    def _calculate_metrics(self, data: pd.DataFrame) -> None:
        """Calculate additional quality metrics"""
        self.metrics.update(
            {
                "timestamp": datetime.now().isoformat(),
                "total_records": len(data),
                "memory_usage": data.memory_usage(deep=True).sum(),
                "column_completeness": (1 - data.isnull().mean()).to_dict(),
                "numeric_columns_stats": {
                    col: {
                        "mean": data[col].mean(),
                        "std": data[col].std(),
                        "min": data[col].min(),
                        "max": data[col].max(),
                    }
                    for col in self.config["numeric_columns"]
                    if col in data.columns
                },
            }
        )

    def get_validation_summary(self) -> str:
        """Generate a formatted validation summary"""
        summary = []
        summary.append("=== Validation Summary ===")

        if self.errors:
            summary.append("\nErrors:")
            summary.extend([f"- {error}" for error in self.errors])

        if self.warnings:
            summary.append("\nWarnings:")
            summary.extend([f"- {warning}" for warning in self.warnings])

        summary.append("\nMetrics:")
        for key, value in self.metrics.items():
            if isinstance(value, dict):
                summary.append(f"\n{key}:")
                for k, v in value.items():
                    summary.append(f"  {k}: {v}")
            else:
                summary.append(f"{key}: {value}")

        return "\n".join(summary)
