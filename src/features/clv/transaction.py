from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..dependencies import Condition, ConditionalFeature, Dependency, DependencyType
from ..preprocessing import NumericPreprocessor
from ..validation import NullRule, RangeRule, UniqueRule


class TransactionMetricsFeature(ConditionalFeature):
    """Calculate transaction metrics with conditional dependencies"""

    def __init__(self):
        # Define conditions
        has_basic_data = Condition(
            lambda ctx: all(
                col in ctx["available_data"].get("transactions", pd.DataFrame())
                for col in ["transaction_amount", "transaction_date", "customer_id"]
            ),
            "Basic transaction data available",
        )

        has_product_data = Condition(
            lambda ctx: all(
                col in ctx["available_data"].get("transactions", pd.DataFrame())
                for col in ["product_id", "category_id", "unit_price", "quantity"]
            ),
            "Product-level data available",
        )

        has_discount_data = Condition(
            lambda ctx: "discount_amount"
            in ctx["available_data"].get("transactions", pd.DataFrame()),
            "Discount data available",
        )

        has_return_data = Condition(
            lambda ctx: "is_return"
            in ctx["available_data"].get("transactions", pd.DataFrame()),
            "Return data available",
        )

        dependencies = [
            # Basic transaction dependencies
            Dependency(
                name="transactions.basic",
                type=DependencyType.GROUP,
                conditions=[has_basic_data],
                validation={
                    "required_features": [
                        "transaction_amount",
                        "transaction_date",
                        "customer_id",
                    ]
                },
            ),
            # Product-level dependencies
            Dependency(
                name="transactions.product",
                type=DependencyType.GROUP,
                conditions=[has_product_data],
                optional=True,
                validation={
                    "required_features": [
                        "product_id",
                        "category_id",
                        "unit_price",
                        "quantity",
                    ]
                },
            ),
            # Discount dependencies
            Dependency(
                name="transactions.discount",
                type=DependencyType.DATA,
                conditions=[has_discount_data],
                optional=True,
            ),
            # Return dependencies
            Dependency(
                name="transactions.returns",
                type=DependencyType.DATA,
                conditions=[has_return_data],
                optional=True,
            ),
        ]

        super().__init__(
            name="transaction_metrics",
            description="Comprehensive transaction metrics",
            dependencies=dependencies,
            conditions=[has_basic_data],  # Require basic data at minimum
            validation_rules=[
                RangeRule(min_value=0, max_value=1e6),
                NullRule(max_null_pct=0.01),
            ],
            preprocessor=NumericPreprocessor(
                method="standard", handle_nulls="mean", handle_outliers=True
            ),
        )

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute transaction metrics based on available data"""
        # Get active dependencies
        active_deps = self.dependency_manager.get_active_dependencies(self.name)

        # Initialize results
        metrics = {}

        # Compute basic metrics (always available)
        basic_metrics = self._compute_basic_metrics(df)
        metrics.update(basic_metrics)

        # Compute additional metrics based on available data
        for dep in active_deps:
            if dep.name == "transactions.product":
                product_metrics = self._compute_product_metrics(df)
                metrics.update(product_metrics)

            if dep.name == "transactions.discount":
                discount_metrics = self._compute_discount_metrics(df)
                metrics.update(discount_metrics)

            if dep.name == "transactions.returns":
                return_metrics = self._compute_return_metrics(df)
                metrics.update(return_metrics)

        return pd.DataFrame(metrics)

    def _compute_basic_metrics(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Compute basic transaction metrics"""
        # Group by customer
        grouped = df.groupby("customer_id")

        # Calculate date-based metrics
        df["transaction_date"] = pd.to_datetime(df["transaction_date"])
        last_purchase = grouped["transaction_date"].max()
        first_purchase = grouped["transaction_date"].min()

        return {
            # Monetary metrics
            "total_revenue": grouped["transaction_amount"].sum(),
            "avg_transaction_value": grouped["transaction_amount"].mean(),
            "max_transaction_value": grouped["transaction_amount"].max(),
            "min_transaction_value": grouped["transaction_amount"].min(),
            "revenue_variance": grouped["transaction_amount"].std().fillna(0),
            # Frequency metrics
            "transaction_count": grouped.size(),
            "purchase_frequency": grouped.size()
            / ((last_purchase - first_purchase).dt.days + 1),
            # Recency metrics
            "days_since_last_purchase": (datetime.now() - last_purchase).dt.days,
            "customer_age_days": (datetime.now() - first_purchase).dt.days,
            # Time between purchases
            "avg_days_between_purchases": grouped.apply(
                lambda x: (
                    x["transaction_date"].diff().dt.days.mean() if len(x) > 1 else 0
                )
            ),
        }

    def _compute_product_metrics(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Compute product-related metrics"""
        grouped = df.groupby("customer_id")

        return {
            # Product diversity
            "unique_products": grouped["product_id"].nunique(),
            "unique_categories": grouped["category_id"].nunique(),
            # Quantity metrics
            "total_items": grouped["quantity"].sum(),
            "avg_items_per_transaction": (grouped["quantity"].sum() / grouped.size()),
            # Price metrics
            "avg_unit_price": grouped["unit_price"].mean(),
            "price_range": (grouped["unit_price"].max() - grouped["unit_price"].min()),
            # Category concentration
            "category_concentration": grouped.apply(
                lambda x: (x.groupby("category_id").size().max() / len(x))
            ),
        }

    def _compute_discount_metrics(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Compute discount-related metrics"""
        grouped = df.groupby("customer_id")

        # Calculate discount percentage
        df["discount_percentage"] = (
            df["discount_amount"] / df["transaction_amount"]
        ).fillna(0)

        return {
            "total_discount_amount": grouped["discount_amount"].sum(),
            "avg_discount_percentage": grouped["discount_percentage"].mean(),
            "discount_frequency": (
                grouped["discount_amount"].apply(lambda x: (x > 0).mean())
            ),
            "max_discount_percentage": grouped["discount_percentage"].max(),
            "discount_variance": grouped["discount_percentage"].std().fillna(0),
        }

    def _compute_return_metrics(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Compute return-related metrics"""
        grouped = df.groupby("customer_id")

        return {
            "return_rate": grouped["is_return"].mean(),
            "total_returns": grouped["is_return"].sum(),
            "days_to_return": grouped.apply(
                lambda x: x[x["is_return"]]["transaction_date"].diff().dt.days.mean()
                if x["is_return"].any()
                else 0
            ),
            "return_amount_ratio": grouped.apply(
                lambda x: (
                    x[x["is_return"]]["transaction_amount"].sum()
                    / x["transaction_amount"].sum()
                    if x["transaction_amount"].sum() > 0
                    else 0
                )
            ),
        }
