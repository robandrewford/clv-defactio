from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..dependencies import Condition, ConditionalFeature, Dependency, DependencyType
from ..preprocessing import NumericPreprocessor
from ..validation import NullRule, RangeRule


class ChannelEngagementFeature(ConditionalFeature):
    """Calculate channel engagement with conditional dependencies"""

    def __init__(self):
        # Define conditions for different channels
        has_online = Condition(
            lambda ctx: "online_transactions" in ctx["available_data"],
            "Online transaction data available",
        )

        has_store = Condition(
            lambda ctx: "store_transactions" in ctx["available_data"],
            "Store transaction data available",
        )

        has_mobile = Condition(
            lambda ctx: "mobile_transactions" in ctx["available_data"],
            "Mobile app transaction data available",
        )

        has_email = Condition(
            lambda ctx: all(
                col in ctx["available_data"].get("marketing", pd.DataFrame())
                for col in ["email_opened", "email_clicked"]
            ),
            "Email engagement data available",
        )

        dependencies = [
            # Online channel dependencies
            Dependency(
                name="online_transactions.amount",
                type=DependencyType.DATA,
                conditions=[has_online],
                optional=True,
                validation={"max_null_pct": 0.1},
            ),
            # Store channel dependencies
            Dependency(
                name="store_transactions.amount",
                type=DependencyType.DATA,
                conditions=[has_store],
                optional=True,
                validation={"max_null_pct": 0.1},
            ),
            # Mobile channel dependencies
            Dependency(
                name="mobile_transactions.amount",
                type=DependencyType.DATA,
                conditions=[has_mobile],
                optional=True,
                validation={"max_null_pct": 0.1},
            ),
            # Email engagement dependencies
            Dependency(
                name="marketing.email_metrics",
                type=DependencyType.GROUP,
                conditions=[has_email],
                optional=True,
                validation={"required_features": ["email_opened", "email_clicked"]},
            ),
        ]

        super().__init__(
            name="channel_engagement",
            description="Multi-channel engagement metrics",
            dependencies=dependencies,
            # Require at least one channel
            conditions=[
                Condition(
                    lambda ctx: any(
                        [
                            has_online.evaluate(ctx),
                            has_store.evaluate(ctx),
                            has_mobile.evaluate(ctx),
                            has_email.evaluate(ctx),
                        ]
                    ),
                    "At least one channel must be available",
                )
            ],
        )

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        # Get active dependencies
        active_deps = self.dependency_manager.get_active_dependencies(self.name)

        # Initialize results dictionary
        engagement_metrics = {}

        # Process each channel based on active dependencies
        for dep in active_deps:
            if dep.name == "online_transactions.amount":
                online_metrics = self._compute_online_metrics(df)
                engagement_metrics.update(online_metrics)

            elif dep.name == "store_transactions.amount":
                store_metrics = self._compute_store_metrics(df)
                engagement_metrics.update(store_metrics)

            elif dep.name == "mobile_transactions.amount":
                mobile_metrics = self._compute_mobile_metrics(df)
                engagement_metrics.update(mobile_metrics)

            elif dep.name == "marketing.email_metrics":
                email_metrics = self._compute_email_metrics(df)
                engagement_metrics.update(email_metrics)

        # Calculate overall engagement score
        if engagement_metrics:
            engagement_metrics[
                "overall_engagement_score"
            ] = self._calculate_overall_score(engagement_metrics)

        return pd.DataFrame(engagement_metrics)

    def _compute_online_metrics(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Compute online channel metrics"""
        online_df = df["online_transactions"]
        grouped = online_df.groupby("customer_id")

        return {
            "online_frequency": grouped.size(),
            "online_recency_days": (
                pd.Timestamp.now() - grouped["transaction_date"].max()
            ).dt.days,
            "online_total_amount": grouped["amount"].sum(),
            "online_avg_amount": grouped["amount"].mean(),
        }

    def _compute_store_metrics(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Compute store channel metrics"""
        store_df = df["store_transactions"]
        grouped = store_df.groupby("customer_id")

        return {
            "store_frequency": grouped.size(),
            "store_recency_days": (
                pd.Timestamp.now() - grouped["transaction_date"].max()
            ).dt.days,
            "store_total_amount": grouped["amount"].sum(),
            "store_avg_amount": grouped["amount"].mean(),
        }

    def _compute_mobile_metrics(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Compute mobile channel metrics"""
        mobile_df = df["mobile_transactions"]
        grouped = mobile_df.groupby("customer_id")

        return {
            "mobile_frequency": grouped.size(),
            "mobile_recency_days": (
                pd.Timestamp.now() - grouped["transaction_date"].max()
            ).dt.days,
            "mobile_total_amount": grouped["amount"].sum(),
            "mobile_avg_amount": grouped["amount"].mean(),
        }

    def _compute_email_metrics(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Compute email engagement metrics"""
        marketing_df = df["marketing"]
        grouped = marketing_df.groupby("customer_id")

        return {
            "email_open_rate": grouped["email_opened"].mean(),
            "email_click_rate": grouped["email_clicked"].mean(),
            "email_engagement_score": (
                grouped["email_opened"].mean() * 0.4
                + grouped["email_clicked"].mean() * 0.6
            ),
        }

    def _calculate_overall_score(self, metrics: Dict[str, pd.Series]) -> pd.Series:
        """Calculate overall engagement score"""
        scores = []
        weights = []

        # Online channel
        if "online_frequency" in metrics:
            online_score = (
                metrics["online_frequency"] * 0.4
                + (1 / (metrics["online_recency_days"] + 1)) * 0.6
            )
            scores.append(online_score)
            weights.append(0.35)

        # Store channel
        if "store_frequency" in metrics:
            store_score = (
                metrics["store_frequency"] * 0.4
                + (1 / (metrics["store_recency_days"] + 1)) * 0.6
            )
            scores.append(store_score)
            weights.append(0.35)

        # Mobile channel
        if "mobile_frequency" in metrics:
            mobile_score = (
                metrics["mobile_frequency"] * 0.4
                + (1 / (metrics["mobile_recency_days"] + 1)) * 0.6
            )
            scores.append(mobile_score)
            weights.append(0.2)

        # Email channel
        if "email_engagement_score" in metrics:
            scores.append(metrics["email_engagement_score"])
            weights.append(0.1)

        # Normalize weights
        weights = [w / sum(weights) for w in weights]

        # Calculate weighted average
        return sum(score * weight for score, weight in zip(scores, weights))


class LoyaltyEngagementFeature(Feature):
    """Calculate loyalty program engagement metrics"""

    def __init__(self):
        super().__init__(
            name="loyalty_engagement",
            description="Customer loyalty program engagement metrics",
        )
        self.dependencies = [
            "customer_id",
            "loyalty_points",
            "is_loyalty_member",
            "loyalty_join_date",
        ]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        self.validate_dependencies(df)

        # Group by customer
        grouped = df.groupby("customer_id")

        loyalty_metrics = pd.DataFrame(
            {
                "total_loyalty_points": grouped["loyalty_points"].sum(),
                "avg_points_per_transaction": grouped["loyalty_points"].mean(),
                "is_loyalty_member": grouped["is_loyalty_member"].last(),
                "loyalty_tenure_days": (
                    pd.Timestamp.now()
                    - pd.to_datetime(grouped["loyalty_join_date"].first())
                ).dt.days,
            }
        )

        # Calculate points earning rate
        loyalty_metrics["points_earning_rate"] = loyalty_metrics[
            "total_loyalty_points"
        ] / loyalty_metrics["loyalty_tenure_days"].clip(lower=1)

        return loyalty_metrics


class MarketingResponseFeature(Feature):
    """Calculate marketing response metrics"""

    def __init__(self):
        super().__init__(
            name="marketing_response", description="Customer marketing response metrics"
        )
        self.dependencies = [
            "customer_id",
            "email_opened",
            "email_clicked",
            "sms_clicked",
            "campaign_id",
        ]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        self.validate_dependencies(df)

        # Group by customer
        grouped = df.groupby("customer_id")

        response_metrics = pd.DataFrame(
            {
                "email_open_rate": grouped["email_opened"].mean(),
                "email_click_rate": grouped["email_clicked"].mean(),
                "sms_click_rate": grouped["sms_clicked"].mean(),
                "campaigns_received": grouped["campaign_id"].nunique(),
                "total_interactions": grouped[
                    ["email_opened", "email_clicked", "sms_clicked"]
                ]
                .sum()
                .sum(),
            }
        )

        # Calculate engagement score
        response_metrics["engagement_score"] = (
            response_metrics["email_open_rate"] * 0.2
            + response_metrics["email_click_rate"] * 0.4
            + response_metrics["sms_click_rate"] * 0.4
        )

        return response_metrics


class CrossSellFeature(Feature):
    """Calculate cross-sell and category engagement metrics"""

    def __init__(self):
        super().__init__(
            name="cross_sell_metrics",
            description="Customer cross-sell and category metrics",
        )
        self.dependencies = [
            "customer_id",
            "product_category",
            "product_id",
            "transaction_amount",
        ]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        self.validate_dependencies(df)

        # Group by customer
        grouped = df.groupby("customer_id")

        category_metrics = pd.DataFrame(
            {
                "unique_categories": grouped["product_category"].nunique(),
                "unique_products": grouped["product_id"].nunique(),
                "favorite_category": grouped["product_category"].agg(
                    lambda x: x.value_counts().index[0]
                ),
            }
        )

        # Calculate category concentration
        category_counts = df.groupby(["customer_id", "product_category"])[
            "transaction_amount"
        ].sum()

        category_concentration = category_counts.groupby("customer_id").apply(
            lambda x: (x / x.sum() ** 2).sum()
        )

        category_metrics["category_concentration"] = category_concentration

        return category_metrics
