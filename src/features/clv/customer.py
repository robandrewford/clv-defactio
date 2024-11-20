from typing import List, Optional

import pandas as pd

from ..base.feature import Feature
from ..dependencies import Condition, ConditionalFeature, Dependency, DependencyType


class CustomerAgeFeature(Feature):
    """Calculate customer age in days"""

    def __init__(self):
        super().__init__(
            name="customer_age_days",
            description="Customer age in days since first purchase",
        )
        self.dependencies = ["first_purchase_date"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        self.validate_dependencies(df)
        today = pd.Timestamp.now()
        return (today - pd.to_datetime(df["first_purchase_date"])).dt.days


class PurchaseFrequencyFeature(Feature):
    """Calculate purchase frequency"""

    def __init__(self):
        super().__init__(
            name="purchase_frequency", description="Number of purchases per time unit"
        )
        self.dependencies = ["transaction_count", "customer_age_days"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        self.validate_dependencies(df)
        return df["transaction_count"] / df["customer_age_days"]


class CustomerValueFeature(ConditionalFeature):
    """Calculate customer value with conditional dependencies"""

    def __init__(self):
        # Define conditions
        has_transactions = Condition(
            lambda ctx: "transactions" in ctx["available_data"],
            "Transactions data is available",
        )

        has_loyalty = Condition(
            lambda ctx: "loyalty_score" in ctx["computed_features"],
            "Loyalty score is computed",
        )

        is_new_customer = Condition(
            lambda ctx: ctx.get("customer_age_days", float("inf")) < 90,
            "Customer is new (less than 90 days)",
        )

        # Define dependencies with conditions
        dependencies = [
            # Basic transaction dependency
            Dependency(
                name="transactions.amount",
                type=DependencyType.DATA,
                conditions=[has_transactions],
            ),
            # Conditional loyalty dependency
            Dependency(
                name="loyalty_score",
                type=DependencyType.FEATURE,
                conditions=[has_loyalty],
                optional=True,
            ),
            # Alternative dependencies for new vs existing customers
            Dependency(
                name="historical_value",
                type=DependencyType.FEATURE,
                conditions=[
                    Condition(
                        lambda ctx: not is_new_customer.evaluate(ctx),
                        "Customer is existing",
                    )
                ],
                alternatives=[
                    Dependency(
                        name="predicted_value",
                        type=DependencyType.FEATURE,
                        conditions=[is_new_customer],
                    )
                ],
            ),
        ]

        super().__init__(
            name="customer_value",
            description="Customer value score",
            dependencies=dependencies,
            conditions=[has_transactions],  # Feature-level conditions
            validation_rules=[RangeRule(min_value=0, max_value=1e6)],
        )

    def compute(self, df: pd.DataFrame) -> pd.Series:
        # Update context with current state
        self.dependency_manager.update_context(
            {"customer_age_days": df["customer_age_days"].mean()}
        )

        # Validate dependencies (will consider conditions)
        self.validate_dependencies(
            available_data={"transactions": df},
            computed_features=self.get_computed_features(),
        )

        # Get active dependencies
        active_deps = self.dependency_manager.get_active_dependencies(self.name)

        # Compute feature based on active dependencies
        if any(dep.name == "loyalty_score" for dep in active_deps):
            # Include loyalty in calculation
            result = self._compute_with_loyalty(df)
        else:
            # Basic calculation without loyalty
            result = self._compute_basic(df)

        return result
