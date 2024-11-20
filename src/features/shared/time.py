from ..dependencies import ConditionalFeature


class SeasonalityFeature(ConditionalFeature):
    """Extract comprehensive seasonality features from date"""

    def __init__(self, date_column: str):
        self.date_column = date_column
        super().__init__(
            name=f"seasonality_{date_column}",
            description=f"Seasonality features from {date_column}",
        )
