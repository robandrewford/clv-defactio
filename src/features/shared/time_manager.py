# from .time import SeasonalityFeature


class TimeFeatureManager:
    """Manages and coordinates time-based features"""

    def __init__(
        self, date_column: str, time_zone: str = "UTC", country_code: str = "US"
    ):
        self.date_column = date_column
        self.time_zone = time_zone
        self.country_code = country_code
