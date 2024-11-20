"""Custom exceptions for the CLV 360 project."""


class CLVError(Exception):
    """Base exception for all CLV-related errors."""

    pass


class DataError(CLVError):
    """Base exception for all data-related errors."""

    pass


class DataLoadError(DataError):
    """Raised when data cannot be loaded."""

    pass


class DataProcessingError(DataError):
    """Raised when data processing fails."""

    pass


class DataValidationError(DataError):
    """Raised when data validation fails."""

    pass


class ModelError(CLVError):
    """Base exception for all model-related errors."""

    pass


class FeatureError(CLVError):
    """Base exception for all feature-related errors."""

    pass


class PipelineError(CLVError):
    """Base exception for all pipeline-related errors."""

    pass
