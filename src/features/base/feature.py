from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import pandas as pd

from ..preprocessing import Preprocessor
from ..validation import ValidationRule


class Feature(ABC):
    """Enhanced base feature class with validation and preprocessing"""

    def __init__(
        self,
        name: str,
        description: str,
        validation_rules: Optional[List[ValidationRule]] = None,
        preprocessor: Optional[Preprocessor] = None,
    ):
        self.name = name
        self.description = description
        self.validation_rules = validation_rules or []
        self.preprocessor = preprocessor
        self.dependencies: List[str] = []
        self.is_computed = False
        self.is_validated = False
        self.is_preprocessed = False

    def validate(self, data: pd.Series) -> bool:
        """Validate feature data"""
        for rule in self.validation_rules:
            if not rule.validate(data):
                raise ValueError(
                    f"Validation failed for feature {self.name}: {rule.message}"
                )
        self.is_validated = True
        return True

    def preprocess(self, data: pd.Series) -> pd.Series:
        """Preprocess feature data"""
        if self.preprocessor:
            if not self.preprocessor.is_fitted:
                self.preprocessor.fit(data)
            data = self.preprocessor.transform(data)
            self.is_preprocessed = True
        return data

    def compute_and_validate(self, df: pd.DataFrame) -> pd.Series:
        """Compute, validate, and preprocess feature"""
        # Compute raw feature
        result = self.compute(df)

        # Validate
        if self.validation_rules:
            self.validate(result)

        # Preprocess
        if self.preprocessor:
            result = self.preprocess(result)

        return result

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Compute the feature value"""
        pass

    def validate_dependencies(self, df: pd.DataFrame) -> bool:
        """Check if all required columns are present"""
        missing = [col for col in self.dependencies if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns for {self.name}: {missing}")
        return True


class FeatureTransformer(ABC):
    """Base class for feature transformers"""

    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False

    @abstractmethod
    def fit(self, series: pd.Series) -> "FeatureTransformer":
        """Fit the transformer to the data"""
        pass

    @abstractmethod
    def transform(self, series: pd.Series) -> pd.Series:
        """Transform the data"""
        pass

    def fit_transform(self, series: pd.Series) -> pd.Series:
        """Fit and transform the data"""
        return self.fit(series).transform(series)
