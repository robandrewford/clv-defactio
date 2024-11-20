import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from .base.feature import Feature
from .dependencies import Dependency, DependencyManager, DependencyType

logger = logging.getLogger(__name__)


@dataclass
class FeatureGroup:
    """Group of related features"""

    name: str
    features: List[Feature]
    dependencies: List[str]  # Required input DataFrames

    def validate_inputs(self, inputs: Dict[str, pd.DataFrame]) -> bool:
        """Validate input DataFrames"""
        missing = [dep for dep in self.dependencies if dep not in inputs]
        if missing:
            raise ValueError(f"Missing required inputs for {self.name}: {missing}")
        return True


class FeaturePipeline:
    """Manages feature computation pipeline"""

    def __init__(
        self, groups: List[FeatureGroup], parallel: bool = True, max_workers: int = 4
    ):
        """
        Initialize feature pipeline

        Parameters
        ----------
        groups : List[FeatureGroup]
            Feature groups to process
        parallel : bool
            Whether to compute features in parallel
        max_workers : int
            Maximum number of parallel workers
        """
        self.groups = {group.name: group for group in groups}
        self.parallel = parallel
        self.max_workers = max_workers
        self.computed_features: Dict[str, pd.DataFrame] = {}
        self.dependency_manager = DependencyManager()

        # Validate feature groups
        self._validate_groups()

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Build dependency graph
        self._build_dependency_graph()

    def _validate_groups(self) -> None:
        """Validate feature group configuration"""
        # Check for duplicate feature names
        all_features = []
        for group in self.groups.values():
            for feature in group.features:
                if feature.name in all_features:
                    raise ValueError(f"Duplicate feature name: {feature.name}")
                all_features.append(feature.name)

    def _build_dependency_graph(self):
        """Build complete dependency graph"""
        for group in self.groups.values():
            for feature in group.features:
                for dep in feature.dependencies:
                    self.dependency_manager.add_dependency(feature.name, dep)

    def compute_features(self, inputs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Compute features in dependency order"""
        # Get computation order
        feature_order = self.dependency_manager.get_computation_order()

        # Compute features in order
        for feature_name in feature_order:
            feature = self._get_feature_by_name(feature_name)
            if feature:
                # Validate dependencies
                feature.validate_dependencies(inputs, self.computed_features)

                # Compute feature
                result = feature.compute_and_validate(inputs)
                self.computed_features[feature_name] = result

        return pd.DataFrame(self.computed_features)

    def _get_feature_by_name(self, feature_name: str) -> Optional[Feature]:
        """Get feature by name"""
        for group in self.groups.values():
            for feature in group.features:
                if feature.name == feature_name:
                    return feature
        return None

    def get_feature_info(self) -> Dict[str, Any]:
        """Get information about computed features"""
        info = {}

        for group_name, group in self.groups.items():
            group_info = {
                "features": [
                    {
                        "name": feature.name,
                        "description": feature.description,
                        "dependencies": feature.dependencies,
                    }
                    for feature in group.features
                ],
                "dependencies": group.dependencies,
                "computed": group_name in self.computed_features,
            }

            if group_name in self.computed_features:
                df = self.computed_features[group_name]
                group_info.update(
                    {
                        "n_features": len(df.columns),
                        "n_rows": len(df),
                        "memory_usage": df.memory_usage().sum() / 1024**2,  # MB
                    }
                )

            info[group_name] = group_info

        return info
