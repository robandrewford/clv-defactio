from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

import networkx as nx
import pandas as pd


class DependencyType(Enum):
    """Types of dependencies"""

    DATA = "data"  # Depends on input data column
    FEATURE = "feature"  # Depends on another feature
    GROUP = "group"  # Depends on feature group
    CONDITIONAL = "conditional"


@dataclass
class Condition:
    """Represents a dependency condition"""

    check: Callable[[Dict[str, Any]], bool]
    message: str

    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate the condition"""
        try:
            return self.check(context)
        except Exception as e:
            raise ValueError(f"Condition evaluation failed: {self.message} - {str(e)}")


@dataclass
class Dependency:
    """Enhanced dependency with conditions"""

    name: str
    type: DependencyType
    optional: bool = False
    validation: Optional[Dict[str, Any]] = None
    conditions: Optional[List[Condition]] = None
    alternatives: Optional[List["Dependency"]] = None

    def validate(self, context: Dict[str, Any]) -> bool:
        """Validate dependency conditions"""
        if not self.conditions:
            return True

        return all(condition.evaluate(context) for condition in self.conditions)


class ConditionalDependencyManager:
    """Manages conditional feature dependencies"""

    def __init__(self):
        self.dependency_graph = nx.DiGraph()
        self.computed_features: Set[str] = set()
        self.context: Dict[str, Any] = {}

    def add_dependency(self, feature_name: str, dependency: Dependency) -> None:
        """Add a dependency to the graph"""
        self.dependency_graph.add_edge(
            dependency.name, feature_name, dependency=dependency
        )

    def update_context(self, context: Dict[str, Any]) -> None:
        """Update the dependency context"""
        self.context.update(context)

    def get_active_dependencies(self, feature_name: str) -> List[Dependency]:
        """Get list of active dependencies based on conditions"""
        active_deps = []

        for _, _, edge_data in self.dependency_graph.in_edges(feature_name, data=True):
            dep = edge_data["dependency"]

            # Check if dependency conditions are met
            if dep.validate(self.context):
                active_deps.append(dep)
            elif dep.alternatives:
                # Try alternative dependencies
                for alt_dep in dep.alternatives:
                    if alt_dep.validate(self.context):
                        active_deps.append(alt_dep)
                        break

        return active_deps

    def get_computation_order(self) -> List[str]:
        """Get ordered list of features considering conditions"""
        # Create a subgraph with only active dependencies
        active_graph = nx.DiGraph()

        for node in self.dependency_graph.nodes():
            active_deps = self.get_active_dependencies(node)
            for dep in active_deps:
                active_graph.add_edge(dep.name, node)

        try:
            return list(nx.topological_sort(active_graph))
        except nx.NetworkXUnfeasible:
            raise ValueError("Circular dependencies detected in active dependencies")

    def validate_dependencies(
        self,
        feature_name: str,
        available_data: Dict[str, pd.DataFrame],
        computed_features: Dict[str, pd.Series],
    ) -> bool:
        """Validate dependencies considering conditions"""
        # Update context with current state
        self.update_context(
            {
                "available_data": available_data,
                "computed_features": computed_features,
                "feature_name": feature_name,
            }
        )

        # Get active dependencies
        active_deps = self.get_active_dependencies(feature_name)

        missing = []
        invalid = []

        for dep in active_deps:
            if dep.type == DependencyType.DATA:
                if not self._validate_data_dependency(dep, available_data):
                    if not dep.optional:
                        missing.append(f"Data: {dep.name}")

            elif dep.type == DependencyType.FEATURE:
                if not self._validate_feature_dependency(dep, computed_features):
                    if not dep.optional:
                        missing.append(f"Feature: {dep.name}")

            elif dep.type == DependencyType.GROUP:
                if not self._validate_group_dependency(dep, computed_features):
                    if not dep.optional:
                        missing.append(f"Group: {dep.name}")

            elif dep.type == DependencyType.CONDITIONAL:
                if not self._validate_conditional_dependency(dep):
                    if not dep.optional:
                        missing.append(f"Conditional: {dep.name}")

        if missing or invalid:
            raise ValueError(
                f"Dependency validation failed for {feature_name}:\n"
                f"Missing: {missing}\nInvalid: {invalid}"
            )

        return True


class ConditionalFeature(EnhancedFeature):
    """Feature with conditional dependencies"""

    def __init__(
        self,
        name: str,
        description: str,
        dependencies: List[Dependency],
        conditions: Optional[List[Condition]] = None,
        validation_rules: Optional[List[ValidationRule]] = None,
        preprocessor: Optional[Preprocessor] = None,
    ):
        super().__init__(
            name, description, dependencies, validation_rules, preprocessor
        )
        self.conditions = conditions or []
        self.dependency_manager = ConditionalDependencyManager()

        # Add dependencies to manager
        for dep in dependencies:
            self.dependency_manager.add_dependency(name, dep)


class EnhancedFeature(Feature):
    """Feature class with enhanced dependency management"""

    def __init__(
        self,
        name: str,
        description: str,
        dependencies: List[Dependency],
        validation_rules: Optional[List[ValidationRule]] = None,
        preprocessor: Optional[Preprocessor] = None,
    ):
        super().__init__(name, description, validation_rules, preprocessor)
        self.dependencies = dependencies

    def validate_dependencies(
        self,
        available_data: Dict[str, pd.DataFrame],
        computed_features: Dict[str, pd.Series],
    ) -> bool:
        """Validate all dependencies"""
        dep_manager = DependencyManager()

        # Add all dependencies
        for dep in self.dependencies:
            dep_manager.add_dependency(self.name, dep)

        # Validate
        return dep_manager.validate_dependencies(
            self.name, available_data, computed_features
        )
