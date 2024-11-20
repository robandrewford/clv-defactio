from dataclasses import dataclass
from typing import Any, Dict, List

from ..models.metrics import ModelMetrics
from ..pipeline.results import ProcessedData


@dataclass
class AnalysisResults:
    """Container for analysis results"""

    models: Dict[str, Any]
    insights: Dict[str, Any]
    metrics: ModelMetrics
    timestamp: str
    analysis_id: str


class AnalysisCoordinator:
    """Coordinates the analysis workflow"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.active_analyses = {}
        self.results_cache = {}

    def create_analysis(self, analysis_config: Dict[str, Any]) -> str:
        """Create a new analysis job"""
        analysis_id = self._generate_analysis_id()
        self.active_analyses[analysis_id] = {
            "status": "initialized",
            "config": analysis_config,
            "timestamp": datetime.now().isoformat(),
        }
        return analysis_id

    def run_analysis(self, analysis_id: str) -> AnalysisResults:
        """Execute an analysis job"""
        try:
            analysis = self.active_analyses.get(analysis_id)
            if not analysis:
                raise ValueError(f"Analysis {analysis_id} not found")

            # Update status
            analysis["status"] = "running"

            # Execute pipeline
            pipeline_results = self.pipeline_manager.execute(
                analysis["config"]["pipeline"]
            )

            # Train models
            model_results = self.model_manager.train(pipeline_results.processed_data)

            # Generate insights
            insights = self._generate_insights(pipeline_results, model_results)

            # Create results
            results = AnalysisResults(
                models=model_results,
                insights=insights,
                metrics=pipeline_results.metrics,
                timestamp=datetime.now().isoformat(),
                analysis_id=analysis_id,
            )

            # Cache results
            self.results_cache[analysis_id] = results
            analysis["status"] = "completed"

            return results

        except Exception as e:
            analysis["status"] = "failed"
            analysis["error"] = str(e)
            raise

    def get_status(self, analysis_id: str) -> Dict[str, Any]:
        """Get the status of an analysis"""
        analysis = self.active_analyses.get(analysis_id)
        if not analysis:
            raise ValueError(f"Analysis {analysis_id} not found")
        return {
            "status": analysis["status"],
            "timestamp": analysis["timestamp"],
            "error": analysis.get("error"),
        }

    def get_results(self, analysis_id: str) -> AnalysisResults:
        """Retrieve analysis results"""
        results = self.results_cache.get(analysis_id)
        if not results:
            raise ValueError(f"Results for analysis {analysis_id} not found")
        return results

    def _generate_analysis_id(self) -> str:
        """Generate unique analysis ID"""
        return f"analysis_{uuid.uuid4().hex[:8]}"

    def _generate_insights(
        self, pipeline_results: "ProcessedData", model_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate insights from results"""
        insights = {
            "data_quality": self._analyze_data_quality(pipeline_results),
            "model_performance": self._analyze_model_performance(model_results),
            "business_metrics": self._analyze_business_metrics(pipeline_results),
        }
        return insights
