from typing import Any, Dict

from ..models.manager import ModelManager
from ..monitoring.service import MonitoringService
from ..pipeline.manager import PipelineManager
from ..storage.manager import StorageManager


class CLVSystem:
    def __init__(self, config: Dict[str, Any]):
        self.pipeline_manager = PipelineManager(config)
        self.model_manager = ModelManager(config)
        self.monitoring = MonitoringService(config)
        self.storage = StorageManager(config)

    def run_analysis(self, data_config: Dict[str, Any]) -> "AnalysisResults":
        try:
            # Start monitoring
            self.monitoring.track_execution(data_config["pipeline_id"])

            # Run pipeline
            pipeline_results = self.pipeline_manager.execute(data_config)

            # Train models
            models = self.model_manager.train(pipeline_results.processed_data)

            # Store results
            results = AnalysisResults(
                models=models,
                insights=pipeline_results.insights,
                metrics=pipeline_results.metrics,
            )
            self.storage.store_results(results)

            return results

        except Exception as e:
            self.monitoring.log_error(e)
            raise
