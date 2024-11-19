classDiagram
    class CLVSystem {
        -PipelineManager pipeline_manager
        -ModelManager model_manager
        -MonitoringService monitoring
        -StorageManager storage
        +run_analysis(data_config: DataConfig) AnalysisResults
    }

    class PipelineManager {
        -Dict active_pipelines
        -VertexAIClient vertex_client
        +create_job(data_config: DataConfig) PipelineJob
        +execute(pipeline_job: PipelineJob) PipelineResults
        -monitor_execution(job_id: str)
        -collect_results(job_id: str) PipelineResults
    }

    class ModelManager {
        -Dict active_models
        -ModelRegistry registry
        +train(data: ProcessedData) Dict[str, Model]
        -train_model(model_type: str, data: ProcessedData) Model
        +predict(model_id: str, data: ProcessedData)
        +evaluate(model_id: str) ModelMetrics
    }

    class MonitoringService {
        -MetricsClient metrics_client
        -AlertManager alerts
        +track_execution(pipeline_id: str) ExecutionMetrics
        +log_error(error: Exception)
        -track_performance(pipeline_id: str) Metrics
        -track_resources(pipeline_id: str) Metrics
    }

    class StorageManager {
        -StorageClient client
        -Dict buckets
        +store_results(results: AnalysisResults) StorageLocations
        +retrieve_results(analysis_id: str) AnalysisResults
        -initialize_buckets()
    }

    class AnalysisCoordinator {
        -CLVSystem system
        -Dict active_analyses
        +start_analysis(config: AnalysisConfig) str
        +get_status(analysis_id: str) AnalysisStatus
        -execute_async(analysis_id: str)
    }

    class ConfigManager {
        -Dict configurations
        +load_config(config_path: str) SystemConfig
        +validate_config(config: SystemConfig)
        +update_config(config_updates: Dict)
    }

    class DataProcessor {
        -PipelineConfig config
        +process_batch(data: DataFrame) ProcessedData
        +validate_data(data: DataFrame) ValidationResults
        -clean_data(data: DataFrame) DataFrame
        -engineer_features(data: DataFrame) DataFrame
    }

    class ModelRegistry {
        -Dict registered_models
        +register(model: Model)
        +get_model(model_id: str) Model
        +list_models() List[Model]
        +delete_model(model_id: str)
    }

    class AlertManager {
        -Dict alert_rules
        -NotificationService notifier
        +check_thresholds(metrics: Metrics)
        +send_alert(alert: Alert)
    }

    class CLVSystemAPI {
        -CLVSystem system
        -AnalysisCoordinator coordinator
        +start_analysis(request: AnalysisRequest) AnalysisResponse
        +get_analysis_status(analysis_id: str) StatusResponse
        -validate_request(request: AnalysisRequest)
    }

    %% Relationships
    CLVSystem --> PipelineManager
    CLVSystem --> ModelManager
    CLVSystem --> MonitoringService
    CLVSystem --> StorageManager
    CLVSystem --> ConfigManager

    PipelineManager --> DataProcessor
    PipelineManager --> MonitoringService

    ModelManager --> ModelRegistry
    ModelManager --> StorageManager

    MonitoringService --> AlertManager

    AnalysisCoordinator --> CLVSystem
    CLVSystemAPI --> AnalysisCoordinator

    %% Configurations
    class SystemConfig {
        +PipelineConfig pipeline
        +ModelConfig model
        +MonitoringConfig monitoring
        +StorageConfig storage
    }

    class PipelineConfig {
        +str pipeline_root
        +int batch_size
        +Dict resource_config
        +Dict monitoring_config
    }

    class ModelConfig {
        +List[str] model_types
        +Dict model_params
        +Dict training_config
    }

    class MonitoringConfig {
        +Dict thresholds
        +int monitoring_frequency
        +Dict alert_rules
    }

    class StorageConfig {
        +str storage_root
        +Dict bucket_config
        +Dict retention_policy
    }

    %% Data Types
    class AnalysisResults {
        +Dict[str, Model] models
        +Dict insights
        +Dict metrics
    }

    class ProcessedData {
        +DataFrame data
        +Dict feature_info
        +Dict processing_metrics
    }

    class ExecutionMetrics {
        +Dict performance
        +Dict resource_usage
        +Dict data_quality
    }

The relationships should show a clean separation of concerns where:
- CLVSystem is the main orchestrator
- PipelineManager handles Vertex AI pipeline operations
- ModelManager manages model lifecycle
- MonitoringService provides observability
- StorageManager handles data persistence
- AnalysisCoordinator coordinates workflows
- CLVSystemAPI provides external interface

Each component should have:
1. Clear responsibilities
2. Well-defined interfaces
3. Configuration management
4. Error handling
5. Monitoring capabilities

Next steps:
1. Detail any specific class/relationship?
2. Add more components/relationships?
3. Show sequence diagrams for key operations?
4. Expand on configuration structures?
