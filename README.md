clv-defactio/
├── src/
│   ├── config/
│   │   ├── segment_config.yaml
│   │   ├── pipeline_config.yaml
│   │   ├── model_config.yaml
│   │   └── system_config.yaml
│   ├── monitoring/
│   │   ├── alerts.py
│   │   └── service.py
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── clv/
│   │       ├── __init__.py
│   │       ├── base.py
│   │       ├── config.py
│   │       ├── model.py
│   │       ├── preprocessing.py
│   │       ├── registry.py
│   │       ├── runner.py
│   │       ├── segmentation.py
│   │       └── vertex_components.py
│   └── infrastructure/
│       ├── __init__.py
│       └── gcp_setup.py
├── tests/
│   ├── conftest.py
│   └── test_clv/
│       ├── test_integration.py
│       ├── test_metadata_analysis.py
│       ├── test_model_diagnostics.py
│       ├── test_performance.py
│       ├── test_report_generation.py
│       ├── test_statistical_analysis.py
│       └── test_statistical_visualization.py
├── .env
├── .gitignore
├── README.md
├── clv-defactio.code-workspace
├── poetry.lock
└── pyproject.toml