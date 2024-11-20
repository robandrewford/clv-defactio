# CLV 360 Project Structure

clv_360/
├── .env                      # Environment variables
├── .gitignore               # Git ignore file
├── pyproject.toml           # Project metadata and dependencies
├── README.md               # Project documentation
│
├── configs/                # Configuration files
│   ├── __init__.py
│   ├── system_config.yaml  # Main system configuration
│   ├── pipeline_config.yaml
│   ├── model_config.yaml
│   └── deployment_config.yaml
│
├── src/                    # Source code
│   ├── __init__.py
│   │
│   ├── infrastructure/    # Infrastructure setup
│   │   ├── __init__.py
│   │   ├── gcp_setup.py
│   │   ├── network.py
│   │   ├── storage.py
│   │   └── monitoring.py
│   │
│   ├── pipeline/         # Pipeline components
│   │   ├── __init__.py
│   │   ├── processor.py
│   │   ├── feature_engineering.py
│   │   └── vertex_pipeline.py
│   │
│   ├── models/          # Model definitions
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── clv_model.py
│   │   └── model_registry.py
│   │
│   ├── data/           # Data handling
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   ├── processor.py
│   │   └── validator.py
│   │
│   └── utils/          # Utilities
│       ├── __init__.py
│       ├── logging.py
│       ├── monitoring.py
│       └── helpers.py
│
├── deployment/         # Deployment configurations
│   ├── __init__.py
│   ├── terraform/     # Infrastructure as Code
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── outputs.tf
│   │
│   └── kubernetes/    # Kubernetes configurations
│       ├── deployments/
│       └── services/
│
├── tests/            # Test files
│   ├── __init__.py
│   ├── test_pipeline.py
│   ├── test_models.py
│   └── test_data.py
│
└── scripts/          # Utility scripts
    ├── deploy.py
    ├── setup.py
    └── cleanup.py
