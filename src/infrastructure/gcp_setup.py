from typing import Any, Dict

from google.cloud import resource_manager


class GCPSetup:
    """Manages GCP infrastructure setup"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = resource_manager.Client()

    def setup_project(self) -> None:
        """Setup GCP project infrastructure"""
        try:
            # Create project
            project = self._create_project()

            # Enable APIs
            self._enable_apis(project.project_id)

            # Setup networking
            self._setup_networking(project.project_id)

            # Setup storage
            self._setup_storage(project.project_id)

        except Exception as e:
            logging.error(f"Project setup failed: {str(e)}")
            raise
