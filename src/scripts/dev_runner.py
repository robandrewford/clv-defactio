# scripts/dev_runner.py
import yaml

from src.models import CLVModel
from src.pipeline import VertexPipeline
from src.utils.gcp_config import GCPDevConfig


class LocalDevelopment:
    """Manages local development environment"""

    def __init__(self):
        self.gcp_config = GCPDevConfig()
        self.load_config()

    def load_config(self):
        """Load development configuration"""
        with open("configs/development.yaml", "r") as f:
            self.config = yaml.safe_load(f)

    def setup_development(self):
        """Setup development environment"""
        try:
            # Initialize GCP clients
            self.gcp_config.initialize_clients()

            # Setup pipeline
            self.pipeline = VertexPipeline(self.config["pipeline"])

            # Setup model
            self.model = CLVModel(self.config["model"])

            print("Development environment ready!")

        except Exception as e:
            print(f"Development setup failed: {str(e)}")
            raise

    def run_local_test(self):
        """Run local development test"""
        try:
            # Test data access
            bucket = self.gcp_config.storage_client.get_bucket(
                self.config["development"]["test_bucket"]
            )

            # Test pipeline components
            self.pipeline.test_components()

            # Test model operations
            self.model.test_operations()

            print("Local tests completed successfully!")

        except Exception as e:
            print(f"Local test failed: {str(e)}")
            raise


if __name__ == "__main__":
    dev = LocalDevelopment()
    dev.setup_development()
    dev.run_local_test()
