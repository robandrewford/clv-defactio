import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class StorageLocation:
    """Represents a single storage location configuration"""

    name: str
    path: str
    type: str  # 'local', 'gcs', 's3', etc.
    credentials: Optional[Dict[str, Any]] = None
    retention_days: Optional[int] = None

    def __post_init__(self):
        """Validate storage location after initialization"""
        self._validate_location()

    def _validate_location(self) -> None:
        """Validate storage location configuration"""
        valid_types = ["local", "gcs", "s3", "azure"]
        if self.type not in valid_types:
            raise ValueError(f"Invalid storage type. Must be one of: {valid_types}")

        if self.type != "local" and not self.credentials:
            raise ValueError(f"Credentials required for {self.type} storage")

    def get_provider(self) -> StorageProvider:
        """Get the appropriate storage provider"""
        if self.type == "local":
            return None
        elif self.type == "gcs":
            return GCSProvider(self.credentials)
        elif self.type == "s3":
            return S3Provider(self.credentials)
        else:
            raise ValueError(f"Unsupported storage type: {self.type}")

    def upload(self, local_path: str, remote_path: str) -> bool:
        """Upload a file to this storage location"""
        if self.type == "local":
            try:
                import shutil

                dest_path = Path(self.path) / remote_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(local_path, dest_path)
                return True
            except Exception as e:
                logging.error(f"Local copy failed: {str(e)}")
                return False
        else:
            provider = self.get_provider()
            return provider.upload_file(local_path, remote_path)

    def download(self, remote_path: str, local_path: str) -> bool:
        """Download a file from this storage location"""
        if self.type == "local":
            try:
                import shutil

                src_path = Path(self.path) / remote_path
                dest_path = Path(local_path)
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dest_path)
                return True
            except Exception as e:
                logging.error(f"Local copy failed: {str(e)}")
                return False
        else:
            provider = self.get_provider()
            return provider.download_file(remote_path, local_path)


class StorageLocations:
    """Manages storage locations for different data types"""

    def __init__(self, config_path: Optional[str] = None):
        self.locations: Dict[str, StorageLocation] = {}
        self.config_path = config_path
        if config_path:
            self.load_config(config_path)
        else:
            self._setup_default_locations()

    def _setup_default_locations(self) -> None:
        """Setup default storage locations"""
        base_path = Path.cwd() / "data"

        default_locations = {
            "models": StorageLocation(
                name="models",
                path=str(base_path / "models"),
                type="local",
                retention_days=90,
            ),
            "processed_data": StorageLocation(
                name="processed_data",
                path=str(base_path / "processed"),
                type="local",
                retention_days=30,
            ),
            "results": StorageLocation(
                name="results",
                path=str(base_path / "results"),
                type="local",
                retention_days=90,
            ),
            "logs": StorageLocation(
                name="logs",
                path=str(base_path / "logs"),
                type="local",
                retention_days=30,
            ),
        }

        self.locations.update(default_locations)
        self._create_directories()

    def load_config(self, config_path: str) -> None:
        """Load storage configuration from file"""
        try:
            with open(config_path, "r") as f:
                if config_path.endswith(".yaml"):
                    config = yaml.safe_load(f)
                else:
                    config = json.load(f)

            self.locations = {
                name: StorageLocation(**location_config)
                for name, location_config in config["locations"].items()
            }
            self._create_directories()

        except Exception as e:
            raise StorageConfigError(f"Failed to load storage config: {str(e)}")

    def save_config(self, config_path: Optional[str] = None) -> None:
        """Save current storage configuration"""
        save_path = config_path or self.config_path
        if not save_path:
            raise ValueError("No config path specified")

        try:
            config = {
                "locations": {
                    name: {
                        "name": loc.name,
                        "path": loc.path,
                        "type": loc.type,
                        "credentials": loc.credentials,
                        "retention_days": loc.retention_days,
                    }
                    for name, loc in self.locations.items()
                }
            }

            with open(save_path, "w") as f:
                if save_path.endswith(".yaml"):
                    yaml.dump(config, f, default_flow_style=False)
                else:
                    json.dump(config, f, indent=2)

        except Exception as e:
            raise StorageConfigError(f"Failed to save storage config: {str(e)}")

    def get_location(self, location_type: str) -> StorageLocation:
        """Get storage location by type"""
        if location_type not in self.locations:
            raise ValueError(f"Unknown storage location type: {location_type}")
        return self.locations[location_type]

    def add_location(self, location_type: str, location: StorageLocation) -> None:
        """Add new storage location"""
        self.locations[location_type] = location
        self._create_directory(location.path)

    def remove_location(self, location_type: str) -> None:
        """Remove storage location"""
        if location_type in self.locations:
            del self.locations[location_type]

    def _create_directories(self) -> None:
        """Create all storage directories"""
        for location in self.locations.values():
            if location.type == "local":
                self._create_directory(location.path)

    def _create_directory(self, path: str) -> None:
        """Create a single directory"""
        if path:
            Path(path).mkdir(parents=True, exist_ok=True)

    def cleanup_old_files(self, location_type: Optional[str] = None) -> Dict[str, int]:
        """Clean up files beyond retention period"""
        from datetime import datetime, timedelta

        cleanup_results = {}
        locations_to_clean = (
            [self.locations[location_type]]
            if location_type
            else self.locations.values()
        )

        for location in locations_to_clean:
            if not location.retention_days:
                continue

            if location.type != "local":
                continue  # Skip non-local storage for now

            path = Path(location.path)
            if not path.exists():
                continue

            cutoff_date = datetime.now() - timedelta(days=location.retention_days)
            deleted_count = 0

            for file_path in path.rglob("*"):
                if file_path.is_file():
                    mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if mtime < cutoff_date:
                        file_path.unlink()
                        deleted_count += 1

            cleanup_results[location.name] = deleted_count

        return cleanup_results

    def get_storage_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get storage statistics for all locations"""
        import shutil

        stats = {}
        for name, location in self.locations.items():
            if location.type != "local":
                continue

            path = Path(location.path)
            if not path.exists():
                stats[name] = {"exists": False}
                continue

            total, used, free = shutil.disk_usage(path)
            file_count = sum(1 for _ in path.rglob("*") if _.is_file())

            stats[name] = {
                "exists": True,
                "total_space": total,
                "used_space": used,
                "free_space": free,
                "file_count": file_count,
                "last_modified": datetime.fromtimestamp(
                    path.stat().st_mtime
                ).isoformat(),
            }

        return stats


# Custom exceptions
class StorageConfigError(Exception):
    pass


class StorageLocationError(Exception):
    pass
