import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import GPUtil
import pandas as pd
import psutil


class ResourceMonitor:
    """Monitor system resources (CPU, Memory, GPU) during model training"""

    def __init__(
        self,
        interval: int = 1,
        log_dir: Optional[str] = None,
        monitor_gpu: bool = False,
    ):
        """
        Initialize the resource monitor.

        Args:
            interval: Sampling interval in seconds
            log_dir: Directory to save monitoring logs
            monitor_gpu: Whether to monitor GPU resources
        """
        self.interval = interval
        self.log_dir = Path(log_dir) if log_dir else Path.cwd() / "logs"
        self.monitor_gpu = monitor_gpu
        self.monitoring = False
        self.metrics: List[Dict[str, Any]] = []
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup logging configuration"""
        self.log_dir.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(self.log_dir / "resource_monitor.log"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger("ResourceMonitor")

    def start(self) -> None:
        """Start resource monitoring in a separate thread"""
        if self.monitoring:
            self.logger.warning("Resource monitoring is already running")
            return

        self.monitoring = True
        self.metrics = []

        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

        self.logger.info("Resource monitoring started")

    def stop(self) -> None:
        """Stop resource monitoring"""
        if not self.monitoring:
            self.logger.warning("Resource monitoring is not running")
            return

        self.monitoring = False
        self.monitor_thread.join()
        self.logger.info("Resource monitoring stopped")

    def _monitor_resources(self) -> None:
        """Main monitoring loop"""
        while self.monitoring:
            try:
                metrics = self._collect_metrics()
                self.metrics.append(metrics)
                time.sleep(self.interval)
            except Exception as e:
                self.logger.error(f"Error collecting metrics: {str(e)}")

    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect current resource metrics"""
        metrics = {
            "timestamp": datetime.now(),
            "cpu_percent": psutil.cpu_percent(interval=None),
            "cpu_freq": psutil.cpu_freq().current if psutil.cpu_freq() else None,
            "memory_percent": psutil.virtual_memory().percent,
            "memory_used": psutil.virtual_memory().used,
            "memory_available": psutil.virtual_memory().available,
            "disk_usage_percent": psutil.disk_usage("/").percent,
            "disk_io_read": psutil.disk_io_counters().read_bytes,
            "disk_io_write": psutil.disk_io_counters().write_bytes,
        }

        # Add per-CPU metrics
        cpu_percentages = psutil.cpu_percent(percpu=True)
        for i, cpu_percent in enumerate(cpu_percentages):
            metrics[f"cpu_{i}_percent"] = cpu_percent

        # Add GPU metrics if requested and available
        if self.monitor_gpu:
            try:
                gpus = GPUtil.getGPUs()
                for i, gpu in enumerate(gpus):
                    metrics.update(
                        {
                            f"gpu_{i}_load": gpu.load,
                            f"gpu_{i}_memory_used": gpu.memoryUsed,
                            f"gpu_{i}_memory_total": gpu.memoryTotal,
                            f"gpu_{i}_temperature": gpu.temperature,
                        }
                    )
            except Exception as e:
                self.logger.warning(f"Failed to collect GPU metrics: {str(e)}")

        return metrics

    def get_metrics(self) -> pd.DataFrame:
        """Return collected metrics as a DataFrame"""
        return pd.DataFrame(self.metrics)

    def save_metrics(self, filename: Optional[str] = None) -> None:
        """Save metrics to CSV file"""
        if not self.metrics:
            self.logger.warning("No metrics to save")
            return

        try:
            df = self.get_metrics()
            save_path = self.log_dir / (
                filename or f"resource_metrics_{datetime.now():%Y%m%d_%H%M%S}.csv"
            )
            df.to_csv(save_path, index=False)
            self.logger.info(f"Metrics saved to {save_path}")
        except Exception as e:
            self.logger.error(f"Failed to save metrics: {str(e)}")

    def get_summary(self) -> Dict[str, Any]:
        """Generate summary statistics of collected metrics"""
        if not self.metrics:
            return {}

        df = self.get_metrics()
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns

        summary = {
            "start_time": df["timestamp"].min(),
            "end_time": df["timestamp"].max(),
            "duration_seconds": (
                df["timestamp"].max() - df["timestamp"].min()
            ).total_seconds(),
            "sample_count": len(df),
        }

        for col in numeric_cols:
            if col != "timestamp":
                summary.update(
                    {
                        f"{col}_mean": df[col].mean(),
                        f"{col}_max": df[col].max(),
                        f"{col}_min": df[col].min(),
                        f"{col}_std": df[col].std(),
                    }
                )

        return summary

    def plot_metrics(
        self, metrics: Optional[List[str]] = None, save_path: Optional[str] = None
    ) -> None:
        """Plot specified metrics over time"""
        try:
            import matplotlib.pyplot as plt

            df = self.get_metrics()
            if not metrics:
                metrics = [col for col in df.columns if col != "timestamp"]

            fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4 * len(metrics)))
            if len(metrics) == 1:
                axes = [axes]

            for ax, metric in zip(axes, metrics):
                ax.plot(df["timestamp"], df[metric])
                ax.set_title(metric)
                ax.grid(True)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path)
                self.logger.info(f"Plot saved to {save_path}")
            else:
                plt.show()

        except ImportError:
            self.logger.error("matplotlib is required for plotting")
        except Exception as e:
            self.logger.error(f"Error plotting metrics: {str(e)}")


class ResourceContext:
    """Context manager for resource monitoring"""

    def __init__(
        self,
        interval: int = 1,
        log_dir: Optional[str] = None,
        monitor_gpu: bool = False,
    ):
        self.monitor = ResourceMonitor(interval, log_dir, monitor_gpu)

    def __enter__(self) -> ResourceMonitor:
        self.monitor.start()
        return self.monitor

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.monitor.stop()
        if exc_type is not None:
            self.monitor.logger.error(f"Error during monitoring: {str(exc_val)}")
