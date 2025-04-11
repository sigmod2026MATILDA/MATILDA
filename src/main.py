import argparse
import threading
import shutil
import datetime
import signal
import sys
import logging  # Added missing import
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from algorithms.base_algorithm import BaseAlgorithm
from algorithms.amie3 import Amie3
from algorithms.ilp import ILP
from algorithms.spider import Spider
from algorithms.matilda import MATILDA

from database.alchemy_utility import AlchemyUtility
from utils.logging_utils import configure_global_logger
from utils.monitor import ResourceMonitor
from utils.config_loader import load_config
from utils.rules import RuleIO


@contextmanager
def mlflow_run_context(use_mlflow: bool, config: dict):
    """
    Context manager to handle MLflow runs.
    """
    if use_mlflow:
        mlflow_tracking_uri = config.get("mlflow", {}).get("tracking_uri", "http://localhost:5000")
        mlflow_experiment = config.get("mlflow", {}).get("experiment_name", "Rule Discovery")
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(mlflow_experiment)
        mlflow.start_run()
        try:
            yield
        finally:
            mlflow.end_run()
    else:
        yield


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run rule discovery on a specified database with a given algorithm."
    )
    parser.add_argument(
        "-c",
        "--config",
        default="config.yaml",
        help="Path to the configuration file (default: config.yaml)",
    )
    return parser.parse_args()


def initialize_directories(results_dir: Path, log_dir: Path) -> None:
    """
    Ensures that results and logs directories exist.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)


class DatabaseProcessor:
    """Handles database rule discovery and result logging."""

    def __init__(
        self,
        algorithm_name: str,
        database_name: Path,
        database_path: Path,
        results_dir: Path,
        logger: logging.Logger,
        use_mlflow: bool = False,
    ):
        self.algorithm_name = algorithm_name
        self.database_name = database_name
        self.database_path = database_path
        self.results_dir = results_dir
        self.logger = logger
        self.use_mlflow = use_mlflow

    def discover_rules(self) -> int:
        """Runs the rule discovery algorithm synchronously."""
        algorithm_map = {
            "POPPER": ILP,
            "ILP": ILP,
            "AMIE3": Amie3,
            "SPIDER": Spider,
            "MATILDA": MATILDA,
        }
        selected_algorithm = algorithm_map.get(self.algorithm_name.upper(), Amie3)

        db_file_path = self.database_path / self.database_name
        db_uri = f"sqlite:///{db_file_path}"
        try:
            self.logger.info(f"Using database URI: {db_uri}")
            with AlchemyUtility(db_uri, database_path=str(self.database_path), create_index=False) as db_util:
                algo: BaseAlgorithm = selected_algorithm(db_util)
                rules = []

                self.logger.debug("Starting rule discovery...")
                for rule in algo.discover_rules(results_dir=str(self.results_dir)):
                    self.logger.info(f"Discovered rule: {rule}")
                    rules.append(rule)

                json_file_name = f"{self.algorithm_name}_{self.database_name.stem}_results.json"
                result_path = self.results_dir / json_file_name

                self.logger.debug(f"Saving rules to {result_path}")
                number_of_rules = RuleIO.save_rules_to_json(rules, result_path)

                self.logger.info(f"Discovered {number_of_rules} rules.")
                if self.algorithm_name.upper() == "SPIDER":
                    self.generate_report(number_of_rules, result_path,[])
                else:
                    top_rules = sorted(rules,key=lambda x:-x.accuracy)[:5]
                    self.generate_report(number_of_rules, result_path,top_rules)

                if self.use_mlflow:
                    mlflow.log_param("algorithm", self.algorithm_name)
                    mlflow.log_param("database", self.database_name.name)
                    mlflow.log_metric("number_of_rules", number_of_rules)

                return number_of_rules

        except Exception as e:
            self.logger.error(f"An error occurred during rule discovery: {e}", exc_info=True)
            if self.use_mlflow:
                mlflow.log_param("error", str(e))
            raise

    def clean_up(self, temp_dirs: Optional[List[Path]] = None) -> None:
        """Cleans up temporary directories synchronously."""
        temp_dirs = temp_dirs or [
            self.database_path / "prolog_tmp",
            self.database_path / "SPIDER_temp",
            self.database_path / "popper",
        ]
        for directory in temp_dirs:
            if directory.exists() and directory.is_dir():
                shutil.rmtree(directory)
                self.logger.info(f"Cleaned up temporary directory: {directory}")

    def generate_report(self, number_of_rules: int, result_path: Path, top_rules: List[dict]) -> None:
        """Generates a report of the run."""
        report_content = f"""
# Run Report

**Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Algorithm:** {self.algorithm_name}
**Database:** {self.database_name.name}
**Number of Rules Discovered:** {number_of_rules}
**Results Path:** {result_path}

## Summary
- **Algorithm:** {self.algorithm_name}
- **Database:** {self.database_name.name}
- **Number of Rules Discovered:** {number_of_rules}
- **Results Path:** {result_path}

## Top 5 Best Rules
Below are the top-5 best rules discovered based on their scores:

| Rank | Rule Description | Support  | Confidence |
|------|------------------|----------| -----------|
"""

        # Add top-5 rules to the report
        for idx, rule in enumerate(top_rules, start=1):
            rule_desc = rule.display.replace('\n', ' ').replace('|', '\\|')  # Escape pipes for markdown tables
            report_content += f"| {idx} | {rule_desc} | {rule.accuracy:.3f} | {rule.confidence:.3f} |\n"

        report_content += f"""

## Details
The rule discovery process was completed successfully. The discovered rules have been saved to the specified results path.

    """

        report_file_name = f"report_{self.algorithm_name}_{self.database_name.stem}.md"
        report_path = self.results_dir / report_file_name

        with report_path.open('w') as report_file:
            report_file.write(report_content)

        self.logger.info(f"Generated report: {report_path}")

        if self.use_mlflow:
            mlflow.log_artifact(str(report_path))
            self.logger.info("Logged report as MLflow artifact.")


def setup_signal_handlers(monitor: ResourceMonitor, logger: logging.Logger) -> None:
    """
    Sets up signal handlers for graceful shutdown.
    """
    def handle_signal(signum, frame):
        logger.info(f"Received signal {signum}. Shutting down gracefully...")
        monitor.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)


def main() -> None:
    """Main entry point of the script."""
    args = parse_arguments()
    config = load_config(args.config)

    # Extract configuration with defaults
    threshold = config.get("monitor", {}).get("memory_threshold", 15 * 1024 * 1024 * 1024)  # 15GB
    timeout = config.get("monitor", {}).get("timeout", 3600)  # 1 hour
    database_path = Path(config.get("database", {}).get("path", "../data/db/"))
    database_name = Path(config.get("database", {}).get("name", "Bupa.db"))
    log_dir = Path(config.get("logging", {}).get("log_dir", "data/logs/"))
    results_dir = Path(config.get("results", {}).get("output_dir", "data/results/"))
    algorithm_name = config.get("algorithm", {}).get("name", "MATILDA")

    # Initialize directories
    initialize_directories(results_dir, log_dir)

    # Configure logger
    logger = configure_global_logger(log_dir)

    # Determine MLflow usage
    use_mlflow = False
    if MLFLOW_AVAILABLE:
        use_mlflow = config.get("mlflow", {}).get("use", False)
        if use_mlflow:
            logger.info("MLflow is enabled.")
    else:
        if config.get("mlflow", {}).get("use", False):
            logger.warning("MLflow is not available. Proceeding without MLflow.")
            use_mlflow = False

    # Initialize Resource Monitor
    monitor = ResourceMonitor(threshold, timeout)  # Changed to positional arguments
    monitor_thread = threading.Thread(target=monitor.monitor, daemon=True)
    monitor_thread.start()
    logger.debug("Resource monitor started.")

    # Setup signal handlers for graceful shutdown
    setup_signal_handlers(monitor, logger)

    # Initialize DatabaseProcessor
    processor = DatabaseProcessor(
        algorithm_name=algorithm_name,
        database_name=database_name,
        database_path=database_path,
        results_dir=results_dir,
        logger=logger,
        use_mlflow=use_mlflow,
    )

    logger.info("Starting rule discovery process.")

    try:
        with mlflow_run_context(use_mlflow, config):
            if use_mlflow:
                logger.info("MLflow run started.")

            number_of_rules = processor.discover_rules()

            processor.clean_up()

            logger.info("Process completed successfully.")

    except Exception as e:
        logger.error("An error occurred during the rule discovery process.", exc_info=True)
        sys.exit(1)
    finally:
        if use_mlflow and mlflow.active_run():
            mlflow.end_run()
            logger.info("MLflow run ended.")


if __name__ == "__main__":
    main()
