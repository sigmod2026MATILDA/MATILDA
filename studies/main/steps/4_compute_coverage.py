import os
import logging
from pathlib import Path
import colorama
from colorama import Fore, Style
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.rule_processors.RuleCoverageCalculator import RuleCoverageCalculator
# Initialize colorama
colorama.init(autoreset=True)

class ColoredFormatter(logging.Formatter):
    def format(self, record):
        if record.levelno == logging.INFO:
            record.msg = f"{Fore.GREEN}{record.msg}{Style.RESET_ALL}"
        elif record.levelno == logging.ERROR:
            record.msg = f"{Fore.RED}{record.msg}{Style.RESET_ALL}"
        elif record.levelno == logging.WARNING:
            record.msg = f"{Fore.YELLOW}{record.msg}{Style.RESET_ALL}"
        return super().format(record)

class CoverageExecutor:
    def __init__(self, rules_dir: str, coverage_output_dir: str, report_dir: str):
        self.rules_dir = Path(rules_dir)
        self.coverage_output_dir = Path(coverage_output_dir)
        self.report_dir = Path(report_dir)
        self.successes = []
        self.failures = []

        self._prepare_directories()
        self._configure_logging()

    def _configure_logging(self):
        """Configure the logging with colored formatter."""
        self.logger = logging.getLogger("CoverageExecutor")
        self.logger.handlers = []
        handler = logging.StreamHandler()
        handler.setFormatter(ColoredFormatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def _prepare_directories(self):
        """Create the coverage output and report directories if they don't exist."""
        if not self.coverage_output_dir.exists():
            os.makedirs(self.coverage_output_dir)
            logging.info(f"Created coverage output directory: {self.coverage_output_dir}")
        else:
            logging.info(f"Coverage output directory already exists: {self.coverage_output_dir}")

        if not self.report_dir.exists():
            os.makedirs(self.report_dir)
            logging.info(f"Created report output directory: {self.report_dir}")
        else:
            logging.info(f"Report output directory already exists: {self.report_dir}")

    def execute_coverage_calculation(self):
        """Execute the coverage calculation using RuleCoverageCalculator."""
        try:
            self.logger.info(f"Starting rule coverage calculation from {self.rules_dir}")
            calculator = RuleCoverageCalculator(
                rules_dir=self.rules_dir,
                coverage_output_dir=self.coverage_output_dir,
                report_dir=self.report_dir
            )
            calculator.main()
            self.successes.extend(calculator.successes)
            self.failures.extend(calculator.failures)
        except Exception as e:
            self.failures.append(f"CoverageExecutor failed: {e}")
            self.logger.error(f"CoverageExecutor failed: {e}")

    def generate_report(self):
        """Generate a Markdown report summarizing the coverage calculation."""
        report_path = self.report_dir / "4_step_compute_coverage_report.md"
        try:
            with open(report_path, "w") as report:
                report.write("# Rapport de Calcul de la Couverture des Règles\n\n")
                
                report.write("## Succès\n")
                for s in self.successes:
                    report.write(f"- {s}\n")
                
                report.write("\n## Échecs\n")
                for f in self.failures:
                    report.write(f"- {f}\n")
            
            self.logger.info(f"Report generated at {report_path}")
            self.successes.append("Coverage report generated successfully.")
        except Exception as e:
            self.failures.append(f"Report generation failed: {e}")
            self.logger.error(f"Report generation failed: {e}")

    def process_all(self):
        """Execute all processing steps."""
        self.execute_coverage_calculation()
        self.generate_report()

if __name__ == "__main__":
    # Directories configuration
    RULES_DIR = "data/results/"
    COVERAGE_OUTPUT_DIR = "data/coverage/"
    REPORT_OUTPUT_DIR = "data/reports/"

    # Instantiate and run the CoverageExecutor
    executor = CoverageExecutor(
        rules_dir=RULES_DIR,
        coverage_output_dir=COVERAGE_OUTPUT_DIR,
        report_dir=REPORT_OUTPUT_DIR
    )
    executor.process_all()