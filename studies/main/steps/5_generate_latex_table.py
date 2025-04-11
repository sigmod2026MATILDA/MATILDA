# call the rulecompatibility checker
import sys
import os
import logging
from pathlib import Path
import colorama
from colorama import Fore, Style
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.generate_reports.LaTeXTableGenerator import LaTeXTableGenerator

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

class LaTeXExecutor:
    def __init__(self, coverage_data_dir: str, latex_output_dir: str, report_dir: str):
        self.coverage_data_dir = Path(coverage_data_dir)
        self.latex_output_dir = Path(latex_output_dir)
        self.report_dir = Path(report_dir)
        self.successes = []
        self.failures = []

        self._prepare_directories()
        self._configure_logging()

    def _configure_logging(self):
        """Configure the logging with colored formatter."""
        self.logger = logging.getLogger("LaTeXExecutor")
        self.logger.handlers = []
        handler = logging.StreamHandler()
        handler.setFormatter(ColoredFormatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def _prepare_directories(self):
        """Create the LaTeX output and report directories if they don't exist."""
        if not self.latex_output_dir.exists():
            os.makedirs(self.latex_output_dir)
            logging.info(f"Created LaTeX output directory: {self.latex_output_dir}")
        else:
            logging.info(f"LaTeX output directory already exists: {self.latex_output_dir}")

        if not self.report_dir.exists():
            os.makedirs(self.report_dir)
            logging.info(f"Created report output directory: {self.report_dir}")
        else:
            logging.info(f"Report output directory already exists: {self.report_dir}")

    def execute_table_generation(self):
        """Execute the table generation using LaTeXTableGenerator."""
        try:
            self.logger.info(f"Starting LaTeX table generation from {self.coverage_data_dir}")
            generator = LaTeXTableGenerator(
                coverage_data_dir=self.coverage_data_dir,
                latex_output_dir=self.latex_output_dir,
                report_dir=self.report_dir
            )
            generator.main()
            # Accumulate successes and failures
            self.successes.extend(generator.successes)
            self.failures.extend(generator.failures)
        except Exception as e:
            self.failures.append(f"LaTeXExecutor failed: {e}")
            self.logger.error(f"LaTeXExecutor failed: {e}")

    def generate_report(self):
        """Generate a Markdown report summarizing the table generation."""
        report_path = self.report_dir / "5_step_generate_latex_table_report.md"
        try:
            with open(report_path, "w") as report:
                report.write("# Rapport de Génération des Tables LaTeX\n\n")
                report.write("## Succès\n")
                for s in self.successes:
                    report.write(f"- {s}\n")
                report.write("\n## Échecs\n")
                for f in self.failures:
                    report.write(f"- {f}\n")
            
            self.logger.info(f"Report generated at {report_path}")
            self.successes.append("LaTeX table report generated successfully.")
        except Exception as e:
            self.failures.append(f"Report generation failed: {e}")
            self.logger.error(f"Report generation failed: {e}")

    def process_all(self):
        """Execute all processing steps."""
        self.execute_table_generation()
        self.generate_report()

if __name__ == "__main__":
    # Directories configuration
    COVERAGE_DATA_DIR = "data/coverage/"        # Directory containing coverage YAML files
    LATEX_OUTPUT_DIR = "data/latex_tables/"    # Directory to save LaTeX table files
    REPORT_OUTPUT_DIR = "data/reports/"        # Directory to save reports

    # Instantiate and run the LaTeXExecutor
    executor = LaTeXExecutor(
        coverage_data_dir=COVERAGE_DATA_DIR,
        latex_output_dir=LATEX_OUTPUT_DIR,
        report_dir=REPORT_OUTPUT_DIR
    )
    executor.process_all()
