import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Add the root directory of the project to the PYTHONPATH
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

import logging
import colorama
from colorama import Fore, Style
from pathlib import Path
import yaml
from utils.rule_processors.LogProcessor import LogProcessor
from utils.rule_processors.SpiderRuleProcessor import SpiderRuleProcessor
from utils.rule_processors.AMIERuleProcessor import AMIERuleProcessor
from utils.rule_processors.ILPRuleProcessor import ILPRuleProcessor

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

class ResultsGatherer:
    def __init__(self, log_dir: str, stats_dir: str, results_dir: str, database_path: str, threshold_min: int, report_dir: str):
        self.log_dir = Path(log_dir)
        self.stats_dir = Path(stats_dir)
        self.results_dir = Path(results_dir)
        self.database_path = Path(database_path)
        self.threshold_min = threshold_min
        self.report_dir = Path(report_dir)
        self.successes = []
        self.failures = []

        self._prepare_directories()
        self._configure_logging()

    def _configure_logging(self):
        """Configure the logging with colored formatter."""
        self.logger = logging.getLogger("ResultsGatherer")
        self.logger.handlers = []
        handler = logging.StreamHandler()
        handler.setFormatter(ColoredFormatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def _prepare_directories(self):
        """Create the stats and report directories if they don't exist."""
        if not self.stats_dir.exists():
            os.makedirs(self.stats_dir)
            logging.info(f"Created stats output directory: {self.stats_dir}")
        else:
            logging.info(f"Stats output directory already exists: {self.stats_dir}")

        if not self.report_dir.exists():
            os.makedirs(self.report_dir)
            logging.info(f"Created report output directory: {self.report_dir}")
        else:
            logging.info(f"Report output directory already exists: {self.report_dir}")

    def process_logs(self):
        """Process the log files to gather statistics."""
        try:
            self.logger.info(f"Processing logs from {self.log_dir} to {self.stats_dir}")
            LogProcessor(self.log_dir, self.stats_dir).process_logs()
            self.successes.append("Log processing completed successfully.")
            self.logger.info("Log processing completed successfully.")
        except Exception as e:
            self.failures.append(f"Log processing failed: {e}")
            self.logger.error(f"Log processing failed: {e}")

    def process_spider_rules(self):
        """Process Spider rules from results."""
        try:
            self.logger.info(f"Processing Spider rules from {self.results_dir} with database at {self.database_path}")
            SpiderRuleProcessor(self.results_dir, self.database_path, self.threshold_min).main()
            self.successes.append("Spider rule processing completed successfully.")
            self.logger.info("Spider rule processing completed successfully.")
        except Exception as e:
            self.failures.append(f"Spider rule processing failed: {e}")
            self.logger.error(f"Spider rule processing failed: {e}")

    def process_amie_rules(self):
        """Process AMIE rules from results."""
        try:
            self.logger.info(f"Processing AMIE rules from {self.results_dir} ")
            AMIERuleProcessor(self.results_dir).main()
            self.successes.append("AMIE rule processing completed successfully.")
            self.logger.info("AMIE rule processing completed successfully.")
        except Exception as e:
            self.failures.append(f"AMIERuleProcessor failed: {e}")
            self.logger.error(f"AMIERuleProcessor failed: {e}")

    def process_ilp_rules(self):
        """Process ILP rules from results."""
        try:
            self.logger.info(f"Processing ILP rules from {self.results_dir} with database at {self.database_path}")
            ILPRuleProcessor(self.results_dir, self.database_path, self.threshold_min).main()
            self.successes.append("ILP rule processing completed successfully.")
            self.logger.info("ILP rule processing completed successfully.")
        except Exception as e:
            self.failures.append(f"ILPRuleProcessor failed: {e}")
            self.logger.error(f"ILPRuleProcessor failed: {e}")

    def generate_report(self):
        """Generate a Markdown report summarizing the processing."""
        report_path = self.report_dir / "3_step_gather_results_report.md"
        try:
            with open(report_path, "w") as report:
                report.write("# Rapport de Récupération des Résultats\n\n")
                
                report.write("## Succès\n")
                for s in self.successes:
                    report.write(f"- {s}\n")
                
                report.write("\n## Échecs\n")
                for f in self.failures:
                    report.write(f"- {f}\n")
            
            self.logger.info(f"Report generated at {report_path}")
            self.successes.append("Report generated successfully.")
        except Exception as e:
            self.failures.append(f"Report generation failed: {e}")
            self.logger.error(f"Report generation failed: {e}")

    def process_all(self):
        """Execute all processing steps."""
        self.process_logs()
        self.process_spider_rules() # compute validity 
        # computation of compatibility 
        # amie 
        # ilp 
        # spider 

        self.process_amie_rules()  # Ajout du traitement AMIE
        #self.process_ilp_rules()    # Ajout du traitement ILP
        self.generate_report()

if __name__ == "__main__":
    # Directories configuration
    LOG_DIR = "data/logs/"
    STATS_DIR = "data/stats/"
    RESULTS_DIR = "data/results/"
    DATABASE_PATH = "../../data/db/"
    THRESHOLD_MIN = 1
    REPORT_OUTPUT_DIR = "data/reports/"

    # Instantiate and run the ResultsGatherer
    gatherer = ResultsGatherer(
        log_dir=LOG_DIR,
        stats_dir=STATS_DIR,
        results_dir=RESULTS_DIR,
        database_path=DATABASE_PATH,
        threshold_min=THRESHOLD_MIN,
        report_dir=REPORT_OUTPUT_DIR
    )
    gatherer.process_all()