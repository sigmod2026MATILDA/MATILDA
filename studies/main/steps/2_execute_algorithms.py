import os
import yaml
import logging
import subprocess
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import colorama
from colorama import Fore, Style

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

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

# Configure logging
handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter('%(asctime)s - %(levelname)s - %(message)s'))
logger = logging.getLogger(__name__)
logger.handlers = []
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Directories
CONFIG_DIR = "data/configs/"  # Directory containing configuration files
RESULTS_DIR = "data/results/"  # Directory to save results
LOGS_DIR = "data/logs/"  # Directory to save logs
MAIN_SCRIPT = "../../src/main.py"  # Path to the main script
TEMP_DIRS = ["temp", "tmp"]  # Temporary directories to clean
REPORT_OUTPUT_DIR="data/reports/"

# Ensure results and logs directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Ensure logs subdirectories are properly created for each config
os.makedirs(Path(LOGS_DIR).parent, exist_ok=True)

def clean_temp_dirs():
    """Clean up temporary directories after execution."""
    for temp_dir in TEMP_DIRS:
        temp_path = Path(temp_dir)
        if temp_path.exists():
            try:
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up {temp_dir} directory")
            except Exception as e:
                logger.warning(f"Failed to clean {temp_dir}: {e}")

class AlgorithmExecutor:
    def __init__(self, database_limit: int = None):
        self.successes = []
        self.failures = []
        self.database_limit = database_limit

    def run_config(self, config_file: Path):
        """Run the main script with the given configuration file."""
        try:
            # Optionnel: Vérifier que le fichier de configuration existe et est valide
            if not config_file.exists():
                logger.error(f"Fichier de configuration manquant: {config_file}")
                self.failures.append(config_file.name)
                return
            
            # Charger la configuration pour vérifier la présence du fichier de base de données
            with open(config_file, "r") as file:
                config = yaml.safe_load(file)
            
            db_path = Path(config["database"]["path"]) / config["database"]["name"]
            if not db_path.exists():
                logger.error(f"Fichier de base de données manquant: {db_path}")
                self.failures.append(config_file.name)
                return

            # Define command to execute the script
            command = ["python", MAIN_SCRIPT, "-c", str(config_file)]
            log_file = LOGS_DIR + f"{config_file.stem}.log"
            with open(log_file, "w") as logfile:
                logger.info(f"Executing config: {config_file}")
                subprocess.run(command, stdout=logfile, stderr=subprocess.STDOUT, check=True, timeout=300)  # Timeout de 5 minutes
                logger.info(f"Execution completed for {config_file}. Results logged to {log_file}")
                
            self.successes.append(config_file.name)
            # Clean temp directories after successful execution
            clean_temp_dirs()
            
        except subprocess.TimeoutExpired:
            self.failures.append(config_file.name)
            logger.error(f"Execution timed out for {config_file}. Check log: {log_file}")
        except subprocess.CalledProcessError as e:
            self.failures.append(config_file.name)
            logger.error(f"Execution failed for {config_file}. Check log: {log_file}")
        except Exception as e:
            self.failures.append(config_file.name)
            logger.error(f"Unexpected error for {config_file}: {e}")

    def execute_all_configs(self):
        """Find all configuration files and run them sequentially."""
        config_files = list(Path(CONFIG_DIR).glob("*.yaml"))
        if not config_files:
            logger.error("No configuration files found in the configs directory.")
            return

        logger.info(f"Found {len(config_files)} configuration files. Starting execution...")

        # Limiter la liste si database_limit est spécifié
        if self.database_limit is not None:
            config_files = config_files[:self.database_limit]

        # Use ThreadPoolExecutor for parallel execution if needed
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(tqdm(executor.map(self.run_config, config_files), total=len(config_files), desc="Executing configs",
                      unit="config"))
        self.generate_report()

    def generate_report(self):
        report_path = Path(REPORT_OUTPUT_DIR) / "2_step_main_report.md"
        with open(report_path, "w") as report:
            report.write("# Rapport des Exécutions\n\n")
            report.write("## Succès\n")
            for s in self.successes:
                report.write(f"- {s}\n")
            report.write("\n## Échecs\n")
            for f in self.failures:
                report.write(f"- {f}\n")

if __name__ == "__main__":
    executor = AlgorithmExecutor(database_limit=None)
    executor.execute_all_configs()
