import os
import yaml
import logging
from pathlib import Path
import colorama
from colorama import Fore, Style

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

# Input YAML file containing database and algorithm information
YAML_FILE = "data/databases_with_results.yaml"
REPORT_OUTPUT_DIR="data/reports/"
CONFIG_OUTPUT_DIR = "data/configs/"  # Directory to save generated configuration files
DATABASE_DIR ="/Users/famat/PycharmProjects/MATILDA_ALL/data/db" # TODO CHANGE THIS #f"data/databases/",

class ConfigGenerator:
    def __init__(self, yaml_file: str, output_dir: str):
        self.yaml_file = yaml_file
        self.output_dir = Path(output_dir)
        self.report_dir = Path(REPORT_OUTPUT_DIR)  # Ajout de l'attribut report_dir
        self._prepare_output_dir()
        self.successes = []  # Initialisation des succès
        self.failures = []   # Initialisation des échecs

    def _prepare_output_dir(self):
        """Create the output and report directories if they don't exist."""
        if not self.output_dir.exists():
            os.makedirs(self.output_dir)
            logger.info(f"Created configuration output directory: {self.output_dir}")
        
        # Création du répertoire de rapports
        if not self.report_dir.exists():
            os.makedirs(self.report_dir)
            logger.info(f"Created report output directory: {self.report_dir}")
        else:
            logger.info(f"Report output directory already exists: {self.report_dir}")

    def read_yaml(self) -> dict:
        """Read the input YAML file containing database information."""
        try:
            with open(self.yaml_file, "r") as file:
                data = yaml.safe_load(file)
                logger.info("YAML file loaded successfully.")
                return data
        except Exception as e:
            logger.error(f"Failed to read YAML file: {e}")
            return {}

    def validate_config(self, config: dict) -> bool:
        """Valide la configuration générée."""
        required_fields = ["monitor", "database", "logging", "results", "algorithm"]
        for field in required_fields:
            if field not in config:
                logger.error(f"Champ manquant dans la configuration: {field}")
                return False
        # Vérifier les sous-champs
        if "path" not in config["database"] or "name" not in config["database"]:
            logger.error("Champs manquants dans 'database' de la configuration.")
            return False
        if "log_dir" not in config["logging"]:
            logger.error("Champ manquant dans 'logging' de la configuration.")
            return False
        if "output_dir" not in config["results"]:
            logger.error("Champ manquant dans 'results' de la configuration.")
            return False
        if "name" not in config["algorithm"]:
            logger.error("Champ manquant dans 'algorithm' de la configuration.")
            return False
        logger.info("Configuration valide.")
        return True

    def generate_configs(self, data: dict):
        """Generate configuration files for each database and each algorithm."""
        algorithms = ["popper", "spider", "amie3", "matilda"]  # List of algorithms
        algorithms = ["matilda"]
        for db_name, db_details in data.items():
            for algo in algorithms:
                config = self._create_config(db_name, algo)
                if config and self.validate_config(config):
                    self._write_config_to_file(db_name, algo, config)
                    self.successes.append(f"{db_name}_{algo}_config.yaml")
                else:
                    self.failures.append(f"{db_name}_{algo}_config.yaml")
                    logger.error(f"Configuration invalide ou base de données manquante pour {db_name} avec l'algorithme {algo}.")
        self.generate_report()

    def generate_report(self):
        report_path = Path(REPORT_OUTPUT_DIR) / "1_step_main_report.md"
        with open(report_path, "w") as report:
            report.write("# Rapport de Génération des Configurations\n\n")
            report.write("## Succès\n")
            for s in self.successes:
                report.write(f"- {s}\n")
            report.write("\n## Échecs\n")
            for f in self.failures:
                report.write(f"- {f}\n")

    def _create_config(self, db_name: str, algorithm: str) -> dict:
        """Create a configuration dictionary for a specific database and algorithm."""
        db_path = Path(DATABASE_DIR) / f"{db_name}.db"
        if not db_path.exists():
            logger.error(f"Fichier de base de données manquant: {db_path}")
            return {}
        logger.info(f"Fichier de base de données trouvé: {db_path}")  # Nouvelle déclaration de journalisation
        
        config = {
            "monitor": {
                "memory_threshold": 16106127360,  # 15GB in bytes
                "timeout": 60  # 1 hour
            },
            "database": {
                "path": str(db_path.parent),
                "name": db_name + ".db"
            },
            "logging": {
                "log_dir": f"data/logs/{db_name}/{algorithm}"
            },
            "results": {
                "output_dir": f"data/results/{db_name}/{algorithm}"
            },
            "algorithm": {
                "name": algorithm
            }
        }
        logger.info(f"Configuration créée pour la base de données '{db_name}' avec l'algorithme '{algorithm}'.")
        return config

    def _write_config_to_file(self, db_name: str, algorithm: str, config: dict):
        """Write the configuration dictionary to a YAML file."""
        config_file_name = f"{db_name}_{algorithm}_config.yaml"
        config_file_path = self.output_dir / config_file_name
        try:
            with open(config_file_path, "w") as file:
                yaml.dump(config, file, default_flow_style=False, sort_keys=False)
                logger.info(f"Configuration file saved: {config_file_path}")
        except Exception as e:
            logger.error(f"Failed to write configuration file for {db_name} and {algorithm}: {e}")


if __name__ == "__main__":
    # Instantiate ConfigGenerator
    generator = ConfigGenerator(YAML_FILE, CONFIG_OUTPUT_DIR)

    # Read database information from YAML file
    database_data = generator.read_yaml()

    # Generate configuration files
    if database_data:
        generator.generate_configs(database_data)
    else:
        logger.error("No database data found in YAML file. Exiting.")
