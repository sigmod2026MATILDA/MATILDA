import json
import logging
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from multiprocessing import Pool, cpu_count
from src.utils.rules import RuleIO #rule_from_dict
from src.database.alchemy_utility import AlchemyUtility
import colorama
from colorama import Fore, Style

colorama.init(autoreset=True)

class ColorFormatter(logging.Formatter):
    COLOR_MAP = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.MAGENTA,
    }

    def format(self, record):
        color = self.COLOR_MAP.get(record.levelno, Fore.WHITE)
        record.msg = f"{color}{record.msg}{Style.RESET_ALL}"
        return super().format(record)

class SpiderRuleProcessor:
    """
    SpiderRuleProcessor is responsible for processing JSON files containing rule definitions,
    validating the rules using database information, and updating the rules with validation results.

    The class uses a database inspector utility to check rule compatibility and supports processing
    multiple files in parallel. It integrates with a specific database structure and rule schema.
    """
    def __init__(self, results_dir, database_path, threshold_min):
        self.results_dir = results_dir
        self.database_path = database_path
        self.threshold_min = threshold_min

        logging.basicConfig(
            level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self._setup_logging()

    def _setup_logging(self):
        formatter = ColorFormatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # Formatter for the main logger
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    def get_overlapping_elements(self, db_inspector, table1, table2, attribute1, attribute2):
        """
        Retrieve overlapping elements between two attributes from different tables using the database inspector.

        This method queries the database to find matching elements between the specified attribute columns of two tables,
        considering the defined compatibility threshold.

        Args:
            db_inspector: An instance of the database inspection utility used to execute queries.
            table1 (str): Name of the first table.
            table2 (str): Name of the second table.
            attribute1 (str): Column name in the first table to check for overlaps.
            attribute2 (str): Column name in the second table to check for overlaps.

        Returns:
            bool or int: The number of overlapping elements if successful, or False in case of an error.
        """
        try:
            logging.debug(
                f"Getting overlapping elements between {table1}.{attribute1} and {table2}.{attribute2}"
            )
            results = db_inspector.check_threshold(
                [
                    (
                        table1,
                        0,
                        attribute1,
                        table2,
                        1,
                        attribute2,
                    )
                ],
                disjoint_semantics=False,
                flag="compatibility",
                threshold=self.threshold_min
            )
            logging.debug(f"Threshold is {results} overlapping elements")
        except Exception as e:
            #logging.error(f"Error getting overlapping elements: {e} - exception type: {type(e)}")
            results = False
        return results

    def is_ind_valid(self, rule, db_inspector):
        """
        Validate a rule by checking the overlap of specific attributes in dependent and referenced tables.

        A rule is considered valid if the overlap of the specified attributes between the two tables
        satisfies the defined threshold conditions. The database inspector is used to perform this check.

        Args:
            rule: The rule object containing the dependent and referenced table and column information.
            db_inspector: An instance of the database inspection utility.

        Returns:
            bool: True if the rule is valid, False otherwise.
        """
        logging.debug(f"Validating rule: {rule} on database {db_inspector.base_name}")
        try:
            results = self.get_overlapping_elements(
                db_inspector,
                rule.table_dependant,
                rule.table_referenced,
                rule.columns_dependant[0],
                rule.columns_referenced[0],
            )
            logging.debug(f"Overlap results: {results}")
        except Exception as e:
            logging.error(f"Error validating rule: {e}")
            return False
        return results

    def should_skip_file(self, filepath):
        """
        Check if the file should be skipped by verifying if any 'correct' field is already populated.

        The JSON file is expected to be an array of rule objects, where each rule is a dictionary
        containing various attributes. This method checks if the 'correct' attribute is already
        set for any of the rules in the file. If so, the file is skipped to avoid redundant processing.

        Args:
            filepath (str): Path to the JSON file to check.

        Returns:
            bool: True if the file should be skipped, False otherwise.
        """
        try:
            with open(filepath, 'r') as file:
                data = json.load(file)
                for rule in data:
                    if rule.get("correct"):
                        logging.info(f"Skipping file {filepath} as it already contains 'correct' values.")
                        return True
        except Exception as e:
            logging.error(f"Error reading JSON file for pre-check {filepath}: {e}")
        return False

    def process_file(self, args):
        """
        Process a single file to validate and update rule definitions.

        Args:
            args (tuple): A tuple containing the database name and the file path to process. The tuple structure is:
                - database_name (str): The name of the database (including the ".db" extension).
                - filepath (str): The path to the JSON file containing rule definitions.
        """
        database_name, filepath = args
        if self.should_skip_file(filepath):
            return  # Skip processing if 'correct' is already populated

        logging.info(f"Processing file: {filepath}")

        try:
            with open(filepath, 'r') as file:
                data = json.load(file)
        except Exception as e:
            logging.error(f"Error reading JSON file {filepath}: {e}")
            return

        updated_data = []
        database_sqlite_path = f"sqlite:///{self.database_path}/{database_name}"
        logging.info(f"Connecting to database at {database_sqlite_path}")

        try:
            db_inspector = AlchemyUtility(database_sqlite_path,
                                          create_csv=False,
                                          create_index=False,
                                          create_tsv=False,
                                          get_data=False)
            for rule in data:
                try:
                    rule_obj = RuleIO.rule_from_dict(rule)
                    logging.debug(f"Rule object: {rule_obj}")
                except Exception as e:
                    logging.error(f"Error converting rule from dict: {e}")
                    continue

                if not self.is_ind_valid(rule_obj, db_inspector):
                    rule["correct"] = False
                    logging.debug("Rule is invalid; set correct to 'False'")
                else:
                    rule["correct"] = True
                    logging.debug("Rule is valid; set correct to 'True'")

                updated_data.append(rule)

            db_inspector.close()
        except Exception as e:
            logging.error(f"Error initializing or closing AlchemyUtility: {e}")
            return

        try:
            with open(filepath, 'w') as file:
                json.dump(updated_data, file, indent=4)
            logging.info(f"Updated data written to {filepath}")
        except Exception as e:
            logging.error(f"Error writing updated data to {filepath}: {e}")

    def main(self):
        files_to_process = []
        for database_name in os.listdir(self.results_dir):
            database_dir = os.path.join(self.results_dir, database_name, "spider")
            if not os.path.isdir(database_dir):
                continue

            filepath = os.path.join(database_dir, f"spider_{database_name}_results.json")
            if os.path.isfile(filepath):
                files_to_process.append((database_name + ".db", filepath))
        for files_to_process in files_to_process:
            self.process_file(files_to_process)
        #with Pool(cpu_count()) as pool:
        #    pool.map(self.process_file, files_to_process)


if __name__ == "__main__":
    results_dir = "../main/data/results/"
    database_path = "../../data/db/"
    threshold_min = 0.5  # Example threshold value

    processor = SpiderRuleProcessor(results_dir, database_path, threshold_min)
    processor.main()
