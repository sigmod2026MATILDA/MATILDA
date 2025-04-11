import logging
import os
import re
import sys

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)

# Add the path to custom modules
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src'))
)

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

# Import custom modules
from src.utils.rules import (
    Predicate,
    TGDRule,
RuleIO,
    InclusionDependency,
)
#from utils.alchemy_utility import AlchemyUtility

class CoverageStats:
    """
    Class to encapsulate coverage statistics for rule comparison.

    Attributes:
        match (int): Number of matching rules.
        error (int): Number of rules with errors (for spider).
        total (int): Total number of rules.
        total_non_error (int): Number of rules without errors.
        correct (int): Number of correct rules.
        compatible (int): Number of compatible rules.
        coverage_percentage (float): Percentage of matched rules among compatible rules.
    """
    def __init__(self, match_count=0, error_count=0, total=0, compatibility_count=0):
        self.match = match_count
        self.error = error_count
        self.total = total
        self.total_non_error = total - error_count
        self.correct = total - error_count
        self.compatible = compatibility_count
        self.coverage_percentage = (
            (match_count / compatibility_count) if compatibility_count > 0 else 1
        )

    def to_dict(self):
        """
        Convert the coverage statistics to a dictionary.

        Returns:
            dict: Dictionary representation of the coverage statistics.
        """
        return {
            "match": self.match,
            "error": self.error,
            "total": self.total,
            "total_non_error": self.total_non_error,
            "correct": self.correct,
            "compatible": self.compatible,
            "coverage_percentage": self.coverage_percentage,
        }

class ResultsComparer:
    """
    Class to compare rules across different methods and generate statistics.

    Attributes:
        source_dir (str): Directory containing result files.
        target_dir (str): Directory to save comparison results.
        output_data (dict): Dictionary to store the comparison results.
    """
    def __init__(self, source_dir, target_dir):
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.output_data = {}
        os.makedirs(self.target_dir, exist_ok=True)

    def compare(self):
        """
        Compare rules from the source directory and generate statistics.

        Iterates through databases in the source directory and processes the rules
        for each database to generate comparison statistics.
        """
        os.makedirs(self.target_dir, exist_ok=True)  # Ensure the target directory exists
        for database_name in os.listdir(self.source_dir):
            database_dir = os.path.join(self.source_dir, database_name)
            if not os.path.isdir(database_dir):
                continue

            logging.info(f"Processing database: {database_name}")
            rules_by_type = self._load_rules(database_name, database_dir)
            self._process_rules(database_name, rules_by_type)

        self._save_results()

    def _load_rules(self, database_name, database_dir):
        """
        Load rules from files in the specified database directory.

        Args:
            database_name (str): Name of the database.
            database_dir (str): Path to the database directory.

        Returns:
            dict: A dictionary mapping rule types to their corresponding rules.
        """
        rules_by_type = {}
        method_dirs = [d for d in os.listdir(database_dir) if os.path.isdir(os.path.join(database_dir, d))]

        for method in method_dirs:
            method_dir = os.path.join(database_dir, method)
            file_name = f"{method}_{database_name}_results.json"
            file_path = os.path.join(method_dir, file_name)

            if not os.path.exists(file_path):
                logging.warning(f"File {file_path} does not exist.")
                continue

            try:
                with open(file_path, 'r') as rule_file:
                    import json
                    rule_data = [RuleIO.rule_from_dict(el) for el in json.load(rule_file)]
                rules_by_type[method] = rule_data
            except Exception as e:
                logging.error(f"Error reading rule data from {file_path}: {e}")
        return rules_by_type
    def _process_rules(self, database_name, rules_by_type):
        """
        Process rules by comparing them across different methods.

        Args:
            database_name (str): Name of the database being processed.
            rules_by_type (dict): Dictionary containing rules categorized by type.
        """
        spiders = rules_by_type.get('spider')
        amie3 = rules_by_type.get('amie3')
        ilps = rules_by_type.get('popper')
        tgds_test = rules_by_type.get('matilda')

        if tgds_test is None:
            return

        for rule_type, rules in {
            "Spider": spiders,
            "Amie3": amie3,
            "ILP": ilps,
        }.items():
            if rules:
                res_data = self._compare_rules(rules, tgds_test)
                res_data_key = "___".join(["MATILDA", database_name, rule_type])
                self.output_data[res_data_key] = res_data

    def _compare_rules(self, source_rules, target_rules):
        """
        Compare rules from a source set to a target set and generate statistics.

        Args:
            source_rules (list): List of rules to be compared.
            target_rules (list): List of target rules to compare against.

        Returns:
            dict: Dictionary containing comparison statistics.
        """
        match_count = 0
        error_count = 0
        compatibility_count = 0

        for rule in source_rules:
            if rule.compatible:
                compatibility_count += 1

        for rule in source_rules:
            for target_rule in target_rules:
                if rule == target_rule:  # Placeholder for actual comparison logic
                    match_count += 1
                    break

        stats = CoverageStats(match_count, error_count, len(source_rules), compatibility_count)
        return stats.to_dict()

    def _save_results(self):
        """
        Save the comparison results to a file in the target directory.
        """
        output_filepath = os.path.join(self.target_dir, 'output.json')
        try:
            with open(output_filepath, 'w') as f:
                import json
                json.dump(self.output_data, f, indent=4)
                # for key, value in self.output_data.items():
                #     f.write(f"{key}: {value}\n")
            logging.info(f"Comparison results saved to {output_filepath}")
        except Exception as e:
            logging.error(f"Error writing output data to {output_filepath}: {e}")

if __name__ == "__main__":
    source_directory = "../main/data/results"  # Replace with actual source directory
    target_directory = "../main/data/coverage"  # Replace with actual target directory
    comparer = ResultsComparer(source_directory, target_directory)
    comparer.compare()
