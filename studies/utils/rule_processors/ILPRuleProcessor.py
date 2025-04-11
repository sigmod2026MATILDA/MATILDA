import os
import sys
import logging
from pathlib import Path
import colorama
from colorama import Fore, Style
import json
from typing import Dict, List, Union

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/utils')))
from rules import Rule, RuleIO

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

class RuleCompatibilityChecker:
    def __init__(self, compatibility_dict: Dict[str, List[str]]):
        self.compatibility_dict = compatibility_dict

    def clean_up_relation(self, relation: str) -> str:
        parts = relation.split("___sep___")
        if len(parts) == 2:
            return parts[0].replace("_", "") + "___sep___" + parts[1]
        return relation

    def is_rule_compatible(self, data) -> bool:
        if type(data) != dict:
            body = data.body
            head = data.head
        else:
            body = data.get("body")
            head = data.get("head")

        indexes = {}
        for predicate in body:
            if predicate.variable2 not in indexes:
                indexes[predicate.variable2] = []
            indexes[predicate.variable2].append(self.clean_up_relation(predicate.relation))
        for predicate in head:
            if predicate.variable2 not in indexes:
                indexes[predicate.variable2] = []
            indexes[predicate.variable2].append(self.clean_up_relation(predicate.relation))

        new_dir = {}
        for key in self.compatibility_dict:
            new_dir[self.clean_up_relation(key)] = [self.clean_up_relation(relation) for relation in self.compatibility_dict[key]]
        self.compatibility_dict = new_dir

        for variable in indexes:
            for sub_variable in indexes[variable]:
                if sub_variable not in self.compatibility_dict:
                    return False
                else:
                    for other_sub_variable in indexes[variable]:
                        if other_sub_variable != sub_variable:
                            if other_sub_variable not in self.compatibility_dict[sub_variable]:
                                return False
        return True

    def check_rule_compatibility(self, filepath: str) -> List[Dict[str, Union[Dict, bool]]]:
        try:
            with open(filepath, 'r') as file:
                data = json.load(file)
        except Exception as e:
            logging.error(f"Error reading JSON file {filepath}: {e}")
            return []

        for rule_dict in data:
            try:
                rule = RuleIO.rule_from_dict(rule_dict)
                compatible = self.is_rule_compatible(rule)
                rule_dict["compatible"] = compatible
            except Exception as e:
                logging.error(f"Error processing rule: {e}. Rule data: {rule_dict}")
                rule_dict["compatible"] = False

        return data

    def is_transitive(self, start: str, target: str, compatibility_dict: Dict[str, List[str]]) -> bool:
        from collections import deque
        visited = set()
        queue = deque([start])

        while queue:
            current = queue.popleft()
            if current == target:
                return True
            if current not in visited:
                visited.add(current)
                queue.extend(compatibility_dict.get(current, []))
        return False

class ILPRuleProcessor:
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.logger = logging.getLogger("ILPRuleProcessor")
        self.logger.handlers = []
        handler = logging.StreamHandler()
        handler.setFormatter(ColoredFormatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        self.successes = []
        self.failures = []

    def process_rules(self):
        """Process the rules generated by ILP."""
        try:
            self.logger.info(f"Processing ILP rules from {self.results_dir} ")
            for database in self.results_dir.iterdir():
                if database.is_dir():
                    self.logger.info(f"Processing database {database}")
                    self.database_path = database
                    print(f"Processing database {database}")
                    self.compute_compatibility(database)

        except Exception as e:
            self.failures.append(f"ILPRuleProcessor failed: {e}")
            self.logger.error(f"ILPRuleProcessor failed: {e}")

    def compute_compatibility(self, database):
        try:
            self.logger.info("Calculating compatibility scores...")
            self.logger.info(f"Reading ILP results from {database}")
            database_name = database.stem
            ilp_results_filename = f"popper_{database_name}_results.json"
            results_json = os.path.join(database, "popper", ilp_results_filename)
            compatibility_file = os.path.join(database, "matilda", f"compatibility_{database_name}.json")

            with open(compatibility_file, "r") as cfile:
                compatibility_dict = json.load(cfile)

            rules = RuleIO.load_rules_from_json(results_json)
            checker = RuleCompatibilityChecker(compatibility_dict)
            
            compatibility_results = checker.check_rule_compatibility(results_json)
            # save the compatibility results as results in the database directory
            output_file = os.path.join(database, "matilda", f"compatibility_results_{database_name}.json")
            with open(output_file, 'w') as file:
                json.dump(compatibility_results, file, indent=4)

        except Exception as e:
            self.logger.error(f"Erreur lors du calcul de compatibilité: {e}")

    def main(self):
        """Main method to execute the processing."""
        self.process_rules()

if __name__ == "__main__":
    results_dir = Path("../main/data/results/")
    processor = ILPRuleProcessor(results_dir)
    processor.main()