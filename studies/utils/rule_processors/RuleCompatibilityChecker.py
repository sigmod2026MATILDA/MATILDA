import logging
import yaml
from pathlib import Path
import colorama
from colorama import Fore, Style
from studies.utils.rule_processors.RuleComparer import RuleComparer

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

class RuleCoverageCalculator:
    def __init__(self, rules_dir: Path, coverage_output_dir: Path, report_dir: Path):
        self.rules_dir = rules_dir
        self.coverage_output_dir = coverage_output_dir
        self.report_dir = report_dir
        self.logger = logging.getLogger("RuleCoverageCalculator")
        self.logger.handlers = []
        handler = logging.StreamHandler()
        handler.setFormatter(ColoredFormatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        self.successes = []
        self.failures = []

    def compare_rule_sets(self, filepath1: str, filepath2: str):
        """Compare two sets of rules using RuleComparer."""
        comparer = RuleComparer()
        result = comparer.compare_rule_sets(filepath1, filepath2)
        self.logger.info(f"Comparison result: {result}")
        return result

    def calculate_coverage(self):
        """Calculate the coverage of the rules."""
        reference_rules_path = str(self.rules_dir / "reference_rules.json")
        database_name = self.rules_dir.name

        amie_results = self.compare_rule_sets(
            str(self.rules_dir / f"results/amie_{database_name}_results.json"),
            reference_rules_path
        )
        spider_results = self.compare_rule_sets(
            str(self.rules_dir / f"results/spider_{database_name}_results.json"),
            reference_rules_path
        )
        ilp_results = self.compare_rule_sets(
            str(self.rules_dir / f"results/ilp_{database_name}_results.json"),
            reference_rules_path
        )

        # Log the results
        self.logger.info(f"AMIE Coverage: {amie_results}")
        self.logger.info(f"Spider Coverage: {spider_results}")
        self.logger.info(f"ILP Coverage: {ilp_results}")

    def generate_report(self):
        """Generate a Markdown report summarizing the coverage calculations."""
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

    def main(self):
        """Main method to execute the coverage calculation."""
        self.calculate_coverage()
        self.generate_report()