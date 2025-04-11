
import logging
import yaml
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

class LaTeXTableGenerator:
    def __init__(self, coverage_data_dir: Path, latex_output_dir: Path, report_dir: Path):
        self.coverage_data_dir = coverage_data_dir
        self.latex_output_dir = latex_output_dir
        self.report_dir = report_dir
        self.logger = logging.getLogger("LaTeXTableGenerator")
        self.logger.handlers = []
        handler = logging.StreamHandler()
        handler.setFormatter(ColoredFormatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        self.successes = []
        self.failures = []

    def generate_tables(self):
        """Generate LaTeX tables from coverage data."""
        try:
            self.logger.info(f"Generating LaTeX tables from {self.coverage_data_dir}")
            coverage_files = list(self.coverage_data_dir.glob("rule_coverage.yaml"))
            if not coverage_files:
                raise FileNotFoundError("Aucun fichier de couverture trouvé.")

            for coverage_file in coverage_files:
                with open(coverage_file, "r") as file:
                    coverage_data = yaml.safe_load(file)
                
                table_latex = self._create_latex_table(coverage_data)
                table_file = self.latex_output_dir / f"{coverage_file.stem}_table.tex"
                
                with open(table_file, "w") as f:
                    f.write(table_latex)
                
                self.logger.info(f"Table LaTeX générée: {table_file}")
            
            self.successes.append("Tables LaTeX générées avec succès.")
            self.logger.info("Toutes les tables LaTeX ont été générées avec succès.")
        except Exception as e:
            self.failures.append(f"LaTeXTableGenerator failed: {e}")
            self.logger.error(f"LaTeXTableGenerator failed: {e}")

    def _create_latex_table(self, coverage_data: dict) -> str:
        """Create a LaTeX table from coverage data."""
        table = "\\begin{table}[h!]\n\\centering\n\\begin{tabular}{|l|c|}\n\\hline\n"
        table += "Fichier & Couverture (%) \\\\ \\hline\n"
        for rule_file, coverage in coverage_data.items():
            table += f"{rule_file} & {coverage} \\\\ \\hline\n"
        table += "\\end{tabular}\n\\caption{Couverture des Règles}\n\\end{table}\n"
        return table

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

    def main(self):
        """Main method to execute table generation and report."""
        self.generate_tables()
        self.generate_report()