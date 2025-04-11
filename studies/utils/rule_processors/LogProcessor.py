import os
import re
import json
import logging
import sys
from datetime import datetime
from tqdm import tqdm
import colorama
colorama.init(autoreset=True)
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LogProcessor:
    def __init__(self, log_dir, output_dir):
        self.log_dir = log_dir
        self.output_dir = output_dir
        self.stats_dir = os.path.join(output_dir, "stats")
        self.combined_stats_file = os.path.join(output_dir, "combined_stats.json")
        os.makedirs(self.stats_dir, exist_ok=True)

    def process_logs(self):
        databases = os.listdir(self.log_dir)
        for database in tqdm(databases, desc="Processing databases results with logs"):
            database_path = os.path.join(self.log_dir, database)
            if os.path.isdir(database_path):
                algorithms = os.listdir(database_path)
                for algorithm in algorithms:
                    algorithm_path = os.path.join(database_path, algorithm, "global.log")
                    if os.path.exists(algorithm_path):
                        self.process_algorithm_log(database, algorithm, algorithm_path)
        # Combine all stats into a single file
        self.combine_stats()
    def combine_stats(self):
        combined_stats = {}
        for database in os.listdir(self.stats_dir):
            database_path = os.path.join(self.stats_dir, database)
            if os.path.isdir(database_path):
                for algorithm in os.listdir(database_path):
                    algorithm_path = os.path.join(database_path, algorithm, "stats.json")
                    if os.path.exists(algorithm_path):
                        with open(algorithm_path, 'r') as stats_file:
                            stats = json.load(stats_file)
                            if database not in combined_stats:
                                combined_stats[database] = {}
                            combined_stats[database][algorithm] = stats

        with open(self.combined_stats_file, 'w') as combined_stats_file:
            json.dump(combined_stats, combined_stats_file, indent=4)
    def process_algorithm_log(self, database_name, algorithm_name, log_path):
        timestamps = self.extract_timestamps(log_path)
        number_of_rules= self.extract_number_of_rules(log_path)
        if timestamps:
            start_time = datetime.strptime(timestamps[0], '%Y-%m-%d %H:%M:%S,%f')
            end_time = datetime.strptime(timestamps[-1], '%Y-%m-%d %H:%M:%S,%f')
            processing_time = (end_time - start_time).total_seconds()

            stats = {
                "processing_time": self.format_time(processing_time),
                "number_of_rules": number_of_rules
            }

            output_stats_dir = os.path.join(self.stats_dir, database_name, algorithm_name)
            os.makedirs(output_stats_dir, exist_ok=True)

            with open(os.path.join(output_stats_dir, "stats.json"), 'w') as stats_file:
                json.dump(stats, stats_file, indent=4)

    def extract_number_of_rules(self, log_path):
        # Correct regular expression to capture the number of discovered rules
        number_of_rules_pattern = re.compile(r'\[INFO\] - Discovered (\d+) rules\.')

        with open(log_path, 'r') as file:
            for line in file:
                number_of_rules_match = number_of_rules_pattern.search(line)
                if number_of_rules_match:
                    # Extract and return the captured number as an integer
                    return int(number_of_rules_match.group(1))
        return 0
    def extract_timestamps(self, log_path):
        timestamp_pattern = re.compile(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})')
        stop_pattern = re.compile(r'Done running')
        timestamps = []

        with open(log_path, 'r') as file:
            for line in file:
                if stop_pattern.search(line):
                    break
                timestamp_match = timestamp_pattern.search(line)
                if timestamp_match:
                    timestamps.append(timestamp_match.group(1))

        return timestamps

    def format_time(self, seconds):
        if seconds < 1:
            return "<1s"

        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60

        if hours > 0:
            return f"{hours}h:{minutes}m"
        elif minutes > 0:
            return f"{minutes}m:{seconds:.2f}s"
        else:
            return f"{seconds:.2f}s"

if __name__ == "__main__":
    if len(sys.argv) != 3:
        logging.error("Usage: python script.py <log_dir> <output_dir>")
        sys.exit(1)

    log_dir = sys.argv[1]
    output_dir = sys.argv[2]

    logging.info("Starting log processing")
    processor = LogProcessor(log_dir, output_dir)
    processor.process_logs()
    logging.info("Log processing completed")
