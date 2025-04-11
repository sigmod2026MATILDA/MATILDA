#0. Download / check that data is available
#python3 steps/0_download_databases.py
#1. Generate configs
python3 steps/1_create_configs.py
#2. Execute algorithms with docker
mkdir logs
python3 steps/2_execute_algorithms.py
#3. Collect results
python3 steps/3_gather_results.py
#4. Compute coverage
python3 steps/4_compute_coverage.py
#5. Generate latex table
python3 steps/5_generate_latex_table.py
