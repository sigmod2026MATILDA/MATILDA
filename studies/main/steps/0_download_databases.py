import os
database_out_path="data/databases/"
download_script_path="../../src/database/"
os.system("python3 "+download_script_path+"download_databases.py "+database_out_path)