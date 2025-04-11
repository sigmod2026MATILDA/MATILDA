import csv
import hashlib
import logging
import os
import time
from typing import Any, Dict, List, Tuple
import sys
sys.path.append("../../")
sys.path.append("../")

import psutil
from sqlalchemy import (
    MetaData,
    alias,
    and_,
    create_engine,
    func,
    select,
    text
)
from sqlalchemy.engine.url import make_url
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm

from src.utils.log_setup import setup_loggers
from src.database.database_connection_manager import DatabaseConnectionManager
from src.database.index_manager import IndexManager
from src.database.data_exporter import DataExporter
from src.database.triple_converter import TripleConverter
from src.database.query_utility import QueryUtility
import colorama   # Added colorama
colorama.init(autoreset=True)

import logging
from colorama import Fore, Style

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



class AlchemyUtility:
    """
    High-level class that composes the other components to:
    - Manage DB connection
    - Export data to CSV and TSV
    - Convert DB to triples
    - Create indexes (SQLite)
    - Perform threshold checks and join counts
    """

    def __init__(
        self,
        db_url: str,
        database_path: str = "",
        create_index: bool = True,
        create_csv: bool = True,
        create_tsv: bool = True,
        get_data: bool = True,
    ):
        setup_loggers()
        self.logger_query_time = logging.getLogger("query_time")
        self.logger_query_results = logging.getLogger("query_results")
        
        self._setup_logging_handlers()  # Configure log handlers

        self.db_url = db_url
        self.database_path = database_path

        # Parse db_url to get base_name
        url = make_url(db_url)
        if (url.drivername == "sqlite"):
            self.base_name = os.path.splitext(os.path.basename(url.database))[0]
        else:
            self.base_name = str(url).split("//")[-1].split(":")[0]

        self.db_manager = DatabaseConnectionManager(db_url)
        self.index_manager = IndexManager(self.db_manager.conn, self.db_manager.metadata)
        self.data_exporter = DataExporter(
            db_path=self.database_path,
            base_name=self.base_name,
            engine=self.db_manager.engine,
            metadata=self.db_manager.metadata,
            logger_query_time=self.logger_query_time,
            logger_query_results=self.logger_query_results
        )
        self.triple_converter = TripleConverter(
            engine=self.db_manager.engine,
            metadata=self.db_manager.metadata,
            logger=self.logger_query_time
        )
        self.query_utility = QueryUtility(
            engine=self.db_manager.engine,
            metadata=self.db_manager.metadata,
            logger_query_time=self.logger_query_time,
            logger_query_results=self.logger_query_results
        )

        # Export CSV
        if create_csv:
            self.data_exporter.export_tables_to_csv()
            self.base_csv_dir = os.path.join(self.database_path, self.base_name,"csv")
        # Create indexes if SQLite
        if url.drivername == "sqlite":
            self._setup_sqlite(create_index)

        # Convert to triples and export to TSV
        if create_tsv:
            triples = self.triple_converter.convert_to_triples()
            self.data_exporter.export_triples_to_tsv(triples)
            self.database_path_tsv = os.path.join(self.database_path,self.base_name, "tsv")
        # Load data if needed
        if get_data:
            self.tables_data = self._extract_table_data()
    def _setup_logging_handlers(self):
        formatter = ColorFormatter('%(asctime)s - %(levelname)s - %(message)s')

        # Formatter for logger_query_time
        handler_time = logging.StreamHandler()
        handler_time.setFormatter(formatter)
        self.logger_query_time.addHandler(handler_time)
        self.logger_query_time.setLevel(logging.DEBUG)

        # Formatter for logger_query_results
        handler_results = logging.StreamHandler()
        handler_results.setFormatter(formatter)
        self.logger_query_results.addHandler(handler_results)
        self.logger_query_results.setLevel(logging.DEBUG)
    def _setup_sqlite(self, create_index: bool):
        """Configure SQLite PRAGMAs and optionally create indexes."""
        self.db_manager.conn.execute(text("PRAGMA temp_store = MEMORY;"))
        max_cache_size = self.get_cache_size()
        self.db_manager.conn.execute(text(f"PRAGMA cache_size = {max_cache_size};"))
        self.db_manager.conn.execute(text("PRAGMA read_uncommitted = 1;"))
        self.db_manager.conn.execute(text("PRAGMA synchronous = OFF;"))
        # Commit or rollback any pending transaction before creating indexes
        self.db_manager.conn.commit()
        if create_index:
            try:
                self.index_manager.create_indexes()
                self.db_manager.conn.commit()
            except Exception as e:
                self.logger_query_time.error(f"Error creating indexes: {e}")
                self.db_manager.conn.rollback()
        self.db_manager.conn.commit()

    @staticmethod
    def get_cache_size() -> int:
        """Get appropriate cache size for SQLite (10% of total memory)."""
        total_memory = psutil.virtual_memory().total
        cache_size_kb = (total_memory * 0.1) // 1024
        cache_size_pages = cache_size_kb // 1.024
        return int(cache_size_pages)

    def _extract_table_data(self) -> Dict[str, Dict[str, Any]]:
        """Extract data from all tables."""
        data = {}
        for table_name in sorted(self.db_manager.metadata.tables.keys()):
            table = self.db_manager.metadata.tables.get(table_name)
            if table is not None:
                columns = [col.name for col in table.columns]
                rows = self._select_query(table_name, columns)
                data[table_name] = {"columns": columns, "rows": rows}
        return data

    def _select_query(self, table_name: str, attributes: List[str]) -> List[Tuple]:
        table_obj = self.db_manager.metadata.tables.get(table_name)
        if table_obj is None:
            return []
        columns = [table_obj.columns[attr] for attr in attributes if attr in table_obj.columns]
        if not columns:
            return []
        query = select(*columns)
        try:
            with self.db_manager.engine.connect() as conn:
                return conn.execute(query).fetchall()
        except Exception as e:
            self.logger_query_time.error(f"Error executing select query on '{table_name}': {e}")
            return []
    def create_composed_indexes(self, cols_list: List[Tuple[str, str, str, str]]):
        """Create composed indexes for tuples of columns."""
        self.index_manager.create_composed_indexes(cols_list)
    def check_threshold(
        self,
        join_conditions: List[Tuple[str, int, str, str, int, str]],
        disjoint_semantics: bool = False,
        distinct: bool = False,
        count_over: List[List[Tuple[str, int, str]]] = None,
        threshold: int = 1,
        flag: str="",
    ) -> int:
        return self.query_utility.check_threshold(
            join_conditions, disjoint_semantics, distinct, count_over, threshold,flag
        )
    def get_join_row_count(self,
        join_conditions: List[Tuple[str, int, str, str, int, str]],
        disjoint_semantics: bool = False,
        distinct: bool = False,
        count_over: List[List[Tuple[str, int, str]]] = None,
       flag: str=""
    ) -> int:
        return self.query_utility.get_join_row_count(join_conditions,disjoint_semantics,distinct,count_over)
    def get_table_names(self) -> List[str]:
        return self.query_utility._get_table_names()
    def get_attribute_names(self, table_name: str) -> List[str]:
        return self.query_utility._get_attribute_names(table_name)
    def get_attribute_domain(self, table_name: str, attribute_name: str) -> str:
        return self.query_utility._get_attribute_domain(table_name, attribute_name)
    def get_attribute_is_key(self, table_name: str, attribute_name: str) -> bool:
        return self.query_utility._get_attribute_is_key(table_name, attribute_name)
    def are_foreign_keys(self, table: str, column: str, other_table: str, other_column: str) -> bool:
        """
        Check if the specified column in a table is a foreign key referencing another table and column.

        :param table: The name of the table containing the column.
        :param column: The name of the column in the first table.
        :param other_table: The name of the referenced table.
        :param other_column: The name of the referenced column.
        :return: True if the column is a foreign key referencing the other table and column, False otherwise.
        """
        foreign_keys = self.query_utility._get_foreign_keys()
        if table in foreign_keys:
            if column in foreign_keys[table]:
                referenced_table, referenced_column = foreign_keys[table][column]
                # Check if the referenced table and column match
                if referenced_table == other_table and referenced_column == other_column:
                    return True
                else:
                    self.logger_query_time.error(f"Incorrect foreign key for table '{table}': '{column}' references '{referenced_table}.{referenced_column}' instead of '{other_table}.{other_column}'.")
                    return False
        self.logger_query_time.error(f"Foreign key '{column}' does not exist in table '{table}'.")
        return False
    def close(self):
        """Close database connection."""
        self.db_manager.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    async def __aenter__(self):
        self.async_engine = create_async_engine(self.db_url)
        self.async_session = sessionmaker(bind=self.async_engine, class_=AsyncSession, expire_on_commit=False)()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.async_session.close()
        await self.async_engine.dispose()


if __name__ == "__main__":
    db_url = "sqlite:///tests/tpcc.db"
    with AlchemyUtility(db_url, database_path="tests") as alchemy_utility:
        print(f"Database exported in 'data/{alchemy_utility.base_name}/'.")
