import csv
import hashlib
import logging
import os
import time
from typing import Any, Dict, List, Tuple

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

#from utils.log_setup import setup_loggers
import colorama   # Added colorama
colorama.init(autoreset=True)




class TripleConverter:
    """
    Converts database tables into RDF-like triples.
    """

    def __init__(self, engine, metadata: MetaData, logger):
        self.engine = engine
        self.metadata = metadata
        self.logger = logger

    def convert_to_triples(self) -> List[Tuple[str, str, str]]:
        triples = []
        foreign_keys = self._get_foreign_keys()
        primary_keys = {table: self._get_primary_keys(table) for table in self._get_table_names()}

        for table_name in self._get_table_names():
            attributes = self._get_attribute_names(table_name)
            pk_columns = primary_keys.get(table_name, [])
            fk_columns = foreign_keys.get(table_name, {})

            if len(pk_columns) == 0 or len(attributes) == 1:
                self.logger.warning(
                    f"Table {table_name} has no PK or only one column. Skipping."
                )
                continue

            rows = self._select_query(table_name, attributes)
            for row in rows:
                row_dict = dict(zip(attributes, row))
                # Skip row if a primary key is missing
                if any(pk not in row_dict or row_dict[pk] is None for pk in pk_columns):
                    self.logger.error(f"Missing primary keys in table {table_name} for row {row_dict}.")
                    continue

                subject = self._generate_rdf_id(table_name, pk_columns, row_dict)

                for attribute, value in row_dict.items():
                    if value is None or attribute is None:
                        continue

                    predicate = f"{self._sanitize_identifier(table_name)}.{self._sanitize_identifier(attribute)}"
                    if attribute in fk_columns:
                        # Foreign key triple
                        try:
                            ref_table, ref_column = fk_columns[attribute]
                            # Skip if foreign key column is missing
                            if ref_column not in row_dict or row_dict[ref_column] is None:
                                #self.logger.warning(
                                #    f"Missing foreign key column '{ref_column}' for row {row_dict}, skipping."
                                #)
                                continue

                            ref_pk_columns = primary_keys.get(ref_table, [])
                            if not ref_pk_columns:
                                self.logger.warning(
                                    f"Referenced table {ref_table} has no PK. Skipping."
                                )
                                continue

                            row_dict_fk = {ref_column: row_dict[ref_column]}
                            ref_subject = self._generate_rdf_id(ref_table, ref_pk_columns, row_dict_fk)
                            triples.append((subject, predicate, ref_subject))
                        except Exception as e:
                            self.logger.error(f"Error processing foreign key for table {table_name}: {e}")
                    elif attribute not in pk_columns:
                        # Literal triple
                        safe_value = str(value).replace('"', '\\"')
                        triples.append((subject, predicate, f'"{safe_value}"'))

        return triples

    def _get_table_names(self) -> List[str]:
        return sorted(self.metadata.tables.keys())

    def _get_foreign_keys(self) -> Dict[str, Dict[str, Tuple[str, str]]]:
        foreign_keys_info = {}
        for table_name, table in self.metadata.tables.items():
            for fk in table.foreign_keys:
                ref_table = fk.column.table.name
                local_column = fk.parent.name
                reference_column = fk.column.name
                if table_name not in foreign_keys_info:
                    foreign_keys_info[table_name] = {}
                foreign_keys_info[table_name][local_column] = (ref_table, reference_column)
        return foreign_keys_info

    def _get_primary_keys(self, table_name: str) -> List[str]:
        table = self.metadata.tables.get(table_name)
        if table is not None and table.primary_key:
            return [key.name for key in table.primary_key.columns]
        return []

    def _get_attribute_names(self, table_name: str) -> List[str]:
        table = self.metadata.tables.get(table_name)
        if table is not None and hasattr(table, "columns"):
            return [column.name for column in table.columns]
        return []

    def _select_query(self, table_name: str, attributes: List[str]) -> List[Tuple]:
        table_obj = self.metadata.tables.get(table_name)
        if table_obj is None:
            return []
        columns = [table_obj.columns[attr] for attr in attributes if attr in table_obj.columns]
        if not columns:
            return []
        query = select(*columns)
        try:
            with self.engine.connect() as conn:
                return conn.execute(query).fetchall()
        except Exception as e:
            self.logger.error(f"Error executing select query on '{table_name}': {e}")
            return []

    def _generate_rdf_id(self, table: str, primary_keys: List[str], row_dict: Dict[str, Any]) -> str:
        try:
            pk_values = "_".join(
                self._sanitize_identifier(str(row_dict[pk])) for pk in primary_keys
            )
        except KeyError as e:
            self.logger.error(f"Missing key {e} in table {table} for row {row_dict}.")
            return self._sanitize_identifier("unknown_id")
        return f"{table}_{pk_values}"

    @staticmethod
    def _sanitize_identifier(identifier: str) -> str:
        return "".join(e if e.isalnum() else "_" for e in identifier)

