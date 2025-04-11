import pytest
from unittest.mock import MagicMock
from sqlalchemy import MetaData, Table, Column, Integer, String, text
from database.index_manager import IndexManager

@pytest.fixture
def setup_index_manager():
    # Create an in-memory MetaData with some tables and columns
    metadata = MetaData()
    table1 = Table('table1', metadata,
                   Column('id', Integer, primary_key=True),
                   Column('name', String),
                   Column('age', Integer))
    table2 = Table('table2', metadata,
                   Column('key', Integer, primary_key=True),
                   Column('value', String))

    # Mock connection
    conn = MagicMock()
    # Mock the context manager for conn.begin()
    conn.begin.return_value.__enter__.return_value = conn
    conn.begin.return_value.__exit__.return_value = False

    index_manager = IndexManager(conn, metadata)
    return index_manager, conn, metadata

def test_create_indexes(setup_index_manager):
    index_manager, conn, metadata = setup_index_manager
    index_manager.create_indexes()

    # There should be one CREATE INDEX per column
    # table1: id, name, age -> 3 indexes
    # table2: key, value -> 2 indexes
    expected_statements = [
        'CREATE INDEX IF NOT EXISTS idx_table1_id ON table1 (id);',
        'CREATE INDEX IF NOT EXISTS idx_table1_name ON table1 (name);',
        'CREATE INDEX IF NOT EXISTS idx_table1_age ON table1 (age);',
        'CREATE INDEX IF NOT EXISTS idx_table2_key ON table2 (key);',
        'CREATE INDEX IF NOT EXISTS idx_table2_value ON table2 (value);'
    ]

    actual_calls = conn.execute.call_args_list
    assert len(actual_calls) == len(expected_statements)

    for (call_args, _), expected_sql in zip(actual_calls, expected_statements):
        executed_text = call_args[0]
        # executed_text is a sqlalchemy.text object, convert to string
        assert str(executed_text) == expected_sql

def test_create_composed_indexes(setup_index_manager):
    index_manager, conn, metadata = setup_index_manager

    # Simulate some composed indexes
    # (t1, c1, t2, c2)
    cols_list = [
        ("table1", "id", "table1", "name"),    # valid (same table, different columns)
        ("table1", "name", "table1", "name"),  # should skip (same column)
        ("table1", "age", "table2", "value"),  # different tables, skip
        ("table2", "key", "table2", "value")   # valid (same table, different columns)
    ]

    index_manager.create_composed_indexes(cols_list)

    # Expected calls:
    # Only first and last entries produce indexes
    expected_statements = [
        'CREATE INDEX IF NOT EXISTS idx_table1_id_name ON table1 (id, name);',
        'CREATE INDEX IF NOT EXISTS idx_table2_key_value ON table2 (key, value);'
    ]

    actual_calls = conn.execute.call_args_list
    assert len(actual_calls) == len(expected_statements)

    for (call_args, _), expected_sql in zip(actual_calls, expected_statements):
        executed_text = call_args[0]
        assert str(executed_text) == expected_sql

    # Also check that commit was called twice (once for each created index)
    assert conn.commit.call_count == 2
