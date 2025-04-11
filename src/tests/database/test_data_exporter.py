import os
import csv
import pytest
from unittest.mock import MagicMock, patch
from sqlalchemy import MetaData, Table, Column, Integer, String, select

from database.data_exporter import DataExporter


@pytest.fixture
def mock_logger():
    """Creates mock loggers for query time and results."""
    logger_query_time = MagicMock()
    logger_query_results = MagicMock()
    return logger_query_time, logger_query_results


@pytest.fixture
def mock_metadata():
    """Creates a mock metadata object with a single table."""
    metadata = MetaData()
    # Example table
    test_table = Table(
        'test_table', metadata,
        Column('id', Integer, primary_key=True),
        Column('name', String),
    )
    return metadata


@pytest.fixture
def mock_engine():
    """Creates a mock synchronous engine with a fake connection."""
    engine = MagicMock()
    connection = MagicMock()
    engine.connect.return_value.__enter__.return_value = connection
    return engine


@pytest.fixture
def data_exporter(tmp_path, mock_metadata, mock_engine, mock_logger):
    """Creates a DataExporter instance with mocked components."""
    db_path = str(tmp_path)
    base_name = "test_db"
    logger_query_time, logger_query_results = mock_logger
    return DataExporter(
        db_path=db_path,
        base_name=base_name,
        engine=mock_engine,
        metadata=mock_metadata,
        logger_query_time=logger_query_time,
        logger_query_results=logger_query_results
    )


def test_export_tables_to_csv_success(data_exporter, mock_engine):
    # Mock fetchall results
    rows = [(1, "Alice"), (2, "Bob")]
    mock_connection = mock_engine.connect.return_value.__enter__.return_value
    mock_connection.execute.return_value.fetchall.return_value = rows

    data_exporter.export_tables_to_csv()

    # Check if CSV was created correctly
    csv_dir = os.path.join(data_exporter.database_path, data_exporter.base_name, "csv")
    csv_path = os.path.join(csv_dir, "test_table.csv")

    assert os.path.exists(csv_path), "CSV file was not created."

    with open(csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        lines = list(reader)

    # Expect header + 2 rows
    assert len(lines) == 3
    assert lines[0] == ["id", "name"]
    assert lines[1] == ["1", "Alice"]
    assert lines[2] == ["2", "Bob"]


def test_export_tables_to_csv_error_on_query(data_exporter, mock_engine, mock_logger):
    # Simulate an exception during fetch
    mock_connection = mock_engine.connect.return_value.__enter__.return_value
    mock_connection.execute.side_effect = Exception("DB Error")

    data_exporter.export_tables_to_csv()

    # Check that error was logged
    logger_query_time, _ = mock_logger
    logger_query_time.error.assert_any_call("Error fetching rows for table 'test_table': DB Error")


def test_export_triples_to_tsv(data_exporter):
    triples = [
        ("subject1", "predicate1", "object1"),
        ("subject2", "predicate2", "object2")
    ]
    data_exporter.export_triples_to_tsv(triples)

    tsv_dir = os.path.join(data_exporter.database_path, data_exporter.base_name, "tsv")
    tsv_path = os.path.join(tsv_dir, f"{data_exporter.base_name}.tsv")

    assert os.path.exists(tsv_path), "TSV file not created."

    with open(tsv_path, "r") as f:
        lines = f.read().strip().split('\n')

    assert len(lines) == 2
    assert lines[0] == "subject1\tpredicate1\tobject1"
    assert lines[1] == "subject2\tpredicate2\tobject2"


def test_export_triples_to_tsv_error(data_exporter, mock_logger):
    # Patch open to raise an exception
    with patch("builtins.open", side_effect=Exception("File write error")):
        data_exporter.export_triples_to_tsv([("s", "p", "o")])

    logger_query_time, _ = mock_logger
    logger_query_time.error.assert_any_call(
        f"Error writing TSV file '{os.path.join(data_exporter.database_path, data_exporter.base_name, 'tsv', f'{data_exporter.base_name}.tsv')}': File write error"
    )


def test_export_triples_to_ttl(data_exporter):
    triples = [
        ("Subject 1", "hasName", '"Alice"'),
        ("Subject 2", "hasFriend", "Bob")
    ]
    data_exporter.export_triples_to_ttl(triples)

    ttl_dir = os.path.join(data_exporter.database_path, data_exporter.base_name, "ttl")
    ttl_path = os.path.join(ttl_dir, f"{data_exporter.base_name}.ttl")

    assert os.path.exists(ttl_path), "TTL file not created."

    with open(ttl_path, "r") as f:
        ttl_content = f.read().strip()

    assert "@prefix ex: <http://example.org/> ." in ttl_content
    assert "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> ." in ttl_content

    # Check first triple line
    assert "ex:Subject_1 ex:hasName \"Alice\" ." in ttl_content
    # Check second triple line
    assert "ex:Subject_2 ex:hasFriend ex:Bob ." in ttl_content


def test_export_triples_to_ttl_error(data_exporter, mock_logger):
    # Patch open to raise an exception
    with patch("builtins.open", side_effect=Exception("File write error")):
        data_exporter.export_triples_to_ttl([("s", "p", "o")])

    logger_query_time, _ = mock_logger
    logger_query_time.error.assert_any_call(
        f"Error writing TTL file '{os.path.join(data_exporter.database_path, data_exporter.base_name, 'ttl', f'{data_exporter.base_name}.ttl')}': File write error"
    )


def test_sanitize_identifier():
    assert DataExporter._sanitize_identifier("Foo Bar") == "Foo_Bar"
    assert DataExporter._sanitize_identifier('Foo"Bar') == "FooBar"
    assert DataExporter._sanitize_identifier("Foo'Bar") == "FooBar"
    assert DataExporter._sanitize_identifier("Foo Bar's") == "Foo_Bars"
