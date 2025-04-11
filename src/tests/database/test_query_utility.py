import pytest
from unittest.mock import MagicMock, patch
from sqlalchemy import MetaData, Table, Column, Integer, String, create_engine
from sqlalchemy.sql import text, and_
from sqlalchemy.orm import sessionmaker
from sqlalchemy import func, select
from typing import List, Tuple

from database.query_utility import QueryUtility

@pytest.fixture
def mock_logger():
    logger_query_time = MagicMock()
    logger_query_results = MagicMock()
    return logger_query_time, logger_query_results

@pytest.fixture
def in_memory_metadata():
    """Create an in-memory metadata with tables for testing."""
    metadata = MetaData()
    # Example tables
    Table('users', metadata,
          Column('id', Integer, primary_key=True),
          Column('name', String),
          Column('age', Integer))

    Table('posts', metadata,
          Column('post_id', Integer, primary_key=True),
          Column('user_id', Integer),
          Column('content', String))
    return metadata

@pytest.fixture
def mock_engine():
    # Mock an SQLAlchemy engine
    engine = MagicMock()
    conn = MagicMock()
    # Mock the engine.connect() to return a mock connection
    engine.connect.return_value.__enter__.return_value = conn

    # Mock a simple scalar result
    # By default, scalar() returns None; you can set return values in tests if needed.
    conn.execute.return_value.scalar.return_value = None
    return engine, conn

@pytest.fixture
def query_utility(mock_engine, in_memory_metadata, mock_logger):
    engine, _ = mock_engine
    logger_query_time, logger_query_results = mock_logger
    return QueryUtility(engine, in_memory_metadata, logger_query_time, logger_query_results)

def test_check_threshold_no_query(query_utility, mock_engine):
    # If _construct_threshold_query returns None, should return 0
    # We can simulate this by passing empty join_conditions that produce no query
    result = query_utility.check_threshold(join_conditions=[])
    assert result == 0

def test_get_join_row_count_no_query(query_utility, mock_engine):
    # Similarly, if no query is constructed, result should be 0
    result = query_utility.get_join_row_count(join_conditions=[])
    assert result == 0

def test_check_threshold(query_utility, mock_engine):
    engine, conn = mock_engine

    # Mock a scenario where a query is constructed. We provide join_conditions that should be valid.
    # Let's join users.id = posts.user_id
    join_conditions = [("users", 1, "id", "posts", 1, "user_id")]

    # Mock the database result
    # The query checks threshold by counting rows > threshold
    # Let's say the result is True (meaning threshold exceeded)
    conn.execute.return_value.scalar.return_value = True

    result = query_utility.check_threshold(
        join_conditions=join_conditions,
        threshold=1
    )
    assert result == 1  # True converted to int is 1

    # Check that a query was executed
    assert conn.execute.called

def test_get_join_row_count(query_utility, mock_engine):
    engine, conn = mock_engine

    # Setup a simple join condition
    join_conditions = [("users", 1, "id", "posts", 1, "user_id")]
    conn.execute.return_value.scalar.return_value = 5  # Suppose 5 rows match

    result = query_utility.get_join_row_count(join_conditions=join_conditions)
    assert result == 5

    assert conn.execute.called

def test_organize_join_conditions(query_utility):
    join_conditions = [
        ("users", 1, "id", "posts", 1, "user_id"),
        ("users", 1, "name", "users", 1, "name")  # same table, same occurrence
    ]
    # Access the internal method for testing (generally not recommended, but okay for unit tests)
    organized = query_utility._organize_join_conditions(join_conditions)
    # Expect keys for each unique set of (table, occurrence)
    # For ("users",1) <-> ("posts",1) and ("users",1) alone
    assert len(organized) == 2

def test_get_or_create_alias_existing_table(query_utility, in_memory_metadata):
    aliases = {}
    alias_obj = query_utility._get_or_create_alias(aliases, "users", 1)
    assert "users_1" in aliases
    # Recalling _get_or_create_alias should return the same alias object
    alias_obj2 = query_utility._get_or_create_alias(aliases, "users", 1)
    assert alias_obj is alias_obj2

def test_get_or_create_alias_non_existing_table(query_utility):
    aliases = {}
    with pytest.raises(ValueError) as exc:
        query_utility._get_or_create_alias(aliases, "non_existent", 1)
    assert "does not exist" in str(exc.value)

def test_attribute_helpers(query_utility):
    # Check attribute helpers on known tables
    tables = query_utility._get_table_names()
    assert "users" in tables
    assert "posts" in tables

    user_columns = query_utility._get_attribute_names("users")
    assert set(user_columns) == {"id", "name", "age"}

    # Check domain
    domain = query_utility._get_attribute_domain("users", "id")
    # Domain will be the SQLAlchemy type name, something like INTEGER
    assert "INTEGER" in domain.upper()

    # Check primary key
    assert query_utility._get_attribute_is_key("users", "id") is True
    assert query_utility._get_attribute_is_key("users", "name") is False

def test_disjoint_semantics_primary_key_conditions(query_utility, in_memory_metadata):
    # Test the creation of primary key conditions under disjoint semantics
    # For simplicity, we will just call _construct_query_base with disjoint_semantics=True
    join_conditions = [
        ("users", 1, "id", "posts", 1, "user_id")
    ]

    query, primary_key_conditions, join_base = query_utility._construct_query_base(
        join_conditions=join_conditions,
        disjoint_semantics=True,
        distinct=False,
        count_over=None
    )

    # We expect query not to be None
    assert query is not None
    # primary_key_conditions might include conditions for ensuring distinct primary keys if multiple occurrences
    # but in this simple case, there's just one occurrence of each table.
    # So no PK inequality conditions expected between multiple occurrences of the same table.
    # We still should have some conditions related to the join.
    # Check that primary_key_conditions is at least not None
    assert primary_key_conditions is not None

def test_count_over_clause(query_utility, mock_engine):
    engine, conn = mock_engine
    # count_over scenario: we specify attributes over which we want a distinct count
    join_conditions = [("users", 1, "id", "posts", 1, "user_id")]
    count_over = [[("users", 1, "id")]]

    conn.execute.return_value.scalar.return_value = 10

    result = query_utility.get_join_row_count(
        join_conditions=join_conditions,
        disjoint_semantics=False,
        distinct=False,
        count_over=count_over
    )
    assert result == 10
    assert conn.execute.called
