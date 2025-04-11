import pytest
from unittest.mock import patch, MagicMock
from sqlalchemy import MetaData

from database.database_connection_manager import DatabaseConnectionManager


@pytest.fixture
def mock_engine():
    """Mock SQLAlchemy engine."""
    engine = MagicMock()
    mock_connection = MagicMock()
    engine.connect.return_value = mock_connection
    return engine, mock_connection


@pytest.fixture
def mock_create_engine(mock_engine):
    """Patch create_engine to return a mock engine."""
    engine, mock_connection = mock_engine
    with patch("database.database_connection_manager.create_engine", return_value=engine) as patched:
        yield patched, engine, mock_connection


@pytest.fixture
def mock_metadata():
    """Patch MetaData.reflect to ensure it is called."""
    metadata = MetaData()
    with patch.object(metadata, 'reflect') as mock_reflect:
        yield metadata, mock_reflect


def test_database_connection_manager_init(mock_create_engine, mock_metadata):
    db_url = "sqlite:///:memory:"
    mock_create_engine_patch, engine, mock_connection = mock_create_engine
    metadata, mock_reflect = mock_metadata

    # Patch MetaData used in DatabaseConnectionManager to our mock
    with patch("database.database_connection_manager.MetaData", return_value=metadata):
        manager = DatabaseConnectionManager(db_url)

        # Assert create_engine was called with db_url
        mock_create_engine_patch.assert_called_once_with(db_url)

        # Assert reflect was called on metadata with the correct bind
        mock_reflect.assert_called_once_with(bind=manager.engine)

        # Assert connection is established
        assert manager.conn is not None
        assert manager.engine is not None


def test_database_connection_manager_close(mock_create_engine, mock_metadata):
    db_url = "sqlite:///:memory:"
    mock_create_engine_patch, engine, mock_connection = mock_create_engine
    metadata, mock_reflect = mock_metadata

    with patch("database.database_connection_manager.MetaData", return_value=metadata):
        manager = DatabaseConnectionManager(db_url)

        # Capture the mock connection before closing
        captured_connection = manager.conn

        # Close the manager
        manager.close()

        # Ensure connection's close() method was called
        captured_connection.close.assert_called_once()

        # Ensure engine's dispose() method was called
        engine.dispose.assert_called_once()

        # Check that manager's engine and conn are set to None
        assert manager.conn is None
        assert manager.engine is None
