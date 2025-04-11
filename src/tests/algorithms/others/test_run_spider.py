# test_spider.py

import pytest
from unittest.mock import patch, MagicMock, mock_open
from datetime import datetime
from algorithms.spider import Spider
from utils.rules import InclusionDependency


@pytest.fixture
def spider_instance():
    """
    Fixture to create a Spider instance with a mocked database.
    """
    mock_database = MagicMock()
    mock_database.base_csv_dir = "/path/to/csv"
    spider = Spider(database=mock_database)
    return spider


@pytest.fixture
def fixed_datetime():
    """
    Fixture to mock datetime to return a fixed current time.
    """
    fixed_time = datetime(2023, 1, 1, 12, 0, 0)
    with patch("algorithms.spider.datetime") as mock_datetime:
        mock_datetime.now.return_value = fixed_time
        mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
        yield mock_datetime


@pytest.fixture
def mock_path_join():
    """
    Fixture to mock os.path.join to behave as a simple string join with "/".
    """
    with patch("algorithms.spider.os.path.join", side_effect=lambda *args: "/".join(args)) as mock_join:
        yield mock_join


@pytest.fixture
def mock_listdir():
    """
    Fixture to mock os.listdir.
    """
    with patch("algorithms.spider.os.listdir") as mock_ls:
        yield mock_ls


@pytest.fixture
def mock_run_cmd():
    """
    Fixture to mock run_cmd function.
    """
    with patch("algorithms.spider.run_cmd") as mock_cmd:
        yield mock_cmd


def test_discover_rules_command_failure(spider_instance, mock_run_cmd, mock_listdir, mock_path_join, fixed_datetime):
    """
    Test that discover_rules returns an empty dictionary when the command fails.
    """
    # Setup the mocks
    mock_run_cmd.return_value = False
    mock_listdir.return_value = ["table1.csv", "table2.csv"]

    # Execute the method
    rules = spider_instance.discover_rules()

    # Assertions
    assert rules == {}

    # Construct the expected command string
    expected_cmd = (
        "java -cp algorithms/bins/metanome/jars/metanome-cli-1.2-SNAPSHOT.jar:"
        "algorithms/bins/metanome/jars/SPIDER-1.2-SNAPSHOT.jar de.metanome.cli.App "
        "--algorithm de.metanome.algorithms.spider.SPIDERFile "
        "--files /path/to/csv/table1.csv /path/to/csv/table2.csv "
        "--table-key INPUT_FILES --separator \",\" --output file:2023-01-01_12-00-00_SPIDER "
        "--header"
    )

    mock_run_cmd.assert_called_once_with(expected_cmd)

    # Ensure no file operations are performed since the command failed
    with patch("algorithms.spider.open") as mock_open_file:
        mock_open_file.assert_not_called()


def test_discover_rules_success(spider_instance, mock_run_cmd, mock_listdir, mock_path_join, fixed_datetime):
    """
    Test that discover_rules successfully parses rules when the command succeeds.
    """
    # Setup the mocks
    mock_run_cmd.return_value = True
    mock_listdir.return_value = ["table1.csv", "table2.csv"]

    # Mock the contents of the result file
    mock_file_content = (
        '{"dependant": {"columnIdentifiers": [{"tableIdentifier": "table1.csv", "columnIdentifier": "col1"}]}, '
        '"referenced": {"columnIdentifiers": [{"tableIdentifier": "table2.csv", "columnIdentifier": "col2"}]}}\n'
    )

    with patch("builtins.open", mock_open(read_data=mock_file_content)) as mock_file, \
            patch("algorithms.spider.os.path.exists") as mock_exists, \
            patch("algorithms.spider.os.remove") as mock_remove:
        # Mock os.path.exists to return True, indicating the result file exists
        mock_exists.return_value = True

        # Execute the method
        rules = spider_instance.discover_rules()

        # Define the expected InclusionDependency
        expected_rule = InclusionDependency(
            table_dependant="table1",
            columns_dependant=("col1",),
            table_referenced="table2",
            columns_referenced=("col2",)
        )

        # Assertions
        assert len(rules) == 1
        assert expected_rule in rules
        assert rules[expected_rule] == (1, 1)

        # Construct the expected command string
        expected_cmd = (
            "java -cp algorithms/bins/metanome/jars/metanome-cli-1.2-SNAPSHOT.jar:"
            "algorithms/bins/metanome/jars/SPIDER-1.2-SNAPSHOT.jar de.metanome.cli.App "
            "--algorithm de.metanome.algorithms.spider.SPIDERFile "
            "--files /path/to/csv/table1.csv /path/to/csv/table2.csv "
            "--table-key INPUT_FILES --separator \",\" --output file:2023-01-01_12-00-00_SPIDER "
            "--header"
        )

        mock_run_cmd.assert_called_once_with(expected_cmd)

        # Ensure file operations
        mock_file.assert_called_with("results/2023-01-01_12-00-00_SPIDER_inds", mode="r")
        mock_exists.assert_called_once_with("results/2023-01-01_12-00-00_SPIDER_inds")
        mock_remove.assert_called_once_with("results/2023-01-01_12-00-00_SPIDER_inds")


def test_discover_rules_no_csv_files(spider_instance, mock_run_cmd, mock_listdir, mock_path_join, fixed_datetime):
    """
    Test that discover_rules handles the case when there are no CSV files.
    """
    # Setup the mocks
    mock_run_cmd.return_value = True
    mock_listdir.return_value = []  # No CSV files

    # Mock the contents of the result file (could be empty or have specific structure)
    mock_file_content = ""  # Assuming no rules are found

    with patch("builtins.open", mock_open(read_data=mock_file_content)) as mock_file, \
            patch("algorithms.spider.os.path.exists") as mock_exists, \
            patch("algorithms.spider.os.remove") as mock_remove:
        # Mock os.path.exists to return False, indicating no file to remove
        mock_exists.return_value = False

        # Execute the method
        rules = spider_instance.discover_rules()

        # Assertions
        assert rules == {}

        # Construct the expected command string (no CSV files)
        expected_cmd = (
            "java -cp algorithms/bins/metanome/jars/metanome-cli-1.2-SNAPSHOT.jar:"
            "algorithms/bins/metanome/jars/SPIDER-1.2-SNAPSHOT.jar de.metanome.cli.App "
            "--algorithm de.metanome.algorithms.spider.SPIDERFile "
            "--files  --table-key INPUT_FILES --separator \",\" "
            "--output file:2023-01-01_12-00-00_SPIDER --header"
        )

        mock_run_cmd.assert_called_once_with(expected_cmd)

        # Ensure file operations
        mock_file.assert_called_with("results/2023-01-01_12-00-00_SPIDER_inds", mode="r")
        mock_exists.assert_called_once_with("results/2023-01-01_12-00-00_SPIDER_inds")
        mock_remove.assert_not_called()


def test_discover_rules_invalid_rule_format(spider_instance, mock_run_cmd, mock_listdir, mock_path_join,
                                            fixed_datetime):
    """
    Test that discover_rules handles invalid rule formats gracefully.
    """
    # Setup the mocks
    mock_run_cmd.return_value = True
    mock_listdir.return_value = ["table1.csv"]

    # Mock the contents of the result file with invalid JSON
    mock_file_content = "invalid_json\n"

    with patch("builtins.open", mock_open(read_data=mock_file_content)) as mock_file, \
            patch("algorithms.spider.os.path.exists") as mock_exists, \
            patch("algorithms.spider.os.remove") as mock_remove, \
            patch("ast.literal_eval", side_effect=ValueError("Invalid format")):
        # Mock os.path.exists to return True
        mock_exists.return_value = True

        # Execute the method
        rules = spider_instance.discover_rules()

        # Assertions
        assert rules == {}

        # Construct the expected command string
        expected_cmd = (
            "java -cp algorithms/bins/metanome/jars/metanome-cli-1.2-SNAPSHOT.jar:"
            "algorithms/bins/metanome/jars/SPIDER-1.2-SNAPSHOT.jar de.metanome.cli.App "
            "--algorithm de.metanome.algorithms.spider.SPIDERFile "
            "--files /path/to/csv/table1.csv "
            "--table-key INPUT_FILES --separator \",\" "
            "--output file:2023-01-01_12-00-00_SPIDER --header"
        )

        mock_run_cmd.assert_called_once_with(expected_cmd)

        # Ensure file operations
        mock_file.assert_called_with("results/2023-01-01_12-00-00_SPIDER_inds", mode="r")
        mock_exists.assert_called_once_with("results/2023-01-01_12-00-00_SPIDER_inds")
        mock_remove.assert_called_once_with("results/2023-01-01_12-00-00_SPIDER_inds")


def test_discover_rules_multiple_rules(spider_instance, mock_run_cmd, mock_listdir, mock_path_join, fixed_datetime):
    """
    Test that discover_rules correctly parses multiple rules from the result file.
    """
    # Setup the mocks
    mock_run_cmd.return_value = True
    mock_listdir.return_value = ["table1.csv", "table2.csv", "table3.csv"]

    # Mock the contents of the result file with multiple rules
    mock_file_content = (
        '{"dependant": {"columnIdentifiers": [{"tableIdentifier": "table1.csv", "columnIdentifier": "col1"}]}, '
        '"referenced": {"columnIdentifiers": [{"tableIdentifier": "table2.csv", "columnIdentifier": "col2"}]}}\n'
        '{"dependant": {"columnIdentifiers": [{"tableIdentifier": "table3.csv", "columnIdentifier": "col3"}]}, '
        '"referenced": {"columnIdentifiers": [{"tableIdentifier": "table1.csv", "columnIdentifier": "col1"}]}}\n'
    )

    with patch("builtins.open", mock_open(read_data=mock_file_content)) as mock_file, \
            patch("algorithms.spider.os.path.exists") as mock_exists, \
            patch("algorithms.spider.os.remove") as mock_remove:
        # Mock os.path.exists to return True
        mock_exists.return_value = True

        # Execute the method
        rules = spider_instance.discover_rules()

        # Define the expected InclusionDependencies
        expected_rule1 = InclusionDependency(
            table_dependant="table1",
            columns_dependant=("col1",),
            table_referenced="table2",
            columns_referenced=("col2",)
        )
        expected_rule2 = InclusionDependency(
            table_dependant="table3",
            columns_dependant=("col3",),
            table_referenced="table1",
            columns_referenced=("col1",)
        )

        # Assertions
        assert len(rules) == 2
        assert expected_rule1 in rules
        assert expected_rule2 in rules
        assert rules[expected_rule1] == (1, 1)
        assert rules[expected_rule2] == (1, 1)

        # Construct the expected command string
        expected_cmd = (
            "java -cp algorithms/bins/metanome/jars/metanome-cli-1.2-SNAPSHOT.jar:"
            "algorithms/bins/metanome/jars/SPIDER-1.2-SNAPSHOT.jar de.metanome.cli.App "
            "--algorithm de.metanome.algorithms.spider.SPIDERFile "
            "--files /path/to/csv/table1.csv /path/to/csv/table2.csv /path/to/csv/table3.csv "
            "--table-key INPUT_FILES --separator \",\" --output file:2023-01-01_12-00-00_SPIDER "
            "--header"
        )

        mock_run_cmd.assert_called_once_with(expected_cmd)

        # Ensure file operations
        mock_file.assert_called_with("results/2023-01-01_12-00-00_SPIDER_inds", mode="r")
        mock_exists.assert_called_once_with("results/2023-01-01_12-00-00_SPIDER_inds")
        mock_remove.assert_called_once_with("results/2023-01-01_12-00-00_SPIDER_inds")
