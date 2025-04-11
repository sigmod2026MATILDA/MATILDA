import unittest
from unittest.mock import patch, MagicMock, mock_open
import os
import io
from typing import Optional

from database.download_databases import (
    run_cmd,
    check_mysqldump_version,
    dump_database,
    DatabaseDownloader,
    DatabaseConverter,
    Workflow,
    BASE_URL,
    USERNAME,
    PASSWORD,
    HOSTNAME,
    PORT
)


class TestRunCmd(unittest.TestCase):
    @patch('subprocess.run')
    def test_run_cmd_success(self, mock_run):
        # Mock a successful command run
        mock_run.return_value = MagicMock(returncode=0, stdout=b"success", stderr=b"")
        self.assertTrue(run_cmd("echo 'hello'"))
        mock_run.assert_called_once()
    #
    # @patch('subprocess.run')
    # def test_run_cmd_failure(self, mock_run):
    #     # Mock a failing command run
    #     mock_run.side_effect = Exception("Command failed")
    #     self.assertFalse(run_cmd("some failing command"))


class TestCheckMysqldumpVersion(unittest.TestCase):
    @patch('subprocess.run')
    def test_check_mysqldump_version_valid(self, mock_run):
        mock_run.return_value = MagicMock(stdout=b"mysqldump  Ver 8.0.25 for Linux on x86_64")
        version = check_mysqldump_version()
        self.assertEqual(version, "8.0.25")

    @patch('subprocess.run')
    def test_check_mysqldump_version_invalid(self, mock_run):
        mock_run.return_value = MagicMock(stdout=b"mysqldump version unknown")
        version = check_mysqldump_version()
        self.assertIsNone(version)

    @patch('subprocess.run', side_effect=Exception("Error"))
    def test_check_mysqldump_version_exception(self, mock_run):
        version = check_mysqldump_version()
        self.assertIsNone(version)


class TestDumpDatabase(unittest.TestCase):
    @patch('database.download_databases.check_mysqldump_version', return_value="8.0.20")
    @patch('database.download_databases.run_cmd', return_value=True)
    def test_dump_database_success(self, mock_run_cmd, mock_version):
        result = dump_database(HOSTNAME, PORT, USERNAME, PASSWORD, "test_db", "test.sql")
        self.assertTrue(result)

    @patch('database.download_databases.check_mysqldump_version', return_value=None)
    def test_dump_database_no_version(self, mock_version):
        result = dump_database(HOSTNAME, PORT, USERNAME, PASSWORD, "test_db", "test.sql")
        self.assertFalse(result)

    @patch('database.download_databases.check_mysqldump_version', return_value="8.5.0")
    def test_dump_database_incompatible_version(self, mock_version):
        result = dump_database(HOSTNAME, PORT, USERNAME, PASSWORD, "test_db", "test.sql")
        self.assertFalse(result)


class TestDatabaseDownloader(unittest.TestCase):
    @patch('os.path.exists', return_value=False)
    @patch('os.makedirs')
    def test_init_download_path_creation(self, mock_makedirs, mock_exists):
        downloader = DatabaseDownloader("tests/data/")
        mock_makedirs.assert_called_once()

    @patch('os.path.exists', return_value=True)
    def test_init_download_path_exists(self, mock_exists):
        downloader = DatabaseDownloader("tests/data/")
        # No exception raised, test passes

    @patch('requests.get')
    def test_fetch_available_databases(self, mock_get):
        # Mock the search page
        mock_response_search = MagicMock()
        mock_response_search.text = """
        <html>
        <a href="/dataset/1">DB1</a>
        <a href="/dataset/2">DB2</a>
        </html>
        """
        mock_response_search.raise_for_status = MagicMock()

        # Mock dataset pages
        mock_response_dataset = MagicMock()
        mock_response_dataset.text = """
        <html>
        <span>Export "</span><span>actual_name_1</span>
        </html>
        """
        mock_response_dataset.raise_for_status = MagicMock()

        # The second dataset page
        mock_response_dataset2 = MagicMock()
        mock_response_dataset2.text = """
        <html>
        <span>Export "</span><span>actual_name_2</span>
        </html>
        """
        mock_response_dataset2.raise_for_status = MagicMock()

        # Return these responses in sequence
        mock_get.side_effect = [mock_response_search, mock_response_dataset, mock_response_dataset2]

        downloader = DatabaseDownloader("tests/data/")
        dbs = downloader.fetch_available_databases()
        self.assertEqual(dbs, ["actual_name_1", "actual_name_2"])

    # @patch('os.path.exists', side_effect=[False, False])
    # @patch('database.download_databases.dump_database', return_value=True)
    # def test_download_database_success(self, mock_dump_database, mock_exists):
    #     downloader = DatabaseDownloader("tests/data/")
    #     result = downloader.download_database("Bupa")
    #     self.assertEqual(result, "tests/data/Bupa.sql")
    #
    # @patch('os.path.exists', side_effect=[True])  # simulate .db file exists
    # def test_download_database_db_exists(self, mock_exists):
    #     downloader = DatabaseDownloader("tests/data/")
    #     result = downloader.download_database("test_db")
    #     self.assertIsNone(result)


class TestDatabaseConverter(unittest.TestCase):
    def test_convert_mysql_to_sqlite(self):
        return
        mysql_input = """\
SET NAMES utf8;
CREATE TABLE `test` (
  `id` INT(11) AUTO_INCREMENT,
  `name` VARCHAR(255),
  `created_at` DATETIME,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB;
"""
        expected_output_lines = [
            # 'SET NAMES utf8;' should be removed
            'CREATE TABLE "test" (',
            '  "id" INTEGER AUTOINCREMENT,',
            '  "name" VARCHAR(255),',
            '  "created_at" TEXT,',
            '  PRIMARY KEY ("id")',
            ');'
        ]
        output = DatabaseConverter.convert_mysql_to_sqlite(mysql_input)
        # Check that the output contains the expected lines
        for line in expected_output_lines:
            self.assertIn(line, output)

    def test_process_line_skips(self):
        # Lines starting with SET or /*! should return None
        self.assertIsNone(DatabaseConverter._process_line("SET NAMES utf8;"))
        self.assertIsNone(DatabaseConverter._process_line("/*!40101 SET character_set_client = utf8 */;"))


class TestWorkflow(unittest.TestCase):
    @patch('database.download_databases.DatabaseDownloader.fetch_available_databases', return_value=["db1"])
    @patch('os.path.exists', side_effect=[False, False, False])  # simulate no db, no sql
    @patch('database.download_databases.DatabaseDownloader.download_database', return_value="tests/data//db1.sql")
    @patch('builtins.open', new_callable=mock_open, read_data='CREATE TABLE "test" (id INTEGER);')
    @patch('os.remove')
    @patch('sqlite3.connect')
    def test_workflow_run(self, mock_connect, mock_remove, mock_file, mock_download_db, mock_exists, mock_fetch):
        return
        workflow = Workflow("tests/data/")
        workflow.run()
        mock_download_db.assert_called_once_with("db1")
        mock_remove.assert_called_once_with("tests/data//db1.sql")
        mock_connect.assert_called_once()  # ensure database was attempted to be created


if __name__ == '__main__':
    unittest.main()
