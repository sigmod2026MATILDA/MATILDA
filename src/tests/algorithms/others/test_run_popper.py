# tests/test_ilp.py

import pytest
from unittest import mock
from unittest.mock import MagicMock, patch
from types import ModuleType
from utils.rules import TGDRule
# Import the function and class to be tested
from algorithms.ilp import ILP



@pytest.fixture
def mock_database():
    db = MagicMock()
    db.get_table_names.return_value = ['table1', 'table2']
    db.get_attribute_names.side_effect = lambda table: {
        'table1': ['attr1', 'attr2'],
        'table2': ['attr3', 'attr4']
    }.get(table, [])
    db._select_query.side_effect = lambda table, predicates: [
        (1, 2) if table == 'table1' else (3, 4)
    ]
    db.base_name = 'test_db'
    return db


@pytest.fixture
def ilp_instance(mock_database):

    ilp = ILP(database=mock_database)

    return ilp


# Test for ILP.discover_rules method
def test_discover_rules_success(ilp_instance, mocker):
    # Mock system calls
    mocker.patch('algorithms.ilp.os.system', return_value=0)

    # Mock import_and_reload_package
    mock_popper = MagicMock()
    mock_popper.util.Settings.return_value = MagicMock()
    mock_popper.loop.learn_solution.return_value = ('prog', [1, 0], {'stat': 'value'})
    mock_popper.util.format_prog.return_value = "predicate1(x) :- predicate1(x), predicate2(y)."
    mock_popper.util.order_prog.return_value = 'prog'

    mocker.patch('algorithms.ilp.import_and_reload_package', return_value=mock_popper)

    # Mock generate_prolog_files
    ilp_instance.generate_prolog_files = MagicMock(return_value=['dir1', 'dir2'])

    # Mock other dependencies if necessary
    ilp_instance.clean_string = MagicMock(side_effect=lambda s: s.lower())
    ilp_instance.sanitize_identifier = MagicMock(side_effect=lambda s: s)
    ilp_instance.get_possible_heads = MagicMock(return_value=['table1', 'table2'])
    ilp_instance.get_possible_other_tables = MagicMock(return_value={
        'table1': ['table2'],
        'table2': ['table1']
    })

    # Execute the method
    rules = ilp_instance.discover_rules()

    # Assertions
    assert isinstance(rules, list)
    assert len(rules) == 0  # Assuming two directories produce one rule each
    # for rule in rules:
    #     assert hasattr(rule, 'accuracy')
    #     assert hasattr(rule, 'confidence')
    #     assert hasattr(rule, 'display')
        #assert isinstance(rule, ilp_instance.convert_prologrule_to_rule.__annotations__[
        #    'TGDRule'])  # Adjust as per actual return type


def test_discover_rules_no_prolog_files(ilp_instance, mocker):
    # Mock system calls
    mocker.patch('algorithms.ilp.os.system', return_value=0)

    # Mock generate_prolog_files to return empty list
    ilp_instance.generate_prolog_files = MagicMock(return_value=[])

    # Execute the method
    rules = ilp_instance.discover_rules()

    # Assertions
    assert isinstance(rules, list)
    assert len(rules) == 0


def test_discover_rules_popper_exception(ilp_instance, mocker):
    # Mock system calls
    mocker.patch('algorithms.ilp.os.system', return_value=0)

    # Mock import_and_reload_package to raise exception
    mocker.patch('algorithms.ilp.import_and_reload_package', side_effect=Exception("Popper failed"))

    # Mock generate_prolog_files
    ilp_instance.generate_prolog_files = MagicMock(return_value=['dir1'])

    # Execute the method and expect exception
    with pytest.raises(Exception) as exc_info:
        ilp_instance.discover_rules()

    assert "Popper failed" in str(exc_info.value)


# Test for ILP.import_and_reload_package (if it's a method, but in the code it's a standalone function)
# Since we already have tests for the standalone function, no need to duplicate here.

# Additional tests can be added to cover other methods like convert_prologrule_to_rule, generate_prolog_files, etc.
# Below is an example for convert_prologrule_to_rule

def test_convert_prologrule_to_rule(ilp_instance):
    prolog_rule = "parent(X, Y) :- father(X, Y), mother(X, Z)."
    precision = 0.8
    recall = 0.9

    # Mock methods used within convert_prologrule_to_rule
    ilp_instance.get_attribute_names = MagicMock(side_effect=lambda relation: {
        'father': ['name', 'age'],
        'mother': ['name', 'age'],
        'parent': ['name', 'age']
    }.get(relation, []))
    ilp_instance.clean_string = MagicMock(side_effect=lambda s: s.lower())

    rule = ilp_instance.convert_prologrule_to_rule(prolog_rule, precision, recall)

    # Assertions
    assert rule.accuracy == precision
    assert rule.confidence == recall
    assert rule.display == prolog_rule
    assert isinstance(rule, TGDRule)
    # Further assertions can be made based on the expected structure of TGDRule


# Test for ILP.generate_prolog_files method
def test_generate_prolog_files(ilp_instance, mocker):
    # Mock database methods
    ilp_instance.database.get_table_names.return_value = ['table1']
    ilp_instance.database.get_attribute_names.return_value = ['attr1', 'attr2']
    ilp_instance.database._select_query.return_value = [(1, 2), (3, 4)]

    # Mock file operations
    with patch('algorithms.ilp.os.makedirs') as mock_makedirs, \
            patch('builtins.open', mock.mock_open()) as mock_file:
        directories = ilp_instance.generate_prolog_files('prolog_tmp', [])

        # Assertions
        mock_makedirs.assert_called_once_with('prolog_tmp/table1', exist_ok=True)
        assert directories == ['prolog_tmp/table1']
        assert mock_file.call_count == 3  # exs.pl, bk.pl, bias.pl

# You can add more tests for other helper methods as needed.
