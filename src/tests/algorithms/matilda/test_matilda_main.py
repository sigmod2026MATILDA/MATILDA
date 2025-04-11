import pytest
from unittest.mock import MagicMock, patch

from algorithms.matilda import MATILDA
from utils.rules import Rule, TGDRuleFactory


@pytest.fixture
def mock_database():
    """Fixture to create a mock database inspector."""
    return MagicMock(name='DatabaseInspector')


@pytest.fixture
def matilda_instance(mock_database):
    """Fixture to create an instance of MATILDA with a mock database."""
    return MATILDA(database=mock_database)


@patch('algorithms.matilda.instantiate_tgd')
@patch('algorithms.matilda.TGDRuleFactory.str_to_tgd')
@patch('algorithms.matilda.split_pruning')
@patch('algorithms.matilda.split_candidate_rule')
@patch('algorithms.matilda.dfs')
@patch('algorithms.matilda.init')
def test_discover_rules_no_jia_list(
    mock_init,
    mock_dfs,
    mock_split_candidate_rule,
    mock_split_pruning,
    mock_str_to_tgd,
    mock_instantiate_tgd,
    matilda_instance
):
    """
    Test discover_rules when jia_list is empty.
    Expect the generator to yield nothing.
    """
    # Setup the mock for init to return empty jia_list
    mock_init.return_value = (MagicMock(), MagicMock(), [])

    # Call discover_rules
    generator = matilda_instance.discover_rules()

    # Convert generator to list to exhaust it
    results = list(generator)

    # Assert that no rules are yielded
    assert len(results) == 0

    # Ensure init was called with correct parameters
    mock_init.assert_called_once_with(
        matilda_instance.db_inspector,
        max_nb_occurrence=3
    )

    # Ensure dfs was not called since jia_list is empty
    mock_dfs.assert_not_called()


import pytest
from unittest.mock import MagicMock, patch

from algorithms.matilda import MATILDA
from utils.rules import Rule, TGDRuleFactory


@pytest.fixture
def mock_database():
    """Fixture to create a mock database inspector."""
    return MagicMock(name='DatabaseInspector')


@pytest.fixture
def matilda_instance(mock_database):
    """Fixture to create an instance of MATILDA with a mock database."""
    return MATILDA(database=mock_database)


def test_init_method(matilda_instance, mock_database):
    """
    Test the initialization of MATILDA class.
    """
    # MATILDA's __init__ only sets db_inspector and settings
    assert matilda_instance.db_inspector == mock_database
    assert matilda_instance.settings == {}

    # Initialize with settings
    settings = {'nb_occurrence': 5}
    matilda_with_settings = MATILDA(database=mock_database, settings=settings)
    assert matilda_with_settings.settings == settings


@patch('algorithms.matilda.instantiate_tgd')
@patch('algorithms.matilda.TGDRuleFactory.str_to_tgd')
@patch('algorithms.matilda.split_pruning')
@patch('algorithms.matilda.split_candidate_rule')
@patch('algorithms.matilda.dfs')
@patch('algorithms.matilda.init')
def test_discover_rules_split_pruning_false(
    mock_init,
    mock_dfs,
    mock_split_candidate_rule,
    mock_split_pruning,
    mock_str_to_tgd,
    mock_instantiate_tgd,
    matilda_instance
):
    """
    Test discover_rules when split_pruning returns res=False.
    Expect such candidate rules to be skipped.
    """
    # Setup the mock for init
    mock_cg = MagicMock(name='cg')
    mock_mapper = MagicMock(name='mapper')
    mock_jia_list = ['jia1']
    mock_init.return_value = (mock_cg, mock_mapper, mock_jia_list)

    # Setup the mock for dfs to return one candidate rule
    candidate_rule = MagicMock(name='CandidateRule')
    mock_dfs.return_value = [candidate_rule]

    # Setup the mock for split_candidate_rule
    split = [('body', 'head')]
    mock_split_candidate_rule.return_value = split

    # Setup the mock for split_pruning to return res=False
    mock_split_pruning.return_value = (False, 10, 0.8)

    # Call discover_rules
    generator = matilda_instance.discover_rules()

    # Convert generator to list to exhaust it
    results = list(generator)

    # Assert that no rules are yielded since split_pruning returned False
    assert len(results) == 0

    # Ensure str_to_tgd was not called
    mock_str_to_tgd.assert_not_called()


@patch('algorithms.matilda.instantiate_tgd')
@patch('algorithms.matilda.TGDRuleFactory.str_to_tgd')
@patch('algorithms.matilda.split_pruning')
@patch('algorithms.matilda.split_candidate_rule')
@patch('algorithms.matilda.dfs')
@patch('algorithms.matilda.init')
def test_discover_rules_invalid_splits(
    mock_init,
    mock_dfs,
    mock_split_candidate_rule,
    mock_split_pruning,
    mock_str_to_tgd,
    mock_instantiate_tgd,
    matilda_instance
):
    """
    Test discover_rules with invalid splits (empty body, empty head, multiple heads).
    Expect such splits to be skipped.
    """
    # Setup the mock for init
    mock_cg = MagicMock(name='cg')
    mock_mapper = MagicMock(name='mapper')
    mock_jia_list = ['jia1']
    mock_init.return_value = (mock_cg, mock_mapper, mock_jia_list)

    # Setup the mock for dfs to return one candidate rule
    candidate_rule = MagicMock(name='CandidateRule')
    mock_dfs.return_value = [candidate_rule]

    # Setup the mock for split_candidate_rule to return invalid splits
    invalid_splits = [
        ([], 'head1'),                    # Empty body
        (['body1'], []),                  # Empty head
        (['body2'], ['head2a', 'head2b']) # Multiple heads
    ]
    mock_split_candidate_rule.return_value = invalid_splits

    # Call discover_rules
    generator = matilda_instance.discover_rules()

    # Convert generator to list to exhaust it
    results = list(generator)

    # Assert that no rules are yielded due to invalid splits
    assert len(results) == 0

    # Ensure split_pruning and str_to_tgd were not called
    mock_split_pruning.assert_not_called()
    mock_str_to_tgd.assert_not_called()
