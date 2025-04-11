import pytest
import copy
from unittest.mock import Mock
from algorithms.MATILDA.constraint_graph import (
    Attribute,
    AttributeMapper,
    ConstraintGraph,
    JoinableIndexedAttributes,
)
from database.alchemy_utility import AlchemyUtility
from algorithms.MATILDA.candidate_rule_chains import CandidateRuleChains
from algorithms.MATILDA.tgd_discovery import *

@pytest.fixture
def mock_db_inspector():
    """Mock database inspector for testing."""
    inspector = Mock(spec=AlchemyUtility)
    inspector.base_name = "test_db"
    inspector.get_table_names.return_value = [0, 1]
    inspector.get_attribute_names.side_effect = lambda table: [0, 1]
    inspector.check_threshold.return_value = True
    inspector.get_join_row_count.return_value = 10
    return inspector

@pytest.fixture
def mock_constraint_graph():
    """Mock constraint graph for testing."""
    cg = ConstraintGraph()
    node1 = JoinableIndexedAttributes(
        IndexedAttribute(0, 0, 0), IndexedAttribute(1, 0, 1)
    )
    node2 = JoinableIndexedAttributes(
        IndexedAttribute(1, 0, 1), IndexedAttribute(0, 0, 0)
    )
    node3 = JoinableIndexedAttributes(
        IndexedAttribute(0, 1, 0), IndexedAttribute(1, 1, 1)
    )
    cg.add_node(node1)
    cg.add_node(node2)
    cg.add_node(node3)
    cg.add_edge(node1, node2)
    cg.add_edge(node2, node3)
    return cg

@pytest.fixture
def mock_mapper():
    """Mock AttributeMapper."""
    return AttributeMapper({0: 0, 1: 1}, {0: {0: 0}, 1: {1: 1}})

# Test for `init`
def test_init(mock_db_inspector):
    cg, mapper, jia_list = init(mock_db_inspector)
    assert cg is not None, "Constraint graph should not be None"
    assert mapper is not None, "Mapper should not be None"
    assert isinstance(jia_list, list), "JIA list should be a list"
    assert len(jia_list) > 0, "JIA list should not be empty"

# Test for `dfs`
def test_dfs(mock_constraint_graph, mock_mapper, mock_db_inspector):
    def mock_pruning_prediction(path, mapper, db_inspector):
        return True

    visited = set()
    candidate_rules = list(
        dfs(
            graph=mock_constraint_graph,
            start_node=None,
            pruning_prediction=mock_pruning_prediction,
            db_inspector=mock_db_inspector,
            mapper=mock_mapper,
            visited=visited,
            max_table=3,
            max_vars=4,
        )
    )
    assert len(candidate_rules) > 0, "DFS should yield candidate rules."

# Test for `prediction`
def test_prediction(mock_mapper, mock_db_inspector):
    candidate_rule = [
        JoinableIndexedAttributes(
            IndexedAttribute(0, 0, 0), IndexedAttribute(1, 0, 1)
        )
    ]
    result = prediction(candidate_rule, mock_mapper, mock_db_inspector)
    assert result >= 0, "Prediction should return a non-negative value."

def test_path_pruning(mock_mapper, mock_db_inspector):
    candidate_rule = [
        JoinableIndexedAttributes(
            IndexedAttribute(0, 0, 0), IndexedAttribute(1, 0, 1)
        )
    ]
    result = path_pruning(candidate_rule, mock_mapper, mock_db_inspector)
    assert isinstance(result, bool), "Path pruning should return a boolean value."

def test_split_pruning(mock_mapper, mock_db_inspector):
    candidate_rule = [
        JoinableIndexedAttributes(
            IndexedAttribute(0, 0, 0), IndexedAttribute(1, 0, 1)
        )
    ]
    body = {(0, 0)}
    head = {(1, 0)}
    result, support, confidence = split_pruning(candidate_rule, body, head, mock_db_inspector, mock_mapper)
    assert isinstance(result, bool), "Split pruning should return a boolean value."
    assert isinstance(support, (float, int)) and support >= 0, "Support should be non-negative."
    assert isinstance(confidence, (float, int)) and confidence >= 0, "Confidence should be non-negative."


# Additional helper tests
def test_duplicate_test():
    tgds = ["TGD1", "TGD2", "TGD1"]
    with pytest.raises(ValueError):
        duplicate_test(tgds)

# Run tests
def run_tests():
    pytest.main([__file__])

if __name__ == "__main__":
    run_tests()
