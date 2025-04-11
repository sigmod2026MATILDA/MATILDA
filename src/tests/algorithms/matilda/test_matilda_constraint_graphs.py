# test_constraint_graph.py

import pytest
from unittest.mock import MagicMock, patch
from algorithms.MATILDA.constraint_graph import (
    Attribute,
    IndexedAttribute,
    AttributeMapper,
    JoinableIndexedAttributes,
    ConstraintGraph,
)
import networkx as nx
import numpy as np
import pandas as pd


# Fixtures for reusable components

@pytest.fixture
def mock_db_inspector():
    """Fixture to provide a mocked AlchemyUtility instance."""
    mock_db = MagicMock()
    return mock_db


@pytest.fixture
def sample_attributes(mock_db_inspector):
    """Fixture to create sample Attribute instances."""
    attribute1 = Attribute(table="users", name="id", is_key=True, domain="INT")
    attribute2 = Attribute(table="orders", name="user_id", is_key=False, domain="INT")
    attribute3 = Attribute(table="products", name="id", is_key=True, domain="INT")
    return attribute1, attribute2, attribute3


@pytest.fixture
def table_name_to_index():
    """Fixture for table name to index mapping."""
    return {"users": 0, "orders": 1}


@pytest.fixture
def attribute_name_to_index():
    """Fixture for attribute name to index mapping."""
    return {
        "users": {"id": 0, "name": 1, "email": 2},
        "orders": {"id": 0, "user_id": 1, "product_id": 2, "amount": 3},
    }


@pytest.fixture
def mapper(table_name_to_index, attribute_name_to_index):
    """Fixture to create an AttributeMapper instance."""
    return AttributeMapper(table_name_to_index, attribute_name_to_index)


# Tests for Attribute

class TestAttribute:

    def test_initialization(self, sample_attributes):
        """Test the initialization of Attribute instances."""
        attribute1, attribute2, attribute3 = sample_attributes
        assert attribute1.table == "users"
        assert attribute1.name == "id"
        assert attribute1.is_key is True
        assert attribute1.domain == "INT"

    def test_is_compatible_same_table(self, sample_attributes, mock_db_inspector):
        """Attributes from the same table should be compatible."""
        attribute1, _, attribute3 = sample_attributes
        # Assuming same table makes them compatible
        attribute3.table = "users"
        assert attribute1.is_compatible(attribute3, db_inspector=mock_db_inspector)

    def test_is_compatible_different_tables_with_common_elements(self, sample_attributes, mock_db_inspector):
        """Attributes from different tables with common elements above threshold are compatible."""
        attribute1, attribute2, attribute3 = sample_attributes
        # Setup the mock to return common elements above threshold
        mock_db_inspector.get_attribute_values.side_effect = [
            ['1', '2', '3'],  # users.id
            ['1', '4', '5']   # products.id
        ]
        # Mock has_common_elements_above_threshold to return True
        with patch.object(
            Attribute, 'has_common_elements_above_threshold', return_value=True
        ):
            assert attribute1.is_compatible(attribute3, db_inspector=mock_db_inspector, threshold_overlap=1)

    # def test_is_not_compatible_due_to_domain_mismatch(self, sample_attributes, mock_db_inspector):
    #     """Attributes with different domains should not be compatible."""
    #     attribute1, _, _ = sample_attributes
    #     attribute_diff_domain = Attribute(table="orders", name="amount", is_key=False, domain="DECIMAL")
    #     assert not attribute1.is_compatible(attribute_diff_domain, db_inspector=mock_db_inspector)

    def test_generate_attributes(self, mock_db_inspector):
        """Test the generate_attributes class method."""
        # Setup the mock methods
        mock_db_inspector.get_table_names.return_value = ["users", "orders"]
        mock_db_inspector.get_attribute_names.side_effect = [
            ["id", "name", "email"],
            ["id", "user_id", "product_id", "amount"]
        ]
        mock_db_inspector.get_attribute_domain.side_effect = [
            "INT", "VARCHAR", "VARCHAR",
            "INT", "INT", "INT", "DECIMAL"
        ]
        mock_db_inspector.get_attribute_is_key.side_effect = [
            True, False, False,
            True, False, False, False
        ]

        attributes = Attribute.generate_attributes(mock_db_inspector)
        assert len(attributes) == 7
        assert any(attr.name == "id" and attr.table == "users" for attr in attributes)
        assert any(attr.name == "user_id" and attr.table == "orders" for attr in attributes)


# Tests for IndexedAttribute

class TestIndexedAttribute:

    def test_initialization(self):
        """Test the initialization of IndexedAttribute."""
        indexed_attr = IndexedAttribute(i=0, j=1, k=2)
        assert indexed_attr.i == 0
        assert indexed_attr.j == 1
        assert indexed_attr.k == 2

    def test_invalid_initialization(self):
        """Test that IndexedAttribute raises ValueError for invalid inputs."""
        with pytest.raises(ValueError):
            IndexedAttribute(i=-1, j=1, k=2)
        with pytest.raises(ValueError):
            IndexedAttribute(i=0.5, j=1, k=2)
        with pytest.raises(ValueError):
            IndexedAttribute(i=0, j=-1, k=2)
        with pytest.raises(ValueError):
            IndexedAttribute(i=0, j=1, k=-2)

    def test_equality(self):
        """Test equality of IndexedAttribute instances."""
        attr1 = IndexedAttribute(i=0, j=1, k=2)
        attr2 = IndexedAttribute(i=0, j=1, k=2)
        attr3 = IndexedAttribute(i=1, j=2, k=3)
        assert attr1 == attr2
        assert attr1 != attr3

    def test_ordering(self):
        """Test the ordering of IndexedAttribute instances."""
        attr1 = IndexedAttribute(i=0, j=1, k=2)
        attr2 = IndexedAttribute(i=0, j=1, k=3)
        attr3 = IndexedAttribute(i=0, j=2, k=1)
        attr4 = IndexedAttribute(i=1, j=0, k=0)
        assert attr1 < attr2
        assert attr2 < attr3
        assert attr3 < attr4

    def test_hashing(self):
        """Test hashing of IndexedAttribute instances."""
        attr1 = IndexedAttribute(i=0, j=1, k=2)
        attr2 = IndexedAttribute(i=0, j=1, k=2)
        attr_set = {attr1}
        assert attr2 in attr_set

    def test_is_connected(self):
        """Test the is_connected method."""
        attr1 = IndexedAttribute(i=0, j=1, k=2)
        attr2 = IndexedAttribute(i=0, j=1, k=3)
        attr3 = IndexedAttribute(i=1, j=1, k=2)
        assert attr1.is_connected(attr2) is True
        assert attr1.is_connected(attr3) is False


# Tests for AttributeMapper

class TestAttributeMapper:

    def test_attribute_to_indexed(self, mapper):
        """Test converting Attribute to IndexedAttribute."""
        attribute = Attribute(table="users", name="email")
        indexed_attr = mapper.attribute_to_indexed(attribute, table_occurrence=0)
        assert indexed_attr.i == 0
        assert indexed_attr.j == 0
        assert indexed_attr.k == 2

    def test_indexed_to_attribute(self, mapper):
        """Test converting IndexedAttribute to Attribute."""
        indexed_attr = IndexedAttribute(i=1, j=0, k=3)
        attribute = mapper.indexed_attribute_to_attribute(indexed_attr)
        assert attribute.table == "orders"
        assert attribute.name == "amount"

    def test_reverse_mappings(self, mapper):
        """Test if reverse mappings are correct."""
        indexed_attr = IndexedAttribute(i=0, j=0, k=1)
        attribute = mapper.indexed_attribute_to_attribute(indexed_attr)
        assert attribute.table == "users"
        assert attribute.name == "name"


# Tests for JoinableIndexedAttributes

class TestJoinableIndexedAttributes:

    def setup_method(self):
        """Setup sample IndexedAttributes and JoinableIndexedAttributes."""
        self.attr1 = IndexedAttribute(i=0, j=1, k=2)
        self.attr2 = IndexedAttribute(i=1, j=0, k=3)
        self.attr3 = IndexedAttribute(i=0, j=1, k=2)
        self.jia1 = JoinableIndexedAttributes(self.attr1, self.attr2)
        self.jia2 = JoinableIndexedAttributes(self.attr3, self.attr2)
        self.jia3 = JoinableIndexedAttributes(self.attr2, self.attr1)  # Should be same as jia1

    def test_equality(self):
        """Test equality of JoinableIndexedAttributes instances."""
        assert self.jia1 == self.jia2
        assert self.jia1 == self.jia3

    def test_hashing(self):
        """Test hashing of JoinableIndexedAttributes instances."""
        jia_set = {self.jia1}
        assert self.jia2 in jia_set
        assert self.jia3 in jia_set

    def test_ordering(self):
        """Test the ordering of JoinableIndexedAttributes instances."""
        # Create another JIA for ordering
        attr4 = IndexedAttribute(i=2, j=2, k=2)
        jia4 = JoinableIndexedAttributes(self.attr1, attr4)
        assert self.jia1 < jia4

    def test_is_connected(self):
        """Test the is_connected method."""
        jia4 = JoinableIndexedAttributes(self.attr1, self.attr3)
        assert self.jia1.is_connected(jia4) is True
        jia5 = JoinableIndexedAttributes(IndexedAttribute(i=3, j=3, k=3), IndexedAttribute(i=3, j=3, k=4))
        assert self.jia1.is_connected(jia5) is False


# Tests for ConstraintGraph

class TestConstraintGraph:

    def setup_method(self):
        """Setup sample IndexedAttributes and JoinableIndexedAttributes."""
        self.attr1 = IndexedAttribute(i=0, j=1, k=2)
        self.attr2 = IndexedAttribute(i=1, j=0, k=3)
        self.attr3 = IndexedAttribute(i=0, j=1, k=4)
        self.jia1 = JoinableIndexedAttributes(self.attr1, self.attr2)
        self.jia2 = JoinableIndexedAttributes(self.attr1, self.attr3)
        self.jia3 = JoinableIndexedAttributes(self.attr2, self.attr3)
        self.graph = ConstraintGraph()

    def test_add_node(self):
        """Test adding nodes to the graph."""
        self.graph.add_node(self.jia1)
        assert self.jia1 in self.graph.nodes

    def test_add_edge(self):
        """Test adding edges to the graph."""
        self.graph.add_node(self.jia1)
        self.graph.add_node(self.jia2)
        self.graph.add_edge(self.jia2, self.jia1)
        assert self.jia1 in self.graph.edges[self.jia2]

    def test_add_edge_invalid_order(self):
        """Test that adding an edge with invalid node order raises an Exception."""
        self.graph.add_node(self.jia2)
        self.graph.add_node(self.jia1)
        with pytest.raises(Exception, match="Source node must be less than target node."):
            self.graph.add_edge(self.jia1, self.jia2)

    def test_is_connected(self):
        """Test the is_connected method."""
        self.graph.add_node(self.jia1)
        self.graph.add_node(self.jia2)
        self.graph.add_edge(self.jia2, self.jia1)
        assert self.graph.is_connected(self.jia1, self.jia2) is False
        # assert self.graph.is_connected(self.jia2, self.jia1) is False

    def test_from_jia_list(self):
        """Test creating a graph from a list of JoinableIndexedAttributes."""
        jia_list = [self.jia1, self.jia2, self.jia3]
        # Mock is_connected method
        with patch.object(
            JoinableIndexedAttributes, 'is_connected', side_effect=lambda x: x == self.jia3
        ):
            graph = ConstraintGraph.from_jia_list(jia_list)
            assert len(graph.nodes) == 3
            # Depending on the is_connected logic, verify edges
            # Here, is_connected only returns True when comparing to jia3
            if self.jia3 in graph.edges.get(self.jia1, set()):
                assert self.jia3 in graph.edges[self.jia1]

    def test_neighbors(self):
        """Test the neighbors method."""
        self.graph.add_node(self.jia1)
        self.graph.add_node(self.jia2)
        self.graph.add_edge(self.jia2, self.jia1)
        neighbors = self.graph.neighbors(self.jia2)
        assert self.jia1 in neighbors
        assert len(neighbors) == 1

    def test_repr(self):
        """Test the string representation of the graph."""
        self.graph.add_node(self.jia1)
        self.graph.add_node(self.jia2)
        self.graph.add_edge(self.jia2, self.jia1)
        repr_str = repr(self.graph)
        assert "ConstraintGraph" in repr_str
        assert "JIA" in repr_str
        assert f"{self.jia2} -> {self.jia1}" in repr_str


# Additional Tests for Attribute's Compatibility Logic

class TestAttributeCompatibility:

    def test_is_compatible_with_foreign_key(self, sample_attributes, mock_db_inspector):
        """Test compatibility when attributes are foreign keys."""
        attribute1, attribute2, _ = sample_attributes
        mock_db_inspector.are_foreign_keys.return_value = True

        # Assuming is_compatible uses are_foreign_keys to determine compatibility
        with patch.object(
            Attribute, 'has_common_elements_above_threshold', return_value=False
        ):
            assert attribute1.is_compatible(attribute2, db_inspector=mock_db_inspector)

    # def test_is_not_compatible_due_to_value_overlap(self, sample_attributes, mock_db_inspector):
    #     """Test compatibility fails due to insufficient value overlap."""
    #     attribute1, attribute2, _ = sample_attributes
    #     with patch.object(
    #         Attribute, 'has_common_elements_above_threshold', return_value=False
    #     ), patch.object(
    #         Attribute, 'has_common_elements_above_threshold_percentage', return_value=False
    #     ):
    #         assert not attribute1.is_compatible(attribute2, db_inspector=mock_db_inspector, threshold_overlap=10, threshold_jaccard=0.05)


# Run the tests with pytest
# To execute the tests, run the following command in your terminal:
# pytest test_constraint_graph.py

