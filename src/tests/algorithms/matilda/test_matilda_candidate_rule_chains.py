import pytest
from algorithms.MATILDA.constraint_graph import (
    AttributeMapper,
    JoinableIndexedAttributes,
    IndexedAttribute,
)
from algorithms.MATILDA.candidate_rule_chains import CandidateRuleChains


# Mock classes
class MockAttributeMapper(AttributeMapper):
    def __init__(self):
        self.table_name_to_index = {}
        self.attribute_name_to_index = {}

    def indexed_attribute_to_attribute(self, indexed_attr):
        # Mocked mapping function
        return type("Attribute", (), {"table": f"table_{indexed_attr.i}", "name": f"attr_{indexed_attr.j}"})


@pytest.fixture
def sample_candidate_rule():
    attr1 = IndexedAttribute(i=1, j=1, k=0)
    attr2 = IndexedAttribute(i=1, j=2, k=0)
    attr3 = IndexedAttribute(i=2, j=1, k=0)
    attr4 = IndexedAttribute(i=2, j=2, k=0)

    pair1 = (attr1, attr2)
    pair2 = (attr2, attr3)
    pair3 = (attr3, attr4)

    return [pair1, pair2, pair3]


@pytest.fixture
def attribute_mapper():
    return MockAttributeMapper()


# Test initialization
def test_candidate_rule_chains_init(sample_candidate_rule):
    chains = CandidateRuleChains(sample_candidate_rule)
    assert chains.cr == sample_candidate_rule
    assert isinstance(chains.cr_chains, list)


# Test find_candidate_rule_chains
def test_find_candidate_rule_chains(sample_candidate_rule):
    chains = CandidateRuleChains(sample_candidate_rule)
    result = chains.find_candidate_rule_chains(sample_candidate_rule)
    assert isinstance(result, list)
    assert all(isinstance(chain, list) for chain in result)
    assert len(result) > 0


# Test is_directly_connected
def test_is_directly_connected(sample_candidate_rule):
    chains = CandidateRuleChains(sample_candidate_rule)

    pair1, pair2, _ = sample_candidate_rule

    assert chains.is_directly_connected(pair1, pair2, sample_candidate_rule)
    assert not chains.is_directly_connected(pair1, (IndexedAttribute(3, 3, k=0), IndexedAttribute(4, 4, k=0)), sample_candidate_rule)


# Test add_to_chain
def test_add_to_chain(sample_candidate_rule):
    chains = CandidateRuleChains(candidate_rule=sample_candidate_rule)
    chain_list = []
    chains.add_to_chain(chain_list, sample_candidate_rule[0], sample_candidate_rule)
    assert len(chain_list) == 1

# Test get_x_chains
def test_get_x_chains(sample_candidate_rule, attribute_mapper):
    chains = CandidateRuleChains(sample_candidate_rule)
    body = {(1, 1)}
    head = {(2, 2)}

    x_chains = chains.get_x_chains(body, head, attribute_mapper, select_body=True, select_head=True)
    assert isinstance(x_chains, list)
    assert len(x_chains) > 0
    assert all(isinstance(item, list) for item in x_chains)
    assert all(len(attr) == 3 for chain in x_chains for attr in chain)


# Test empty candidate rule
def test_empty_candidate_rule():
    chains = CandidateRuleChains(candidate_rule=[])
    assert chains.cr_chains == []


if __name__ == "__main__":
    pytest.main()
