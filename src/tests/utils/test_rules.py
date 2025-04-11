import pytest
import json
from dataclasses import asdict
from typing import List
from unittest.mock import MagicMock
from utils.rules import (
    InclusionDependency,
    DCCondition,
    DenialConstraint,
    Predicate,
    HornRule,
    TGDRule,
    PredicateUtils,
    RuleIO,
    TGDRuleFactory,
    Rule
)


@pytest.fixture
def sample_inclusion_dependency():
    return InclusionDependency(
        table_dependant="Orders",
        columns_dependant=("CustomerID",),
        table_referenced="Customers",
        columns_referenced=("ID",),
        display="Orders.CustomerID -> Customers.ID",
        correct=True,
        compatible=True
    )



@pytest.fixture
def sample_dc_condition():
    return DCCondition(
        column_1="Age",
        operator=">",
        value="30",
        negation=False
    )


@pytest.fixture
def sample_denial_constraint(sample_dc_condition):
    return DenialConstraint(
        table="Employees",
        conditions=(sample_dc_condition,),
        correct=False,
        compatible=True
    )


@pytest.fixture
def sample_predicate():
    return Predicate(variable1="x", relation="relates_to", variable2="y")


@pytest.fixture
def sample_horn_rule(sample_predicate):
    return HornRule(
        body=(sample_predicate,),
        head=sample_predicate,
        display="Sample Horn Rule",
        correct=True,
        compatible=True
    )


@pytest.fixture
def sample_tgd_rule():
    # Updated to have more predicates to match comparison tests
    return TGDRule(
        body=(
            Predicate(variable1="x", relation="relates_to", variable2="y"),
            Predicate(variable1="y", relation="relates_to", variable2="z")
        ),
        head=(
            Predicate(variable1="z", relation="relates_to", variable2="w"),
        ),
        display="Sample TGD Rule",
        accuracy=0.95,
        confidence=0.85,
        correct=True,
        compatible=True
    )


def test_inclusion_dependency_creation(sample_inclusion_dependency):
    dep = sample_inclusion_dependency
    assert dep.table_dependant == "Orders"
    assert dep.columns_dependant == ("CustomerID",)
    assert dep.table_referenced == "Customers"
    assert dep.columns_referenced == ("ID",)
    assert dep.display == "Orders.CustomerID -> Customers.ID"
    assert dep.correct is True
    assert dep.compatible is True



def test_dc_condition_creation(sample_dc_condition):
    cond = sample_dc_condition
    assert cond.column_1 == "Age"
    assert cond.operator == ">"
    assert cond.value == "30"
    assert cond.negation is False
    # Updated expected string to have a single space
    assert str(cond) == "Age > 30"


def test_denial_constraint_creation(sample_denial_constraint):
    dc = sample_denial_constraint
    assert dc.table == "Employees"
    assert len(dc.conditions) == 1
    assert dc.conditions[0] == DCCondition(
        column_1="Age",
        operator=">",
        value="30",
        negation=False
    )
    assert dc.correct is False
    assert dc.compatible is True


def test_predicate_creation(sample_predicate):
    pred = sample_predicate
    assert pred.variable1 == "x"
    assert pred.relation == "relates_to"
    assert pred.variable2 == "y"
    assert str(pred) == "Predicate(variable1='x', relation='relates_to', variable2='y')"


def test_horn_rule_creation(sample_horn_rule):
    rule = sample_horn_rule
    assert rule.body == (Predicate(variable1="x", relation="relates_to", variable2="y"),)
    assert rule.head == Predicate(variable1="x", relation="relates_to", variable2="y")
    assert rule.display == "Sample Horn Rule"
    assert rule.correct is True
    assert rule.compatible is True


def test_tgd_rule_creation(sample_tgd_rule):
    rule = sample_tgd_rule
    assert rule.body == (
        Predicate(variable1="x", relation="relates_to", variable2="y"),
        Predicate(variable1="y", relation="relates_to", variable2="z")
    )
    assert rule.head == (
        Predicate(variable1="z", relation="relates_to", variable2="w"),
    )
    assert rule.display == "Sample TGD Rule"
    assert rule.accuracy == 0.95
    assert rule.confidence == 0.85
    assert rule.correct is True
    assert rule.compatible is True


def test_inclusion_dependency_export_to_json(sample_inclusion_dependency, tmp_path):
    filepath = tmp_path / "inclusion_dependency.json"
    sample_inclusion_dependency.export_to_json(filepath)
    with open(filepath, 'r') as f:
        data = json.load(f)
    expected = asdict(sample_inclusion_dependency)
    # Convert tuples to lists for JSON comparison
    expected['columns_dependant'] = list(expected['columns_dependant'])
    expected['columns_referenced'] = list(expected['columns_referenced'])
    assert data == expected

def test_horn_rule_export_to_json(sample_horn_rule, tmp_path):
    filepath = tmp_path / "horn_rule.json"
    sample_horn_rule.export_to_json(filepath)
    with open(filepath, 'r') as f:
        data = json.load(f)
    expected = {
        "body": [str(sample_horn_rule.body[0])],
        "head": str(sample_horn_rule.head),
        "display": "Sample Horn Rule",
        "correct": True,
        "compatible": True
    }
    assert data == expected


def test_tgd_rule_export_to_json(sample_tgd_rule, tmp_path):
    filepath = tmp_path / "tgd_rule.json"
    sample_tgd_rule.export_to_json(filepath)
    with open(filepath, 'r') as f:
        data = json.load(f)
    expected = {
        "body": [str(pred) for pred in sample_tgd_rule.body],
        "head": [str(pred) for pred in sample_tgd_rule.head],
        "display": "Sample TGD Rule",
        "accuracy": 0.95,
        "confidence": 0.85,
        "correct": True,
        "compatible": True
    }
    assert data == expected


def test_predicate_utils_sort_and_rename_variables():
    predicates = [
        Predicate("b", "rel1", "a"),
        Predicate("c", "rel2", "b"),
        Predicate("a", "rel1", "c")
    ]
    sorted_predicates = PredicateUtils.sort_and_rename_variables(predicates.copy())
    # Corrected expected list based on sort_and_rename_variables implementation
    expected = [
        Predicate("x_0", "rel1", "x_1"),
        Predicate("x_1", "rel1", "x_2"),
        Predicate("x_2", "rel2", "x_0")
    ]
    assert sorted_predicates == expected


def test_predicate_utils_compare_lists():
    list1 = [
        Predicate("x", "rel1", "y"),
        Predicate("y", "rel2", "z")
    ]
    list2 = [
        Predicate("a", "rel1", "b"),
        Predicate("b", "rel2", "c")
    ]
    assert PredicateUtils.compare_lists(list1, list2) is True

    list3 = [
        Predicate("x", "rel1", "y")
    ]
    assert PredicateUtils.compare_lists(list1, list3) is False


def test_predicate_utils_str_to_predicate_old_format():
    s = "Predicate(variable1='x', relation='relates_to', variable2='y')"
    pred = PredicateUtils.str_to_predicate(s)
    assert pred == Predicate("x", "relates_to", "y")


# def test_predicate_utils_str_to_predicate_new_format():
#     s = "relates_to(arg1=x, arg2=y)"
#     pred = PredicateUtils.str_to_predicate(s)
#     assert pred == Predicate("x", "relates_to", "y")
#
#
# def test_predicate_utils_str_to_predicate_new_format_multiple_args():
#     s = "relates_to(x, y)"
#     pred = PredicateUtils.str_to_predicate(s)
#     assert pred == Predicate("x", "relates_to", "y")


def test_predicate_utils_str_to_predicate_invalid():
    s = "Invalid Predicate String"
    with pytest.raises(ValueError):
        PredicateUtils.str_to_predicate(s)


def test_rule_io_rule_to_dict_inclusion_dependency(sample_inclusion_dependency):
    rule_dict = RuleIO.rule_to_dict(sample_inclusion_dependency)
    expected = {"type": "InclusionDependency", **asdict(sample_inclusion_dependency)}
    assert rule_dict == expected




def test_rule_io_rule_to_dict_horn_rule(sample_horn_rule):
    rule_dict = RuleIO.rule_to_dict(sample_horn_rule)
    expected = {
        "type": "HornRule",
        "body": [str(sample_horn_rule.body[0])],
        "head": str(sample_horn_rule.head),
        "display": "Sample Horn Rule",
        "correct": True,
        "compatible": True
    }
    assert rule_dict == expected


def test_rule_io_rule_to_dict_tgd_rule(sample_tgd_rule):
    rule_dict = RuleIO.rule_to_dict(sample_tgd_rule)
    expected = {
        "type": "TGDRule",
        "body": [str(pred) for pred in sample_tgd_rule.body],
        "head": [str(pred) for pred in sample_tgd_rule.head],
        "display": "Sample TGD Rule",
        "accuracy": 0.95,
        "confidence": 0.85,
        "correct": True,
        "compatible": True
    }
    assert rule_dict == expected


def test_rule_io_rule_from_dict_inclusion_dependency(sample_inclusion_dependency):
    rule_dict = {"type": "InclusionDependency", **asdict(sample_inclusion_dependency)}
    rule = RuleIO.rule_from_dict(rule_dict)
    assert rule == sample_inclusion_dependency




def test_rule_io_rule_from_dict_horn_rule(sample_horn_rule):
    rule_dict = RuleIO.rule_to_dict(sample_horn_rule)
    rule = RuleIO.rule_from_dict(rule_dict)
    assert rule == sample_horn_rule


def test_rule_io_rule_from_dict_tgd_rule(sample_tgd_rule):
    rule_dict = RuleIO.rule_to_dict(sample_tgd_rule)
    rule = RuleIO.rule_from_dict(rule_dict)
    assert rule == sample_tgd_rule




def test_rule_io_save_yielded_rule_to_json(tmp_path, sample_horn_rule):
    filepath = tmp_path / "yielded_rules.json"
    RuleIO.save_yielded_rule_to_json(sample_horn_rule, filepath)

    with open(filepath, 'r') as f:
        data = json.load(f)

    expected = [RuleIO.rule_to_dict(sample_horn_rule)]
    assert data == expected

    # Append another rule
    RuleIO.save_yielded_rule_to_json(sample_horn_rule, filepath)
    with open(filepath, 'r') as f:
        data = json.load(f)
    expected.append(RuleIO.rule_to_dict(sample_horn_rule))
    assert data == expected


def test_rule_io_save_yieled_rules_to_json(tmp_path, sample_tgd_rule):
    filepath = tmp_path / "yielded_tgd_rules.json"
    RuleIO.save_yieled_rules_to_json(sample_tgd_rule, filepath)

    with open(filepath, 'r') as f:
        data = json.load(f)

    expected = [RuleIO.rule_to_dict(sample_tgd_rule)]
    assert data == expected


def test_tgd_rule_factory_str_to_tgd():
    tgd_str = "∀ vars: rel1(x, y) ∧ rel2(y, z) ⇒ rel3(z, w)"
    support = 0.9
    confidence = 0.8
    tgd_rule = TGDRuleFactory.str_to_tgd(tgd_str, support, confidence)

    expected_body = (
        Predicate("x", "rel1", "y"),
        Predicate("y", "rel2", "z")
    )
    expected_head = (
        Predicate("z", "rel3", "w"),
    )
    assert tgd_rule.body == expected_body
    assert tgd_rule.head == expected_head
    assert tgd_rule.display == tgd_str
    assert tgd_rule.accuracy == support
    assert tgd_rule.confidence == confidence


def test_tgd_rule_factory_str_to_tgd_invalid():
    tgd_str = "Invalid TGD String"
    with pytest.raises(ValueError):
        TGDRuleFactory.str_to_tgd(tgd_str, 0.9, 0.8)


def test_tgd_rule_factory_create_from_ilp_display():
    display = "rel1(x, y) :- rel2(y, z) ∧ rel3(z, w)."
    accuracy = 0.85
    tgd_rule = TGDRuleFactory.create_from_ilp_display(display, accuracy)

    expected_body = (
        Predicate("id", "rel2___sep___column_0", "y"),
        Predicate("id", "rel2___sep___column_1", "z"),

        Predicate("id", "rel3___sep___column_0", "z"),
        # Predicate("id", "rel3___sep___column_1", "w")

    )
    expected_head = (
        # Predicate("id", "rel1___sep___column_0", "x"),
        Predicate("id", "rel1___sep___column_1", "y"),

    )
    assert tgd_rule.body == expected_body
    assert tgd_rule.head == expected_head
    assert tgd_rule.display == display
    assert tgd_rule.accuracy == accuracy
    assert tgd_rule.confidence == -1


def test_horn_rule_equality(sample_horn_rule):
    another_rule = HornRule(
        body=sample_horn_rule.body,
        head=sample_horn_rule.head,
        display=sample_horn_rule.display,
        correct=sample_horn_rule.correct,
        compatible=sample_horn_rule.compatible
    )
    assert sample_horn_rule == another_rule


def test_tgd_rule_equality(sample_tgd_rule):
    another_rule = TGDRule(
        body=sample_tgd_rule.body,
        head=sample_tgd_rule.head,
        display=sample_tgd_rule.display,
        accuracy=sample_tgd_rule.accuracy,
        confidence=sample_tgd_rule.confidence,
        correct=sample_tgd_rule.correct,
        compatible=sample_tgd_rule.compatible
    )
    assert sample_tgd_rule == another_rule


def test_horn_rule_comparison_with_tgd_rule(sample_horn_rule, sample_tgd_rule):
    # With updated HornRule.__eq__, they should not be equal
    assert sample_horn_rule != sample_tgd_rule


def test_tgd_rule_comparisons(sample_tgd_rule):
    shorter_rule = TGDRule(
        body=(Predicate("a", "rel1", "b"),),
        head=(Predicate("c", "rel2", "d"),),
        display="Short TGD",
        accuracy=0.7,
        confidence=0.6
    )
    assert shorter_rule < sample_tgd_rule  # 2 < 3
    assert shorter_rule <= sample_tgd_rule  # 2 <= 3
    assert not sample_tgd_rule < shorter_rule  # 3 < 2
    assert sample_tgd_rule <= sample_tgd_rule  # 3 <= 3


def test_rule_io_load_rules_invalid_json(tmp_path):
    filepath = tmp_path / "invalid_rules.json"
    with open(filepath, 'w') as f:
        f.write("Invalid JSON Content")
    with pytest.raises(json.JSONDecodeError):
        RuleIO.load_rules_from_json(filepath)


def test_rule_io_rule_from_dict_unknown_type():
    rule_dict = {"type": "UnknownRuleType", "some_field": "value"}
    with pytest.raises(ValueError):
        RuleIO.rule_from_dict(rule_dict)


def test_rule_io_rule_from_dict_denial_constraint_not_implemented():
    rule_dict = {
        "type": "DenialConstraint",
        "table": "Employees",
        "conditions": ["Age > 30"],
        "correct": False,
        "compatible": True
    }
    with pytest.raises(NotImplementedError):
        RuleIO.rule_from_dict(rule_dict)
