# rule_comparer.py

import json
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
import sys
import os 
import sys 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src/utils/')))

from rules import (
    Rule, HornRule, TGDRule, InclusionDependency, FunctionalDependency, DenialConstraint
)

from rules import RuleIO#, is_equivalent_to_tgd, is_equivalent_to_horn

class RuleComparer:
    """
    A versatile, instance-based class for comparing sets of rules 
    (InclusionDependency, FunctionalDependency, DenialConstraint, HornRule, TGDRule).

    Features:
      - Ignores specified fields (e.g., display, accuracy) when comparing
      - Allows fuzzy matching of numeric fields (accuracy/confidence) via numeric_tolerance
      - Calculates two coverage metrics:
         1) coverage_all = match_count / total_in_file1
         2) coverage_correct_and_compatible = match_count / (# in file1 that have correct==True and compatible==True)
      - Records matched/unmatched rules in detail for debugging or reporting.
    """

    def __init__(
        self,
        ignore_fields: Optional[List[str]] = None,
        numeric_tolerance: float = 0.0,
        logger: Optional[logging.Logger] = None,
        verbosity: int = 0
    ):
        """
        :param ignore_fields: A list of string field names to ignore in comparison 
                              (e.g. ["display", "accuracy", "confidence"])
        :param numeric_tolerance: If >0, allow approximate matches for numeric fields 
                                  (like accuracy/confidence) within this tolerance
        :param logger: Optional logger to use. If None, uses __name__ logger.
        :param verbosity: Verbosity level (0=quiet, 1=info, 2=debug).
        """
        self.ignore_fields = ignore_fields or []
        self.numeric_tolerance = numeric_tolerance
        self.logger = logger if logger else logging.getLogger(__name__)
        self.verbosity = verbosity

    def _log(self, level: int, msg: str):
        """Helper to log at given level if verbosity is high enough."""
        if self.verbosity >= level:
            if level == 1:
                self.logger.info(msg)
            elif level == 2:
                self.logger.debug(msg)
            else:
                self.logger.warning(msg)

    def _equal_numeric(self, val1: float, val2: float) -> bool:
        """
        Check approximate equality of two numeric values 
        if self.numeric_tolerance > 0, else exact equality.
        """
        if self.numeric_tolerance > 0:
            return abs(val1 - val2) <= self.numeric_tolerance
        return val1 == val2

    def _compare_same_type(self, r1: Rule, r2: Rule) -> bool:
        """
        Compare two rules that are guaranteed to have the same type,
        respecting 'ignore_fields' and 'numeric_tolerance' for certain fields.
        """

        # Convert them to dicts to handle ignoring fields
        d1 = self._rule_to_filtered_dict(r1)
        d2 = self._rule_to_filtered_dict(r2)

        # Now compare the filtered dictionaries
        # For numeric fields we consider self.numeric_tolerance
        # for everything else, we do direct equality
        if type(r1) in [HornRule, TGDRule]:  
            # Field-by-field check
            for k in d1:
                if k not in d2:
                    return False
                if isinstance(d1[k], float) and isinstance(d2[k], float):
                    if not self._equal_numeric(d1[k], d2[k]):
                        return False
                else:
                    if d1[k] != d2[k]:
                        return False
            # Also check no extra keys in d2
            for k in d2:
                if k not in d1:
                    return False
            return True

        # For other rule types, let's do a direct dict comparison 
        # but approximate compare for numeric fields if present
        return self._compare_dicts_approx(d1, d2)

    def _compare_dicts_approx(self, d1: Dict[str, Any], d2: Dict[str, Any]) -> bool:
        """Compares two dictionaries field-by-field, with optional numeric tolerance."""
        if d1.keys() != d2.keys():
            return False
        for k in d1:
            v1 = d1[k]
            v2 = d2[k]
            if isinstance(v1, float) and isinstance(v2, float):
                if not self._equal_numeric(v1, v2):
                    return False
            else:
                if v1 != v2:
                    return False
        return True

    def _rule_to_filtered_dict(self, rule: Rule) -> Dict[str, Any]:
        """
        Convert a rule to dict but ignore 
        certain fields (like "display" or "accuracy") if in self.ignore_fields.
        """
        d = {}
        if isinstance(rule, (HornRule, TGDRule)):
            d["body"] = tuple(rule.body)  # these are Predicates
            d["head"] = (
                rule.head if isinstance(rule, HornRule) 
                else tuple(rule.head)  # TGDRule has a tuple for head
            )
            if "display" not in self.ignore_fields:
                d["display"] = rule.display
            if "correct" not in self.ignore_fields:
                d["correct"] = rule.correct
            if "compatible" not in self.ignore_fields:
                d["compatible"] = rule.compatible
            if isinstance(rule, TGDRule):
                if "accuracy" not in self.ignore_fields and rule.accuracy is not None:
                    d["accuracy"] = rule.accuracy
                if "confidence" not in self.ignore_fields and rule.confidence is not None:
                    d["confidence"] = rule.confidence

        elif isinstance(rule, InclusionDependency):
            d["table_dependant"] = rule.table_dependant
            d["columns_dependant"] = rule.columns_dependant
            d["table_referenced"] = rule.table_referenced
            d["columns_referenced"] = rule.columns_referenced
            if "display" not in self.ignore_fields:
                d["display"] = rule.display
            if "correct" not in self.ignore_fields:
                d["correct"] = rule.correct
            if "compatible" not in self.ignore_fields:
                d["compatible"] = rule.compatible
            if "accuracy" not in self.ignore_fields and rule.accuracy is not None:
                d["accuracy"] = rule.accuracy
            if "confidence" not in self.ignore_fields and rule.confidence is not None:
                d["confidence"] = rule.confidence

        elif isinstance(rule, FunctionalDependency):
            d["table"] = rule.table
            d["determinant"] = rule.determinant
            d["dependent"] = rule.dependent
            if "correct" not in self.ignore_fields:
                d["correct"] = rule.correct
            if "compatible" not in self.ignore_fields:
                d["compatible"] = rule.compatible

        elif isinstance(rule, DenialConstraint):
            d["table"] = rule.table
            d["conditions"] = tuple(str(c) for c in rule.conditions)
            if "correct" not in self.ignore_fields:
                d["correct"] = rule.correct
            if "compatible" not in self.ignore_fields:
                d["compatible"] = rule.compatible

        else:
            # Fallback: asdict if you want
            from dataclasses import asdict
            fallback_dict = asdict(rule)
            # remove ignored fields
            for key, val in fallback_dict.items():
                if key not in self.ignore_fields:
                    d[key] = val

        return d

    def compare_rules(self, r1: Rule, r2: Rule) -> bool:
        """
        Master method to compare two rules for equivalence or a match,
        accounting for ignoring certain fields and numeric tolerances.
        Also includes cross-type logic for HornRule<->TGDRule or 
        IND<->TGD/Horn.
        """
        # 1) If they are literally the same object
        if r1 is r2:
            return True

        # 2) If same type, rely on specialized comparison 
        if type(r1) == type(r2):
            return self._compare_same_type(r1, r2)

        # 3) Check cross-type known scenarios
        # HornRule <-> TGDRule
        if isinstance(r1, HornRule) and isinstance(r2, TGDRule):
            return self._compare_same_type(r1, r2)
        if isinstance(r1, TGDRule) and isinstance(r2, HornRule):
            return self._compare_same_type(r1, r2)
        #"TODOOOO"
        # # InclusionDependency <-> TGDRule/Horn
        # if isinstance(r1, InclusionDependency) and isinstance(r2, TGDRule):
        #     return is_equivalent_to_tgd(r1, r2)
        # if isinstance(r1, InclusionDependency) and isinstance(r2, HornRule):
        #     return is_equivalent_to_horn(r1, r2)

        # if isinstance(r2, InclusionDependency) and isinstance(r1, TGDRule):
        #     return is_equivalent_to_tgd(r2, r1)
        # if isinstance(r2, InclusionDependency) and isinstance(r1, HornRule):
        #     return is_equivalent_to_horn(r2, r1)

        # else no known cross-compare => false
        return False

    def compare_rule_sets(self, filepath1: str, filepath2: str) -> Dict[str, Any]:
        """
        Compare two sets of rules in JSON files (File 1 = reference).
        
        Stats Returned:
          - match_count: # of rules in File 1 that match something in File 2
          - total_in_file1
          - total_in_file2
          - unmatched_in_1_count
          - unmatched_in_2_count
          - matched_pairs: List of (Rule1, Rule2) pairs that matched
          - coverage_all = match_count / total_in_file1
          - total_in_file_that_are_correct_and_compatible
          - coverage_correct_and_compatible = match_count / total_in_file_that_are_correct_and_compatible
        """
        self._log(1, f"Loading rules from {filepath1} and {filepath2} ...")

        try:
            rules1 = RuleIO.load_rules_from_json(filepath1)
            rules2 = RuleIO.load_rules_from_json(filepath2)
        except Exception as e:
            self._log(0, f"Error loading rule files: {e}")
            return {
                "file1": filepath1,
                "file2": filepath2,
                "error": str(e)
            }

        total_in_file1 = len(rules1)
        total_in_file2 = len(rules2)
        self._log(1, f"File 1 has {total_in_file1} rules; File 2 has {total_in_file2} rules")

        matched_in_file1: List[Rule] = []
        unmatched_in_file1: List[Rule] = []
        matched_pairs: List[Tuple[Rule, Rule]] = []

        # 1) For each rule in file1, see if there's a match in file2
        for r1 in rules1:
            found_match = False
            for r2 in rules2:
                if self.compare_rules(r1, r2):
                    found_match = True
                    matched_in_file1.append(r1)
                    matched_pairs.append((r1, r2))
                    break
            if not found_match:
                unmatched_in_file1.append(r1)

        # 2) Identify unmatched in file2 as well
        unmatched_in_file2: List[Rule] = []
        # We consider a rule in file2 unmatched if it is NOT among 
        # the second elements in matched_pairs
        matched_file2_set = {pair[1] for pair in matched_pairs}

        for r2 in rules2:
            if r2 not in matched_file2_set:
                unmatched_in_file2.append(r2)

        match_count = len(matched_in_file1)

        # coverage_all = (# matched) / (total_in_file1)
        coverage_all = (float(match_count) / float(total_in_file1)) if total_in_file1 > 0 else 0.0

        # 3) total_in_file_that_are_correct_and_compatible
        rules1_correct_and_compatible = [
            r for r in rules1 if (getattr(r, 'correct', False) and getattr(r, 'compatible', False))
        ]
        total_correct_compatible = len(rules1_correct_and_compatible)

        # coverage_correct_and_compatible
        coverage_correct_and_compatible = (
            float(match_count) / float(total_correct_compatible)
            if total_correct_compatible > 0 else 0.0
        )

        # Build final result dict
        result = {
            "file1": filepath1,
            "file2": filepath2,
            "match_count": match_count,
            "total_in_file1": total_in_file1,
            "total_in_file2": total_in_file2,
            "unmatched_in_1_count": len(unmatched_in_file1),
            "unmatched_in_2_count": len(unmatched_in_file2),
            "coverage_all": coverage_all,
            "total_in_file_that_are_correct_and_compatible": total_correct_compatible,
            "coverage_correct_and_compatible": coverage_correct_and_compatible,
            # Uncomment below to include detailed matched/unmatched rules
            # "unmatched_in_1": [RuleIO.rule_to_dict(r) for r in unmatched_in_file1],
            # "unmatched_in_2": [RuleIO.rule_to_dict(r) for r in unmatched_in_file2],
            # "matched_pairs": [
            #     {
            #         "file1_rule": RuleIO.rule_to_dict(r1),
            #         "file2_rule": RuleIO.rule_to_dict(r2),
            #     }
            #     for (r1, r2) in matched_pairs
            # ]
        }
        return result
