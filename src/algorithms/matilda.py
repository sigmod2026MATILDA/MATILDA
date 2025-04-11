from typing import Generator, Optional

from algorithms.base_algorithm import BaseAlgorithm
from algorithms.MATILDA.tgd_discovery import (
    init,
    dfs,
    path_pruning,
    split_candidate_rule,
    split_pruning,
    instantiate_tgd,
)
from utils.rules import Rule, TGDRuleFactory


class MATILDA(BaseAlgorithm):
    """
    MATILDA algorithm for discovering tuple-generating dependencies (TGDs) in a database.
    """

    def __init__(self, database: object, settings: Optional[dict] = None):
        """
        Initialize the MATILDA algorithm with a database inspector and optional settings.

        :param database: The database inspector object.
        :param settings: Optional settings for the algorithm.
        """
        self.db_inspector = database
        self.settings = settings or {}

    def discover_rules(self, **kwargs) -> Generator[Rule, None, None]:
        """
        Discover TGDRules based on the provided database and settings.

        :param kwargs: Optional parameters to override default settings.
            - nb_occurrence (int): Minimum number of occurrences for a rule to be considered.
            - max_table (int): Maximum number of tables involved in a rule.
            - max_vars (int): Maximum number of variables in a rule.
        :return: A generator yielding discovered TGDRules.
        """
        nb_occurrence = kwargs.get("nb_occurrence", self.settings.get("nb_occurrence", 3))
        max_table = kwargs.get("max_table", self.settings.get("max_table", 3))
        max_vars = kwargs.get("max_vars", self.settings.get("max_vars", 6))
        results_path = kwargs.get("results_dir", self.settings.get("results_dir", None))
        # create a results folder if it does not exist
        if results_path:
            import os
            os.makedirs(results_path, exist_ok=True)

        cg, mapper, jia_list = init(
            self.db_inspector,
            max_nb_occurrence=nb_occurrence,
            results_path=results_path
        )

        if not jia_list:
            return

        for candidate_rule in dfs(
            cg,
            None,
            path_pruning,
            self.db_inspector,
            mapper,
            max_table=max_table,
            max_vars=max_vars,
        ):
            if not candidate_rule:
                continue

            splits = split_candidate_rule(candidate_rule)
            for body, head in splits:
                if not body or not head or len(head) != 1:
                    continue

                res, support, confidence = split_pruning(
                    candidate_rule, body, head, self.db_inspector, mapper
                )

                if not res:
                    debug = True
                    if debug:
                        print("removed")
                        a = instantiate_tgd(candidate_rule, (body, head), mapper)
                    continue

                tgd = instantiate_tgd(candidate_rule, (body, head), mapper)
                yield TGDRuleFactory.str_to_tgd(tgd, support, confidence)
