import numpy as np

from typing import List, Union, Optional

from dover_lap.libs.turn import Turn, merge_turns
from dover_lap.src.label_mapping import LabelMapping
from dover_lap.src.label_voting import LabelVoting


class DOVERLap:
    @classmethod
    def combine_turns_list(
        cls,
        turns_list: List[List[Turn]],
        file_id: str,
        label_mapping: Optional[str] = "greedy",
        second_maximal: Optional[bool] = False,
        voting_method: Optional[str] = "average",
        weight_type: Optional[str] = "rank",
        dover_weight: Optional[float] = 0.1,
        custom_weight: Optional[List[str]] = None,
        gaussian_filter_std: float = 0.01,
    ) -> List[List[Turn]]:

        # Label mapping stage
        mapped_turns_list, weights = LabelMapping.get_mapped_turns_list(
            turns_list,
            file_id,
            method=label_mapping,
            second_maximal=second_maximal,
        )

        # Compute weights based on rank
        if weight_type == "rank":
            ranks = cls.__get_ranks(weights)
            weights = cls.__compute_weights(ranks, dover_weight)
        elif weight_type == "custom":
            assert isinstance(custom_weight, list)
            weights = np.array([float(x) for x in custom_weight])
        elif weight_type == "norm":
            weights /= np.linalg.norm(weights, ord=1)  # use normalized weights

        # Label voting stage
        combined_turns_list = LabelVoting.get_combined_turns(
            mapped_turns_list,
            file_id,
            voting_method,
            weights,
            gaussian_filter_std=gaussian_filter_std,
        )
        # Merge consecutive turns with the same label
        combined_turns_list = merge_turns(combined_turns_list)
        return combined_turns_list

    def __get_ranks(weights: np.array) -> np.array:

        weights /= np.linalg.norm(weights, ord=1)
        temp = weights.argsort()
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(weights)) + 1
        return ranks

    def __compute_weights(ranks: np.array, weight: float) -> np.array:

        out_weights = 1 / np.power(ranks, weight)
        out_weights /= np.linalg.norm(out_weights, ord=1)
        return out_weights
