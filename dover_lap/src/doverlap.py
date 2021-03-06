import numpy as np

from typing import List, Union, Optional

from dover_lap.libs.turn import Turn
from dover_lap.src.label_mapping import LabelMapping
from dover_lap.src.label_voting import LabelVoting


class DOVERLap:
    @classmethod
    def combine_turns_list(
        cls,
        turns_list: List[List[Turn]],
        file_id: str,
        label_mapping: Optional[str] = "greedy",
        sort_first: Optional[bool] = False,
        second_maximal: Optional[bool] = False,
        tie_breaking: Optional[str] = "uniform",
        weight_type: Optional[str] = "rank",
        dover_weight: Optional[float] = 0.1,
        custom_weight: Optional[List[str]] = None,
    ) -> List[List[Turn]]:

        # Label mapping stage
        mapped_turns_list, ranks = LabelMapping.get_mapped_turns_list(
            turns_list,
            file_id,
            method=label_mapping,
            second_maximal=second_maximal,
            sort_first=sort_first,
        )

        # Compute weights based on rank
        if weight_type == "rank":
            weights = cls.__compute_weights(ranks, dover_weight)
        else:
            assert isinstance(custom_weight, list)
            weights = np.array([float(x) for x in custom_weight])

        # Label voting stage
        combined_turns_list = LabelVoting.get_combined_turns(
            mapped_turns_list, file_id, tie_breaking, weights
        )

        return combined_turns_list

    def __compute_weights(ranks: np.array, weight: float) -> np.array:

        out_weights = 1 / np.power(ranks, weight)
        out_weights /= np.linalg.norm(out_weights, ord=1)
        return out_weights
