import numpy as np
import sys

from typing import List, Dict, Tuple, Optional

from dover_lap.libs.utils import groupby, info
from dover_lap.libs.turn import Turn

from .randomized import RandomizedLocalSearchMap
from .hungarian import HungarianMap
from .greedy import GreedyMap


class LabelMapping:

    @classmethod
    def get_mapped_turns_list(cls,
        turns_list: List[List[Turn]],
        file_id: str,
        method: Optional[str]='greedy',
        run_second_maximal: Optional[bool]=False
    ) -> List[List[Turn]]:
        """
        This function takes turns list from all RTTMs and applies an n-dimensional
        matching approximation algorithm to map all to a common label space.
        """

        import time
        start_time = time.time()
        if (len(turns_list) == 2) or (method == 'hungarian'):
            # We replace the original turns list with one sorted by average DER
            label_mapping, weights, turns_list = HungarianMap.compute_mapping(turns_list)
        elif (method == 'greedy'):
            label_mapping, weights = GreedyMap.compute_mapping(turns_list, run_second_maximal)
        elif (method == 'randomized'):
            label_mapping, weights = RandomizedLocalSearchMap.compute_mapping(turns_list)
        time_taken = (time.time() - start_time)*1000

        # Get mapped speaker labels using the mapping
        mapped_turns_list = []
        for i, turns in enumerate(turns_list):
            spk_groups = {
                key: list(group)
                for key, group in groupby(turns, lambda x: x.speaker_id)
            }
            mapped_turns = []
            for spk_id in spk_groups.keys():
                new_spk_id = label_mapping[(i, spk_id)]
                for turn in spk_groups[spk_id]:
                    mapped_turns.append(
                        Turn(
                            turn.onset,
                            turn.offset,
                            speaker_id=new_spk_id,
                            file_id=file_id,
                        )
                    )
            mapped_turns_list.append(mapped_turns)
        
        ranks = cls.__get_ranks(weights)
        return mapped_turns_list, ranks, time_taken


    def __get_ranks(
        weights: np.array
    ) -> np.array:
        
        weights /= np.linalg.norm(weights, ord=1)
        temp = weights.argsort()
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(weights)) + 1
        return ranks
