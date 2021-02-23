import numpy as np
import sys

from typing import List, Dict, Tuple, Optional

from dover_lap.libs.utils import groupby, info
from dover_lap.libs.turn import Turn

from dover_lap.src.mapping import HungarianMap, GreedyMap, RandomizedMap


class LabelMapping:

    @classmethod
    def get_mapped_turns_list(cls,
        turns_list: List[List[Turn]],
        file_id: str,
        method: Optional[str]='greedy',
        sort_first: Optional[bool]=False,
        second_maximal: Optional[bool]=False,
        random_init: Optional[str]='none',
        random_epochs: Optional[int]=100
    ) -> List[List[Turn]]:
        """
        This function takes turns list from all RTTMs and applies an n-dimensional
        matching approximation algorithm to map all to a common label space.
        """

        import time
        start_time = time.time()
        
        if (len(turns_list) == 2) or (method == 'hungarian'):
            # We replace the original turns list with one sorted by average DER
            hungarian_map = HungarianMap(sort_first=sort_first)
            label_mapping, weights = hungarian_map.compute_mapping(turns_list)
            turns_list = hungarian_map.sorted_turns_list
        
        elif (method == 'greedy'):
            greedy_map = GreedyMap(second_maximal=second_maximal)
            label_mapping, weights = greedy_map.compute_mapping(turns_list)
        
        elif (method == 'randomized'):
            randomized_map = RandomizedMap(init_method=random_init)
            label_mapping, weights = randomized_map.compute_mapping(turns_list, num_epochs=random_epochs)
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
