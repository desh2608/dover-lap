from typing import List, Dict, Optional

from dover_lap.libs.utils import groupby
from dover_lap.libs.turn import Turn

from dover_lap.src.mapping import HungarianMap, GreedyMap


class LabelMapping:
    @classmethod
    def get_mapped_turns_list(
        cls,
        turns_list: List[List[Turn]],
        file_id: str,
        method: Optional[str] = "greedy",
        second_maximal: Optional[bool] = False,
    ) -> List[List[Turn]]:
        """
        This function takes turns list from all RTTMs and applies an n-dimensional
        matching approximation algorithm to map all to a common label space.
        """

        if (len(turns_list) == 2) or (method == "hungarian"):
            # We replace the original turns list with one sorted by average DER
            hungarian_map = HungarianMap()
            label_mapping, weights = hungarian_map.compute_mapping(turns_list)
            turns_list = hungarian_map.sorted_turns_list

        elif method == "greedy":
            greedy_map = GreedyMap(second_maximal=second_maximal)
            label_mapping, weights = greedy_map.compute_mapping(turns_list)

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

        return mapped_turns_list, weights
