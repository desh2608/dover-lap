import numpy as np

from typing import List, Dict, Tuple, Optional

from dover_lap.libs.utils import groupby
from dover_lap.libs.turn import Turn, merge_turns
from spyder import DER

from .map_utils import *


class HungarianMap:
    def __init__(self) -> None:
        pass

    def compute_mapping(
        self,
        turns_list: List[List[Turn]],
    ) -> Tuple[Dict[Tuple[int, str], int], np.ndarray]:
        """
        Use Hungarian algorithm for label mapping for 2 system special case.
        """
        self.turns_list = turns_list

        weights = self._compute_weights()
        # Sort the hypotheses by their weights
        sorted_idx = weights.argsort().tolist()
        self.sorted_turns_list = [self.turns_list[i] for i in sorted_idx]

        cur_turns = self.sorted_turns_list[0]
        self.global_mapping = dict()

        for i in range(1, len(self.sorted_turns_list)):
            next_turns = self.sorted_turns_list[i]
            local_mapping = self._map_pair(cur_turns, next_turns)
            cur_turns = self._merge_pair(cur_turns, next_turns, local_mapping)
            self._update_global_map(local_mapping)

        if not self._validate_global_mapping():
            raise Exception("Some speakers have not been mapped")
        return self.global_mapping, weights

    def _validate_global_mapping(self) -> bool:
        for i, turns in enumerate(self.sorted_turns_list):
            groups = {
                key: list(group)
                for key, group in groupby(turns, lambda x: x.speaker_id)
            }
            for spk in groups:
                if (i, spk) not in self.global_mapping:
                    return False
        return True

    def _compute_weights(self) -> np.ndarray:
        N = len(self.turns_list)
        DERs = np.zeros(N)
        for i in range(N):
            DER_i = []
            hyp = [
                (turn.speaker_id, turn.onset, turn.offset)
                for turn in self.turns_list[i]
            ]
            for j in range(N):
                if i == j:
                    continue
                ref = [
                    (turn.speaker_id, turn.onset, turn.offset)
                    for turn in self.turns_list[j]
                ]
                der = DER(ref, hyp).der
                DER_i.append(der)
            DERs[i] = np.mean(DER_i)
        return DERs

    def _map_pair(
        self, ref_turns: List[Turn], sys_turns: List[Turn]
    ) -> Dict[Tuple[int, str], int]:
        ref = [(turn.speaker_id, turn.onset, turn.offset) for turn in ref_turns]
        sys = [(turn.speaker_id, turn.onset, turn.offset) for turn in sys_turns]
        metrics = DER(ref, sys)
        ref_map = metrics.ref_map
        sys_map = metrics.hyp_map

        label_mapping = {}
        for k, v in ref_map.items():
            label_mapping[(0, k)] = v
        for k, v in sys_map.items():
            label_mapping[(1, k)] = v

        return label_mapping

    def _merge_pair(
        self,
        ref_turns: List[Turn],
        sys_turns: List[Turn],
        label_map: Dict[Tuple[int, int], int],
    ) -> List[Turn]:
        ref_turns_mapped = [
            Turn(
                onset=turn.onset,
                offset=turn.offset,
                speaker_id=label_map[(0, turn.speaker_id)],
            )
            for turn in ref_turns
        ]
        sys_turns_mapped = [
            Turn(
                onset=turn.onset,
                offset=turn.offset,
                speaker_id=label_map[(1, turn.speaker_id)],
            )
            for turn in sys_turns
        ]
        all_turns = merge_turns(ref_turns_mapped + sys_turns_mapped)
        return all_turns

    def _update_global_map(self, local_map: Dict[Tuple[int, str], int]) -> None:
        if not self.global_mapping:
            self.global_mapping = local_map.copy()
            return
        new_global_map = {}
        max_file_id = 0
        for key, old_id in self.global_mapping.items():
            file_id, spk_id = key
            max_file_id = max(max_file_id, file_id)
            new_global_map[key] = local_map[(0, old_id)]
        for key, val in local_map.items():
            file_id, spk_id = key
            if file_id == 1:
                new_global_map[(max_file_id + 1, spk_id)] = val
        self.global_mapping = new_global_map.copy()
