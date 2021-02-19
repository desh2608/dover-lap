import numpy as np
import sys

from typing import List, Dict, Tuple, Optional
from scipy.optimize import linear_sum_assignment

from dover_lap.libs.utils import groupby, info
from dover_lap.libs.turn import Turn, merge_turns
from dover_lap.libs.metrics import DER

from .map_utils import *


class HungarianMap:

    @classmethod
    def compute_mapping(cls,
        turns_list: List[List[Turn]],
    ) -> Tuple[Dict[Tuple[int, str], int], np.ndarray]:
        """
        Use Hungarian algorithm for label mapping for 2 system special case.
        """
        weights = cls.__compute_weights(turns_list)
        sorted_idx = weights.argsort().tolist()
        sorted_turns_list = [turns_list[i] for i in sorted_idx]
        cur_turns = sorted_turns_list[0]
        global_mapping = {}
        
        for i in range(1,len(sorted_turns_list)):
            next_turns = sorted_turns_list[i]
            local_mapping = cls.__map_pair(cur_turns, next_turns)
            cur_turns = cls.__merge_pair(cur_turns, next_turns, local_mapping)
            global_mapping = cls.__update_global_map(global_mapping, local_mapping)

        if not cls.__validate_global_mapping(global_mapping, sorted_turns_list):
            raise Exception("Some speakers have not been mapped")
        return global_mapping, weights, sorted_turns_list


    def __validate_global_mapping(
        map: Dict[Tuple[int,str], int],
        turns_list: List[List[Turn]]
    ) -> bool:
        for i, turns in enumerate(turns_list):
            groups = {
                key: list(group) for key, group in groupby(turns, lambda x: x.speaker_id)
            }
            for spk in groups:
                if (i,spk) not in map:
                    return False
        return True

    @classmethod
    def __compute_weights(cls,
        turns_list: List[List[Turn]]
    ) -> List[List[Turn]]:
        N = len(turns_list)
        DERs = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                DERs[i,j] = DER(turns_list[i], turns_list[j])
        mean_ders = DERs.mean(axis=0)
        return -1*mean_ders


    def __map_pair(
        ref_turns: List[Turn],
        sys_turns: List[Turn]
    ) -> Dict[Tuple[int,str], int]:
        ref_groups = {
            key: list(group) for key, group in groupby(ref_turns, lambda x: x.speaker_id)
        }
        sys_groups = {
            key: list(group) for key, group in groupby(sys_turns, lambda x: x.speaker_id)
        }
        ref_keys = sorted(ref_groups.keys())
        sys_keys = sorted(sys_groups.keys())
        M, N = len(ref_keys), len(sys_keys)
        cost_matrix = np.zeros((M,N))
        for i, ref_spk_id in enumerate(ref_keys):
            cur_row = []
            ref_spk_turns = ref_groups[ref_spk_id]
            for j, sys_spk_id in enumerate(sys_keys):
                sys_spk_turns = sys_groups[sys_spk_id]
                total_overlap = compute_spk_overlap(ref_spk_turns, sys_spk_turns)
                cost_matrix[i,j] = -1*total_overlap

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Keep track of remaining row or col indices
        row_indices_remaining = list(range(M))
        col_indices_remaining = list(range(N))
        label_mapping = {}

        for i in range(len(row_ind)):
            label_mapping[(0, ref_keys[row_ind[i]])] = i
            row_indices_remaining.remove(row_ind[i])
            label_mapping[(1, sys_keys[col_ind[i]])] = i
            col_indices_remaining.remove(col_ind[i])

        next_label = i + 1

        # Assign labels to remaining row indices
        while len(row_indices_remaining) != 0:
            label_mapping[(0, ref_keys[row_indices_remaining[0]])] = next_label
            next_label += 1
            del row_indices_remaining[0]

        # Assign labels to remaining col indices
        while len(col_indices_remaining) != 0:
            label_mapping[(1, sys_keys[col_indices_remaining[0]])] = next_label
            next_label += 1
            del col_indices_remaining[0]

        return label_mapping

    def __merge_pair(
        ref_turns: List[Turn],
        sys_turns: List[Turn],
        label_map: Dict[Tuple[int,int], int]
    ) -> List[Turn]:
        ref_turns_mapped = [
            Turn(onset=turn.onset, offset=turn.offset, speaker_id=label_map[(0,turn.speaker_id)])
            for turn in ref_turns
        ]
        sys_turns_mapped = [
            Turn(onset=turn.onset, offset=turn.offset, speaker_id=label_map[(1,turn.speaker_id)])
            for turn in sys_turns
        ]
        all_turns = merge_turns(ref_turns_mapped + sys_turns_mapped)
        return all_turns

    def __update_global_map(
        global_map: Dict[Tuple[int,str], int],
        local_map: Dict[Tuple[int,str], int]
    ) -> Dict[Tuple[int,str], int]:
        if not global_map:
            return local_map
        new_global_map = {}
        max_file_id = 0
        for key, old_id in global_map.items():
            file_id, spk_id = key
            max_file_id = max(max_file_id, file_id)
            new_global_map[key] = local_map[(0,old_id)]
        for key, val in local_map.items():
            file_id, spk_id = key
            if (file_id == 1):
                new_global_map[(max_file_id+1,spk_id)] = val
        return new_global_map
