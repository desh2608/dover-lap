import numpy as np
import sys

from typing import List, Dict, Tuple, Optional

from dover_lap.libs.utils import groupby, info
from dover_lap.libs.turn import Turn
from .map_utils import *


class GreedyMap:
    def __init__(self, second_maximal: Optional[bool] = False) -> None:
        self.second_maximal = second_maximal

    def compute_mapping(
        self,
        turns_list: List[List[Turn]],
    ) -> Tuple[Dict[Tuple[int, str], int], np.ndarray]:
        """
        Use the DOVER-Lap greedy label mapping algorithm. Returns a map from
        old labels to new, and the weights for the hypotheses.
        """
        self.turns_list = turns_list
        N = len(self.turns_list)
        cost_tensor, pairwise_costs = self.compute_cost_tensor(turns_list)

        # The weight of each hypothesis is computed by computing its total
        # overlap with all other hypotheses
        weights = np.array([0] * N, dtype=float)
        for i in range(N):
            cur_pairwise_costs = [
                np.squeeze(x) for x in pairwise_costs.values() if x.shape[i] != 1
            ]
            weights[i] = -1 * sum([np.sum(x) for x in cur_pairwise_costs])

        label_mapping = self._apply_maximal_matching(
            cost_tensor,
            get_speaker_keys(turns_list),
        )
        return (label_mapping, weights)

    def compute_cost_tensor(self, turns_list: List[List[Turn]]) -> np.ndarray:

        N = len(turns_list)
        k = int((N * (N - 1) / 2))
        pairwise_costs = {}

        has_single_speaker = False

        for i, ref_turns in enumerate(turns_list):
            for j, sys_turns in enumerate(turns_list):
                if j <= i:
                    continue
                cost = []
                ref_groups = {
                    key: list(group)
                    for key, group in groupby(ref_turns, lambda x: x.speaker_id)
                }
                sys_groups = {
                    key: list(group)
                    for key, group in groupby(sys_turns, lambda x: x.speaker_id)
                }

                if len(ref_groups.keys()) == 1 or len(sys_groups.keys()) == 1:
                    has_single_speaker = True

                for ref_spk_id in sorted(ref_groups.keys()):
                    cur_row = []
                    ref_spk_turns = ref_groups[ref_spk_id]
                    for sys_spk_id in sorted(sys_groups.keys()):
                        sys_spk_turns = sys_groups[sys_spk_id]
                        total_overlap = compute_spk_overlap(
                            ref_spk_turns, sys_spk_turns
                        )
                        cur_row.append(-1 * total_overlap)
                    cost.append(cur_row)

                new_axis = list(range(N))
                new_axis.remove(i)
                new_axis.remove(j)

                # The expand_dims is for easy broadcasting
                pairwise_costs[(i, j)] = np.expand_dims(
                    np.array(cost), axis=tuple(k for k in new_axis)
                )

        if has_single_speaker:
            # iterate and add since numpy cannot broadcast with 2 dummy dimensions
            vals = list(pairwise_costs.values())
            cost_tensor = vals[0]
            for val in vals[1:]:
                cost_tensor = np.add(cost_tensor, val)
        else:
            # otherwise use broadcasting
            cost_tensor = np.sum(np.fromiter(pairwise_costs.values(), dtype=object))
        return cost_tensor, pairwise_costs

    def _apply_maximal_matching(
        self,
        cost_tensor: np.ndarray,
        speakers_dict: Dict[Tuple[int, int], str],
    ) -> List[List[Turn]]:

        # Sort the cost tensor
        sorted_idx = np.transpose(
            np.unravel_index(np.argsort(cost_tensor, axis=None), cost_tensor.shape)
        )

        # Get the maximal matching using an approximation algorithm
        M = []
        remaining_idx = {
            i: list(range(cost_tensor.shape[i])) for i in range(len(cost_tensor.shape))
        }

        iter = 1
        while len(remaining_idx.keys()) != 0:
            info(
                "Iteration {}: {} labels left to be mapped".format(
                    iter,
                    sum([len(v) for v in remaining_idx.values()]),
                ),
                file=sys.stderr,
            )
            sorted_idx_filtered = self._filter_sorted_index_list(
                sorted_idx, remaining_idx
            )

            # find initial maximal matching
            M_cur = []
            for idx in sorted_idx_filtered:
                if not self._contradicts(M_cur, idx):
                    M_cur.append(idx)

            if self.second_maximal:
                # find second maximal matching
                change = True
                while change:
                    change = False
                    for idx in list(M_cur):
                        M_cur.remove(idx)
                        M_r = self._find_remaining_maximal_matching(
                            M_cur, sorted_idx_filtered
                        )
                        if len(M_r) > 1:
                            M_cur = M_cur + M_r
                            change = True
                        else:
                            M_cur.append(idx)

            for idx in M_cur:
                for i, j in enumerate(idx):
                    if i in remaining_idx and j in remaining_idx[i]:
                        remaining_idx[i].remove(j)

            for i in list(remaining_idx.keys()):
                if len(remaining_idx[i]) == 0:
                    del remaining_idx[i]

            M += M_cur
            iter += 1

        label_mapping = {}
        for k, idx_tuple in enumerate(M):
            # For each speaker j in hypothesis i, assign new label k
            for i, j in enumerate(idx_tuple):
                old_spk_id = speakers_dict[(i, j)]
                if (i, old_spk_id) not in label_mapping:
                    label_mapping[(i, old_spk_id)] = k

        return label_mapping

    def _find_remaining_maximal_matching(
        self, M: List[Dict[int, int]], idx_list: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """
        Given a list of index tuples and a matching M, find a maximal
        matching on the "remaining" list, i.e., using only those index
        tuples which are not present in the original matching.
        """
        S_r = []
        for idx in list(idx_list):
            if not self._contradicts(M, idx):
                S_r.append(idx)

        M_r = []
        for idx in S_r:
            if not self._contradicts(M_r, idx):
                M_r.append(idx)

        return M_r

    def _filter_sorted_index_list(
        self, sorted_idx: List[np.ndarray], remaining_idx: List[Tuple[int, int]]
    ) -> List[np.ndarray]:
        """
        Filter the sorted list of index tuples to only retain tuples for which
        at least one element is in the remaining list.
        """
        sorted_idx_filtered = []
        for idx_tuple in sorted_idx:
            remaining = False
            for i, j in enumerate(idx_tuple):
                if i in remaining_idx and j in remaining_idx[i]:
                    remaining = True
                    break
            if remaining:
                sorted_idx_filtered.append(idx_tuple)
        return sorted_idx_filtered

    def _contradicts(self, M: List[Dict[int, int]], idx_tuple: List[int]) -> bool:
        """
        Check if an index tuple contradicts a matching, i.e. return True if
        any index in the tuple is already present in the matching.
        """
        for i, index in enumerate(idx_tuple):
            existing_idx = [idx[i] for idx in M]
            if index in existing_idx:
                return True
        return False
