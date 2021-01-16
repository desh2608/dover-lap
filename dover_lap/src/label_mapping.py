import numpy as np
import sys

from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple, Optional

from dover_lap.libs.utils import groupby, info
from dover_lap.libs.turn import Turn


class LabelMapping:

    @classmethod
    def get_mapped_turns_list(cls,
        turns_list: List[List[Turn]],
        file_id: str,
        run_second_maximal: Optional[bool]=False
    ) -> List[List[Turn]]:
        """
        This function takes turns list from all RTTMs and applies an n-dimensional
        matching approximation algorithm to map all to a common label space.
        """

        N = len(turns_list) # number of input hypotheses
        if N == 2:
            # if only 2 inputs need to be combined, we use the Hungarian algorithm
            # since it is provably optimal. Also, we assign both the systems
            # equal weight to prevent the voting to be dominated by one method.
            label_mapping = self.__map_hungarian(*turns_list)
            weights = np.array([0.5, 0.5])

        else:
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
                            total_overlap = cls.__compute_spk_overlap(
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
                cost_tensor = np.sum(list(pairwise_costs.values()))

            # The weight of each hypothesis is computed by computing its total
            # overlap with all other hypotheses
            weights = np.array([0] * N, dtype=float)
            for i in range(N):
                cur_pairwise_costs = [
                    np.squeeze(x) for x in pairwise_costs.values() if x.shape[i] != 1
                ]
                weights[i] = -1 * sum([np.sum(x) for x in cur_pairwise_costs])

            label_mapping = cls.__apply_maximal_matching(cost_tensor, run_second_maximal)

        # Get mapped speaker labels using the mapping
        mapped_turns_list = []
        for i, turns in enumerate(turns_list):
            spk_groups = {
                key: list(group)
                for key, group in groupby(turns, lambda x: x.speaker_id)
            }
            mapped_turns = []
            for j, spk_id in enumerate(spk_groups.keys()):
                new_spk_id = label_mapping[(i, j)]
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
        return mapped_turns_list, ranks


    def __map_hungarian(
        ref_turns: List[Turn],
        sys_turns: List[Turn]
    ) -> Dict[Tuple[int, int], int]:
        """
        Use Hungarian algorithm for label mapping for 2 system special case.
        """
        cost_matrix = []
        ref_groups = {
            key: list(group) for key, group in groupby(ref_turns, lambda x: x.speaker_id)
        }
        sys_groups = {
            key: list(group) for key, group in groupby(sys_turns, lambda x: x.speaker_id)
        }
        for ref_spk_id in sorted(ref_groups.keys()):
            cur_row = []
            ref_spk_turns = ref_groups[ref_spk_id]
            for sys_spk_id in sorted(sys_groups.keys()):
                sys_spk_turns = sys_groups[sys_spk_id]
                total_overlap = _compute_spk_overlap(ref_spk_turns, sys_spk_turns)
                cur_row.append(-1 * total_overlap)
            cost_matrix.append(cur_row)

        cost_matrix = np.array(cost_matrix)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Keep track of remaining row or col indices
        row_indices_remaining = list(range(len(cost_matrix)))
        col_indices_remaining = list(range(len(cost_matrix[0])))
        label_mapping = {}

        for i in range(len(row_ind)):
            label_mapping[(0, row_ind[i])] = i
            row_indices_remaining.remove(row_ind[i])
            label_mapping[(1, col_ind[i])] = i
            col_indices_remaining.remove(col_ind[i])

        next_label = i + 1

        # Assign labels to remaining row indices
        while len(row_indices_remaining) != 0:
            label_mapping[(0, row_indices_remaining[0])] = next_label
            next_label += 1
            del row_indices_remaining[0]

        # Assign labels to remaining col indices
        while len(col_indices_remaining) != 0:
            label_mapping[(1, col_indices_remaining[0])] = next_label
            next_label += 1
            del col_indices_remaining[0]

        return label_mapping

    @classmethod
    def __apply_maximal_matching(cls,
        cost_tensor: np.ndarray,
        run_second_maximal: Optional[bool]=False
    ) -> List[List[Turn]]:

        # Sort the cost tensor
        sorted_idx = np.transpose(
            np.unravel_index(np.argsort(cost_tensor, axis=None), cost_tensor.shape)
        )

        # Get the maximal matching using an approximation algorithm
        M = []
        remaining_idx = {
            i: list(range(cost_tensor.shape[i]))
            for i in range(len(cost_tensor.shape))
        }

        iter = 1
        while len(remaining_idx.keys()) != 0:
            info(
                "Iteration {}: {} labels left to be mapped".format(
                    iter, sum([len(v) for v in remaining_idx.values()]),
                ),
                file=sys.stderr
            )
            sorted_idx_filtered = cls.__filter_sorted_index_list(
                sorted_idx, remaining_idx
            )

            # find initial maximal matching
            M_cur = []
            for idx in sorted_idx_filtered:
                if not cls.__contradicts(M_cur, idx):
                    M_cur.append(idx)

            if run_second_maximal:
                # find second maximal matching
                change = True
                while change:
                    change = False
                    for idx in list(M_cur):
                        M_cur.remove(idx)
                        M_r = cls.__find_remaining_maximal_matching(
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
                if (i, j) not in label_mapping:
                    label_mapping[(i, j)] = k

        return label_mapping

    @classmethod
    def __find_remaining_maximal_matching(cls,
        M: List[Dict[int,int]],
        idx_list: List[Tuple[int, int]]
    ) -> List[Tuple[int,int]]:
        """
        Given a list of index tuples and a matching M, find a maximal
        matching on the "remaining" list, i.e., using only those index
        tuples which are not present in the original matching.
        """
        S_r = []
        for idx in list(idx_list):
            if not cls.__contradicts(M, idx):
                S_r.append(idx)

        M_r = []
        for idx in S_r:
            if not cls.__contradicts(M_r, idx):
                M_r.append(idx)

        return M_r


    def __filter_sorted_index_list(
        sorted_idx: List[np.ndarray],
        remaining_idx: List[Tuple[int,int]]
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


    def __contradicts(
        M: List[Dict[int,int]],
        idx_tuple: List[int]
    ) -> bool:
        """
        Check if an index tuple contradicts a matching, i.e. return True if
        any index in the tuple is already present in the matching.
        """
        for i, index in enumerate(idx_tuple):
            existing_idx = [idx[i] for idx in M]
            if index in existing_idx:
                return True
        return False


    def __compute_spk_overlap(
        ref_spk_turns: List[Turn],
        sys_spk_turns: List[Turn]
    ) -> float:
        
        tokens = []
        all_turns = ref_spk_turns + sys_spk_turns
        for turn in all_turns:
            tokens.append(("BEG", turn.onset))
            tokens.append(("END", turn.offset))
        spk_count = 0
        ovl_duration = 0
        for token in sorted(tokens, key=lambda x: x[1]):
            if token[0] == "BEG":
                spk_count += 1
                if spk_count == 2:
                    ovl_begin = token[1]
            else:
                spk_count -= 1
                if spk_count == 1:
                    ovl_duration += token[1] - ovl_begin
        return ovl_duration / sum([turn.dur for turn in all_turns])


    def __get_ranks(
        weights: np.array
    ) -> np.array:
        
        weights /= np.linalg.norm(weights, ord=1)
        temp = weights.argsort()
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(weights)) + 1
        return ranks
