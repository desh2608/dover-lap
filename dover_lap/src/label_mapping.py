from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import itertools, sys

from dover_lap.libs.utils import groupby
from dover_lap.libs.turn import Turn

__all__ = ['get_mapped_turns_list']

def get_mapped_turns_list(file_to_turns_list, run_second_maximal=False, dover_weight=0.1):
    """
    This function takes turns list from all RTTMs and applies an n-dimensional
    matching approximation algorithm to map all to a common label space.
    """
    file_to_mapped_turns_list = {}
    file_to_weights = {}

    for file_id in file_to_turns_list.keys():
        turns_list = file_to_turns_list[file_id]

        pairwise_costs = {}
        n = len(turns_list)
        k = int((n * (n-1)/2))

        has_single_speaker = False

        for i, ref_turns in enumerate(turns_list):
            for j, sys_turns in enumerate(turns_list):
                if (j <= i):
                    continue
                cost = []
                ref_groups = {key: list(group) for key, group in groupby(ref_turns, lambda x: x.speaker_id)};
                sys_groups = {key: list(group) for key, group in groupby(sys_turns, lambda x: x.speaker_id)};

                if (len(ref_groups.keys())==1 or len(sys_groups.keys())==1):
                    has_single_speaker = True
                for ref_spk_id in sorted(ref_groups.keys()):
                    cur_row = []
                    ref_spk_turns = ref_groups[ref_spk_id]
                    for sys_spk_id in sorted(sys_groups.keys()):
                        sys_spk_turns = sys_groups[sys_spk_id]
                        total_overlap = _compute_spk_overlap(ref_spk_turns, sys_spk_turns)
                        cur_row.append(-1*total_overlap)
                    cost.append(cur_row)

                new_axis = list(range(n))
                new_axis.remove(i)
                new_axis.remove(j)
                pairwise_costs[(i,j)] = np.expand_dims(np.array(cost), axis=tuple(k for k in new_axis))
        
        if has_single_speaker: # iterate and add since numpy cannot broadcast with 2 dummy dimensions
            vals = list(pairwise_costs.values())
            cost_tensor = vals[0]
            for val in vals[1:]:
                cost_tensor = np.add(cost_tensor,val)
        else: # otherwise use broadcasting
            cost_tensor = np.sum(list(pairwise_costs.values()))

        weights = np.array([0]*len(turns_list), dtype=float)
        for i in range(len(turns_list)):
            cur_pairwise_costs = [np.squeeze(x) for x in pairwise_costs.values() if x.shape[i] != 1]
            weights[i] = -1*sum([np.sum(x) for x in cur_pairwise_costs])

        out_weights = _compute_weights(weights, dover_weight)
        file_to_weights[file_id] = out_weights

        # Sort the cost tensor
        sorted_idx = np.transpose(np.unravel_index(np.argsort(cost_tensor, axis=None), cost_tensor.shape))
        
        # Get the maximal matching Apx3DM-Second algorithm
        M = []
        remaining_idx = {}
        for i in range(len(turns_list)):
            remaining_idx[i] = []
            for j in range (cost_tensor.shape[i]):
                remaining_idx[i].append(j)
        
        while (len(remaining_idx.keys()) != 0):
            print ("{}: {} labels left to be mapped".format(file_id,sum([len(v) for v in remaining_idx.values()])))
            sorted_idx_filtered = _filter_sorted_index_list(sorted_idx, remaining_idx)
            
            # find initial maximal matching
            M_cur = []
            for idx in sorted_idx_filtered:
                if not _contradicts(M_cur, idx):
                    M_cur.append(idx)
            
            if run_second_maximal:
                # find second maximal matching
                change = True
                while change:
                    change = False
                    for idx in list(M_cur):
                        M_cur.remove(idx)
                        M_r = _find_remaining_maximal_matching(M_cur, sorted_idx_filtered)
                        if len(M_r) > 1:
                            M_cur = M_cur + M_r
                            change = True
                        else:
                            M_cur.append(idx)

            for idx in M_cur:
                for i,j in enumerate(idx):
                    if i in remaining_idx and j in remaining_idx[i]:
                        remaining_idx[i].remove(j)

            for i in list(remaining_idx.keys()):
                if len(remaining_idx[i]) == 0:
                    del remaining_idx[i]

            M += M_cur

        new_label_map = {}
        for k,idx_tuple in enumerate(M):
            for i,j in enumerate(idx_tuple):
                if (i,j) not in new_label_map:
                    new_label_map[(i,j)] = k


        mapped_turns_list = []
        for i, turns in enumerate(turns_list):
            spk_groups = {key: list(group) for key, group in groupby(turns, lambda x: x.speaker_id)}
            mapped_turns = []
            for j,spk_id in enumerate(spk_groups.keys()):
                new_spk_id = new_label_map[(i,j)]
                for turn in spk_groups[spk_id]:
                    mapped_turns.append(Turn(turn.onset, turn.offset, speaker_id=new_spk_id, file_id=file_id))
            mapped_turns_list.append(mapped_turns)
        file_to_mapped_turns_list[file_id] = mapped_turns_list

    return file_to_mapped_turns_list, file_to_weights


##############################################################################################################
# HELPER FUNCTIONS FOR LABEL MAPPING
##############################################################################################################

def _compute_weights(weights, dover_weight):
    """
    Compute DOVER-style weights for each hypothesis.
    """
    weights /= np.linalg.norm(weights, ord=1)
    temp = weights.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(weights)) + 1
    out_weights = 1/np.power(ranks,dover_weight)
    out_weights /= np.linalg.norm(out_weights, ord=1)
    return out_weights

def _find_remaining_maximal_matching(M, idx_list):
    """
    Given a list of index tuples and a matching M, find a maximal
    matching on the "remaining" list, i.e., using only those index
    tuples which are not present in the original matching.
    """
    S_r = []
    for idx in list(idx_list):
        if not _contradicts(M, idx):
            S_r.append(idx)

    M_r = []
    for idx in S_r:
        if not _contradicts(M_r, idx):
            M_r.append(idx)

    return M_r

def _filter_sorted_index_list(sorted_idx, remaining_idx):
    """
    Filter the sorted list of index tuples to only retain tuples for which
    at least one element is in the remaining list.
    """
    sorted_idx_filtered = []
    for idx_tuple in sorted_idx:
        remaining = False
        for i,j in enumerate(idx_tuple):
            if i in remaining_idx and j in remaining_idx[i]:
                remaining = True
                break
        if remaining:
            sorted_idx_filtered.append(idx_tuple)
    return sorted_idx_filtered
        

def _contradicts(M, idx_tuple):
    """
    Check if an index tuple contradicts a matching, i.e. return True if
    any index in the tuple is already present in the matching.
    """
    for i,index in enumerate(idx_tuple):
        existing_idx = [idx[i] for idx in M]
        if index in existing_idx:
            return True
    return False


def _compute_spk_overlap(ref_spk_turns, sys_spk_turns):
    tokens = []
    all_turns = ref_spk_turns + sys_spk_turns
    for turn in all_turns:
        tokens.append(('BEG', turn.onset))
        tokens.append(('END', turn.offset))
    spk_count = 0
    ovl_duration = 0
    for token in sorted(tokens, key=lambda x:x[1]):
        if token[0] == 'BEG':
            spk_count += 1
            if spk_count == 2:
                ovl_begin = token[1]
        else:
            spk_count -= 1
            if spk_count == 1:
                ovl_duration += (token[1] - ovl_begin)
    return ovl_duration/sum([turn.dur for turn in all_turns])