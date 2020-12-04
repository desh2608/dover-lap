from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from collections import defaultdict, namedtuple, Counter

import numpy as np
import itertools, sys

from dover_lap.libs.six import iteritems, itervalues
from dover_lap.libs.utils import groupby
from dover_lap.libs.turn import Turn

__all__ = ['get_combined_turns']
EPS = 1e-3 # 0.0001 because float comparisons are inaccurate

def get_combined_turns(file_to_turns_list, file_to_weights):
    """
    This function takes turns list for all input RTTMs and performs
    the DOVER-Lap label voting for getting a combined output turns list.
    """
    file_to_combined_turns = {}
    for file_id in file_to_turns_list.keys():
        turns_list = file_to_turns_list[file_id]
        weights = file_to_weights[file_id]
        combined_turns = _combine_turns(turns_list, file_id, weights)
        file_to_combined_turns[file_id] = combined_turns
    return file_to_combined_turns


def _get_regions(turns_list, weights=None):
    """
    Returns homogeneous time regions of the inputs as list of tuples. Regions are 
    demarcated by start or end times of speaker intervals in any of the input
    hypothesis.
    """
    if (weights is None):
        weights = np.array([1]*len(turns_list))
    weighted_turns_list = []

    for turns, weight in zip(turns_list, weights):
        weighted_turns = [Turn(turn.onset, offset=turn.offset, speaker_id=turn.speaker_id, \
            file_id=turn.file_id, weight=weight) for turn in turns]
        weighted_turns_list.append(weighted_turns)
    all_turns = [turn for turns in weighted_turns_list for turn in turns]

    tokens = []
    for turn in all_turns:
        # Name is 'START' (not 'BEG') so that 'END' tokens come first for same timestamp
        tokens.append(('START', turn.onset, turn.speaker_id, turn.weight)) 
        tokens.append(('END', turn.offset, turn.speaker_id, turn.weight))


    sorted_tokens = sorted(tokens, key=lambda x:(x[1],x[0]))
    
    regions = []
    region_start = sorted_tokens[0][1]
    running_speakers_dict = {sorted_tokens[0][2]:sorted_tokens[0][3]}
    for token in sorted_tokens[1:]:
        if (token[1] > region_start):
            running_speakers = [(k,v) for k,v in running_speakers_dict.items() if v>0]
            if len(running_speakers) > 0:
                regions.append((region_start, token[1], running_speakers.copy()))
        if token[0] == 'START':
            if (token[2] in running_speakers_dict):
                running_speakers_dict[token[2]] += token[3]
            else:
                running_speakers_dict[token[2]] = token[3]
        else:
            running_speakers_dict[token[2]] -= token[3]
            if running_speakers_dict[token[2]] <= EPS:
                running_speakers_dict[token[2]] = 0
        region_start = token[1]

    return regions


def _combine_turns(turns_list, file_id, weights=None):
    """
    Implements combination using the DOVER-Lap label voting method
    """
    print ("Applying label voting on {}".format(file_id))
    combined_turns = []
    regions = _get_regions(turns_list, weights)
    
    for region in regions:
        spk_weights = sorted(region[2], key=lambda x:x[1], reverse=True)
        num_spk = max(1,int(round(sum([spk_weights[1] for spk_weights in region[2]]))))
        i = 0
        while i < num_spk and len(spk_weights) > 0:
            cur_weight = spk_weights[0][1]
            filtered_ids = [spk_id for (spk_id,weight) in spk_weights if np.abs(weight-cur_weight)<EPS]

            dur = (region[1] - region[0])/len(filtered_ids)
            if (np.abs(region[1]>region[0]+EPS):
                for j,spk_id in enumerate(filtered_ids):
                    turn = Turn(region[0]+j*dur, offset=region[0]+(j+1)*dur, speaker_id=spk_id, file_id=file_id)
                    combined_turns.append(turn)
            i += 1
            spk_weights = spk_weights[len(filtered_ids):] # remove the spk ids that have been seen

    return combined_turns
