"""Functions for combining multiple RTTMs into one using a DOVER variant."""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

from dover_lap.libs.utils import groupby

from .label_mapping import get_mapped_turns_list
from .label_voting import get_combined_turns

__all__ = ['DOVERLap']


class DOVERLap:

    def __init__(self, second_maximal, dover_weight):
        self.second_maximal = second_maximal
        self.dover_weight = dover_weight

    def combine_turns_list(self, turns_list, file_ids):

        file_to_turns_list = {}
        for turns in turns_list:
            for fid, g in groupby(turns, lambda x: x.file_id):
                if fid in file_to_turns_list:
                    file_to_turns_list[fid].append(list(g))
                else:
                    file_to_turns_list[fid] = [list(g)]
        
        # Label mapping stage
        print ("Mapping speaker labels..")
        file_to_mapped_turns_list, file_to_weights = get_mapped_turns_list(file_to_turns_list, self.second_maximal, self.dover_weight)

        # Label voting stage
        print ("Performing speaker voting")
        file_to_combined_turns = get_combined_turns(file_to_mapped_turns_list, file_to_weights)

        return file_to_combined_turns

