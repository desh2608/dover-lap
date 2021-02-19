import numpy as np
import sys

from typing import List, Dict, Tuple, Optional

from dover_lap.libs.utils import groupby, info
from dover_lap.libs.turn import Turn

import networkx as nx


class HypothesisGraph:

    def __init__(self,
        turns_list: List[List[Turn]]
    ) -> None:
        
        self.graph = nx.Graph()
        self.node_index = dict()
        self.turns_list = turns_list

        # Initialize graph from nodes
        node_id = 0
        for i, turns in enumerate(turns_list):
            spk_groups = {
                key: list(group)
                for key, group in groupby(turns, lambda x: x.speaker_id)
            }
            for spk_id in sorted(spk_groups):
                self.graph.add_node(node_id, hyp=i, spk=spk_id)
                self.node_index[(i,spk_id)] = node_id
                node_id += 1
        
        # Initialize edges
        for i, ref_turns in enumerate(turns_list):
            for j, sys_turns in enumerate(turns_list):
                if (j<=i):
                    continue
                ref_groups = {
                    key: list(group)
                    for key, group in groupby(ref_turns, lambda x: x.speaker_id)
                }
                sys_groups = {
                    key: list(group)
                    for key, group in groupby(sys_turns, lambda x: x.speaker_id)
                }
                for ref_spk_id in sorted(ref_groups.keys()):
                    ref_spk_turns = ref_groups[ref_spk_id]
                    node_i = self.node_index[(i, ref_spk_id)]
                    for sys_spk_id in sorted(sys_groups.keys()):
                        node_j = self.node_index[(j, sys_spk_id)]
                        sys_spk_turns = sys_groups[sys_spk_id]
                        total_overlap = compute_spk_overlap(
                            ref_spk_turns, sys_spk_turns
                        )
                        self.graph.add_edge(node_i, node_j, weight=total_overlap)

    def initialize_cliques(self,
        init_map: Optional[Dict[Tuple[int, str], int]] = None
    ) -> None:
        pass

def compute_spk_overlap(
    ref_spk_turns: List[Turn],
    sys_spk_turns: List[Turn]
) -> float:
    """
    Computes 'relative' overlap, i.e. Intersection Over Union
    """
    tokens = []
    all_turns = ref_spk_turns + sys_spk_turns
    for turn in all_turns:
        tokens.append(("BEG", turn.onset))
        tokens.append(("END", turn.offset))
    spk_count = 0
    ovl_duration = 0
    total_duration = 0
    for token in sorted(tokens, key=lambda x: x[1]):
        if token[0] == "BEG":
            spk_count += 1
            if spk_count == 2:
                ovl_begin = token[1]
            if spk_count == 1:
                speech_begin = token[1]
        else:
            spk_count -= 1
            if spk_count == 1:
                ovl_duration += token[1] - ovl_begin
            if spk_count == 0:
                total_duration += token[1] - speech_begin
    return (ovl_duration / total_duration)

def get_speaker_keys(
    turns_list: List[List[Turn]]
) -> Dict[Tuple[int,int], str]:
    """
    Returns a dictionary which maps a file id (relative) and speaker id (relative)
    to absolute speaker id.
    """
    speakers_dict = {}
    for i,turns in enumerate(turns_list):
        turn_groups = {
            key: list(group) for key, group in groupby(turns, lambda x: x.speaker_id)
        }
        for j,key in enumerate(sorted(turn_groups.keys())):
            speakers_dict[(i,j)] = key
    return speakers_dict