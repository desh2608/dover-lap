import sys
import random

from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from itertools import combinations
from copy import deepcopy

from dover_lap.libs.utils import groupby, info
from dover_lap.libs.turn import Turn

import numpy as np
import networkx as nx
from networkx.algorithms.clique import find_cliques


class HypothesisGraph:
    def __init__(self, turns_list: List[List[Turn]]) -> None:

        self.graph = nx.Graph()
        self.node_index = dict()
        self.turns_list = turns_list
        self.max_num_speakers = 0

        # Initialize graph from nodes
        node_id = 0
        for i, turns in enumerate(turns_list):
            spk_groups = {
                key: list(group)
                for key, group in groupby(turns, lambda x: x.speaker_id)
            }
            self.max_num_speakers = max(len(spk_groups), self.max_num_speakers)
            for spk_id in sorted(spk_groups):
                self.graph.add_node(node_id, hyp=i, spk=spk_id)
                self.node_index[(i, spk_id)] = node_id
                node_id += 1

        # Initialize edges
        for i, ref_turns in enumerate(turns_list):
            for j, sys_turns in enumerate(turns_list):
                if j <= i:
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

    def initialize_cliques(
        self, init_map: Optional[Dict[Tuple[int, str], int]] = None
    ) -> None:
        self.cliques = list()
        if not init_map:
            H = deepcopy(self.graph)
            while len(H) > 0:
                all_cliques = list(find_cliques(H))
                selected_clique = random.choice(all_cliques)
                self.cliques.append(selected_clique)
                H.remove_nodes_from(selected_clique)
        else:
            grouped_nodes = defaultdict(list)
            for key, val in init_map.items():
                grouped_nodes[val].append(key)

            for node_list in grouped_nodes.values():
                self.cliques.append([self.node_index[node] for node in node_list])

    def search_cliques(self, max_iter: int) -> float:
        S_A = []  # edges within the cliques
        for clique in self.cliques:
            S_A.extend(list(combinations(clique, 2)))
        S_C = list(set(self.graph.edges()) - set(S_A))  # edges going across cliques

        max_weight = sum(
            [self.__compute_clique_weight(clique) for clique in self.cliques]
        )
        max_clique_set = deepcopy(self.cliques)
        i = 0
        while len(S_C) > 0 and i < max_iter:
            i += 1

            # select any "crossing" edge with prob. weighted by edge weights
            prob_dist = np.array([self.graph[u][v]["weight"] for u, v in S_C])
            if sum(prob_dist) == 0:
                break
            prob_dist = prob_dist / sum(prob_dist)
            u, v = S_C[np.random.choice(len(S_C), p=prob_dist)]
            S_C.remove((u, v))

            # select one of the nodes on the edge
            x = random.choice([u, v])

            # select some y from the same independent set as x
            hyp = self.graph.nodes[x]["hyp"]
            choices = list(
                set(
                    [
                        node
                        for node, data in self.graph.nodes(data=True)
                        if data["hyp"] == hyp
                    ]
                )
                - set([x])
            )
            if len(choices) == 0:
                continue
            y = random.choice(choices)

            # swap x and y in the clique set
            for clique in self.cliques:
                if x in clique:
                    clique[clique.index(x)] = y
                elif y in clique:
                    clique[clique.index(y)] = x

            # compute clique set weight
            weight = sum(
                [self.__compute_clique_weight(clique) for clique in self.cliques]
            )
            # print (weight, max_weight)
            # Update max if current weight is higher
            if weight > max_weight:
                max_weight = weight
                max_clique_set = deepcopy(self.cliques)

        self.cliques = deepcopy(max_clique_set)
        return max_weight

    def __compute_clique_weight(self, clique: List[int]) -> float:
        clique_weight = sum(
            [
                data["weight"]
                for u, v, data in self.graph.edges(data=True)
                if u in clique and v in clique
            ]
        )
        return clique_weight


def compute_spk_overlap(ref_spk_turns: List[Turn], sys_spk_turns: List[Turn]) -> float:
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
    return ovl_duration / total_duration


def get_speaker_keys(turns_list: List[List[Turn]]) -> Dict[Tuple[int, int], str]:
    """
    Returns a dictionary which maps a file id (relative) and speaker id (relative)
    to absolute speaker id.
    """
    speakers_dict = {}
    for i, turns in enumerate(turns_list):
        turn_groups = {
            key: list(group) for key, group in groupby(turns, lambda x: x.speaker_id)
        }
        for j, key in enumerate(sorted(turn_groups.keys())):
            speakers_dict[(i, j)] = key
    return speakers_dict