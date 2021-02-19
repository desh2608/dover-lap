import numpy as np
import sys

from typing import List, Dict, Tuple, Optional
from collections import namedtuple

from dover_lap.libs.utils import groupby, info
from dover_lap.libs.turn import Turn
from .map_utils import *

import networkx as nx


class RandomizedMap:

    def __init__(self,
        init_method: Optional[str] = 'greedy'
    ) -> None:
        self.init_method = init_method
    
    def compute_mapping(self,
        turns_list: List[List[Turn]]
    ) -> Tuple[Dict[Tuple[int, str], int], np.ndarray]:
        self.turns_list = turns_list
        N = len(self.turns_list)
        self.hyp_graph = HypothesisGraph(self.turns_list)

        greedy_map = GreedyMap()
        if self.init_method == 'greedy':
            self.greedy_map, weights = greedy_map.compute_mapping(turns_list)
            self.hyp_graph.initialize_cliques(init_map=self.greedy_map)
        else:
            _, pairwise_costs = greedy_map.compute_cost_tensor(turns_list)
            weights = np.array([0] * N, dtype=float)
            for i in range(N):
                cur_pairwise_costs = [
                    np.squeeze(x) for x in pairwise_costs.values() if x.shape[i] != 1
                ]
                weights[i] = -1 * sum([np.sum(x) for x in cur_pairwise_costs])
            self.hyp_graph.initialize_cliques(init_map=None)
        
        return label_mapping, weights
        