import sys

import numpy as np
import networkx as nx
from scipy.special import gammaln

from typing import List, Dict, Tuple, Optional
from collections import namedtuple
from copy import deepcopy

from dover_lap.libs.utils import groupby, info
from dover_lap.libs.turn import Turn

from .greedy import GreedyMap
from .map_utils import *


EPS = 0.05
EPS_SQ = np.power(EPS,2)

class RandomizedMap:

    def __init__(self,
        init_method: Optional[str] = 'greedy'
    ) -> None:
        self.init_method = init_method
    
    def compute_mapping(self,
        turns_list: List[List[Turn]],
        num_epochs: Optional[int]=100
    ) -> Tuple[Dict[Tuple[int, str], int], np.ndarray]:
        self.turns_list = turns_list
        self.N = len(self.turns_list)
        self.HG = HypothesisGraph(self.turns_list)

        if self.init_method == 'greedy':
            greedy_map = GreedyMap()
            init_map, weights = greedy_map.compute_mapping(turns_list)
        else:
            init_map = None
            weights = self.__compute_weights()
        # n_iter = self.__get_num_iterations(self.N, self.HG.max_num_speakers)
        max_iter_per_epoch = self.HG.max_num_speakers*(2*self.N - 1)
        best_overall_set = None
        max_overall_weight = 0
        for i in range(num_epochs):
            self.HG.initialize_cliques(init_map=init_map)
            max_epoch_weight = self.HG.search_cliques(max_iter=max_iter_per_epoch)
            if (max_epoch_weight > max_overall_weight):
                print (f'Epoch {i}: weight={max_epoch_weight}, max_weight={max_overall_weight}')
                best_overall_set = deepcopy(self.HG.cliques)
                max_overall_weight = max_epoch_weight
        
        # Compute mapping from best clique configuration
        label_mapping = dict()
        spk_label = 0
        for nodelist in best_overall_set:
            for node in nodelist:
                hyp = self.HG.graph.nodes[node]['hyp']
                spk = self.HG.graph.nodes[node]['spk']
                label_mapping[(hyp,spk)] = spk_label
            spk_label += 1
        return label_mapping, weights

    def __compute_weights(self) -> None:
        weights = np.zeros(self.N)
        for i in range(self.N):
            node_list = []
            for key,val in self.HG.node_index.items():
                if key[0] == i:
                    node_list.append(val)
                edges = self.HG.graph.edges(node_list, data=True)
                weights[i] = sum([data['weight'] for u,v,data in edges])
        return weights
        
    def __get_num_iterations(self,
        r: int, # number of hypotheses (independent sets)
        k: int  # maximum number of speakers in any hypothesis
    ) -> int:
        log_numerator = (r-1)*gammaln(k+1)
        log_denominator = k*(r-1)*(np.log(
            1 + (
                EPS_SQ / ( 4*(np.power(k-1,2))*(np.power(1+EPS,2)) )
            )
        ))
        print (log_numerator, log_denominator)
        logN = log_numerator - log_denominator
        N = int(np.exp(logN))
        return N
