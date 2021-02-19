import numpy as np
import sys

from typing import List, Dict, Tuple, Optional

from dover_lap.libs.utils import groupby, info
from dover_lap.libs.turn import Turn
from .map_utils import *

import networkx as nx
from networkx.algorithms.clique import find_cliques


class GreedyNxMap:
    
    def __init__(self,
        second_maximal: Optional[bool] = False
    ) -> None:
        self.second_maximal = second_maximal
    
    def compute_mapping(self,
        turns_list: List[List[Turn]],
    ) -> Tuple[Dict[Tuple[int, str], int], np.ndarray]:
        """
        Use the DOVER-Lap greedy label mapping algorithm. Returns a map from
        old labels to new, and the weights for the hypotheses.
        """
        self.turns_list = turns_list
        N = len(self.turns_list)
        self.HG = HypothesisGraph(self.turns_list)
        
        weights = np.zeros(N)
        for i in range(N):
            node_list = []
            for key,val in self.HG.node_index.items():
                if key[0] == i:
                    node_list.append(val)
                edges = self.HG.graph.edges(node_list, data=True)
                weights[i] = sum([data['weight'] for u,v,data in edges])
        
        label_mapping = dict()
        spk_label = 0
        while len(list(self.HG.graph)) > 0:
            # Get maximum weighted clique
            max_weight = 0
            max_clique = None
            for clique in find_cliques(self.HG.graph):
                clique_weight = sum(
                    [data['weight'] for u,v,data in self.HG.graph.edges(data=True)
                    if u in clique and v in clique]
                )
                if clique_weight >= max_weight:
                    max_clique = clique.copy()
                    max_weight = clique_weight
            
            # Add max clique to label map
            for node in max_clique:
                hyp = self.HG.graph.nodes[node]['hyp']
                spk = self.HG.graph.nodes[node]['spk']
                label_mapping[(hyp,spk)] = spk_label
            
            # Increment speaker label
            spk_label += 1

            # Remove nodes contained in max clique from the graph
            self.HG.graph.remove_nodes_from(max_clique) 

        return (label_mapping, weights)
