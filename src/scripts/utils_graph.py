# scripts/ec_utils_graph.py

import networkx as nx
from itertools import combinations

def hyperedges_to_2section(hyperedges):
    G = nx.Graph()
    for edge in hyperedges:
        for u, v in combinations(edge, 2):
            if G.has_edge(u, v):
                G[u][v]['weight'] += 1
            else:
                G.add_edge(u, v, weight=1)
    return G
