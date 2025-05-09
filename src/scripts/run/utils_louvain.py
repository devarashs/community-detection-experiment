# scripts/ec_refinement/utils_louvain.py

import json
import networkx as nx
import community as community_louvain

def load_partition(path):
    with open(path) as f:
        part = json.load(f)
    return {int(k): v for k, v in part.items()}

def compute_modularity(G, partition_dict):
    return community_louvain.modularity(partition_dict, G, weight='weight')

def run_louvain(G):
    return community_louvain.best_partition(G, weight='weight')
