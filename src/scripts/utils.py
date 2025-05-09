# scripts/ec_utils.py

import json
import networkx as nx
import numpy as np
from itertools import combinations
from node2vec import Node2Vec
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score


def hyperedges_to_2section(hyperedges):
    G = nx.Graph()
    for edge in hyperedges:
        for u, v in combinations(edge, 2):
            if G.has_edge(u, v):
                G[u][v]['weight'] += 1
            else:
                G.add_edge(u, v, weight=1)
    return G


def run_node2vec(G, dimensions=16, walk_length=10, num_walks=50, window=5):
    node2vec = Node2Vec(G, dimensions=dimensions, walk_length=walk_length,
                        num_walks=num_walks, workers=1, weight_key='weight', quiet=True)
    model = node2vec.fit(window=window, min_count=1)
    nodes = list(G.nodes())
    embeddings = np.array([model.wv[str(node)] for node in nodes])
    return embeddings, nodes, model


def cluster_embeddings(embeddings, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    return labels


def compute_ami(true_labels, predicted_labels):
    return adjusted_mutual_info_score(true_labels, predicted_labels)


def save_partition(nodes, labels, output_path):
    partition = {str(node): int(label) for node, label in zip(nodes, labels)}
    with open(output_path, "w") as f:
        json.dump(partition, f, indent=2)
    return partition
