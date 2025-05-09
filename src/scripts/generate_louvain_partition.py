import os
import json
import networkx as nx
import matplotlib.pyplot as plt
from node2vec import Node2Vec
from itertools import combinations
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.decomposition import PCA
import numpy as np
import random
import csv
import time
import community as community_louvain  # For modularity calculation

# === Config ===
dataset_name = "synthetic_200_10comm_txt"
experiment_id = "generate_louvian_partition_txt"
data_file = "data/synthetic_hypergraph_for_hlouvain.txt"
num_communities = 10

# Record start time
start_time = time.time()
os.makedirs(f"results/{experiment_id}", exist_ok=True)

# === Step 1: Load hyperedges from .txt ===
hyperedges = []
with open(data_file, "r") as f:
    for line in f:
        parts = line.strip().split(",")
        if parts:
            edge = list(map(int, parts))
            hyperedges.append(edge)

print(f"[✓] Loaded {len(hyperedges)} hyperedges from {data_file}")

# === Step 2: Convert to 2-section graph ===
def hyperedges_to_2section(hyperedges):
    G = nx.Graph()
    for edge in hyperedges:
        for u, v in combinations(edge, 2):
            if G.has_edge(u, v):
                G[u][v]['weight'] += 1
            else:
                G.add_edge(u, v, weight=1)
    return G

G = hyperedges_to_2section(hyperedges)
nodes = list(G.nodes())
print(f"[i] 2-section graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

# === Step 3: Node2Vec embedding ===
node2vec = Node2Vec(G, dimensions=16, walk_length=10, num_walks=50, workers=1, weight_key='weight', quiet=True)
model = node2vec.fit(window=5, min_count=1)
embeddings = np.array([model.wv[str(node)] for node in nodes])

# === Step 4: KMeans clustering ===
kmeans = KMeans(n_clusters=num_communities, random_state=42)
labels = kmeans.fit_predict(embeddings)

# === Step 5: Visualization (PCA) ===
pca = PCA(n_components=2)
reduced = pca.fit_transform(embeddings)
plt.figure(figsize=(10, 8))
plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', s=50, edgecolor='k')
plt.title("Node Embeddings Clustered via KMeans")
plt.savefig(f"results/{experiment_id}/plot_clusters.png")
plt.close()

# === Step 6: Stats and Partition ===
partition = {node: int(cluster) for node, cluster in zip(nodes, labels)}

# Ground-truth by node index (if known structure)
ground_truth = {i: i // (len(nodes) // num_communities) for i in nodes}
true_labels = [ground_truth[n] for n in nodes]
ami_score = adjusted_mutual_info_score(true_labels, labels)
modularity_score = community_louvain.modularity(partition, G)

print(f"[i] Adjusted Mutual Info (AMI): {ami_score:.4f}")
print(f"[i] Modularity: {modularity_score:.4f}")

stats = {
    "n_nodes": len(G.nodes),
    "n_edges": len(G.edges),
    "embedding_dim": 16,
    "k_clusters": num_communities,
    "ami": ami_score,
    "modularity": modularity_score
}

# === Save outputs ===
with open(f"results/{experiment_id}/ec_partition.json", "w") as f:
    json.dump(partition, f, indent=2)

with open(f"results/{experiment_id}/clustering_stats.json", "w") as f:
    json.dump(stats, f, indent=2)

# === Step 7: Leaderboard logging ===
runtime = time.time() - start_time
leaderboard_row = [
    experiment_id,
    dataset_name,
    "Node2Vec",
    "KMeans",
    num_communities,
    round(ami_score, 4),
    round(modularity_score, 4),
    round(runtime, 2),
    "loaded_txt_for_hlouvain"
]

with open("results/experiment_leaderboard.csv", "a", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(leaderboard_row)

print("[✓] All steps complete. EC partition generated from .txt file.")
