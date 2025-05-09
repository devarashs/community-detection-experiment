import os
import json
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score
import networkx as nx
from node2vec import Node2Vec
from itertools import combinations
import numpy as np
import csv
import time

# Parameters
num_communities = 10
embedding_dim = 16
data_file = "data/synthetic_hypergraph_for_hlouvain.txt"
out_dir = "results/experiment_002"

os.makedirs(out_dir, exist_ok=True)

# Step 1: Load hyperedges from file
hyperedges = []
with open(data_file, "r") as f:
    for line in f:
        nodes = list(map(int, line.strip().split(",")))
        if len(nodes) >= 2:
            hyperedges.append(nodes)

print(f"[✓] Loaded {len(hyperedges)} hyperedges from {data_file}")

# Step 2: Convert hypergraph to 2-section graph
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
print(f"[i] 2-section graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

# Step 3: Run DeepWalk (node2vec with p=1, q=1)
node2vec = Node2Vec(G, dimensions=embedding_dim, walk_length=10, num_walks=50,
                    workers=1, weight_key='weight', p=1, q=1, quiet=True)
model = node2vec.fit(window=5, min_count=1)

nodes = list(G.nodes())
embeddings = np.array([model.wv[str(node)] for node in nodes])

# Step 4: Clustering
kmeans = KMeans(n_clusters=num_communities, random_state=42)
labels = kmeans.fit_predict(embeddings)

# Step 5: Ground-truth labels
max_node_id = max(nodes)
nodes_per_community = (max_node_id + 1) // num_communities + 1
ground_truth = {n: int(n) // nodes_per_community for n in nodes}
true_labels = [ground_truth[n] for n in nodes]

ami = adjusted_mutual_info_score(true_labels, labels)
print(f"[i] Adjusted Mutual Info (AMI): {ami:.4f}")

# Step 6: Save partition
partition = {str(node): int(label) for node, label in zip(nodes, labels)}
with open(f"{out_dir}/ec_partition.json", "w") as f:
    json.dump(partition, f, indent=2)

# Step 7: Visualization
pca = PCA(n_components=2)
reduced = pca.fit_transform(embeddings)

plt.figure(figsize=(10, 8))
plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', s=100, edgecolor='k')
plt.title(f"DeepWalk Clusters (AMI = {ami:.2f})")
plt.savefig(f"{out_dir}/plot_clusters.png")
plt.close()

# Step 8: Save stats
stats = {
    "n_nodes": G.number_of_nodes(),
    "n_edges": G.number_of_edges(),
    "embedding_dim": embedding_dim,
    "k_clusters": num_communities,
    "ami": ami,
    "modularity": "TBD"
}
with open(f"{out_dir}/clustering_stats.json", "w") as f:
    json.dump(stats, f, indent=2)

print("[✓] DeepWalk experiment complete.")

# Step 9: Leaderboard update
start_time = time.time()
runtime = time.time() - start_time

leaderboard_row = [
    "002",                     # experiment_id
    "synthetic_txt",           # dataset
    "DeepWalk",                # embedding
    "KMeans",                  # clustering
    num_communities,          # k_clusters
    round(ami, 4),             # ami_score
    "TBD",                     # modularity
    round(runtime, 2),         # runtime_sec
    "loaded_txt"               # notes
]

with open("results/experiment_leaderboard.csv", "a", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(leaderboard_row)

print("[✓] Updated experiment leaderboard.")
