import os
import sys
import json
import csv
from collections import defaultdict, Counter
import networkx as nx
import hypernetx as hnx

# Add path to h-louvain repo
sys.path.append("h-louvain")  # Adjust this if your repo is elsewhere
from h_louvain import hLouvain

# === Config ===
dataset_name = "hyperedges-contact-high-school"
hyperedge_file = "data/synthetic_hypergraph_for_hlouvain.txt"
experiment_id = "pure_hlouvain"
out_dir = f"results/{experiment_id}"
os.makedirs(out_dir, exist_ok=True)

# === Load hyperedges from .txt ===
raw_hyperedges = []
with open(hyperedge_file, "r") as f:
    for line in f:
        parts = line.strip().split(",")
        if parts:
            edge = list(map(int, parts))
            raw_hyperedges.append(edge)

print(f"[✓] Loaded {len(raw_hyperedges)} hyperedges from {hyperedge_file}")
print(f"Average hyperedge size: {sum(len(e) for e in raw_hyperedges) / len(raw_hyperedges):.2f}")

# === Create HyperNetX hypergraph ===
edges_dict = {i: list(edge) for i, edge in enumerate(raw_hyperedges)}
HG = hnx.Hypergraph(edges_dict)

if len(list(HG.edges)) == 0:
    print("[ERROR] Hypergraph has no edges. Exiting.")
    sys.exit(1)

print(f"[i] Hypergraph has {len(HG.nodes)} nodes and {len(HG.edges)} edges.")

# === Run H-Louvain on the Hypergraph ===
hlouvain_model = hLouvain(HG)
partition_sets, hmod, alpha_seq = hlouvain_model.h_louvain_community()

# === Convert to {node: cluster_id} format ===
hlouvain_partition = {}
for cluster_id, node_set in enumerate(partition_sets):
    for node in node_set:
        hlouvain_partition[int(node)] = cluster_id

counts = Counter(hlouvain_partition.values())
print(f"[✓] H-Louvain found {len(counts)} clusters")
print("Top 10 cluster sizes:", counts.most_common(10))

# === Convert to 2-section graph ===
def hyperedges_to_2section(hyperedges):
    G = nx.Graph()
    for edge in hyperedges:
        for i in range(len(edge)):
            for j in range(i + 1, len(edge)):
                u, v = edge[i], edge[j]
                if G.has_edge(u, v):
                    G[u][v]["weight"] += 1
                else:
                    G.add_edge(u, v, weight=1)
    return G

G = hyperedges_to_2section(raw_hyperedges)
print(f"[i] 2-section graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

# === Compute modularity using Louvain utils ===
def compute_modularity(G, partition_dict):
    try:
        import community as community_louvain
    except ImportError:
        raise ImportError("You need to install the `python-louvain` package")

    return community_louvain.modularity(partition_dict, G)

modularity_score = compute_modularity(G, hlouvain_partition)
print(f"[✓] Modularity of H-Louvain partition: {modularity_score:.4f}")
print(f"[✓] H-modularity score: {hmod:.4f}")

# === Save results ===
with open(f"{out_dir}/hlouvain_partition.json", "w") as f:
    json.dump(hlouvain_partition, f, indent=2)

with open(f"{out_dir}/modularity_stats.json", "w") as f:
    json.dump({
        "dataset": dataset_name,
        "num_clusters": len(counts),
        "modularity_2section": modularity_score,
        "h_modularity": hmod,
        "alpha_sequence": alpha_seq
    }, f, indent=2)

# === Update leaderboard ===
with open("results/pure_hlouvain_leaderboard.csv", "a", newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        experiment_id,
        dataset_name,
        len(counts),
        round(modularity_score, 4),
        round(hmod, 4),
        ""
    ])

print("[✓] Saved partition, stats, and leaderboard entry.")
