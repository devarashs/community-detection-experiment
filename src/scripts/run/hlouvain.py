import os
import sys
import csv
import json
from collections import defaultdict, Counter
import networkx as nx
import hypernetx as hnx

from scripts.utils_graph import hyperedges_to_2section
from scripts.run.utils_louvain import load_partition, compute_modularity

# Add h-louvain path
sys.path.append("h-louvain")
from h_louvain import hLouvain

# === Config ===
dataset_name = "hyperedges-contact-high-school"
k_clusters = 10
experiment_id = "generate_louvian_partition"
embedding_type = "Node2Vec"
partition_file = f"results/{experiment_id}_txt/ec_partition.json"
hyperedge_txt_file = "data/synthetic_hypergraph_for_hlouvain.txt"

# === Load EC partition ===
ec_partition = load_partition(partition_file)
ec_partition = {int(node): cluster_id for node, cluster_id in ec_partition.items()}

# === Load hyperedges from .txt file ===
hyperedges = []
with open(hyperedge_txt_file, "r") as f:
    for line in f:
        nodes = list(map(int, line.strip().split(",")))
        if nodes:
            hyperedges.append(nodes)

print(f"Loaded {len(hyperedges)} hyperedges from {hyperedge_txt_file}")

# Print sample coherence
for i, edge in enumerate(hyperedges[:5]):
    cluster_ids = [ec_partition.get(node, -1) for node in edge]
    print(f"Edge {i}: nodes={edge} → cluster_ids={cluster_ids} → coherence={len(edge) / len(set(cluster_ids)):.2f}")

# Compute coherence weights for edges
def ec_coherence(edge):
    cluster_ids = [ec_partition.get(node, -1) for node in edge]
    return len(edge) / len(set(cluster_ids)) if len(set(cluster_ids)) > 0 else 1.0

edges_dict = {i: list(edge) for i, edge in enumerate(hyperedges)}
edge_props = {i: {"weight": ec_coherence(edge)} for i, edge in enumerate(hyperedges)}

HG = hnx.Hypergraph(edges_dict, edge_props=edge_props)
for eid in HG.edges:
    HG.edges[eid].weight = edge_props[eid]["weight"]

# === 2-section graph ===
G = hyperedges_to_2section(hyperedges)
print(f"2-section graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# === Pad EC partition ===
missing_nodes = [n for n in G.nodes() if n not in ec_partition]
if missing_nodes:
    print(f"[!] {len(missing_nodes)} missing nodes in EC partition. Assigning dummy cluster.")
    max_cluster = max(ec_partition.values(), default=0)
    for node in missing_nodes:
        ec_partition[node] = max_cluster + 1

# === H-Louvain ===
hlouvain_model = hLouvain(HG)
partition_sets, hmod, alpha_seq = hlouvain_model.h_louvain_community()

hlouvain_partition = {int(node): i for i, cluster in enumerate(partition_sets) for node in cluster}

missing_from_hlouvain = [n for n in G.nodes() if n not in hlouvain_partition]
if missing_from_hlouvain:
    print(f"[!] {len(missing_from_hlouvain)} nodes missing from H-Louvain partition. Assigning dummy cluster.")
    max_cluster = max(hlouvain_partition.values(), default=0)
    for node in missing_from_hlouvain:
        hlouvain_partition[node] = max_cluster + 1

assert all(n in hlouvain_partition for n in G.nodes()), "Missing nodes in final partition"

# === Modularity ===
mod_before = compute_modularity(G, ec_partition)
mod_after = compute_modularity(G, hlouvain_partition)

# === Save output ===
out_dir = f"results/ec_refinement_{experiment_id}"
os.makedirs(out_dir, exist_ok=True)

with open(f"{out_dir}/refined_partition.json", "w") as f:
    json.dump(hlouvain_partition, f, indent=2)

stats = {
    "dataset": dataset_name,
    "ec_experiment_id": experiment_id,
    "embedding": embedding_type,
    "k_clusters": k_clusters,
    "modularity_before_hlouvain": mod_before,
    "modularity_after_hlouvain": mod_after,
    "h_modularity": hmod,
    "alpha_sequence": alpha_seq
}
with open(f"{out_dir}/modularity_stats.json", "w") as f:
    json.dump(stats, f, indent=2)

# === Leaderboard ===
with open("results/refinement_leaderboard.csv", "a", newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        f"{experiment_id}-hlouvain",
        dataset_name,
        embedding_type,
        "KMeans",
        k_clusters,
        round(mod_before, 4),
        round(mod_after, 4),
        len(missing_nodes),
        "H-Louvain refinement from EC"
    ])

# === Summary ===
print("\n[✓] H-Louvain refinement complete")
print(f"    Modularity before:  {mod_before:.4f}")
print(f"    Modularity after:   {mod_after:.4f}")
print(f"    H-modularity score: {hmod:.4f}")