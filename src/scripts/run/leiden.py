# scripts/ec_refinement/refine_with_leiden.py

import os
import json
import csv
import igraph as ig
import leidenalg
import networkx as nx
from scripts.utils_graph import hyperedges_to_2section
from scripts.run.utils_louvain import load_partition

# === Config ===
dataset_name = "hyperedges-contact-high-school"
experiment_id = "002"
embedding_type = "deepwalk"
k_clusters = 10
partition_file = f"results/experiment_{experiment_id}/ec_partition.json"
output_dir = f"results/ec_refinement_{experiment_id}_leiden"
os.makedirs(output_dir, exist_ok=True)

# === Load hyperedges from .txt ===
hyperedge_file = "data/synthetic_hypergraph_for_hlouvain.txt"
raw_hyperedges = []
with open(hyperedge_file, "r") as f:
    for line in f:
        parts = line.strip().split(",")
        if parts:
            edge = list(map(int, parts))
            raw_hyperedges.append(edge)
print(f"[✓] Loaded {len(raw_hyperedges)} hyperedges from {hyperedge_file}")

# === Convert to 2-section graph ===
G_nx = hyperedges_to_2section(raw_hyperedges)

# === Convert to iGraph ===
G_ig = ig.Graph()
nodes_nx = list(G_nx.nodes())
G_ig.add_vertices(nodes_nx)

# Create a mapping from node name to iGraph vertex ID
name_to_id = {name: i for i, name in enumerate(nodes_nx)}

# Convert edges to use iGraph vertex IDs
edges_ig = [(name_to_id[u], name_to_id[v]) for u, v in G_nx.edges()]
weights = [G_nx[u][v].get("weight", 1) for u, v in G_nx.edges()]
G_ig.add_edges(edges_ig)
G_ig.es["weight"] = weights

# === Load EC partition and pad missing nodes ===
ec_partition = load_partition(partition_file)
missing_nodes = [v for v in G_nx.nodes() if v not in ec_partition]
if missing_nodes:
    print(f"[!] {len(missing_nodes)} nodes missing from EC partition. Assigning dummy cluster.")
    max_cluster = max(ec_partition.values(), default=0)
    for node in missing_nodes:
        ec_partition[node] = max_cluster + 1

# === Convert EC partition to iGraph format ===
membership = [ec_partition[v] for v in G_ig.vs["name"]]
initial_partition = leidenalg.ModularityVertexPartition(G_ig, membership, weights="weight")

# === Run Leiden optimization from EC start ===
optimiser = leidenalg.Optimiser()
mod_after = optimiser.optimise_partition(initial_partition, n_iterations=-1)

mod_after_trials = []
for _ in range(10):  # Run 10 trials
    optimiser = leidenalg.Optimiser()
    optimiser.optimise_partition(initial_partition, n_iterations=-1)
    mod_after_trials.append(initial_partition.modularity)
avg_mod = sum(mod_after_trials) / len(mod_after_trials)

# === Modularity scores ===
mod_before = initial_partition.modularity

# === Save refined partition ===
refined_dict = {int(G_ig.vs[i]["name"]): comm for i, comm in enumerate(initial_partition.membership)}

with open(f"{output_dir}/refined_partition.json", "w") as f:
    json.dump(refined_dict, f, indent=2)

# === Save stats ===
stats = {
    "dataset": dataset_name,
    "ec_experiment_id": experiment_id,
    "embedding": embedding_type,
    "k_clusters": k_clusters,
    "modularity_before_leiden": mod_before,
    "modularity_after_leiden": mod_after,
    "modularity_avg_10_trials": avg_mod,
    "missing_nodes_handled": len(missing_nodes)
}
with open(f"{output_dir}/modularity_stats.json", "w") as f:
    json.dump(stats, f, indent=2)

# === Print summary ===
print(f"[✓] Leiden refinement complete for experiment {experiment_id}")
print(f"    Modularity before: {mod_before:.4f}")
print(f"    Modularity after:  {mod_after:.4f}")
print(f"    Average after 10 trials: {avg_mod:.4f}")

# === Append to refinement leaderboard ===
with open("results/refinement_leaderboard.csv", "a", newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        f"{experiment_id}-leiden",
        dataset_name,
        embedding_type,
        "KMeans",
        k_clusters,
        round(mod_before, 4),
        round(mod_after, 4),
        len(missing_nodes),
        "Leiden refinement from EC"
    ])
print(f"[✓] Appended to refinement leaderboard.")
