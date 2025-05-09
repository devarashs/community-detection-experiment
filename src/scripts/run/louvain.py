# scripts/ec_refinement/refine_with_louvain.py

import os
import json
import csv
import networkx as nx
from scripts.utils_graph import hyperedges_to_2section
from scripts.run.utils_louvain import load_partition, compute_modularity, run_louvain

# === Config ===
dataset_name = "hyperedges-contact-high-school"
experiment_id = "001"
embedding_type = "Node2Vec"
k_clusters = 10
hyperedge_file = "data/synthetic_hypergraph_for_hlouvain.txt"
partition_file = f"results/experiment_{experiment_id}/ec_partition.json"
output_dir = f"results/ec_refinement_{experiment_id}_louvain"
os.makedirs(output_dir, exist_ok=True)

# === Load hypergraph from TXT ===
def load_hyperedges_txt(path):
    with open(path, "r") as f:
        return [list(map(int, line.strip().replace(",", " ").split())) for line in f if line.strip()]

hyperedges = load_hyperedges_txt(hyperedge_file)

# === Build 2-section graph ===
G = hyperedges_to_2section(hyperedges)

# === Load EC-based partition ===
ec_partition = load_partition(partition_file)

# === Pad partition so all nodes are included ===
missing_nodes = [n for n in G.nodes() if n not in ec_partition]
if missing_nodes:
    print(f"[!] {len(missing_nodes)} nodes missing from EC partition. Assigning dummy cluster.")
    max_cluster = max(ec_partition.values(), default=0)
    for node in missing_nodes:
        ec_partition[node] = max_cluster + 1

# === Compute modularity before refinement ===
mod_before = compute_modularity(G, ec_partition)

# === Run Louvain refinement ===
louvain_partition = run_louvain(G)

# === Compute modularity after refinement ===
mod_after = compute_modularity(G, louvain_partition)

# === Save refined partition ===
with open(f"{output_dir}/refined_partition.json", "w") as f:
    json.dump(louvain_partition, f, indent=2)

# === Save modularity stats ===
stats = {
    "dataset": dataset_name,
    "ec_experiment_id": experiment_id,
    "embedding": embedding_type,
    "k_clusters": k_clusters,
    "modularity_before_louvain": mod_before,
    "modularity_after_louvain": mod_after,
    "missing_nodes_handled": len(missing_nodes)
}
with open(f"{output_dir}/modularity_stats.json", "w") as f:
    json.dump(stats, f, indent=2)

# === Log to refinement leaderboard ===
leaderboard_path = "results/refinement_leaderboard.csv"
with open(leaderboard_path, "a", newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        f"{experiment_id}-louvain",
        dataset_name,
        embedding_type,
        "KMeans",
        k_clusters,
        round(mod_before, 4),
        round(mod_after, 4),
        len(missing_nodes),
        "Louvain refinement from EC"
    ])

# === Print summary ===
print(f"[✓] Louvain refinement complete for experiment {experiment_id}")
print(f"    Modularity before: {mod_before:.4f}")
print(f"    Modularity after:  {mod_after:.4f}")
print(f"[✓] Appended to refinement leaderboard.")
