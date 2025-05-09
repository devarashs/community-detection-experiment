# scripts/generate_leaderboard_plots.py

import pandas as pd
import matplotlib.pyplot as plt
import os

# Make sure the output folder exists
output_dir = "results/leaderboard_plots"
os.makedirs(output_dir, exist_ok=True)

# Load the leaderboard
df = pd.read_csv("results/experiment_leaderboard.csv")

# Clean missing modularity values
df['modularity'] = df['modularity'].replace('TBD', None)
df['modularity'] = pd.to_numeric(df['modularity'], errors='coerce')

# ===============================
# 1. Plot AMI Score by Embedding
# ===============================

plt.figure(figsize=(10,6))
plt.bar(df['embedding'], df['ami_score'], color='skyblue', edgecolor='black')
plt.title('AMI Score by Embedding Method')
plt.xlabel('Embedding Method')
plt.ylabel('AMI Score')
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(f"{output_dir}/ami_score_by_embedding.png")
plt.close()

print("[✓] Saved AMI score plot.")

# ===============================
# 2. Plot Modularity by Embedding (if available)
# ===============================

if df['modularity'].notna().any():
    plt.figure(figsize=(10,6))
    plt.bar(df['embedding'], df['modularity'], color='lightcoral', edgecolor='black')
    plt.title('Modularity by Embedding Method')
    plt.xlabel('Embedding Method')
    plt.ylabel('Modularity')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/modularity_by_embedding.png")
    plt.close()
    print("[✓] Saved Modularity plot.")
else:
    print("[i] No modularity data available yet, skipped modularity plot.")

# ===============================
# 3. Optional: Plot Runtime by Embedding
# ===============================

if 'runtime_sec' in df.columns:
    plt.figure(figsize=(10,6))
    plt.bar(df['embedding'], df['runtime_sec'], color='mediumseagreen', edgecolor='black')
    plt.title('Runtime by Embedding Method')
    plt.xlabel('Embedding Method')
    plt.ylabel('Runtime (seconds)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/runtime_by_embedding.png")
    plt.close()
    print("[✓] Saved Runtime plot.")

print("[🏁] Leaderboard visualizations complete!")
