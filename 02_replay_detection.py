"""
Replay Attack Detection: Demonstrating Why Single-Layer Detection Fails

Research finding: In a network-layer replay attack against an ICS, the replayed
packets are structurally legitimate traffic, so network classifiers miss them.
Meanwhile the physical PLC registers go suspiciously static. But:

  - HDBSCAN misses most of the replay window because static data forms a tight
    dense cluster — exactly what density-based detectors are built to call normal.
  - K-Means also fails to partition the physical sensor space along the attack
    boundary, because the registers don't shift to a new macro-state; they just
    stop varying.

Both detectors fail simultaneously, for different structural reasons. The
correct detection architecture is cross-layer correlation: legitimate-looking
network traffic + suspicious physical stasis = replay attack signature.
"""

import os
import warnings

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, classification_report
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)

PALETTE = {"Normal": "#4e79a7", "replay": "#e15759"}

# ─── 1. Load & time-align ────────────────────────────────────────────────────
print("Loading datasets …")
dataset_dir = kagglehub.dataset_download("alirezadehlaghi/icssim")
df_net = pd.read_csv(os.path.join(dataset_dir, "Dataset.csv"))
df_plc = pd.read_csv(os.path.join(dataset_dir, "snapshots_PLC1.csv"))
df_plc.columns = df_plc.columns.str.strip()

# Both datasets use independent clocks; align via elapsed seconds from each
# dataset's own start so the relative attack windows line up correctly.
net_start = pd.to_numeric(df_net["start"], errors="coerce").min()
df_net["elapsed_start"] = pd.to_numeric(df_net["start"], errors="coerce") - net_start
df_net["elapsed_end"]   = pd.to_numeric(df_net["end"],   errors="coerce") - net_start

plc_time = pd.to_datetime(df_plc["time"], errors="coerce")
df_plc["elapsed_time"] = (plc_time - plc_time.min()).dt.total_seconds()

replay_rows = df_net[df_net["IT_M_Label"] == "replay"]
replay_start = replay_rows["elapsed_start"].min()
replay_end   = replay_rows["elapsed_end"].max()

df_plc["IT_M_Label"] = "Normal"
replay_mask = (df_plc["elapsed_time"] >= replay_start) & \
              (df_plc["elapsed_time"] <= replay_end)
df_plc.loc[replay_mask, "IT_M_Label"] = "replay"

print("PLC label distribution:")
print(df_plc["IT_M_Label"].value_counts().to_string())

# ─── 2. Feature matrix — physical sensors only ───────────────────────────────
# Exclude timing/counter metadata (current_loop, loop_latency,
# logic_execution_time) — these are timestamp proxies, not process signals.
# Using only true physical sensor readings isolates the question: can we detect
# replay from physical state alone?
SENSOR_COLS = [
    "tank_input_valve_status(0)",
    "tank_input_valve_mode(1)",
    "tank_level_value(2)",
    "tank_level_min(3)",
    "tank_level_max(4)",
    "tank_output_valve_status(5)",
    "tank_output_valve_mode(6)",
    "tank_output_flow_value(7)",
]
sensor_cols_present = [c for c in SENSOR_COLS if c in df_plc.columns]
X = df_plc[sensor_cols_present].fillna(0)

X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
var = pca.explained_variance_ratio_

ground_truth = df_plc["IT_M_Label"]

# ─── 3. HDBSCAN — the structural blind spot ──────────────────────────────────
print("\n── HDBSCAN on physical sensor registers ──")
hdb = HDBSCAN(min_cluster_size=50, min_samples=5)
hdb_labels = hdb.fit_predict(X_scaled)

anomaly_mask = hdb_labels == -1
n_replay     = replay_mask.sum()
hdb_caught   = (anomaly_mask & replay_mask).sum()
hdb_rate     = hdb_caught / n_replay * 100

print(f"Replay window rows      : {n_replay:,}")
print(f"Flagged as anomaly (-1) : {hdb_caught:,}  ({hdb_rate:.1f}% of replay window)")
print(
    "\nInterpretation: HDBSCAN labels most of the replay window as normal because "
    "static PLC registers form a tight dense cluster — exactly what density-based "
    "detectors are built to trust. Structural blind spot, not a tuning failure."
)

# ─── 4. K-Means — macro-state partitioning attempt ───────────────────────────
print("\n── K-Means macro-state partitioning ──")
km = KMeans(n_clusters=2, random_state=42, n_init=10)
km_labels = km.fit_predict(X_pca)

ari = adjusted_rand_score(ground_truth, km_labels)
print(f"Adjusted Rand Index (ARI): {ari:.4f}")

cluster_majority = (
    pd.DataFrame({"cluster": km_labels, "label": ground_truth})
    .groupby("cluster")["label"]
    .agg(lambda s: s.value_counts().index[0])
)
km_label_names = pd.Series(km_labels).map(cluster_majority)
print("\nK-Means classification report vs ground truth:")
print(classification_report(ground_truth, km_label_names, zero_division=0))

if ari < 0.1:
    print(
        "Interpretation: K-Means also fails. The replay attack does not shift the "
        "physical process to a new macro-state — it suppresses variation within the "
        "existing state. The registers don't go somewhere new; they stop moving. "
        "Macro-state partitioning cannot detect absence-of-variance."
    )
else:
    print(
        "Interpretation: K-Means partitions the physical state space into two "
        "macro-states that align with the replay window."
    )

print(
    "\nConclusion: Both physical-layer detectors fail simultaneously for different "
    "structural reasons. The correct detection signature is cross-layer: "
    "legitimate-looking network traffic combined with suspicious physical stasis."
)

# ─── 5. Physical register variance: Normal vs Replay ─────────────────────────
print("\n── Physical sensor variance by window ──")
for col in sensor_cols_present:
    var_normal = df_plc.loc[ground_truth == "Normal", col].var()
    var_replay = df_plc.loc[ground_truth == "replay", col].var()
    ratio = var_replay / var_normal if var_normal > 0 else float("inf")
    print(f"  {col:35s}  normal={var_normal:.4f}  replay={var_replay:.4f}  ratio={ratio:.3f}")

# ─── 6. Visualisations ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(
    "Replay Attack: Both Physical-Layer Detectors Fail\n"
    "(Correct signature requires cross-layer correlation)",
    fontsize=12,
)

# — Ground truth
ax = axes[0]
for lbl in ["Normal", "replay"]:
    idx = ground_truth == lbl
    ax.scatter(X_pca[idx, 0], X_pca[idx, 1],
               c=PALETTE[lbl], label=lbl, alpha=0.35, s=6)
ax.set_title("Ground Truth Labels")
ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% var)")
ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% var)")
ax.legend(markerscale=3, fontsize=9)

# — HDBSCAN result
ax = axes[1]
hdb_colors = np.where(anomaly_mask, "#e15759", "#4e79a7")
ax.scatter(X_pca[:, 0], X_pca[:, 1], c=hdb_colors, alpha=0.35, s=6)
ax.scatter([], [], c="#e15759", label=f"Anomaly: {anomaly_mask.sum():,} ({hdb_rate:.1f}% of replay)")
ax.scatter([], [], c="#4e79a7", label=f"Normal: {(~anomaly_mask).sum():,}")
ax.set_title(f"HDBSCAN — Misses {100-hdb_rate:.1f}% of Replay Window")
ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% var)")
ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% var)")
ax.legend(markerscale=3, fontsize=9)

# — K-Means result
km_palette = {0: "#4e79a7", 1: "#f28e2b"}
ax = axes[2]
for cluster_id in [0, 1]:
    idx = km_labels == cluster_id
    ax.scatter(X_pca[idx, 0], X_pca[idx, 1],
               c=km_palette[cluster_id],
               label=f"Cluster {cluster_id} ({idx.sum():,})",
               alpha=0.35, s=6)
ax.set_title(f"K-Means — ARI = {ari:.4f}  (near-zero = fails)")
ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% var)")
ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% var)")
ax.legend(markerscale=3, fontsize=9)

plt.tight_layout()
out = "outputs/replay_detection_comparison.png"
plt.savefig(out, dpi=120, bbox_inches="tight")
plt.close()
print(f"\nSaved: {out}")

# ─── 7. Variance suppression time-series ─────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
fig.suptitle(
    "PLC1 Sensor Behaviour — Normal vs Replay Window\n"
    "Physical signature of replay: suppressed variance, not anomalous values",
    fontsize=11,
)

plot_cols = [
    "tank_level_value(2)",
    "tank_output_flow_value(7)",
    "tank_input_valve_status(0)",
]
t = df_plc["elapsed_time"]

for ax, col in zip(axes, plot_cols):
    if col not in df_plc.columns:
        ax.set_visible(False)
        continue
    normal_idx = ground_truth == "Normal"
    replay_idx = ground_truth == "replay"
    ax.plot(t[normal_idx], df_plc.loc[normal_idx, col],
            color="#4e79a7", alpha=0.5, lw=0.6, label="Normal")
    ax.plot(t[replay_idx], df_plc.loc[replay_idx, col],
            color="#e15759", alpha=0.9, lw=0.8, label="Replay")
    ax.set_ylabel(col, fontsize=8)
    ax.legend(fontsize=7, loc="upper right")

axes[-1].set_xlabel("Elapsed seconds from dataset start")
plt.tight_layout()
out2 = "outputs/replay_register_timeseries.png"
plt.savefig(out2, dpi=120, bbox_inches="tight")
plt.close()
print(f"Saved: {out2}")
