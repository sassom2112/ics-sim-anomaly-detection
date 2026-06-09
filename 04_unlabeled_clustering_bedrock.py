"""
04_unlabeled_clustering_bedrock.py
====================================
Unsupervised clustering pipeline for ICS/OT network behavior analysis.

Scenario: raw network data arrives WITHOUT attack labels — the common
production reality for a newly-monitored OT environment. Goal: discover
behavioral clusters, evaluate cluster quality, and use AWS Bedrock
foundation models for embedding-based clustering and cluster interpretation.

Pipeline:
  1.  Load & strip labels  (simulate unlabeled production data)
  2.  EDA                  null audit, skewness inventory
  3.  Preprocessing        impute + log1p + StandardScaler
  4.  PCA                  explained-variance elbow, 2-D projection
  5.  K-Means              elbow + silhouette-score selection
  6.  DBSCAN               ε via k-distance plot; noise = anomaly signal
  7.  HDBSCAN              soft memberships; low-probability = anomaly candidate
  8.  Agglomerative        Ward linkage + dendrogram
  9.  Evaluation dashboard silhouette / Davies-Bouldin / Calinski-Harabasz
  10. Bedrock §A           Amazon Titan Embed Text v2: rows → embeddings → cluster
  11. Bedrock §B           Claude (haiku): describe cluster behavioral profiles
  12. Post-hoc eval        reveal hidden labels → ARI / NMI per method
  13. Export               CSV assignments + JSON summary

JD vocabulary this script demonstrates:
  "clustering network behaviors"     → §5-8
  "surfacing anomalies that matter"  → DBSCAN / HDBSCAN noise points
  "appropriate observability"        → §9 evaluation dashboard
  "sane failure modes"               → Bedrock graceful fallback
  "data contracts"                   → typed function signatures throughout
  "batch processing patterns"        → full-dataset sklearn clustering
  "evaluate model trustworthiness"   → silhouette < 0.2 warning
"""

import json
import logging
import os
import time
import warnings
from typing import Dict, List, Optional, Tuple

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import DBSCAN, HDBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
OUTPUT_DIR         = "outputs/unlabeled_clustering"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BEDROCK_REGION     = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
EMBED_MODEL_ID     = "amazon.titan-embed-text-v2:0"
INFERENCE_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
EMBED_SAMPLE_SIZE  = 300     # rows to embed via Bedrock (cost control)
K_RANGE            = range(2, 9)
RANDOM_STATE       = 42


# ── Bedrock client setup (sane failure mode) ──────────────────────────────────
def _init_bedrock():
    """Try to create a bedrock-runtime client; return None if unavailable."""
    try:
        import boto3
        client = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
        client.meta.service_model       # force lazy init — validates credentials
        log.info("Bedrock client ready (region=%s)", BEDROCK_REGION)
        return client
    except Exception as exc:
        log.warning(
            "Bedrock unavailable (%s) — §10/11 will be skipped.\n"
            "  To enable: export AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, "
            "AWS_DEFAULT_REGION", exc
        )
        return None

bedrock = _init_bedrock()


# ══════════════════════════════════════════════════════════════════════════════
# §1  DATA LOADING — strip labels, simulate unlabeled ingest
# ══════════════════════════════════════════════════════════════════════════════
log.info("[1/13] Loading ICSSim dataset …")
dataset_dir = kagglehub.dataset_download("alirezadehlaghi/icssim")
df_raw = pd.read_csv(os.path.join(dataset_dir, "Dataset.csv"))

# Ground-truth labels kept SEPARATE — not used until §12 post-hoc eval
hidden_labels: pd.Series = df_raw["IT_M_Label"].copy()

# Drop all label and timestamp columns to simulate arriving unlabeled data
_drop = [c for c in df_raw.columns
         if any(k in c for k in ("Label", "label", "start", "end", "Date",
                                  "Offset", "sAddress", "rAddress",
                                  "sMACs", "rMACs", "sIPs", "rIPs", "protocol"))]
df_num = (
    df_raw.drop(columns=_drop, errors="ignore")
          .select_dtypes(include=[np.number])
)

log.info("  Feature matrix: %d rows × %d columns", *df_num.shape)
log.info("  True labels (held out):\n%s", hidden_labels.value_counts().to_string())


# ══════════════════════════════════════════════════════════════════════════════
# §2  EDA — without peeking at labels
# ══════════════════════════════════════════════════════════════════════════════
log.info("[2/13] EDA …")


def null_audit(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column null counts. Data contract: input all-numeric DataFrame."""
    s = df.isnull().sum()
    return (
        pd.DataFrame({"null_count": s, "null_pct": s / len(df) * 100})
        .query("null_count > 0")
        .sort_values("null_pct", ascending=False)
    )


def skewness_inventory(df: pd.DataFrame, threshold: float = 1.0) -> pd.Series:
    """Features with |skewness| > threshold — candidates for log transform."""
    sk = df.skew(numeric_only=True)
    return sk[sk.abs() > threshold].sort_values(key=abs, ascending=False)


null_rpt = null_audit(df_num)
skew_rpt = skewness_inventory(df_num)
log.info("  Columns with nulls: %d", len(null_rpt))
log.info("  High-skew features (|skew|>1): %d", len(skew_rpt))
null_rpt.to_csv(f"{OUTPUT_DIR}/eda_null_audit.csv")
skew_rpt.to_csv(f"{OUTPUT_DIR}/eda_skew_inventory.csv")


# ══════════════════════════════════════════════════════════════════════════════
# §3  PREPROCESSING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
log.info("[3/13] Preprocessing …")


def build_feature_matrix(
    df: pd.DataFrame,
    log_skew_threshold: float = 2.0,
) -> Tuple[np.ndarray, List[str], StandardScaler]:
    """
    Impute medians → log1p high-skew features → StandardScaler.

    Data contract:
      Input:  all-numeric DataFrame, any shape.
      Output: (X_scaled float64, feature_names list, fitted StandardScaler).
              X_scaled is zero-mean, unit-variance.
    """
    X = df.fillna(df.median())
    skew = X.skew()
    log_cols = skew[skew.abs() > log_skew_threshold].index.tolist()
    X = X.copy()
    X.loc[:, log_cols] = np.log1p(X[log_cols].clip(lower=0))
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, list(X.columns), scaler


X_scaled, feature_names, scaler = build_feature_matrix(df_num)
log.info("  Scaled matrix: %s", X_scaled.shape)


# ══════════════════════════════════════════════════════════════════════════════
# §4  PCA — explained-variance elbow, 2-D projection
# ══════════════════════════════════════════════════════════════════════════════
log.info("[4/13] PCA …")

pca_full = PCA(random_state=RANDOM_STATE).fit(X_scaled)
cum_var  = np.cumsum(pca_full.explained_variance_ratio_)
n_95     = int(np.searchsorted(cum_var, 0.95)) + 1

# 2-D projection for all subsequent scatter plots
pca_2d = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca  = pca_2d.fit_transform(X_scaled)

# Reduced space for actual clustering (keeps 95 % variance)
pca_nd   = PCA(n_components=n_95, random_state=RANDOM_STATE)
X_red    = pca_nd.fit_transform(X_scaled)
log.info("  Components for 95%% variance: %d / %d features", n_95, X_scaled.shape[1])

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
k20 = min(20, len(pca_full.explained_variance_ratio_))
axes[0].bar(range(1, k20 + 1), pca_full.explained_variance_ratio_[:k20],
            color="#4e79a7", alpha=0.8)
axes[0].set(xlabel="Component", ylabel="Explained Variance Ratio",
            title="Scree Plot (top-20)")
axes[1].plot(range(1, len(cum_var) + 1), cum_var, color="#e15759", lw=2)
axes[1].axvline(n_95, color="orange", ls="--", label=f"95% @ PC{n_95}")
axes[1].axhline(0.95, color="gray", ls=":")
axes[1].set(xlabel="n_components", ylabel="Cumulative Variance",
            title="Cumulative Explained Variance")
axes[1].legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/pca_scree.png", dpi=110, bbox_inches="tight")
plt.close()
log.info("  Saved pca_scree.png")


# ══════════════════════════════════════════════════════════════════════════════
# §5  K-MEANS — elbow + silhouette selection
# ══════════════════════════════════════════════════════════════════════════════
log.info("[5/13] K-Means …")

inertias, sil_scores = [], []
for k in K_RANGE:
    km  = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    lbl = km.fit_predict(X_red)
    inertias.append(km.inertia_)
    sil_scores.append(
        silhouette_score(X_red, lbl, sample_size=2000, random_state=RANDOM_STATE)
    )

best_k   = list(K_RANGE)[int(np.argmax(sil_scores))]
km_best  = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
km_labels = km_best.fit_predict(X_red)
log.info("  Best k=%d (silhouette=%.3f)", best_k, max(sil_scores))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(list(K_RANGE), inertias, marker="o", color="#4e79a7")
axes[0].set(xlabel="k", ylabel="Inertia (WCSS)", title="K-Means Elbow")
axes[1].plot(list(K_RANGE), sil_scores, marker="o", color="#e15759")
axes[1].axvline(best_k, color="orange", ls="--", label=f"best k={best_k}")
axes[1].set(xlabel="k", ylabel="Silhouette Score", title="Silhouette vs k")
axes[1].legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/kmeans_selection.png", dpi=110, bbox_inches="tight")
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# §6  DBSCAN — ε via k-distance plot; noise points = anomaly candidates
# ══════════════════════════════════════════════════════════════════════════════
log.info("[6/13] DBSCAN …")

# Subsample before NearestNeighbors: kneighbors(X_red) on 45K rows builds an
# O(n²) distance matrix that exhausts RAM. 5K rows is enough to estimate ε
# reliably — the k-distance distribution converges well before that.
DBSCAN_SAMPLE = 5000
db_idx  = np.random.RandomState(RANDOM_STATE).choice(len(X_red),
                                                      size=min(DBSCAN_SAMPLE, len(X_red)),
                                                      replace=False)
X_db    = X_red[db_idx]

nbrs     = NearestNeighbors(n_neighbors=4).fit(X_db)
dists, _ = nbrs.kneighbors(X_db)
knn_dist = np.sort(dists[:, -1])[::-1]

# Heuristic: ε at the maximum second-derivative of the sorted k-distance curve
d2 = np.gradient(np.gradient(knn_dist))
eps_idx  = int(np.argmax(d2[: len(d2) // 2]))   # elbow in first half
eps_auto = float(knn_dist[eps_idx])

dbscan    = DBSCAN(eps=eps_auto, min_samples=10)
db_labels_sub = dbscan.fit_predict(X_db)

# Assign full-dataset labels via nearest-neighbour lookup on the subsample
from sklearn.neighbors import KNeighborsClassifier
knn_assign = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
knn_assign.fit(X_db, db_labels_sub)
db_labels = knn_assign.predict(X_red)
n_db_clust = len(set(db_labels_sub)) - (1 if -1 in db_labels_sub else 0)
n_db_noise = int((db_labels == -1).sum())
log.info("  DBSCAN ε=%.4f → %d clusters, %d noise projected to full set (sample n=%d)",
         eps_auto, n_db_clust, n_db_noise, DBSCAN_SAMPLE)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(knn_dist[:3000], color="#4e79a7", lw=0.7)
ax.axvline(eps_idx, color="orange", ls="--", label=f"ε={eps_auto:.4f} (auto)")
ax.set(xlabel="Points (sorted by 4th-NN distance, desc)",
       ylabel="4th-NN Distance",
       title="DBSCAN k-Distance Plot — ε at the knee")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/dbscan_epsilon.png", dpi=110, bbox_inches="tight")
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# §7  HDBSCAN — soft memberships; low-probability = anomaly candidate
# ══════════════════════════════════════════════════════════════════════════════
log.info("[7/13] HDBSCAN …")

hdb       = HDBSCAN(min_cluster_size=50, min_samples=5, store_centers="medoid")
hdb_labels = hdb.fit_predict(X_red)
hdb_proba  = hdb.probabilities_
n_hdb_clust = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)
n_hdb_noise = int((hdb_labels == -1).sum())
# Soft anomaly signal: legitimate cluster members with low membership confidence
soft_anomalies = int(((hdb_labels != -1) & (hdb_proba < 0.3)).sum())
log.info("  HDBSCAN → %d clusters, %d noise, %d soft anomalies (prob<0.3)",
         n_hdb_clust, n_hdb_noise, soft_anomalies)


# ══════════════════════════════════════════════════════════════════════════════
# §8  AGGLOMERATIVE (Hierarchical) — Ward linkage
# ══════════════════════════════════════════════════════════════════════════════
log.info("[8/13] Agglomerative clustering …")

rng        = np.random.RandomState(RANDOM_STATE)
samp_idx   = rng.choice(len(X_red), size=min(500, len(X_red)), replace=False)
Z          = linkage(X_red[samp_idx], method="ward")

fig, ax = plt.subplots(figsize=(12, 4))
dendrogram(Z, ax=ax, truncate_mode="lastp", p=20, show_leaf_counts=True,
           color_threshold=0.7 * float(Z[:, 2].max()))
ax.set(title="Hierarchical Clustering — Ward Linkage (n=500 sample, truncated)",
       xlabel="Cluster Size", ylabel="Ward Distance")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/hierarchical_dendrogram.png", dpi=110, bbox_inches="tight")
plt.close()

agg_labels = AgglomerativeClustering(n_clusters=best_k, linkage="ward").fit_predict(X_red)
log.info("  Agglomerative Ward k=%d", best_k)


# ── Combined scatter: all 4 methods in 2-D PCA ───────────────────────────────
PALETTE     = plt.cm.tab10.colors
NOISE_COLOR = "#bbbbbb"
all_methods = {
    f"K-Means (k={best_k})":                    km_labels,
    f"DBSCAN (ε={eps_auto:.4f}, noise={n_db_noise:,})": db_labels,
    f"HDBSCAN ({n_hdb_clust} clusters)":        hdb_labels,
    f"Agglomerative Ward (k={best_k})":         agg_labels,
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, (title, lbl) in zip(axes.flatten(), all_methods.items()):
    for cid in sorted(set(lbl)):
        mask  = lbl == cid
        color = NOISE_COLOR if cid == -1 else PALETTE[cid % len(PALETTE)]
        label = "Noise" if cid == -1 else f"C{cid}"
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=color, s=4, alpha=0.3, label=label)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)", fontsize=7)
    ax.set_ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)", fontsize=7)
    ax.legend(markerscale=3, fontsize=7, ncol=2)
fig.suptitle("Cluster Assignments — 2-D PCA Projection (unlabeled ICS/OT data)", fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/cluster_scatters.png", dpi=110, bbox_inches="tight")
plt.close()
log.info("  Saved cluster_scatters.png")

cluster_results: Dict[str, np.ndarray] = {
    "kmeans":          km_labels,
    "dbscan":          db_labels,
    "hdbscan":         hdb_labels,
    "agglomerative":   agg_labels,
}


# ══════════════════════════════════════════════════════════════════════════════
# §9  CLUSTER EVALUATION DASHBOARD
# Observability question: are these outputs trustworthy for downstream use?
# ══════════════════════════════════════════════════════════════════════════════
log.info("[9/13] Evaluation dashboard …")

LOW_SIL_THRESHOLD = 0.20


def evaluate_clustering(
    X: np.ndarray,
    labels: np.ndarray,
    method: str,
) -> Dict[str, float]:
    """
    Silhouette, Davies-Bouldin, and Calinski-Harabasz scores.
    Noise points (label == -1) are excluded from metric computation.
    Returns NaN for degenerate solutions (< 2 non-noise clusters).

    Data contract: X float64 (n_samples, n_features), labels int (n_samples,).
    """
    non_noise_mask = labels != -1
    unique_clusters = set(labels[non_noise_mask])
    if len(unique_clusters) < 2:
        log.warning("  %s: <2 non-noise clusters — metrics undefined", method)
        return {"silhouette": float("nan"), "davies_bouldin": float("nan"),
                "calinski_harabasz": float("nan")}
    Xm, ym = X[non_noise_mask], labels[non_noise_mask]
    return {
        "silhouette":        float(silhouette_score(Xm, ym, sample_size=2000,
                                                    random_state=RANDOM_STATE)),
        "davies_bouldin":    float(davies_bouldin_score(Xm, ym)),
        "calinski_harabasz": float(calinski_harabasz_score(Xm, ym)),
    }


metrics_rows = []
for method_name, labels in cluster_results.items():
    row = evaluate_clustering(X_red, labels, method_name)
    row["method"]     = method_name
    row["n_clusters"] = len(set(labels[labels != -1]))
    row["n_noise"]    = int((labels == -1).sum())
    metrics_rows.append(row)

metrics_df = pd.DataFrame(metrics_rows).set_index("method")
metrics_df.to_csv(f"{OUTPUT_DIR}/cluster_metrics.csv")
log.info("\n%s", metrics_df.to_string())

# Trustworthiness gate — flag weak solutions before they propagate downstream
for method_name, row in metrics_df.iterrows():
    sil = row["silhouette"]
    if not np.isnan(sil) and sil < LOW_SIL_THRESHOLD:
        log.warning(
            "  ⚠  %s silhouette=%.3f < %.2f — separation is weak; "
            "outputs may not be trustworthy for asset classification.",
            method_name, sil, LOW_SIL_THRESHOLD
        )

# Evaluation bar chart
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
valid = metrics_df.dropna(subset=["silhouette", "davies_bouldin", "calinski_harabasz"])
for ax, (col, better) in zip(axes, [
    ("silhouette",        "higher"),
    ("davies_bouldin",    "lower"),
    ("calinski_harabasz", "higher"),
]):
    vals   = valid[col]
    best_v = vals.max() if better == "higher" else vals.min()
    colors = ["#4e79a7" if v == best_v else "#aaaaaa" for v in vals]
    ax.bar(vals.index, vals, color=colors, alpha=0.85)
    ax.set_title(f"{col.replace('_', ' ').title()}\n({better} is better)", fontsize=9)
    ax.tick_params(axis="x", rotation=15)
    ax.set_ylabel("Score")
fig.suptitle("Cluster Quality — Evaluation Dashboard", fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/evaluation_dashboard.png", dpi=110, bbox_inches="tight")
plt.close()
log.info("  Saved evaluation_dashboard.png")


# ══════════════════════════════════════════════════════════════════════════════
# §10  AWS BEDROCK — Amazon Titan Embed Text v2
#      Embed rows in semantic space, then cluster the embedding vectors.
#      Demonstrates using a Bedrock FM as a feature extractor (inference,
#      not training) — exactly the "put existing model types to work" pattern.
# ══════════════════════════════════════════════════════════════════════════════
log.info("[10/13] Bedrock §A — Titan Embeddings …")


def serialize_row(row: pd.Series, top_k: int = 12) -> str:
    """
    Convert a feature vector to text for embedding.
    Titan Embed Text v2 ingests strings, so we describe the flow in natural form.
    Only the top-k features by absolute value are included (8K-token budget).
    """
    top_feats = row.abs().nlargest(top_k).index
    parts = [f"{feat}={row[feat]:.4g}" for feat in top_feats]
    return "ICS network flow — " + ", ".join(parts)


def embed_row_bedrock(text: str, client) -> Optional[List[float]]:
    """
    Call Titan Embed Text v2 → 256-dim embedding.
    Returns None on any error (API error, throttle, malformed response).
    Data contract: input str ≤ 8192 tokens; output List[float] length 256.
    """
    try:
        body = json.dumps({"inputText": text, "dimensions": 256, "normalize": True})
        resp = client.invoke_model(
            modelId=EMBED_MODEL_ID,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        return json.loads(resp["body"].read())["embedding"]
    except Exception as exc:
        log.debug("  Embedding error: %s", exc)
        return None


bedrock_cluster_labels: Optional[np.ndarray] = None

if bedrock is not None:
    log.info("  Sampling %d rows for embedding …", EMBED_SAMPLE_SIZE)
    samp_emb = rng.choice(len(df_num), size=min(EMBED_SAMPLE_SIZE, len(df_num)),
                           replace=False)
    df_emb_sample = df_num.iloc[samp_emb].reset_index(drop=True)

    embeddings: List[List[float]] = []
    valid_emb_idx: List[int] = []

    for i, (_, row) in enumerate(df_emb_sample.iterrows()):
        emb = embed_row_bedrock(serialize_row(row), bedrock)
        if emb is not None:
            embeddings.append(emb)
            valid_emb_idx.append(i)
        if (i + 1) % 50 == 0:
            log.info("    Embedded %d/%d …", i + 1, EMBED_SAMPLE_SIZE)
        time.sleep(0.05)    # stay within Bedrock default rate limits

    if len(embeddings) >= 10:
        E = np.array(embeddings)
        km_emb = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
        bedrock_cluster_labels = km_emb.fit_predict(E)
        sil_emb = silhouette_score(E, bedrock_cluster_labels)
        log.info("  Embedding clusters: k=%d, silhouette=%.3f", best_k, sil_emb)

        pca_emb = PCA(n_components=2, random_state=RANDOM_STATE)
        E_pca   = pca_emb.fit_transform(E)
        fig, ax = plt.subplots(figsize=(8, 6))
        for cid in range(best_k):
            mask = bedrock_cluster_labels == cid
            ax.scatter(E_pca[mask, 0], E_pca[mask, 1], s=20, alpha=0.55,
                       color=PALETTE[cid % len(PALETTE)], label=f"C{cid}")
        ax.set_title(
            f"Bedrock Titan Embedding Clusters\n"
            f"k={best_k}, silhouette={sil_emb:.3f} "
            f"(n={len(embeddings)} rows)"
        )
        ax.legend(markerscale=2)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/bedrock_embedding_clusters.png",
                    dpi=110, bbox_inches="tight")
        plt.close()
        log.info("  Saved bedrock_embedding_clusters.png")

        # Compare embedding-cluster solution to sklearn K-Means
        # (can only compare on the overlapping sample rows)
        ari_vs_km = adjusted_rand_score(km_labels[samp_emb][valid_emb_idx],
                                        bedrock_cluster_labels)
        log.info("  Bedrock-embedding vs sklearn K-Means ARI=%.3f "
                 "(1=identical, 0=random)", ari_vs_km)
    else:
        log.warning("  Only %d embeddings succeeded — skipping embedding clustering.",
                    len(embeddings))
else:
    log.info("  Bedrock unavailable — skipping Titan Embeddings section.")


# ══════════════════════════════════════════════════════════════════════════════
# §11  AWS BEDROCK — Claude cluster interpretation
#      After clustering, use a Bedrock FM to translate feature statistics
#      into human-readable behavioral descriptions — "what does cluster 2 mean
#      for an ICS/OT analyst?"
# ══════════════════════════════════════════════════════════════════════════════
log.info("[11/13] Bedrock §B — Claude cluster interpretation …")


def cluster_profile(
    df_feats: pd.DataFrame,
    labels: np.ndarray,
    cluster_id: int,
    top_n: int = 8,
) -> Dict[str, float]:
    """
    Return mean values of the top-n most discriminative features for one cluster.
    Discriminative = largest absolute deviation of cluster mean from global mean.
    Data contract: labels int array aligned with df_feats index.
    """
    mask = labels == cluster_id
    c    = df_feats.loc[mask]
    diff = (c.mean() - df_feats.mean()).abs().nlargest(top_n).index
    return {col: round(float(c[col].mean()), 4) for col in diff}


def describe_cluster_bedrock(
    profile: Dict[str, float],
    client,
    cluster_id: int,
) -> str:
    """
    Invoke Claude on Bedrock to produce a 2-3 sentence behavioral description.
    Data contract: profile is {feature_name: mean_value}; returns plain-text str.
    """
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 300,
        "messages": [{
            "role": "user",
            "content": (
                "You are an ICS/OT security analyst. "
                f"Cluster {cluster_id} in an unlabeled network dataset has "
                "this behavioral profile (top discriminative feature means):\n"
                f"{json.dumps(profile, indent=2)}\n\n"
                "In 2-3 sentences: what type of network behavior does this cluster "
                "likely represent? Could it indicate normal ICS traffic, DDoS, "
                "port scan, replay, MITM, or IP scan? State your confidence level."
            ),
        }],
    })
    try:
        resp = client.invoke_model(
            modelId=INFERENCE_MODEL_ID,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        return json.loads(resp["body"].read())["content"][0]["text"].strip()
    except Exception as exc:
        return f"[Bedrock inference unavailable: {exc}]"


interpretations: Dict[int, str] = {}

if bedrock is not None:
    cluster_ids = sorted(c for c in set(km_labels) if c != -1)
    for cid in cluster_ids:
        profile = cluster_profile(df_num, km_labels, cid)
        desc    = describe_cluster_bedrock(profile, bedrock, cid)
        interpretations[cid] = desc
        preview = desc[:120] + "…" if len(desc) > 120 else desc
        log.info("  Cluster %d: %s", cid, preview)
    with open(f"{OUTPUT_DIR}/cluster_interpretations.json", "w") as f:
        json.dump(interpretations, f, indent=2)
    log.info("  Saved cluster_interpretations.json")
else:
    log.info("  Bedrock unavailable — showing feature-based heuristics:")
    for cid in sorted(c for c in set(km_labels) if c != -1):
        profile  = cluster_profile(df_num, km_labels, cid)
        top_feat = max(profile, key=profile.__getitem__)
        log.info("  Cluster %d dominant feature: %s=%.2f", cid, top_feat, profile[top_feat])


# ══════════════════════════════════════════════════════════════════════════════
# §12  POST-HOC EVALUATION — reveal hidden labels
#      ARI and NMI tell us whether our clusters align with actual attack types.
#      This is the "evaluate whether model outputs are trustworthy" step.
# ══════════════════════════════════════════════════════════════════════════════
log.info("[12/13] Post-hoc evaluation (revealing hidden labels) …")
log.info("  True label distribution:\n%s", hidden_labels.value_counts().to_string())

eval_rows = []
for method_name, labels in cluster_results.items():
    ari = adjusted_rand_score(hidden_labels, labels)
    nmi = normalized_mutual_info_score(hidden_labels, labels)
    eval_rows.append({"method": method_name, "ARI": round(ari, 4), "NMI": round(nmi, 4)})

posthoc_df = pd.DataFrame(eval_rows).set_index("method")
posthoc_df.to_csv(f"{OUTPUT_DIR}/posthoc_evaluation.csv")
log.info("\nPost-hoc scores (ARI and NMI: 1=perfect, 0=random):\n%s",
         posthoc_df.to_string())

fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(posthoc_df))
w = 0.35
ax.bar(x - w/2, posthoc_df["ARI"], w, label="ARI", color="#4e79a7", alpha=0.85)
ax.bar(x + w/2, posthoc_df["NMI"], w, label="NMI", color="#e15759", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(posthoc_df.index, rotation=10)
ax.axhline(0, color="black", lw=0.5)
ax.set_ylabel("Score (1 = perfect alignment with true labels)")
ax.set_title("Post-hoc: How Well Do Clusters Align with True Attack Labels?")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/posthoc_ari_nmi.png", dpi=110, bbox_inches="tight")
plt.close()
log.info("  Saved posthoc_ari_nmi.png")

best_method = posthoc_df["ARI"].idxmax()
best_ari    = float(posthoc_df["ARI"].max())
log.info(
    "\n  Interpretation:\n"
    "  Best: %s (ARI=%.3f)\n"
    "  ARI < 0.3 is common — attack classes don't always form tight geometric\n"
    "  clusters in raw feature space. The replay-detection script shows why:\n"
    "  some attacks (replay) suppress variance within existing states rather\n"
    "  than shifting to a new region. Unsupervised clustering surfaces\n"
    "  behavioral groups; cross-layer correlation is still required to\n"
    "  distinguish some attack signatures from normal variation.",
    best_method, best_ari
)


# ══════════════════════════════════════════════════════════════════════════════
# §13  EXPORT
# ══════════════════════════════════════════════════════════════════════════════
log.info("[13/13] Exporting …")

export_df = df_num.copy()
for method_name, labels in cluster_results.items():
    export_df[f"cluster_{method_name}"] = labels
export_df["true_label"] = hidden_labels.values
export_df.to_csv(f"{OUTPUT_DIR}/cluster_assignments.csv", index=False)

summary = {
    "n_rows":             int(len(df_num)),
    "n_features":         int(df_num.shape[1]),
    "n_components_95pct": int(n_95),
    "kmeans_best_k":      int(best_k),
    "dbscan_auto_eps":    float(eps_auto),
    "dbscan_n_noise":     int(n_db_noise),
    "hdbscan_n_noise":    int(n_hdb_noise),
    "hdbscan_soft_anomalies": int(soft_anomalies),
    "cluster_metrics":    metrics_df.fillna("NA").to_dict(),
    "posthoc_scores":     posthoc_df.to_dict(),
}
with open(f"{OUTPUT_DIR}/pipeline_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

log.info("  Written to %s/", OUTPUT_DIR)
log.info("  cluster_assignments.csv — all methods + true labels")
log.info("  pipeline_summary.json  — metrics + k choices")
log.info("\nDone.")
