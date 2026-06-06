"""
Full EDA Suite — ICSSim v2 Dataset

Generates seven chart families across three data sources (network flows, PLC1,
PLC2) and saves them to outputs/charts/:

  1. Histograms       — per-feature density by attack class
  2. Correlation      — Pearson heatmap (top features by label correlation)
  3. Mann-Whitney U   — effect size + significance for every numeric feature
  4. Chi-square       — categorical feature association with attack label
  5. Regression       — per-class OLS lines showing moderation by label
  6. PCA              — 2-D projection coloured by class
  7. Cronbach α       — internal consistency per feature group, Normal vs Attack
"""

import os
import warnings
import kagglehub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")
os.makedirs("outputs/charts", exist_ok=True)

PALETTE = {
    "Normal": "#4e79a7",
    "replay": "#e15759",
    "ddos": "#f28e2b",
    "port-scan": "#76b7b2",
    "mitm": "#59a14f",
    "ip-scan": "#b07aa1",
}
ATTACK_TYPES = ["replay", "ddos", "port-scan", "mitm", "ip-scan"]

# =========================================================
# 0. LOAD & TIME-ALIGN ALL DATASETS
# =========================================================
print("Loading datasets …")
dataset_dir = kagglehub.dataset_download("alirezadehlaghi/icssim")
df_net  = pd.read_csv(os.path.join(dataset_dir, "Dataset.csv"))
df_plc1 = pd.read_csv(os.path.join(dataset_dir, "snapshots_PLC1.csv"))
df_plc2 = pd.read_csv(os.path.join(dataset_dir, "snapshots_PLC2.csv"))
# Strip leading/trailing whitespace from column names (PLC CSVs have leading spaces)
df_plc1.columns = df_plc1.columns.str.strip()
df_plc2.columns = df_plc2.columns.str.strip()

# -- Elapsed-time alignment (same approach as test.py, extended to all labels)
net_start = pd.to_numeric(df_net["start"], errors="coerce").min()
df_net["elapsed_start"] = pd.to_numeric(df_net["start"], errors="coerce") - net_start
df_net["elapsed_end"]   = pd.to_numeric(df_net["end"],   errors="coerce") - net_start

def label_plc(df_plc, df_net_ref):
    t = pd.to_datetime(df_plc["time"], errors="coerce")
    t0 = t.min()
    df_plc = df_plc.copy()
    df_plc["elapsed_time"] = (t - t0).dt.total_seconds()
    df_plc["IT_M_Label"] = "Normal"
    for atk in ATTACK_TYPES:
        rows = df_net_ref[df_net_ref["IT_M_Label"] == atk]
        if rows.empty:
            continue
        mask = (df_plc["elapsed_time"] >= rows["elapsed_start"].min()) & \
               (df_plc["elapsed_time"] <= rows["elapsed_end"].max())
        df_plc.loc[mask, "IT_M_Label"] = atk
    return df_plc

df_plc1 = label_plc(df_plc1, df_net)
df_plc2 = label_plc(df_plc2, df_net)

print("Network label dist:\n", df_net["IT_M_Label"].value_counts().to_string())
print("\nPLC1 label dist:\n",   df_plc1["IT_M_Label"].value_counts().to_string())
print("\nPLC2 label dist:\n",   df_plc2["IT_M_Label"].value_counts().to_string())

# Helper: colour list aligned to a label series
def label_colors(series):
    return series.map(PALETTE).fillna("#999999")

# =========================================================
# 1. HISTOGRAMS — each numeric feature, faceted by label
# =========================================================
print("\n[1/7] Histograms …")

def plot_histograms(df, label_col, title_prefix, out_stem, n_cols=5):
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                if c not in ("elapsed_time",)]
    n = len(num_cols)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 2.8))
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        ax = axes[i]
        for lbl, grp in df.groupby(label_col):
            ax.hist(grp[col].dropna(), bins=40, alpha=0.5,
                    color=PALETTE.get(lbl, "#999"), label=lbl, density=True)
        ax.set_title(col, fontsize=7, pad=2)
        ax.tick_params(labelsize=6)
        ax.set_xlabel("")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    # single legend on first axis
    handles = [plt.Rectangle((0,0),1,1, color=PALETTE.get(l,"#999"), alpha=0.6)
               for l in df[label_col].unique()]
    labels_u = list(df[label_col].unique())
    fig.legend(handles, labels_u, loc="lower right", ncol=3, fontsize=7,
               title="Label", title_fontsize=7)
    fig.suptitle(f"{title_prefix} — Feature Histograms by Label", fontsize=11, y=1.01)
    plt.tight_layout()
    path = f"outputs/charts/{out_stem}_histograms.png"
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")

# Select a meaningful subset for Network (53 cols → top 20 by variance)
net_num = df_net.select_dtypes(include=[np.number]).drop(
    columns=["start", "end", "startOffset", "endOffset",
             "elapsed_start", "elapsed_end", "IT_B_Label", "NST_B_Label"], errors="ignore")
top20_net = net_num.var().nlargest(20).index.tolist()
plot_histograms(df_net[top20_net + ["IT_M_Label"]], "IT_M_Label",
                "Network (top-20 by variance)", "network")
plot_histograms(df_plc1, "IT_M_Label", "PLC1", "plc1")
plot_histograms(df_plc2, "IT_M_Label", "PLC2", "plc2")

# =========================================================
# 2. CORRELATION HEATMAPS
# =========================================================
print("\n[2/7] Correlation heatmaps …")

def plot_corr(df, label_col, title, out_path, max_cols=25):
    num = df.select_dtypes(include=[np.number])
    # encode label numerically for correlation
    label_enc = pd.Categorical(df[label_col]).codes
    num["__label__"] = label_enc
    if len(num.columns) > max_cols:
        # keep top features by abs-corr with label
        corr_with_label = num.corr()["__label__"].abs().drop("__label__").nlargest(max_cols - 1)
        num = num[corr_with_label.index.tolist() + ["__label__"]]
    corr = num.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(max(10, len(corr) * 0.45),
                                    max(8,  len(corr) * 0.42)))
    sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0, linewidths=0.3,
                annot=(len(corr) <= 15), fmt=".2f", annot_kws={"size": 7},
                ax=ax, cbar_kws={"shrink": 0.7})
    ax.set_title(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")

plot_corr(df_net, "IT_M_Label",
          "Network — Correlation Heatmap (top-24 features + label)",
          "outputs/charts/network_corr_heatmap.png", max_cols=25)
plot_corr(df_plc1, "IT_M_Label",
          "PLC1 — Correlation Heatmap",
          "outputs/charts/plc1_corr_heatmap.png", max_cols=30)
plot_corr(df_plc2, "IT_M_Label",
          "PLC2 — Correlation Heatmap",
          "outputs/charts/plc2_corr_heatmap.png", max_cols=30)

# =========================================================
# 3. HYPOTHESIS TESTS — Mann-Whitney U (attack vs Normal)
# =========================================================
print("\n[3/7] Mann-Whitney U tests …")

def mannwhitney_chart(df, label_col, title, out_path, top_n=20):
    normal_mask = df[label_col] == "Normal"
    attack_mask = ~normal_mask
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                if c not in ("elapsed_time",)]
    results = []
    for col in num_cols:
        a = df.loc[normal_mask, col].dropna()
        b = df.loc[attack_mask,  col].dropna()
        if len(a) < 5 or len(b) < 5:
            continue
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        effect = (u / (len(a) * len(b))) * 2 - 1  # rank-biserial r
        results.append({"feature": col, "p_value": p, "effect_r": abs(effect)})
    res = pd.DataFrame(results).sort_values("effect_r", ascending=False).head(top_n)

    fig, axes = plt.subplots(1, 2, figsize=(14, top_n * 0.35 + 2))
    # Effect size
    axes[0].barh(res["feature"][::-1], res["effect_r"][::-1], color="#4e79a7", alpha=0.8)
    axes[0].set_xlabel("Rank-biserial |r| (effect size)")
    axes[0].set_title("Effect Size (Attack vs Normal)")
    axes[0].axvline(0.1, color="orange", ls="--", lw=0.8, label="small (0.1)")
    axes[0].axvline(0.3, color="red",    ls="--", lw=0.8, label="medium (0.3)")
    axes[0].legend(fontsize=7)
    # -log10(p)
    neg_log_p = -np.log10(res["p_value"].clip(lower=1e-300))
    colors = ["#e15759" if p < 0.05 else "#999" for p in res["p_value"]]
    axes[1].barh(res["feature"][::-1], neg_log_p[::-1], color=colors, alpha=0.85)
    axes[1].set_xlabel("–log₁₀(p-value)  [red = significant]")
    axes[1].axvline(-np.log10(0.05), color="orange", ls="--", lw=0.8, label="p=0.05")
    axes[1].set_title("Statistical Significance")
    axes[1].legend(fontsize=7)
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")
    return res

mannwhitney_chart(df_net,  "IT_M_Label", "Network — Mann-Whitney U: Attack vs Normal",
                  "outputs/charts/network_mannwhitney.png")
mannwhitney_chart(df_plc1, "IT_M_Label", "PLC1 — Mann-Whitney U: Attack vs Normal",
                  "outputs/charts/plc1_mannwhitney.png")
mannwhitney_chart(df_plc2, "IT_M_Label", "PLC2 — Mann-Whitney U: Attack vs Normal",
                  "outputs/charts/plc2_mannwhitney.png")

# =========================================================
# 4. CHI-SQUARE — categorical features vs label
# =========================================================
print("\n[4/7] Chi-square tests …")

def chisquare_chart(df, label_col, cat_cols, title, out_path):
    results = []
    for col in cat_cols:
        if col not in df.columns:
            continue
        ct = pd.crosstab(df[col], df[label_col])
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        n = ct.values.sum()
        cramers_v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
        results.append({"feature": col, "chi2": chi2, "p_value": p,
                         "cramers_v": cramers_v, "dof": dof})
    if not results:
        print(f"  No categorical columns found for {title}")
        return
    res = pd.DataFrame(results).sort_values("cramers_v", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, max(4, len(res) * 0.6 + 2)))
    # Cramer's V
    axes[0].barh(res["feature"][::-1], res["cramers_v"][::-1], color="#76b7b2", alpha=0.85)
    axes[0].set_xlabel("Cramér's V (association strength)")
    axes[0].axvline(0.1, color="orange", ls="--", lw=0.8, label="weak")
    axes[0].axvline(0.3, color="red",    ls="--", lw=0.8, label="moderate")
    axes[0].set_title("Association Strength")
    axes[0].legend(fontsize=7)
    # -log10(p)
    neg_log_p = -np.log10(res["p_value"].clip(lower=1e-300))
    colors = ["#e15759" if p < 0.05 else "#999" for p in res["p_value"]]
    axes[1].barh(res["feature"][::-1], neg_log_p[::-1], color=colors, alpha=0.85)
    axes[1].set_xlabel("–log₁₀(p-value)")
    axes[1].axvline(-np.log10(0.05), color="orange", ls="--", lw=0.8, label="p=0.05")
    axes[1].set_title("Significance")
    axes[1].legend(fontsize=7)
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")

    # Stacked bar: class distribution by protocol / sAddress top-10
    for col in cat_cols[:2]:
        if col not in df.columns:
            continue
        top_vals = df[col].value_counts().head(10).index
        sub = df[df[col].isin(top_vals)]
        ct_norm = pd.crosstab(sub[col], sub[label_col], normalize="index")
        ct_norm.plot(kind="bar", stacked=True, figsize=(10, 4),
                     color=[PALETTE.get(c,"#999") for c in ct_norm.columns])
        plt.title(f"Label Distribution within {col} (top-10)")
        plt.ylabel("Proportion")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        p2 = f"outputs/charts/network_chi2_{col}_stacked.png"
        plt.savefig(p2, dpi=110, bbox_inches="tight")
        plt.close()
        print(f"  Saved {p2}")

chisquare_chart(df_net, "IT_M_Label",
                ["protocol", "sAddress", "rAddress"],
                "Network — Chi-Square: Categorical Features vs Label",
                "outputs/charts/network_chisquare.png")

# =========================================================
# 5. REGRESSION + MODERATION ANALYSIS
# =========================================================
print("\n[5/7] Regression + moderation …")

def regression_moderation(df, label_col, pairs, title_prefix, out_stem):
    """
    For each (X, Y) pair:
    - scatter coloured by label (moderation visual)
    - OLS regression line per class
    - print Pearson r and slope per class
    """
    n = len(pairs)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, (xcol, ycol) in zip(axes, pairs):
        if xcol not in df.columns or ycol not in df.columns:
            ax.set_visible(False)
            continue
        for lbl, grp in df.groupby(label_col):
            x = grp[xcol].dropna()
            y = grp[ycol].dropna()
            common = x.index.intersection(y.index)
            x, y = x.loc[common], y.loc[common]
            if len(x) < 10:
                continue
            color = PALETTE.get(lbl, "#999")
            ax.scatter(x, y, c=color, alpha=0.25, s=6, label=lbl)
            # OLS line
            if x.nunique() < 2:
                continue
            m, b, r, p, _ = stats.linregress(x, y)
            xr = np.linspace(x.quantile(0.01), x.quantile(0.99), 100)
            ax.plot(xr, m * xr + b, color=color, lw=1.5, alpha=0.85)
        ax.set_xlabel(xcol, fontsize=9)
        ax.set_ylabel(ycol, fontsize=9)
        ax.set_title(f"{xcol}  →  {ycol}", fontsize=9)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center",
               ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(f"{title_prefix} — Regression + Moderation by Label", fontsize=11)
    plt.tight_layout()
    path = f"outputs/charts/{out_stem}_regression.png"
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")

regression_moderation(
    df_net, "IT_M_Label",
    [("duration", "sBytesSum"),
     ("sLoad",    "rLoad"),
     ("sSynRate", "sRstRate")],
    "Network", "network"
)
regression_moderation(
    df_plc1, "IT_M_Label",
    [("tank_level_value(2)", "tank_output_flow_value(7)"),
     ("loop_latency",        "logic_execution_time"),
     ("tank_input_valve_status(0)", "tank_level_value(2)")],
    "PLC1", "plc1"
)
regression_moderation(
    df_plc2, "IT_M_Label",
    [("bottle_level_value(10)", "bottle_distance_to_filler_value(12)"),
     ("conveyor_belt_engine_status(8)", "bottle_level_value(10)"),
     ("loop_latency",                   "logic_execution_time")],
    "PLC2", "plc2"
)

# =========================================================
# 6. PCA — 2-D class separation
# =========================================================
print("\n[6/7] PCA …")

def pca_plot(df, label_col, title, out_path, drop_extra=()):
    drop = list(drop_extra) + ["elapsed_time", "IT_M_Label",
                                "IT_B_Label", "NST_B_Label", "NST_M_Label",
                                "start", "end", "startOffset", "endOffset",
                                "elapsed_start", "elapsed_end"]
    X = df.drop(columns=[c for c in drop if c in df.columns], errors="ignore")
    X = X.select_dtypes(include=[np.number]).fillna(0)
    if X.shape[1] < 2:
        return
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    Xp = pca.fit_transform(Xs)
    labels = df[label_col]
    fig, ax = plt.subplots(figsize=(9, 6))
    for lbl in labels.unique():
        idx = labels == lbl
        ax.scatter(Xp[idx, 0], Xp[idx, 1], c=PALETTE.get(lbl, "#999"),
                   label=lbl, alpha=0.4, s=8)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.set_title(title)
    ax.legend(markerscale=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")

pca_plot(df_net,  "IT_M_Label", "Network PCA — 2D Class Separation",
         "outputs/charts/network_pca.png",
         drop_extra=["sAddress","rAddress","sMACs","rMACs","sIPs","rIPs",
                     "protocol","startDate","endDate"])
pca_plot(df_plc1, "IT_M_Label", "PLC1 PCA — 2D Class Separation",
         "outputs/charts/plc1_pca.png")
pca_plot(df_plc2, "IT_M_Label", "PLC2 PCA — 2D Class Separation",
         "outputs/charts/plc2_pca.png")

# =========================================================
# 7. CRONBACH'S ALPHA — reliability per feature group
# =========================================================
print("\n[7/7] Cronbach's alpha …")

def cronbach_alpha(df_items):
    """Compute Cronbach's alpha for a DataFrame of items (rows=obs, cols=items)."""
    df_items = df_items.dropna()
    k = df_items.shape[1]
    if k < 2:
        return np.nan
    item_variances = df_items.var(axis=0, ddof=1).sum()
    total_variance  = df_items.sum(axis=1).var(ddof=1)
    if total_variance == 0:
        return np.nan
    return (k / (k - 1)) * (1 - item_variances / total_variance)

# Network feature groups (by conceptual domain)
net_num_all = df_net.select_dtypes(include=[np.number])
groups_net = {
    "Send-side bytes\n(sBytes*)":     [c for c in net_num_all if c.startswith("sBytes")],
    "Recv-side bytes\n(rBytes*)":     [c for c in net_num_all if c.startswith("rBytes")],
    "Send-side payload\n(sPayload*)": [c for c in net_num_all if c.startswith("sPayload")],
    "Recv-side payload\n(rPayload*)": [c for c in net_num_all if c.startswith("rPayload")],
    "Send-side flags\n(s*Rate)":      [c for c in net_num_all if c.startswith("s") and c.endswith("Rate")],
    "Recv-side flags\n(r*Rate)":      [c for c in net_num_all if c.startswith("r") and c.endswith("Rate")],
    "ACK delays":                     [c for c in net_num_all if "AckDelay" in c],
    "Load":                           [c for c in net_num_all if "Load" in c],
}
groups_plc1 = {
    "Valve status/mode":   [c for c in df_plc1.select_dtypes(include=[np.number]).columns
                            if "valve" in c.lower()],
    "Tank level":          [c for c in df_plc1.select_dtypes(include=[np.number]).columns
                            if "tank_level" in c.lower()],
    "Timing":              [c for c in df_plc1.select_dtypes(include=[np.number]).columns
                            if "latency" in c.lower() or "execution" in c.lower()],
}
groups_plc2 = {
    "Conveyor":    [c for c in df_plc2.select_dtypes(include=[np.number]).columns
                    if "conveyor" in c.lower()],
    "Bottle":      [c for c in df_plc2.select_dtypes(include=[np.number]).columns
                    if "bottle" in c.lower()],
    "Timing":      [c for c in df_plc2.select_dtypes(include=[np.number]).columns
                    if "latency" in c.lower() or "execution" in c.lower()],
}

def cronbach_bar(group_dict, df_dict, labels_list, title, out_path):
    """
    group_dict: {group_name: [col_names]}
    df_dict:    {label: df}  or just pass a single df
    """
    records = []
    for grp_name, cols in group_dict.items():
        valid_cols = [c for c in cols if c in list(df_dict.values())[0].columns]
        if len(valid_cols) < 2:
            continue
        for lbl, df_sub in df_dict.items():
            a = cronbach_alpha(df_sub[valid_cols])
            records.append({"group": grp_name, "label": lbl, "alpha": a,
                             "n_items": len(valid_cols)})
    if not records:
        return
    res = pd.DataFrame(records)
    pivot = res.pivot(index="group", columns="label", values="alpha")
    fig, ax = plt.subplots(figsize=(max(8, len(pivot) * 1.3), 5))
    pivot.plot(kind="bar", ax=ax,
               color=[PALETTE.get(l, "#999") for l in pivot.columns],
               alpha=0.8, edgecolor="white")
    ax.axhline(0.7,  color="orange", ls="--", lw=1, label="acceptable (0.7)")
    ax.axhline(0.9,  color="green",  ls="--", lw=1, label="excellent (0.9)")
    ax.set_ylim(-0.1, 1.15)
    ax.set_ylabel("Cronbach's α")
    ax.set_title(title)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=20)
    handles, legend_labels = ax.get_legend_handles_labels()
    ax.legend(handles, legend_labels, fontsize=7, ncol=3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")

# Split by Normal vs Attack for Cronbach
def split_label(df, label_col):
    return {
        "Normal": df[df[label_col] == "Normal"],
        "Attack": df[df[label_col] != "Normal"],
    }

cronbach_bar(groups_net,  split_label(df_net,  "IT_M_Label"),
             ["Normal", "Attack"],
             "Network — Cronbach's α by Feature Group (Normal vs Attack)",
             "outputs/charts/network_cronbach.png")
cronbach_bar(groups_plc1, split_label(df_plc1, "IT_M_Label"),
             ["Normal", "Attack"],
             "PLC1 — Cronbach's α by Feature Group (Normal vs Attack)",
             "outputs/charts/plc1_cronbach.png")
cronbach_bar(groups_plc2, split_label(df_plc2, "IT_M_Label"),
             ["Normal", "Attack"],
             "PLC2 — Cronbach's α by Feature Group (Normal vs Attack)",
             "outputs/charts/plc2_cronbach.png")

print("\nAll charts saved to charts/")
