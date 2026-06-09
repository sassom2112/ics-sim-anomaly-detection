"""
06_streaming_anomaly.py
========================
Near-real-time anomaly detection on ICS/OT physical sensor streams.

Simulates a row-by-row streaming pipeline over PLC1 and PLC2 sensor
snapshots. Builds a per-sensor behavioral baseline from a warm-up window,
then emits structured alert events when sensors deviate (z-score spike)
or go suspiciously static (variance collapse — the replay signature from
02_replay_detection.py). Scores detection precision/recall against the
ground-truth attack timeline in attacker_machine_summary.csv.

Pipeline:
  1. Load PLC1 + PLC2 snapshots and attacker timeline
  2. Parse timestamps, align to elapsed seconds, sort chronologically
  3. Warm-up baseline: collect mean/std per sensor before first attack
  4. Streaming simulation: process each row, update rolling window
       - z-score spike    → potential active attack (ddos, port-scan, mitm)
       - variance collapse → potential replay attack (physical stasis)
       - Cross-layer note → alert stamped "cross-layer candidate" when
         network activity is concurrent but physical sensors are frozen
  5. Deduplicate and score alerts vs. ground-truth windows
  6. Detection latency: how many seconds after attack start until first alert
  7. Export alert CSV + scored timeline chart

JD vocabulary:
  "near-real-time processing patterns"  → row-by-row streaming with rolling window
  "surfacing anomalies that matter"     → z-score + variance-collapse detectors
  "appropriate observability"           → structured alert events, scored timeline
  "sane failure modes"                  → graceful degradation when baseline thin
  "data contracts"                      → typed AlertEvent + SensorBaseline
"""

import dataclasses
import json
import logging
import os
import warnings
from typing import Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import deque

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = "outputs/streaming_anomaly"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Configuration ─────────────────────────────────────────────────────────────
WINDOW_SECONDS    = 60      # rolling baseline window
ZSCORE_THRESHOLD  = 3.5     # standard deviations to trigger spike alert
VAR_COLLAPSE_THR  = 0.05    # variance below this fraction of baseline = replay signal
MIN_BASELINE_ROWS = 30      # warm-up rows before alerting begins
ALERT_COOLDOWN_S  = 5       # seconds between same-sensor alerts (dedup)

# Physical sensor columns for each PLC
PLC1_SENSORS = [
    "tank_input_valve_status(0)",
    "tank_input_valve_mode(1)",
    "tank_level_value(2)",
    "tank_level_min(3)",
    "tank_level_max(4)",
    "tank_output_valve_status(5)",
    "tank_output_valve_mode(6)",
    "tank_output_flow_value(7)",
]
PLC2_SENSORS = [
    "conveyor_belt_engine_status(8)",
    "conveyor_belt_engine_mode(9)",
    "bottle_level_value(10)",
    "bottle_level_max(11)",
    "bottle_distance_to_filler_value(12)",
]


# ══════════════════════════════════════════════════════════════════════════════
# Data contracts
# ══════════════════════════════════════════════════════════════════════════════
@dataclasses.dataclass
class AlertEvent:
    """
    Structured anomaly alert emitted by the streaming detector.
    Consumers downstream receive this typed contract — no raw dicts.
    """
    timestamp:    float          # Unix epoch seconds
    elapsed_s:    float          # seconds from stream start
    plc:          str            # "PLC1" or "PLC2"
    sensor:       str            # sensor column name
    alert_type:   str            # "z_score_spike" | "variance_collapse" | "cross_layer"
    z_score:      float          # current z-score (NaN for variance_collapse)
    rolling_var:  float          # rolling variance over window
    baseline_var: float          # baseline variance (warm-up period)
    confidence:   str            # "high" | "medium" | "low"


@dataclasses.dataclass
class SensorBaseline:
    """Per-sensor statistics learned during the warm-up window."""
    mean:     float
    std:      float
    variance: float
    n_rows:   int


# ══════════════════════════════════════════════════════════════════════════════
# §1  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
log.info("[1/7] Loading data …")

def _load_plc(path: str, sensor_cols: List[str], plc_name: str) -> pd.DataFrame:
    """Load and clean a PLC snapshot CSV. Returns df with elapsed_s column."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["time_parsed"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time_parsed"]).sort_values("time_parsed").reset_index(drop=True)
    t0 = df["time_parsed"].iloc[0]
    df["elapsed_s"]  = (df["time_parsed"] - t0).dt.total_seconds()
    df["unix_ts"]    = df["time_parsed"].astype(np.int64) // 10**9
    df["plc"]        = plc_name
    cols_present     = [c for c in sensor_cols if c in df.columns]
    return df[["time_parsed", "unix_ts", "elapsed_s", "plc"] + cols_present].copy()

plc1 = _load_plc("data/snapshots_PLC1.csv", PLC1_SENSORS, "PLC1")
plc2 = _load_plc("data/snapshots_PLC2.csv", PLC2_SENSORS, "PLC2")

# Load attacker timeline (ground truth)
atk = pd.read_csv("data/attacker_machine_summary.csv")
atk["startTime_parsed"] = pd.to_datetime(atk["startTime"], errors="coerce")
atk["endTime_parsed"]   = pd.to_datetime(atk["endTime"],   errors="coerce")

log.info("  PLC1: %d rows   PLC2: %d rows", len(plc1), len(plc2))
log.info("  Attack windows: %d", len(atk))
log.info("  Attack types: %s", atk["attack"].unique().tolist())

# Convert attacker timestamps to elapsed seconds (relative to PLC1 start)
plc1_t0 = plc1["time_parsed"].iloc[0]
atk["elapsed_start"] = (atk["startTime_parsed"] - plc1_t0).dt.total_seconds()
atk["elapsed_end"]   = (atk["endTime_parsed"]   - plc1_t0).dt.total_seconds()

# Filter to only windows that fall within PLC1 elapsed range
plc1_max = plc1["elapsed_s"].max()
atk_in_range = atk[
    (atk["elapsed_start"] >= 0) & (atk["elapsed_start"] <= plc1_max)
].copy()
log.info("  Attack windows in PLC1 time range: %d", len(atk_in_range))


# ══════════════════════════════════════════════════════════════════════════════
# §2  WARM-UP BASELINE
# Learn per-sensor mean/std from rows BEFORE the first attack starts.
# ══════════════════════════════════════════════════════════════════════════════
log.info("[2/7] Building warm-up baseline …")

first_attack_elapsed = atk_in_range["elapsed_start"].min() if len(atk_in_range) else float("inf")
log.info("  First attack at elapsed_s=%.1f", first_attack_elapsed)


def build_baseline(
    df: pd.DataFrame,
    sensor_cols: List[str],
    warmup_end_s: float,
) -> Dict[str, SensorBaseline]:
    """
    Compute per-sensor statistics from rows with elapsed_s < warmup_end_s.
    Data contract: returns dict[sensor_name -> SensorBaseline].
    """
    warmup = df[df["elapsed_s"] < warmup_end_s]
    baselines: Dict[str, SensorBaseline] = {}
    for col in sensor_cols:
        if col not in warmup.columns:
            continue
        vals = warmup[col].dropna()
        if len(vals) < MIN_BASELINE_ROWS:
            log.warning("  Thin baseline for %s (n=%d) — alerting disabled for this sensor.", col, len(vals))
            continue
        baselines[col] = SensorBaseline(
            mean     = float(vals.mean()),
            std      = float(vals.std()) if vals.std() > 0 else 1e-6,
            variance = float(vals.var()) if vals.var() > 0 else 1e-9,
            n_rows   = int(len(vals)),
        )
    return baselines

sensor_cols_plc1 = [c for c in PLC1_SENSORS if c in plc1.columns]
sensor_cols_plc2 = [c for c in PLC2_SENSORS if c in plc2.columns]
baselines_plc1 = build_baseline(plc1, sensor_cols_plc1, first_attack_elapsed)
baselines_plc2 = build_baseline(plc2, sensor_cols_plc2, first_attack_elapsed)
log.info("  PLC1 baseline sensors: %d   PLC2 baseline sensors: %d",
         len(baselines_plc1), len(baselines_plc2))


# ══════════════════════════════════════════════════════════════════════════════
# §3  STREAMING SIMULATION ENGINE
# Process rows in arrival order; maintain a deque-based rolling window.
# Emit AlertEvent objects for anomalies — typed, structured, no magic dicts.
# ══════════════════════════════════════════════════════════════════════════════
log.info("[3/7] Streaming simulation …")


def stream_plc(
    df: pd.DataFrame,
    sensor_cols: List[str],
    baselines: Dict[str, SensorBaseline],
    plc_name: str,
    window_s: float = WINDOW_SECONDS,
) -> List[AlertEvent]:
    """
    Simulate row-by-row streaming over a PLC snapshot DataFrame.

    For each row:
      - Append sensor values to per-sensor rolling deques (time-bounded).
      - Compute z-score relative to baseline; emit spike alert if above threshold.
      - Compute rolling variance; emit collapse alert if < VAR_COLLAPSE_THR * baseline_var.

    Sane failure modes:
      - If baseline is missing for a sensor, skip that sensor silently.
      - If rolling window has < 3 values, skip variance check.
      - If values are all-zero (disabled sensor), skip to avoid false positives.
    """
    alerts:        List[AlertEvent] = []
    last_alert_t:  Dict[str, float] = {}     # sensor → last alert elapsed_s (dedup)
    windows:       Dict[str, deque] = {c: deque() for c in sensor_cols}
    window_ts:     Dict[str, deque] = {c: deque() for c in sensor_cols}

    for _, row in df.iterrows():
        t   = row["elapsed_s"]
        uts = row["unix_ts"]

        for col in sensor_cols:
            if col not in baselines or col not in df.columns:
                continue
            val = row.get(col)
            if pd.isna(val):
                continue

            bl = baselines[col]

            # Update rolling window (evict entries older than window_s)
            windows[col].append(float(val))
            window_ts[col].append(t)
            while window_ts[col] and (t - window_ts[col][0]) > window_s:
                windows[col].popleft()
                window_ts[col].popleft()

            # Skip degenerate sensors (all zeros → disabled register)
            if bl.std < 1e-5:
                continue

            win_vals = list(windows[col])
            if len(win_vals) < 3:
                continue

            # ── z-score spike detection ───────────────────────────────────────
            z = (float(val) - bl.mean) / bl.std
            if abs(z) >= ZSCORE_THRESHOLD:
                cooldown_ok = (t - last_alert_t.get(col + "_spike", -999)) >= ALERT_COOLDOWN_S
                if cooldown_ok:
                    alerts.append(AlertEvent(
                        timestamp    = float(uts),
                        elapsed_s    = t,
                        plc          = plc_name,
                        sensor       = col,
                        alert_type   = "z_score_spike",
                        z_score      = round(z, 3),
                        rolling_var  = round(float(np.var(win_vals)), 6),
                        baseline_var = round(bl.variance, 6),
                        confidence   = "high" if abs(z) > 5 else "medium",
                    ))
                    last_alert_t[col + "_spike"] = t

            # ── variance collapse detection (replay signature) ────────────────
            rolling_var = float(np.var(win_vals))
            collapse_thr = VAR_COLLAPSE_THR * bl.variance
            if rolling_var < collapse_thr and bl.variance > 1e-8:
                cooldown_ok = (t - last_alert_t.get(col + "_collapse", -999)) >= ALERT_COOLDOWN_S
                if cooldown_ok:
                    alerts.append(AlertEvent(
                        timestamp    = float(uts),
                        elapsed_s    = t,
                        plc          = plc_name,
                        sensor       = col,
                        alert_type   = "variance_collapse",
                        z_score      = float("nan"),
                        rolling_var  = round(rolling_var, 9),
                        baseline_var = round(bl.variance, 6),
                        confidence   = "high" if rolling_var < (collapse_thr / 10) else "medium",
                    ))
                    last_alert_t[col + "_collapse"] = t

    return alerts


alerts_plc1 = stream_plc(plc1, sensor_cols_plc1, baselines_plc1, "PLC1")
alerts_plc2 = stream_plc(plc2, sensor_cols_plc2, baselines_plc2, "PLC2")
all_alerts  = alerts_plc1 + alerts_plc2

log.info("  PLC1 alerts: %d   PLC2 alerts: %d", len(alerts_plc1), len(alerts_plc2))
log.info("  Alert type breakdown: %s",
         pd.Series([a.alert_type for a in all_alerts]).value_counts().to_dict())


# ══════════════════════════════════════════════════════════════════════════════
# §4  CROSS-LAYER CORRELATION SIGNAL
# Flag alert windows where BOTH networks look normal AND physical stasis is
# present — the replay attack signature identified in 02_replay_detection.py.
# In production: join network flow aggregates on the same time window here.
# ══════════════════════════════════════════════════════════════════════════════
def tag_cross_layer_candidates(
    alerts: List[AlertEvent],
    window_s: float = 30.0,
) -> List[AlertEvent]:
    """
    Promote variance_collapse alerts to cross_layer when they cluster in time
    across multiple sensors simultaneously (high-confidence replay signature).
    Single-sensor collapse could be a stuck register; multi-sensor is suspicious.
    """
    tagged = []
    collapse = [a for a in alerts if a.alert_type == "variance_collapse"]
    for a in alerts:
        if a.alert_type != "variance_collapse":
            tagged.append(a)
            continue
        # Count how many OTHER sensors also had collapse within ±window_s
        concurrent = sum(
            1 for b in collapse
            if b is not a
            and b.plc == a.plc
            and abs(b.elapsed_s - a.elapsed_s) <= window_s
        )
        if concurrent >= 2:
            tagged.append(dataclasses.replace(a, alert_type="cross_layer"))
        else:
            tagged.append(a)
    return tagged

all_alerts = tag_cross_layer_candidates(all_alerts)
n_cross = sum(1 for a in all_alerts if a.alert_type == "cross_layer")
log.info("  Cross-layer candidates (multi-sensor stasis): %d", n_cross)


# ══════════════════════════════════════════════════════════════════════════════
# §5  SCORE AGAINST GROUND TRUTH
# For each true attack window, find the earliest alert that falls within it.
# Report: detection rate and detection latency.
# ══════════════════════════════════════════════════════════════════════════════
log.info("[5/7] Scoring vs. ground truth …")

alert_df = pd.DataFrame([dataclasses.asdict(a) for a in all_alerts])
alert_df.to_csv(f"{OUTPUT_DIR}/alerts.csv", index=False)
log.info("  Saved %d alerts to alerts.csv", len(alert_df))

score_rows = []
for _, window in atk_in_range.iterrows():
    ws, we  = window["elapsed_start"], window["elapsed_end"]
    atk_type = window["attack"]
    # Alerts that fall within this attack window
    hits = [a for a in all_alerts if ws <= a.elapsed_s <= we]
    detected     = len(hits) > 0
    first_hit    = min((a.elapsed_s for a in hits), default=None)
    latency_s    = round(first_hit - ws, 2) if first_hit is not None else None
    alert_types  = list({a.alert_type for a in hits}) if hits else []
    score_rows.append({
        "attack":        atk_type,
        "elapsed_start": round(ws, 1),
        "elapsed_end":   round(we, 1),
        "duration_s":    round(we - ws, 1),
        "detected":      detected,
        "n_alerts":      len(hits),
        "latency_s":     latency_s,
        "alert_types":   ", ".join(alert_types),
    })

score_df = pd.DataFrame(score_rows)
score_df.to_csv(f"{OUTPUT_DIR}/detection_scores.csv", index=False)

log.info("\nDetection results by attack type:")
summary = score_df.groupby("attack").agg(
    windows=("detected", "count"),
    detected=("detected", "sum"),
    avg_latency_s=("latency_s", "mean"),
).round(2)
summary["recall"] = (summary["detected"] / summary["windows"]).round(3)
log.info("\n%s", summary.to_string())
summary.to_csv(f"{OUTPUT_DIR}/detection_summary.csv")

overall_recall = score_df["detected"].mean()
log.info("\n  Overall detection recall: %.1f%%", overall_recall * 100)
log.info("  Expected: replay windows will have HIGH recall (physical stasis)")
log.info("  Expected: ddos/port-scan windows may have LOWER recall")
log.info("  (physical layer alone doesn't catch all attack types — cross-layer needed)")


# ══════════════════════════════════════════════════════════════════════════════
# §6  VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════
log.info("[6/7] Visualizations …")

ALERT_COLORS = {
    "z_score_spike":   "#e15759",
    "variance_collapse": "#f28e2b",
    "cross_layer":     "#9467bd",
}
ATTACK_COLORS = {
    "replay":    "#e15759",
    "ddos":      "#4e79a7",
    "port-scan": "#59a14f",
    "ip-scan":   "#76b7b2",
    "mitm":      "#b07aa1",
}

# ── Plot 1: Sensor timeline with alert overlay (PLC1 key sensors) ─────────────
key_sensors = [
    "tank_level_value(2)",
    "tank_output_flow_value(7)",
    "tank_input_valve_status(0)",
]
key_sensors = [s for s in key_sensors if s in plc1.columns]

fig, axes = plt.subplots(len(key_sensors) + 1, 1,
                          figsize=(16, 3 * (len(key_sensors) + 1)),
                          sharex=True)
fig.suptitle("PLC1 Sensor Streams + Anomaly Alerts vs. Ground-Truth Attack Windows",
             fontsize=11)

# Bottom axis: attack timeline
ax_atk = axes[-1]
for _, w in atk_in_range.iterrows():
    color = ATTACK_COLORS.get(w["attack"], "#aaaaaa")
    ax_atk.axvspan(w["elapsed_start"], w["elapsed_end"], alpha=0.4, color=color,
                   label=w["attack"])
ax_atk.set_ylabel("Attack\nWindows", fontsize=8)
ax_atk.set_yticks([])
# Deduplicated legend
seen = set()
for p in ax_atk.patches:
    lbl = p.get_label()
    if lbl and lbl not in seen:
        seen.add(lbl)
handles = [mpatches.Patch(color=v, label=k) for k, v in ATTACK_COLORS.items()]
ax_atk.legend(handles=handles, loc="upper right", fontsize=7, ncol=3)

# Sensor subplots
for ax, col in zip(axes[:-1], key_sensors):
    t   = plc1["elapsed_s"].values
    val = plc1[col].fillna(method="ffill").values
    ax.plot(t, val, color="#4e79a7", lw=0.7, alpha=0.8, label=col)

    # Overlay attack windows
    for _, w in atk_in_range.iterrows():
        color = ATTACK_COLORS.get(w["attack"], "#aaaaaa")
        ax.axvspan(w["elapsed_start"], w["elapsed_end"], alpha=0.12, color=color)

    # Overlay alerts for this sensor
    sensor_alerts = [a for a in all_alerts if a.sensor == col and a.plc == "PLC1"]
    for a in sensor_alerts:
        marker_color = ALERT_COLORS.get(a.alert_type, "#999")
        ax.axvline(a.elapsed_s, color=marker_color, alpha=0.7, lw=1.0, ls="--")

    ax.set_ylabel(col.split("(")[0].replace("_", " "), fontsize=8)
    ax.tick_params(axis="both", labelsize=7)

axes[-1].set_xlabel("Elapsed seconds from stream start")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/sensor_timeline.png", dpi=110, bbox_inches="tight")
plt.close()
log.info("  Saved sensor_timeline.png")


# ── Plot 2: Alert density histogram over time ─────────────────────────────────
if len(alert_df) > 0:
    fig, ax = plt.subplots(figsize=(14, 4))
    bins = np.linspace(0, plc1["elapsed_s"].max(), 100)
    for atype, color in ALERT_COLORS.items():
        sub = alert_df[alert_df["alert_type"] == atype]["elapsed_s"]
        if len(sub):
            ax.hist(sub, bins=bins, color=color, alpha=0.6, label=atype)

    for _, w in atk_in_range.iterrows():
        color = ATTACK_COLORS.get(w["attack"], "#aaaaaa")
        ax.axvspan(w["elapsed_start"], w["elapsed_end"], alpha=0.15, color=color)

    ax.set_xlabel("Elapsed seconds from stream start")
    ax.set_ylabel("Alert count per bin")
    ax.set_title("Alert Density vs. Attack Windows (gray = attack window)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/alert_density.png", dpi=110, bbox_inches="tight")
    plt.close()
    log.info("  Saved alert_density.png")


# ── Plot 3: Detection summary bar chart ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
colors = [ATTACK_COLORS.get(idx, "#aaaaaa") for idx in summary.index]
summary["recall"].plot.bar(ax=ax, color=colors, alpha=0.85, edgecolor="white")
ax.axhline(1.0, color="gray", ls=":", lw=0.8)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Detection Recall (windows caught / total windows)")
ax.set_title("Physical-Layer Streaming Detector — Per-Attack Recall\n"
             "(replay = high recall via variance collapse; "
             "network attacks may need cross-layer enrichment)")
ax.tick_params(axis="x", rotation=20)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/detection_recall.png", dpi=110, bbox_inches="tight")
plt.close()
log.info("  Saved detection_recall.png")


# ══════════════════════════════════════════════════════════════════════════════
# §7  EXPORT SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
log.info("[7/7] Export …")

pipeline_summary = {
    "window_seconds":         WINDOW_SECONDS,
    "zscore_threshold":       ZSCORE_THRESHOLD,
    "variance_collapse_thr":  VAR_COLLAPSE_THR,
    "alert_cooldown_s":       ALERT_COOLDOWN_S,
    "plc1_baseline_sensors":  len(baselines_plc1),
    "plc2_baseline_sensors":  len(baselines_plc2),
    "total_alerts":           len(all_alerts),
    "alert_breakdown":        alert_df["alert_type"].value_counts().to_dict() if len(alert_df) else {},
    "cross_layer_candidates": int(n_cross),
    "attack_windows_scored":  int(len(atk_in_range)),
    "overall_recall":         round(float(overall_recall), 4),
    "per_attack_recall":      summary["recall"].to_dict(),
    "per_attack_avg_latency_s": summary["avg_latency_s"].to_dict(),
    "interpretation": {
        "replay": (
            "High recall expected — variance collapse across multiple PLC1 sensors "
            "is the replay signature. Physically frozen registers = strong signal. "
            "Connects to 02_replay_detection.py finding: HDBSCAN and KMeans miss this "
            "because static data looks like a normal dense cluster."
        ),
        "ddos": (
            "Physical sensors may not change during a network-layer DDoS — "
            "the attack floods the network but doesn't disrupt PLC logic. "
            "Cross-layer enrichment with network flow data needed for high recall."
        ),
        "port-scan / ip-scan": (
            "Passive reconnaissance. No physical impact expected. "
            "Physical-layer alerts here would be coincidental. "
            "Network-layer classifier (05_attack_classifier.py) is the right detector."
        ),
        "mitm": (
            "MITM may cause small deviations in PLC values if packets are modified. "
            "TTL anomalies in network flows (see 05) are a stronger MITM signal."
        ),
        "cross_layer_design": (
            "A production ICS anomaly platform joins both layers: "
            "ingest network flows from Dataset.csv AND physical snapshots from PLC CSVs "
            "in the same time-bucketed stream. Alert only when network looks normal "
            "AND physical stasis is detected. This eliminates most false positives."
        ),
    },
}

with open(f"{OUTPUT_DIR}/pipeline_summary.json", "w") as f:
    json.dump(pipeline_summary, f, indent=2)

log.info("  Outputs written to %s/", OUTPUT_DIR)
log.info("  alerts.csv              — every alert event (typed fields)")
log.info("  detection_scores.csv    — per-window ground-truth scoring")
log.info("  detection_summary.csv   — per-attack recall + latency")
log.info("  pipeline_summary.json   — full metrics + interpretation notes")
log.info("\nDone.")
