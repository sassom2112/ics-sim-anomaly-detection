"""
mitl.datasets.hai — Reference MITL encoding for HAI 22.04 (steam turbine HIL).

Constraints derived from:
  "HAI Security Dataset Technical Manual v4.0"
  Dataset: HAI (HIL-Based Augmented ICS) 22.04
  Hardware: steam-turbine + HIL (Hardware-in-the-Loop) simulator

Spec sources:
  Table 1 (pp 12–15) — Data points: per-tag [min, max, unit] for 86 tags.
  Figures 4–13 (pp 7–11) — Control loop block diagrams; every loop shows
    Saturation and Rate Limiter / Ramp blocks.
  Attack table (pp 25–27) — AP01–AP47 attack scenarios; AP27 is the key
    cross-layer replay-class attack (AutoSD SP + SIT01 PV manipulation).

Four constraints (mirror of ICSSim C1–C4, re-derived from a different manual):
  C1 — Saturation bounds      (explicit, from Table 1 data-points table, p 12–15)
  C2 — Rate limiter invariant (diagram-inferred, from Figures 4–13, p 7–11)
  C3 — P2-SC tracking (AP27) (text + diagram, Figure 11 + AP27 row, p 10 + 27)
  C4 — Cross-layer P4→P2     (architecture, Figure 10, p 10)

HAI column naming convention:
  Tags follow the format <layer>_<instrument>, e.g. P2_SIT01, P4_ST_PS.
  Label column: 'attack' (int 0/1).  Timestamp: 'timestamp'.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from ..calibrate import BehavioralBaseline
from ..spec import (ConstraintSource, ConstraintSpec, ConstraintViolation,
                    ControlLoopSpec, TagSpec)


# ── Spec sources (page-level provenance) ─────────────────────────────────────

def _src(page: int, figure: str, quote: str, by: str = "manual",
         conf: float = 1.0) -> ConstraintSource:
    return ConstraintSource(
        document="HAI Security Dataset Technical Manual v4.0",
        page_number=page,
        figure_id=figure,
        quote=quote,
        extracted_by=by,
        confidence=conf,
    )


# ── Tag bounds from Table 1 (pp 12–15) ───────────────────────────────────────
# Format: tag → (min, max, unit, page, description)

HAI_TAG_BOUNDS = {
    # P1 — Boiler / Steam Generation
    "P1_B2016":  (0.0,   10.0,  "bar", 12, "pressure demand SP"),
    "P1_PIT01":  (0.0,   10.0,  "bar", 13, "pressure PV"),
    "P1_PCV01D": (0.0,  100.0,  "%",   13, "pressure CV valve position"),
    "P1_B3004":  (0.0,  720.0,  "mm",  12, "boiler level SP"),
    "P1_LIT01":  (0.0,  720.0,  "mm",  13, "boiler level PV"),
    "P1_LCV01D": (0.0,  100.0,  "%",   13, "level CV valve position"),
    "P1_B3005":  (0.0,   10.0,  "m³/h",12,"flow SP"),
    "P1_FCV03D": (0.0,  100.0,  "%",   13, "flow CV valve position"),
    "P1_FIT01":  (0.0,   10.0,  "m³/h",13,"flow PV"),
    "P1_B3003":  (0.0,  500.0,  "°C",  12, "temperature SP"),
    "P1_TIT01":  (0.0,  500.0,  "°C",  13, "temperature PV"),
    # P2 — Turbine / Speed Control
    "P2_AutoSD": (0.0, 3200.0,  "RPM", 13, "turbine speed demand (setpoint)"),
    "P2_SIT01":  (0.0, 3200.0,  "RPM", 14, "turbine speed PV (speed probe)"),
    "P2_RTR":    (0.0, 2880.0,  "RPM", 14, "RPM trip rate — hard limit"),
    "P2_SCO":    (0.0,  100.0,  "%",   14, "speed control output"),
    # P3 — Water Treatment
    "P3_LIT01":  (0.0, 1000.0,  "mm",  14, "water storage level"),
    "P3_LCV01":  (0.0,  100.0,  "%",   14, "water level CV valve"),
    # P4 — HIL Simulator outputs
    "P4_ST_PS":  (0.0,  100.0,  "%",   15, "turbine scheduled power demand"),
    "P4_HT_PS":  (0.0,  100.0,  "%",   15, "heating scheduled power demand"),
}

# Tags that have Rate Limiter or Ramp blocks in their respective loop diagrams.
# Source: Figures 4 (P1-PC), 5 (P1-LC), 6 (P1-FC), 11 (P2-SC Ramp), 13 (P3-LC).
RATE_LIMITED_TAGS = [
    "P1_PCV01D",   # Figure 4 — Rate Limiter after PID
    "P1_LCV01D",   # Figure 5 — Rate Limiter after Saturation
    "P1_FCV03D",   # Figure 6 — Rate Limiter after PID
    "P2_AutoSD",   # Figure 11 — Ramp block (constant-rate speed ramp)
    "P3_LCV01",    # Figure 13 — Rate Limiter
]


# ── ConstraintSpec ─────────────────────────────────────────────────────────────

def build_hai_spec() -> ConstraintSpec:
    """Build the HAI 22.04 ConstraintSpec from the technical manual."""
    tag_specs = {}
    for tag, (lo, hi, unit, page, desc) in HAI_TAG_BOUNDS.items():
        tag_specs[tag] = TagSpec(
            name=tag, min_val=lo, max_val=hi, unit=unit, description=desc,
            source=_src(page, "Table 1", f"{tag}: [{lo}, {hi}] {unit}"),
        )

    loop_specs = {
        "P1-PC": ControlLoopSpec(
            loop_id="P1-PC", setpoint_tag="P1_B2016",
            process_var_tag="P1_PIT01", control_var_tag="P1_PCV01D",
            has_saturation=True, has_rate_limiter=True, cross_layer_inputs=[],
            source=_src(7, "Figure 4", "P1-PC: B2016 → PID → Saturation → Rate Limiter → PCV01D"),
        ),
        "P1-LC": ControlLoopSpec(
            loop_id="P1-LC", setpoint_tag="P1_B3004",
            process_var_tag="P1_LIT01", control_var_tag="P1_LCV01D",
            has_saturation=True, has_rate_limiter=True, cross_layer_inputs=[],
            source=_src(7, "Figure 5", "P1-LC: B3004 → Saturation → Rate Limiter → LCV01D"),
        ),
        "P1-FC": ControlLoopSpec(
            loop_id="P1-FC", setpoint_tag="P1_B3005",
            process_var_tag="P1_FIT01", control_var_tag="P1_FCV03D",
            has_saturation=True, has_rate_limiter=True, cross_layer_inputs=[],
            source=_src(7, "Figure 6", "P1-FC: B3005 → PID → Saturation → Rate Limiter → FCV03D"),
        ),
        "P2-SC": ControlLoopSpec(
            loop_id="P2-SC", setpoint_tag="P2_AutoSD",
            process_var_tag="P2_SIT01", control_var_tag="P2_SCO",
            has_saturation=True, has_rate_limiter=True,
            cross_layer_inputs=["P4_ST_PS"],
            source=_src(10, "Figure 11",
                        "P2-SC: AutoSD → Saturation → Ramp → Σ → SCO → Motor → SIT01. "
                        "PID controller maintains SIT01 ≈ AutoSD."),
        ),
        "P3-LC": ControlLoopSpec(
            loop_id="P3-LC", setpoint_tag="P3_LIT01",
            process_var_tag="P3_LIT01", control_var_tag="P3_LCV01",
            has_saturation=True, has_rate_limiter=True,
            cross_layer_inputs=["P4_HT_PS"],
            source=_src(11, "Figure 13", "P3-LC driven by P4-HTM discharge/pumping demand"),
        ),
    }

    return ConstraintSpec(
        dataset_name="HAI-22.04",
        manual_version="HAI Security Dataset Technical Manual v4.0",
        tag_specs=tag_specs,
        loop_specs=loop_specs,
        extraction_confidence=1.0,
    )


# ── Constraint functions ──────────────────────────────────────────────────────

def c1_saturation_bounds(
    window_df: pd.DataFrame,
    baseline:  BehavioralBaseline,
    spec:      ConstraintSpec,
) -> List[ConstraintViolation]:
    """C1: all sensor/actuator values must stay within spec-defined physical limits.

    Source: Table 1 (pp 12–15).  Confidence: 1.0 — explicit numeric table.
    """
    viols: List[ConstraintViolation] = []
    for tag, (lo, hi, unit, page, desc) in HAI_TAG_BOUNDS.items():
        if tag not in window_df.columns:
            continue
        vals = window_df[tag].dropna()
        if vals.empty:
            continue
        lo_viol = vals[vals < lo]
        hi_viol = vals[vals > hi]
        if not lo_viol.empty or not hi_viol.empty:
            observed = f"[{vals.min():.3f}, {vals.max():.3f}] {unit}"
            allowed  = f"[{lo}, {hi}] {unit}"
            viols.append(ConstraintViolation(
                constraint_id="C1",
                constraint_name="saturation-bounds",
                severity="definite",
                evidence=f"{tag}: observed {observed} outside spec {allowed}",
                spec_page=page,
                spec_figure="Table 1",
                spec_quote=f"{desc}: bounds [{lo}, {hi}] {unit}",
            ))
    return viols


c1_saturation_bounds.constraint_id  = "C1"
c1_saturation_bounds.spec_confidence = 1.0


def c2_rate_limiter_invariant(
    window_df: pd.DataFrame,
    baseline:  BehavioralBaseline,
    spec:      ConstraintSpec,
) -> List[ConstraintViolation]:
    """C2: control outputs cannot step-change — every loop has Rate Limiter / Ramp.

    Source: Figures 4–13 (pp 7–11).  Confidence: 0.85 — diagram topology.
    The threshold is 3× the 99th-percentile rate from warm-up (not a magic number).
    """
    viols: List[ConstraintViolation] = []
    for tag in RATE_LIMITED_TAGS:
        if tag not in window_df.columns:
            continue
        vals = window_df[tag].dropna().values
        if len(vals) < 2:
            continue
        max_rate = baseline.max_rate_of(tag)
        if max_rate <= 0:
            continue
        observed_max = float(np.abs(np.diff(vals)).max())
        if observed_max > 3.0 * max_rate:
            viols.append(ConstraintViolation(
                constraint_id="C2",
                constraint_name="rate-limiter-invariant",
                severity="probable",
                evidence=(f"{tag} Δ/step={observed_max:.4f} "
                          f"vs baseline 99th-pct {max_rate:.4f} (×{observed_max/max_rate:.1f})"),
                spec_page=9,
                spec_figure="Figures 4–13",
                spec_quote="Rate Limiter / Ramp blocks present in all P1-PC, P1-LC, P1-FC, P2-SC, P3-LC loops",
            ))
    return viols


c2_rate_limiter_invariant.constraint_id  = "C2"
c2_rate_limiter_invariant.spec_confidence = 0.85


def c3_tracking_invariant_p2sc(
    window_df: pd.DataFrame,
    baseline:  BehavioralBaseline,
    spec:      ConstraintSpec,
) -> List[ConstraintViolation]:
    """C3 (AP27): if P2_AutoSD is ramping, P2_SIT01 must track it.

    Frozen SIT01 during active AutoSD ramp = sensor spoofing / replay.
    This is the HAI equivalent of the ICSSim valve-stasis invariant.

    Source: Figure 11 (p 10) + AP27 in attack table (p 27).
    Confidence: 0.95 — prose, diagram, and attack scenario all agree.
    """
    viols: List[ConstraintViolation] = []
    if "P2_AutoSD" not in window_df.columns or "P2_SIT01" not in window_df.columns:
        return viols

    autosd = window_df["P2_AutoSD"].dropna()
    sit01  = window_df["P2_SIT01"].dropna()
    if len(autosd) < 2 or len(sit01) < 2:
        return viols

    autosd_range = float(autosd.max() - autosd.min())
    sit01_var    = float(sit01.var())
    sit01_base_v = baseline.variance_of("P2_SIT01")

    # AutoSD is actively ramping (>50 RPM swing = meaningful speed change)
    if autosd_range > 50.0:
        # SIT01 frozen: var < 5% of its warm-up variance
        if sit01_var < 0.05 * (sit01_base_v + 1e-9):
            viols.append(ConstraintViolation(
                constraint_id="C3",
                constraint_name="tracking-invariant-P2-SC",
                severity="definite",
                evidence=(f"AutoSD range={autosd_range:.1f} RPM (active ramp) "
                          f"but SIT01 var={sit01_var:.4f} ≈ 0 (frozen). "
                          f"Baseline SIT01 var={sit01_base_v:.4f}"),
                spec_page=10,
                spec_figure="Figure 11",
                spec_quote=(
                    "PID controller to maintain motor speed value (SIT01) "
                    "as close as possible to speed setpoint value (AutoSD)"
                ),
            ))
    return viols


c3_tracking_invariant_p2sc.constraint_id  = "C3"
c3_tracking_invariant_p2sc.spec_confidence = 0.95


def c4_cross_layer_p4_p2(
    window_df: pd.DataFrame,
    baseline:  BehavioralBaseline,
    spec:      ConstraintSpec,
) -> List[ConstraintViolation]:
    """C4: P4-STM scheduled power drives AutoSD → P2 speed must follow P4 commands.

    If P4_ST_PS changes significantly but P2_SIT01 is frozen, the cross-layer
    coupling between the HIL simulator (P4) and the turbine (P2) is broken —
    indicating either a cross-layer attack or sensor compromise.

    Source: Figure 10 (p 10) — turbine process architecture.  Confidence: 0.80.
    """
    viols: List[ConstraintViolation] = []
    if "P4_ST_PS" not in window_df.columns or "P2_SIT01" not in window_df.columns:
        return viols

    p4_ps = window_df["P4_ST_PS"].dropna()
    sit01 = window_df["P2_SIT01"].dropna()
    if len(p4_ps) < 2 or len(sit01) < 2:
        return viols

    p4_delta  = float(abs(p4_ps.iloc[-1] - p4_ps.iloc[0]))
    p4_thresh = 3.0 * baseline.std_of("P4_ST_PS") + 1e-9

    if p4_delta > p4_thresh:
        sit01_var   = float(sit01.var())
        sit01_bvar  = baseline.variance_of("P2_SIT01")
        if sit01_var < 0.05 * (sit01_bvar + 1e-9):
            viols.append(ConstraintViolation(
                constraint_id="C4",
                constraint_name="cross-layer-P4-P2-discrepancy",
                severity="definite",
                evidence=(f"P4_ST_PS Δ={p4_delta:.2f}% > {p4_thresh:.2f}% (3σ) "
                          f"but P2_SIT01 var={sit01_var:.4f} ≈ 0 (frozen)"),
                spec_page=10,
                spec_figure="Figure 10",
                spec_quote="P4-STM provides scheduled power demand (P4_ST_PS) driving P2-SC AutoSD",
            ))
    return viols


c4_cross_layer_p4_p2.constraint_id  = "C4"
c4_cross_layer_p4_p2.spec_confidence = 0.80


def hai_constraints() -> list:
    """Return the ordered list of HAI 22.04 MITL constraint functions."""
    return [
        c1_saturation_bounds,
        c2_rate_limiter_invariant,
        c3_tracking_invariant_p2sc,
        c4_cross_layer_p4_p2,
    ]
