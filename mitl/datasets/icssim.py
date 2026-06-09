"""
mitl.datasets.icssim — Reference MITL encoding for ICSSim v2 (water treatment).

Constraints derived from the ICSSim v2 PLC register layout and the
water-treatment fill/drain cycle specification.  This is the dataset used
for the ICSSim experiments in the MITL paper (Sprints 3–8).

Four constraints:
  C1 — Saturation bounds (static, from PLC register min/max columns)
  C2 — State-flow consistency (static, from instrument range spec)
  C3 — Valve cycle invariant (spec structure + warm-up calibration)
  C4 — Cross-layer network/valve discrepancy (spec + warm-up)

Replay attack mechanism:
  Attacker replays legitimate Modbus traffic verbatim.  Replayed commands
  freeze the input valve at the captured position while network traffic
  volume INCREASES (replayed traffic on top of normal).
  Physical invariant: elevated network commands → valve must actuate.
  C4 encodes this cross-layer contradiction.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from ..calibrate import BehavioralBaseline
from ..spec import (ConstraintSource, ConstraintSpec, ConstraintViolation,
                    ControlLoopSpec, TagSpec)

# ── Spec source references ────────────────────────────────────────────────────

_SRC_REGISTER = ConstraintSource(
    document="ICSSim v2 PLC register layout",
    page_number=1,
    figure_id="",
    quote="tank_level_min / tank_level_max are setpoint registers in PLC1",
    extracted_by="manual",
    confidence=1.0,
)
_SRC_INSTRUMENT = ConstraintSource(
    document="ICSSim v2 water-treatment instrument spec",
    page_number=1,
    figure_id="",
    quote="Nominal max flow = 0.0001 m³/s; deadband = 50% of nominal",
    extracted_by="manual",
    confidence=1.0,
)
_SRC_PROCESS = ConstraintSource(
    document="ICSSim v2 water-treatment process description",
    page_number=1,
    figure_id="",
    quote="Modbus write commands to valve registers cause physical valve actuation",
    extracted_by="manual",
    confidence=1.0,
)

# ── ConstraintSpec ─────────────────────────────────────────────────────────────

def build_icssim_spec() -> ConstraintSpec:
    """Build the ICSSim v2 ConstraintSpec from the PLC register layout."""
    tag_specs = {
        "tank_level_value(2)": TagSpec(
            name="tank_level_value(2)", min_val=0.0, max_val=1.0,
            unit="m", description="water level in tank",
            source=_SRC_REGISTER,
        ),
        "tank_output_flow_value(7)": TagSpec(
            name="tank_output_flow_value(7)", min_val=0.0, max_val=0.0001,
            unit="m³/s", description="output flow measured by flow meter",
            source=_SRC_INSTRUMENT,
        ),
    }
    loop_specs = {
        "fill-drain-cycle": ControlLoopSpec(
            loop_id="fill-drain-cycle",
            setpoint_tag="tank_level_min(3)",
            process_var_tag="tank_level_value(2)",
            control_var_tag="tank_input_valve_status(0)",
            has_saturation=False,
            has_rate_limiter=False,
            cross_layer_inputs=["network_flow_count"],
            source=_SRC_PROCESS,
        ),
    }
    return ConstraintSpec(
        dataset_name="ICSSim-v2",
        manual_version="ICSSim v2 (Kaggle: alirezadehlaghi/icssim)",
        tag_specs=tag_specs,
        loop_specs=loop_specs,
        extraction_confidence=1.0,
    )


# ── Constraint functions ──────────────────────────────────────────────────────

FLOW_DEADBAND       = 0.00005   # 50% of nominal 0.0001 m³/s (instrument spec)
VALVE_STASIS_RATIO  = 0.05      # rolling var < 5% of baseline var → valve frozen
NETWORK_SPIKE_MULT  = 1.8       # net flows > 1.8× baseline mean → elevated traffic

VALVE_COLS  = ["tank_input_valve_status(0)", "tank_output_valve_status(5)"]
FLOW_COL    = "tank_output_flow_value(7)"
LEVEL_COL   = "tank_level_value(2)"
LEVEL_MIN   = "tank_level_min(3)"
LEVEL_MAX   = "tank_level_max(4)"


def c1_saturation_bounds(
    window_df: pd.DataFrame,
    baseline:  BehavioralBaseline,
    spec:      ConstraintSpec,
) -> List[ConstraintViolation]:
    """C1: tank level must be within PLC-configured setpoint window."""
    viols: List[ConstraintViolation] = []
    for col in [LEVEL_COL, LEVEL_MIN, LEVEL_MAX]:
        if col not in window_df.columns:
            return viols

    level = window_df[LEVEL_COL].dropna()
    lo    = window_df[LEVEL_MIN].dropna().median()
    hi    = window_df[LEVEL_MAX].dropna().median()

    if pd.isna(lo) or pd.isna(hi) or level.empty:
        return viols

    out_hi = level[level > hi]
    out_lo = level[level < lo]
    if not out_hi.empty or not out_lo.empty:
        viols.append(ConstraintViolation(
            constraint_id="C1",
            constraint_name="saturation-bounds",
            severity="definite",
            evidence=(f"level ∈ [{level.min():.4f}, {level.max():.4f}] m "
                      f"outside setpoint window [{lo:.4f}, {hi:.4f}] m"),
            spec_page=1,
            spec_figure="",
            spec_quote=_SRC_REGISTER.quote,
        ))
    return viols


c1_saturation_bounds.constraint_id  = "C1"
c1_saturation_bounds.spec_confidence = 1.0


def c2_state_flow_consistency(
    window_df: pd.DataFrame,
    baseline:  BehavioralBaseline,
    spec:      ConstraintSpec,
) -> List[ConstraintViolation]:
    """C2: output valve open → output flow > deadband."""
    viols: List[ConstraintViolation] = []
    out_valve = "tank_output_valve_status(5)"
    if out_valve not in window_df.columns or FLOW_COL not in window_df.columns:
        return viols

    valve_open = window_df[out_valve].dropna()
    flow       = window_df[FLOW_COL].dropna()
    if valve_open.empty or flow.empty:
        return viols

    if valve_open.mean() > 0.8 and flow.mean() < FLOW_DEADBAND:
        viols.append(ConstraintViolation(
            constraint_id="C2",
            constraint_name="state-flow-consistency",
            severity="probable",
            evidence=(f"output_valve=open ({valve_open.mean():.2f}) "
                      f"but flow={flow.mean():.6f} < deadband {FLOW_DEADBAND}"),
            spec_page=1,
            spec_figure="",
            spec_quote=_SRC_INSTRUMENT.quote,
        ))
    return viols


c2_state_flow_consistency.constraint_id  = "C2"
c2_state_flow_consistency.spec_confidence = 1.0


def c3_valve_cycle_invariant(
    window_df: pd.DataFrame,
    baseline:  BehavioralBaseline,
    spec:      ConstraintSpec,
) -> List[ConstraintViolation]:
    """C3: input valve must not be frozen relative to its warm-up variance baseline."""
    viols: List[ConstraintViolation] = []
    in_valve = "tank_input_valve_status(0)"
    if in_valve not in window_df.columns:
        return viols

    vals         = window_df[in_valve].dropna()
    rolling_var  = float(vals.var()) if len(vals) > 1 else 0.0
    baseline_var = baseline.variance_of(in_valve)

    if rolling_var < VALVE_STASIS_RATIO * baseline_var:
        viols.append(ConstraintViolation(
            constraint_id="C3",
            constraint_name="valve-cycle-invariant",
            severity="probable",
            evidence=(f"input_valve var={rolling_var:.2e} < "
                      f"{VALVE_STASIS_RATIO}×baseline {baseline_var:.2e}"),
            spec_page=1,
            spec_figure="",
            spec_quote=_SRC_PROCESS.quote,
        ))
    return viols


c3_valve_cycle_invariant.constraint_id  = "C3"
c3_valve_cycle_invariant.spec_confidence = 0.9


def c4_cross_layer_discrepancy(
    window_df: pd.DataFrame,
    baseline:  BehavioralBaseline,
    spec:      ConstraintSpec,
) -> List[ConstraintViolation]:
    """C4: elevated network traffic + valve stasis = physically impossible."""
    viols: List[ConstraintViolation] = []
    net_col  = "net_flow_count"
    in_valve = "tank_input_valve_status(0)"

    if net_col not in window_df.columns or in_valve not in window_df.columns:
        return viols

    net_count    = float(window_df[net_col].sum())
    net_baseline = baseline.tag_baselines.get(net_col)
    if net_baseline is None:
        return viols

    net_spike = net_count > NETWORK_SPIKE_MULT * net_baseline.mean

    valve_vals   = window_df[in_valve].dropna()
    rolling_var  = float(valve_vals.var()) if len(valve_vals) > 1 else 0.0
    baseline_var = baseline.variance_of(in_valve)
    valve_frozen = rolling_var < VALVE_STASIS_RATIO * baseline_var

    if net_spike and valve_frozen:
        viols.append(ConstraintViolation(
            constraint_id="C4",
            constraint_name="cross-layer-network-valve-discrepancy",
            severity="definite",
            evidence=(f"net_flows={net_count:.0f} > {NETWORK_SPIKE_MULT}× "
                      f"baseline {net_baseline.mean:.0f} AND "
                      f"valve_var={rolling_var:.2e} (frozen)"),
            spec_page=1,
            spec_figure="",
            spec_quote=_SRC_PROCESS.quote,
        ))
    return viols


c4_cross_layer_discrepancy.constraint_id  = "C4"
c4_cross_layer_discrepancy.spec_confidence = 0.9


def icssim_constraints() -> list:
    """Return the ordered list of ICSSim MITL constraint functions."""
    return [
        c1_saturation_bounds,
        c2_state_flow_consistency,
        c3_valve_cycle_invariant,
        c4_cross_layer_discrepancy,
    ]
