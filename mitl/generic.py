"""
mitl.generic — Dataset-agnostic constraint functions derived from ConstraintSpec.

These functions read everything they need from the ConstraintSpec and
BehavioralBaseline at evaluation time, so they work on ANY dataset that
has been ingested into the library.  No hardcoded column names, no dataset-
specific logic.

Users who ingest a new manual via `mitl ingest` get these four constraints
automatically.  Dataset-specific constraint modules (mitl.datasets.hai,
mitl.datasets.icssim) override individual constraints for tighter precision
when the dataset pattern is known.

Four generic constraints:
  GC1 — Saturation bounds         (from spec.tag_specs — explicit bounds)
  GC2 — Rate limiter invariant     (from spec.loop_specs — Rate Limiter flag)
  GC3 — SP/PV tracking invariant   (from spec.loop_specs — SP vs PV divergence)
  GC4 — Cross-layer coupling       (from spec.loop_specs — cross_layer_inputs)
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from .calibrate import BehavioralBaseline
from .spec import ConstraintSpec, ConstraintViolation


# ── GC1 — Saturation bounds ───────────────────────────────────────────────────

def gc1_saturation_bounds(
    window_df: pd.DataFrame,
    baseline:  BehavioralBaseline,
    spec:      ConstraintSpec,
) -> List[ConstraintViolation]:
    """
    GC1: every tag in spec.tag_specs must stay within [min_val, max_val].
    Reads bounds directly from the ConstraintSpec — works for any dataset.
    """
    viols: List[ConstraintViolation] = []
    for tag, ts in spec.tag_specs.items():
        if tag not in window_df.columns:
            continue
        vals = window_df[tag].dropna()
        if vals.empty:
            continue
        if (vals < ts.min_val).any() or (vals > ts.max_val).any():
            observed = f"[{vals.min():.3g}, {vals.max():.3g}]"
            allowed  = f"[{ts.min_val}, {ts.max_val}]"
            viols.append(ConstraintViolation(
                constraint_id="GC1",
                constraint_name="saturation-bounds",
                severity="definite",
                evidence=f"{tag}: {observed} {ts.unit} outside spec {allowed} {ts.unit}",
                spec_page=ts.source.page_number,
                spec_figure=ts.source.figure_id,
                spec_quote=ts.source.quote,
            ))
    return viols


gc1_saturation_bounds.constraint_id  = "GC1"
gc1_saturation_bounds.spec_confidence = 0.95


# ── GC2 — Rate limiter invariant ──────────────────────────────────────────────

def gc2_rate_limiter(
    window_df: pd.DataFrame,
    baseline:  BehavioralBaseline,
    spec:      ConstraintSpec,
) -> List[ConstraintViolation]:
    """
    GC2: control outputs in loops with has_rate_limiter=True cannot step-change.
    Threshold = 3× warm-up 99th-percentile rate of the control variable.
    """
    viols: List[ConstraintViolation] = []
    for lid, ls in spec.loop_specs.items():
        if not ls.has_rate_limiter:
            continue
        cv_tag = ls.control_var_tag
        if not cv_tag or cv_tag not in window_df.columns:
            continue
        vals = window_df[cv_tag].dropna().values
        if len(vals) < 2:
            continue
        max_rate = baseline.max_rate_of(cv_tag)
        if max_rate <= 0:
            continue
        observed = float(np.abs(np.diff(vals)).max())
        if observed > 3.0 * max_rate:
            viols.append(ConstraintViolation(
                constraint_id="GC2",
                constraint_name=f"rate-limiter-{lid}",
                severity="probable",
                evidence=(f"{cv_tag} Δ/step={observed:.4g} "
                          f"vs baseline×3 = {3*max_rate:.4g}"),
                spec_page=ls.source.page_number,
                spec_figure=ls.source.figure_id,
                spec_quote=ls.source.quote or f"Loop {lid} has Rate Limiter block",
            ))
    return viols


gc2_rate_limiter.constraint_id  = "GC2"
gc2_rate_limiter.spec_confidence = 0.85


# ── GC3 — SP/PV tracking invariant ───────────────────────────────────────────

def gc3_sp_pv_tracking(
    window_df: pd.DataFrame,
    baseline:  BehavioralBaseline,
    spec:      ConstraintSpec,
) -> List[ConstraintViolation]:
    """
    GC3: when a loop's setpoint is actively changing, the process variable
    must also change (tracking invariant).

    Frozen PV during active SP ramp indicates sensor spoofing / replay.
    Threshold: SP range > 2× baseline SP std → PV variance must be non-negligible.
    """
    viols: List[ConstraintViolation] = []
    for lid, ls in spec.loop_specs.items():
        sp_tag = ls.setpoint_tag
        pv_tag = ls.process_var_tag
        if not sp_tag or not pv_tag:
            continue
        if sp_tag not in window_df.columns or pv_tag not in window_df.columns:
            continue
        if sp_tag == pv_tag:
            continue

        sp_vals = window_df[sp_tag].dropna()
        pv_vals = window_df[pv_tag].dropna()
        if len(sp_vals) < 2 or len(pv_vals) < 2:
            continue

        sp_range   = float(sp_vals.max() - sp_vals.min())
        sp_std_bl  = baseline.std_of(sp_tag)
        if sp_range < 2.0 * sp_std_bl:
            continue  # SP not actively ramping — invariant doesn't apply

        pv_var    = float(pv_vals.var())
        pv_var_bl = baseline.variance_of(pv_tag)
        if pv_var < 0.05 * (pv_var_bl + 1e-12):
            viols.append(ConstraintViolation(
                constraint_id="GC3",
                constraint_name=f"tracking-invariant-{lid}",
                severity="definite",
                evidence=(f"SP={sp_tag} range={sp_range:.3g} (>{2*sp_std_bl:.3g}=2σ) "
                          f"but PV={pv_tag} var={pv_var:.3g} ≈ 0 (frozen). "
                          f"Baseline PV var={pv_var_bl:.3g}"),
                spec_page=ls.source.page_number,
                spec_figure=ls.source.figure_id,
                spec_quote=ls.source.quote or f"Loop {lid}: PV must track SP",
            ))
    return viols


gc3_sp_pv_tracking.constraint_id  = "GC3"
gc3_sp_pv_tracking.spec_confidence = 0.90


# ── GC4 — Cross-layer coupling invariant ──────────────────────────────────────

def gc4_cross_layer_coupling(
    window_df: pd.DataFrame,
    baseline:  BehavioralBaseline,
    spec:      ConstraintSpec,
) -> List[ConstraintViolation]:
    """
    GC4: if a loop has cross_layer_inputs, a significant change in those
    inputs must produce a corresponding response in the loop's PV.

    Cross-layer architecture is declared in ControlLoopSpec.cross_layer_inputs.
    A large input delta + frozen PV = cross-layer invariant violated.
    """
    viols: List[ConstraintViolation] = []
    for lid, ls in spec.loop_specs.items():
        if not ls.cross_layer_inputs:
            continue
        pv_tag = ls.process_var_tag
        if not pv_tag or pv_tag not in window_df.columns:
            continue

        for input_tag in ls.cross_layer_inputs:
            if input_tag not in window_df.columns:
                continue
            inp  = window_df[input_tag].dropna()
            pv   = window_df[pv_tag].dropna()
            if len(inp) < 2 or len(pv) < 2:
                continue

            inp_delta  = float(abs(inp.iloc[-1] - inp.iloc[0]))
            inp_thresh = 3.0 * baseline.std_of(input_tag) + 1e-9
            if inp_delta <= inp_thresh:
                continue

            pv_var    = float(pv.var())
            pv_var_bl = baseline.variance_of(pv_tag)
            if pv_var < 0.05 * (pv_var_bl + 1e-12):
                viols.append(ConstraintViolation(
                    constraint_id="GC4",
                    constraint_name=f"cross-layer-{input_tag}->{pv_tag}",
                    severity="definite",
                    evidence=(f"{input_tag} Δ={inp_delta:.3g} > 3σ={inp_thresh:.3g} "
                              f"but {pv_tag} var={pv_var:.3g} ≈ 0 (frozen)"),
                    spec_page=ls.source.page_number,
                    spec_figure=ls.source.figure_id,
                    spec_quote=ls.source.quote or f"Loop {lid}: {input_tag} drives {pv_tag}",
                ))
    return viols


gc4_cross_layer_coupling.constraint_id  = "GC4"
gc4_cross_layer_coupling.spec_confidence = 0.80


# ── Bundle ────────────────────────────────────────────────────────────────────

def generic_constraints() -> list:
    """Return all four generic constraints in evaluation order."""
    return [
        gc1_saturation_bounds,
        gc2_rate_limiter,
        gc3_sp_pv_tracking,
        gc4_cross_layer_coupling,
    ]
