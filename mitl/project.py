"""
mitl.project — Constraint projection over windowed time-series data.

A ConstraintProjector applies a list of constraint functions to every
time window in a DataFrame, producing per-window WindowConstraintResults
that can be evaluated against ground-truth labels or fed into AutoML.
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional

import pandas as pd

from .calibrate import BehavioralBaseline
from .spec import ConstraintSpec, ConstraintViolation, WindowConstraintResult

log = logging.getLogger(__name__)

ConstraintFn = Callable[
    [pd.DataFrame, BehavioralBaseline, ConstraintSpec],
    List[ConstraintViolation],
]


class ConstraintProjector:
    """
    Apply a list of constraint functions to a windowed DataFrame.

    Usage::

        projector = ConstraintProjector(
            spec=hai_spec,
            baseline=baseline,
            constraints=[c1_bounds, c2_rate_limiter, c3_tracking, c4_cross_layer],
            window_col="bucket",
        )
        results = projector.evaluate(test_df)
    """

    def __init__(
        self,
        spec:        ConstraintSpec,
        baseline:    BehavioralBaseline,
        constraints: List[ConstraintFn],
        window_col:  str = "bucket",
    ):
        self.spec        = spec
        self.baseline    = baseline
        self.constraints = constraints
        self.window_col  = window_col

    def evaluate(self, df: pd.DataFrame) -> List[WindowConstraintResult]:
        """Evaluate all constraints over every window bucket in df."""
        results: List[WindowConstraintResult] = []
        buckets = sorted(df[self.window_col].unique())

        for i, bucket in enumerate(buckets):
            window_df = df[df[self.window_col] == bucket]
            all_violations: List[ConstraintViolation] = []

            for fn in self.constraints:
                try:
                    viols = fn(window_df, self.baseline, self.spec)
                    if viols:
                        all_violations.extend(viols)
                except Exception as exc:
                    log.debug("Constraint %s raised in bucket %d: %s", fn.__name__, bucket, exc)

            # Per-constraint confidence: fraction of windows with this constraint available
            conf: Dict[str, float] = {}
            for fn in self.constraints:
                cid = getattr(fn, "constraint_id", fn.__name__)
                # Confidence is static per constraint type (set on the function object)
                conf[cid] = getattr(fn, "spec_confidence", 1.0)

            results.append(WindowConstraintResult(
                window_id=i,
                bucket_ts=int(bucket),
                flagged=bool(all_violations),
                violations=all_violations,
                per_constraint_confidence=conf,
            ))

        flagged = sum(r.flagged for r in results)
        log.info("Projected %d windows → %d flagged (%.1f%%)",
                 len(results), flagged, 100 * flagged / max(len(results), 1))
        return results

    @staticmethod
    def to_feature_df(results: List[WindowConstraintResult]) -> pd.DataFrame:
        """Convert evaluation results to a feature DataFrame for AutoML."""
        rows = []
        for r in results:
            row = {"bucket_ts": r.bucket_ts, "mitl_flagged": int(r.flagged)}
            row.update(r.to_feature_row())
            rows.append(row)
        return pd.DataFrame(rows)
