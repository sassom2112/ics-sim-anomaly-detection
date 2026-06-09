"""
mitl.calibrate — Warm-up baseline calibration.

Fits per-tag statistics from the first WARMUP_FRACTION of training data
(no attack labels needed).  The resulting BehavioralBaseline is the
WHAT-IS-NORMAL complement to the spec's WHAT-IS-INVARIANT.
"""
from __future__ import annotations

import dataclasses
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .spec import ConstraintSpec

log = logging.getLogger(__name__)


@dataclasses.dataclass
class TagBaseline:
    name:               str
    mean:               float
    std:                float
    variance:           float
    max_delta_per_step: float   # max |Δ/step| observed during warm-up
    n_samples:          int


@dataclasses.dataclass
class BehavioralBaseline:
    """Calibrated behavioral norms derived from unlabeled warm-up data."""
    tag_baselines:   Dict[str, TagBaseline]
    warmup_fraction: float
    n_warmup_rows:   int
    dataset_name:    str

    def get(self, tag: str) -> Optional[TagBaseline]:
        return self.tag_baselines.get(tag)

    def variance_of(self, tag: str, fallback: float = 1e-6) -> float:
        tb = self.tag_baselines.get(tag)
        return tb.variance if tb is not None else fallback

    def std_of(self, tag: str, fallback: float = 1e-3) -> float:
        tb = self.tag_baselines.get(tag)
        return tb.std if tb is not None else fallback

    def max_rate_of(self, tag: str, fallback: float = 1.0) -> float:
        tb = self.tag_baselines.get(tag)
        return tb.max_delta_per_step if tb is not None else fallback


class BaselineCalibrator:
    """
    Fit a BehavioralBaseline from the warm-up slice of a DataFrame.

    Usage::

        calibrator = BaselineCalibrator(warmup_fraction=0.15)
        baseline   = calibrator.fit(train_df, spec)
    """

    def __init__(self, warmup_fraction: float = 0.15, skip_cols: Optional[List[str]] = None):
        self.warmup_fraction = warmup_fraction
        self.skip_cols = set(skip_cols or ["timestamp", "attack",
                                           "attack_P1", "attack_P2",
                                           "attack_P3", "attack_P4"])

    def fit(self, df: pd.DataFrame, spec: Optional[ConstraintSpec] = None) -> BehavioralBaseline:
        n_warm = max(10, int(len(df) * self.warmup_fraction))
        warm   = df.iloc[:n_warm]
        log.info("Warm-up calibration on %d rows (%.0f%% of %d)",
                 n_warm, self.warmup_fraction * 100, len(df))

        tag_baselines: Dict[str, TagBaseline] = {}
        feature_cols = [c for c in warm.columns
                        if c not in self.skip_cols
                        and pd.api.types.is_numeric_dtype(warm[c])]

        for col in feature_cols:
            vals = warm[col].dropna().values
            if len(vals) < 2:
                continue
            deltas = np.abs(np.diff(vals))
            tag_baselines[col] = TagBaseline(
                name=col,
                mean=float(np.mean(vals)),
                std=float(np.std(vals) + 1e-12),
                variance=float(np.var(vals) + 1e-12),
                max_delta_per_step=float(np.percentile(deltas, 99) if len(deltas) else 0.0),
                n_samples=len(vals),
            )

        dataset_name = spec.dataset_name if spec else "unknown"
        log.info("  Calibrated %d tags for dataset '%s'", len(tag_baselines), dataset_name)
        return BehavioralBaseline(
            tag_baselines=tag_baselines,
            warmup_fraction=self.warmup_fraction,
            n_warmup_rows=n_warm,
            dataset_name=dataset_name,
        )
