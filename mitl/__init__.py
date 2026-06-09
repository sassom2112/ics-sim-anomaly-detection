"""
mitl — Manual-in-the-Loop constraint projection for ICS anomaly detection.

Companion library to the MITL paper.  Encodes specification-derived
physical invariants from ICS engineering manuals and evaluates them over
windowed sensor/network time-series data.

Companion to CATT (Constrained-Adversarial Tabular Telemetry, AISec @ CCS 2026):
  CATT shows constraint projection exposes inflated EVASION in adversarial NIDS.
  MITL shows constraint projection closes DETECTION GAPS in ICS anomaly detection.

Quickstart::

    from mitl.spec       import ConstraintSpec
    from mitl.calibrate  import BaselineCalibrator
    from mitl.project    import ConstraintProjector
    from mitl.metrics    import etapr_report
    from mitl.datasets.hai import build_hai_spec, hai_constraints

    spec      = build_hai_spec()
    baseline  = BaselineCalibrator(warmup_fraction=0.15).fit(train_df, spec)
    projector = ConstraintProjector(spec, baseline, hai_constraints())
    results   = projector.evaluate(test_df)
"""

from .spec      import (ConstraintSpec, ConstraintSource, TagSpec,
                        ControlLoopSpec, ConstraintViolation, WindowConstraintResult)
from .calibrate import BaselineCalibrator, BehavioralBaseline, TagBaseline
from .project   import ConstraintProjector
from .metrics   import etapr_f1, etapr_report

__version__ = "0.1.0"
__all__ = [
    "ConstraintSpec", "ConstraintSource", "TagSpec", "ControlLoopSpec",
    "ConstraintViolation", "WindowConstraintResult",
    "BaselineCalibrator", "BehavioralBaseline", "TagBaseline",
    "ConstraintProjector",
    "etapr_f1", "etapr_report",
]
