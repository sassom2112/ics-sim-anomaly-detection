"""
mitl.spec — Core data contracts for the Manual-in-the-Loop constraint layer.

Every constraint must carry provenance: the document page and figure that
justify it.  ConstraintSpec is the single object a ConstraintProjector
consumes; it is either hand-authored or produced by mitl.extract.bedrock.
"""
from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional


@dataclasses.dataclass
class ConstraintSource:
    """Provenance of one constraint — where in the spec document it lives."""
    document:     str
    page_number:  int
    figure_id:    str            # "Figure 11" or "" if from a table
    quote:        str            # verbatim text from the manual
    extracted_by: str            # "table" | "diagram" | "text" | "llm" | "manual"
    confidence:   float = 1.0   # 1.0=explicit table row, 0.5=diagram inference


@dataclasses.dataclass
class TagSpec:
    """Per-sensor/actuator specification from the data-points table."""
    name:        str
    min_val:     float
    max_val:     float
    unit:        str
    description: str
    source:      ConstraintSource


@dataclasses.dataclass
class ControlLoopSpec:
    """A feedback control loop as described in the manual."""
    loop_id:             str
    setpoint_tag:        str
    process_var_tag:     str
    control_var_tag:     str
    has_saturation:      bool
    has_rate_limiter:    bool
    cross_layer_inputs:  List[str]   # tags from other process layers that drive this loop
    source:              ConstraintSource


@dataclasses.dataclass
class ConstraintViolation:
    """One constraint fire inside a single time window."""
    constraint_id:   str           # "C1", "C2", "C3", "C4"
    constraint_name: str
    severity:        str           # "definite" | "probable"
    evidence:        str           # human-readable description of what was observed
    spec_page:       int
    spec_figure:     str
    spec_quote:      str


@dataclasses.dataclass
class WindowConstraintResult:
    """MITL evaluation for one time window."""
    window_id:   int
    bucket_ts:   int
    flagged:     bool
    violations:  List[ConstraintViolation]
    per_constraint_confidence: Dict[str, float] = dataclasses.field(default_factory=dict)

    def flag_reason(self) -> str:
        if not self.violations:
            return "clean"
        return "+".join(sorted({v.constraint_id for v in self.violations}))

    def to_feature_row(self) -> Dict[str, float]:
        """AutoML integration: one flat dict suitable as a feature-matrix row."""
        row: Dict[str, float] = {
            "mitl_flagged":    float(self.flagged),
            "mitl_n_viols":    float(len(self.violations)),
        }
        for k, v in self.per_constraint_confidence.items():
            row[f"mitl_{k}_conf"] = v
        for viol in self.violations:
            row[f"mitl_{viol.constraint_id}_flag"] = 1.0
        return row


@dataclasses.dataclass
class ConstraintSpec:
    """
    Complete MITL specification compiled from one system manual.

    Hand-author this for a new dataset, or generate it via
    mitl.extract.bedrock.extract_constraint_spec().
    """
    dataset_name:    str
    manual_version:  str
    tag_specs:       Dict[str, TagSpec]       = dataclasses.field(default_factory=dict)
    loop_specs:      Dict[str, ControlLoopSpec] = dataclasses.field(default_factory=dict)
    extraction_confidence: float = 1.0        # fraction of expected tags found

    def describe(self) -> str:
        return (
            f"ConstraintSpec({self.dataset_name} / {self.manual_version})\n"
            f"  Tags: {len(self.tag_specs)}  Loops: {len(self.loop_specs)}  "
            f"Confidence: {self.extraction_confidence:.2f}"
        )
