"""
mitl.library — Persistent library of ConstraintSpecs.

Stores extracted and hand-coded specs as JSON under ~/.mitl/library/<name>/.
Tracks review state: LLM-extracted items start at confidence 0.85 and are
promoted to 1.0 after a human runs `mitl review` + `mitl approve`.

Library layout::

    ~/.mitl/
      library/
        HAI-22.04/
          entry.json     ← spec + metadata in one file
        ICSSim-v2/
          entry.json
      config.json        ← global defaults (region, model_id, etc.)

The MITL_LIBRARY_PATH env var overrides the default ~/.mitl/library path.
"""
from __future__ import annotations

import dataclasses
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .spec import (ConstraintSource, ConstraintSpec, ControlLoopSpec, TagSpec)


# ── Library root ──────────────────────────────────────────────────────────────

def library_root() -> Path:
    env = os.environ.get("MITL_LIBRARY_PATH")
    if env:
        return Path(env)
    return Path.home() / ".mitl" / "library"


def _entry_path(name: str) -> Path:
    return library_root() / name / "entry.json"


# ── Serialisation ─────────────────────────────────────────────────────────────

def _src_to_dict(src: ConstraintSource) -> dict:
    return dataclasses.asdict(src)


def _src_from_dict(d: dict) -> ConstraintSource:
    return ConstraintSource(**d)


def _tag_to_dict(ts: TagSpec) -> dict:
    return {
        "name":        ts.name,
        "min_val":     ts.min_val,
        "max_val":     ts.max_val,
        "unit":        ts.unit,
        "description": ts.description,
        "source":      _src_to_dict(ts.source),
    }


def _tag_from_dict(d: dict) -> TagSpec:
    return TagSpec(
        name=d["name"], min_val=d["min_val"], max_val=d["max_val"],
        unit=d["unit"], description=d["description"],
        source=_src_from_dict(d["source"]),
    )


def _loop_to_dict(ls: ControlLoopSpec) -> dict:
    return {
        "loop_id":            ls.loop_id,
        "setpoint_tag":       ls.setpoint_tag,
        "process_var_tag":    ls.process_var_tag,
        "control_var_tag":    ls.control_var_tag,
        "has_saturation":     ls.has_saturation,
        "has_rate_limiter":   ls.has_rate_limiter,
        "cross_layer_inputs": ls.cross_layer_inputs,
        "source":             _src_to_dict(ls.source),
    }


def _loop_from_dict(d: dict) -> ControlLoopSpec:
    return ControlLoopSpec(
        loop_id=d["loop_id"],
        setpoint_tag=d["setpoint_tag"],
        process_var_tag=d["process_var_tag"],
        control_var_tag=d["control_var_tag"],
        has_saturation=d["has_saturation"],
        has_rate_limiter=d["has_rate_limiter"],
        cross_layer_inputs=d.get("cross_layer_inputs", []),
        source=_src_from_dict(d["source"]),
    )


def spec_to_dict(spec: ConstraintSpec) -> dict:
    return {
        "dataset_name":          spec.dataset_name,
        "manual_version":        spec.manual_version,
        "extraction_confidence": spec.extraction_confidence,
        "tag_specs":  {k: _tag_to_dict(v) for k, v in spec.tag_specs.items()},
        "loop_specs": {k: _loop_to_dict(v) for k, v in spec.loop_specs.items()},
    }


def spec_from_dict(d: dict) -> ConstraintSpec:
    return ConstraintSpec(
        dataset_name=d["dataset_name"],
        manual_version=d["manual_version"],
        extraction_confidence=d.get("extraction_confidence", 1.0),
        tag_specs={k: _tag_from_dict(v) for k, v in d.get("tag_specs", {}).items()},
        loop_specs={k: _loop_from_dict(v) for k, v in d.get("loop_specs", {}).items()},
    )


# ── Entry (spec + metadata) ───────────────────────────────────────────────────

def _blank_meta(name: str, system: str, source_pdf: str, model_id: str) -> dict:
    return {
        "name":            name,
        "system":          system,
        "source_pdf":      str(source_pdf),
        "model_id":        model_id,
        "ingested_at":     datetime.now(timezone.utc).isoformat(),
        "review_status":   "pending",   # pending | reviewed | approved
        "reviewed_tags":   [],          # tags a human has spot-checked
        "reviewed_loops":  [],
        "approved_at":     None,
    }


def _load_entry(name: str) -> dict:
    p = _entry_path(name)
    if not p.exists():
        raise KeyError(f"No spec named '{name}' in library at {library_root()}")
    with open(p) as f:
        return json.load(f)


def _save_entry(name: str, entry: dict) -> None:
    p = _entry_path(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(entry, f, indent=2)


# ── Public API ────────────────────────────────────────────────────────────────

def save(
    spec:       ConstraintSpec,
    name:       str,
    system:     str = "",
    source_pdf: str = "",
    model_id:   str = "",
) -> Path:
    """
    Save a ConstraintSpec to the library.  Overwrites if name already exists.
    Returns the path to the entry file.
    """
    # Preserve existing metadata if re-saving
    try:
        existing = _load_entry(name)
        meta = existing["meta"]
    except KeyError:
        meta = _blank_meta(name, system, source_pdf, model_id)

    meta["source_pdf"] = str(source_pdf) or meta.get("source_pdf", "")
    meta["model_id"]   = model_id or meta.get("model_id", "")
    meta["system"]     = system   or meta.get("system", "")

    entry = {"meta": meta, "spec": spec_to_dict(spec)}
    _save_entry(name, entry)
    return _entry_path(name)


def load(name: str) -> ConstraintSpec:
    """Load a ConstraintSpec from the library by name."""
    entry = _load_entry(name)
    return spec_from_dict(entry["spec"])


def load_with_meta(name: str):
    """Return (ConstraintSpec, meta_dict)."""
    entry = _load_entry(name)
    return spec_from_dict(entry["spec"]), entry["meta"]


def list_specs() -> List[Dict[str, Any]]:
    """Return summary rows for all specs in the library."""
    root = library_root()
    if not root.exists():
        return []
    rows = []
    for entry_file in sorted(root.glob("*/entry.json")):
        try:
            entry = json.loads(entry_file.read_text())
            meta  = entry["meta"]
            spec  = entry["spec"]
            rows.append({
                "name":            meta["name"],
                "system":          meta.get("system", ""),
                "n_tags":          len(spec.get("tag_specs", {})),
                "n_loops":         len(spec.get("loop_specs", {})),
                "confidence":      spec.get("extraction_confidence", 1.0),
                "review_status":   meta.get("review_status", "pending"),
                "ingested_at":     meta.get("ingested_at", "")[:10],
                "source_pdf":      Path(meta.get("source_pdf", "")).name,
            })
        except Exception:
            pass
    return rows


def delete(name: str) -> None:
    """Remove a spec from the library."""
    p = _entry_path(name)
    if not p.exists():
        raise KeyError(f"No spec named '{name}'")
    p.unlink()
    try:
        p.parent.rmdir()
    except OSError:
        pass


def review_report(name: str) -> dict:
    """
    Generate a review report for a spec: what needs human verification.

    Returns a dict with:
      - tags_needing_review:  list of tag names with confidence < 1.0
      - loops_needing_review: list of loop IDs with confidence < 1.0
      - suspicious_bounds:    tags where the range looks unusual
    """
    spec, meta = load_with_meta(name)

    needs_review_tags  = []
    suspicious_bounds  = []
    reviewed_tags      = set(meta.get("reviewed_tags", []))

    for tag, ts in spec.tag_specs.items():
        if ts.source.confidence < 1.0 and tag not in reviewed_tags:
            needs_review_tags.append({
                "tag":        tag,
                "bounds":     [ts.min_val, ts.max_val],
                "unit":       ts.unit,
                "confidence": ts.source.confidence,
                "page":       ts.source.page_number,
            })
        # Flag suspiciously wide or zero-width bounds
        span = ts.max_val - ts.min_val
        if span <= 0 or span > 1e6:
            suspicious_bounds.append({"tag": tag, "bounds": [ts.min_val, ts.max_val]})

    needs_review_loops = []
    reviewed_loops     = set(meta.get("reviewed_loops", []))
    for lid, ls in spec.loop_specs.items():
        if ls.source.confidence < 1.0 and lid not in reviewed_loops:
            needs_review_loops.append({
                "loop_id":          lid,
                "has_rate_limiter": ls.has_rate_limiter,
                "has_saturation":   ls.has_saturation,
                "cross_layer":      ls.cross_layer_inputs,
                "confidence":       ls.source.confidence,
                "figure":           ls.source.figure_id,
            })

    return {
        "name":                meta["name"],
        "review_status":       meta.get("review_status", "pending"),
        "n_tags":              len(spec.tag_specs),
        "n_loops":             len(spec.loop_specs),
        "confidence":          spec.extraction_confidence,
        "tags_needing_review": needs_review_tags,
        "loops_needing_review":needs_review_loops,
        "suspicious_bounds":   suspicious_bounds,
        "reviewed_tags":       sorted(reviewed_tags),
        "reviewed_loops":      sorted(reviewed_loops),
    }


def approve(
    name:         str,
    tags:         Optional[List[str]] = None,
    loops:        Optional[List[str]] = None,
    approve_all:  bool = False,
) -> dict:
    """
    Mark tags/loops as human-reviewed (promotes confidence to 1.0 in metadata).
    Use approve_all=True to approve everything at once.

    Returns the updated review report.
    """
    entry = _load_entry(name)
    meta  = entry["meta"]
    spec_d = entry["spec"]

    reviewed_tags  = set(meta.get("reviewed_tags", []))
    reviewed_loops = set(meta.get("reviewed_loops", []))

    if approve_all:
        reviewed_tags  = set(spec_d.get("tag_specs", {}).keys())
        reviewed_loops = set(spec_d.get("loop_specs", {}).keys())
    else:
        if tags:
            reviewed_tags.update(tags)
        if loops:
            reviewed_loops.update(loops)

    meta["reviewed_tags"]  = sorted(reviewed_tags)
    meta["reviewed_loops"] = sorted(reviewed_loops)

    # Promote confidence in spec for approved items
    for tag in reviewed_tags:
        if tag in spec_d.get("tag_specs", {}):
            spec_d["tag_specs"][tag]["source"]["confidence"] = 1.0
    for lid in reviewed_loops:
        if lid in spec_d.get("loop_specs", {}):
            spec_d["loop_specs"][lid]["source"]["confidence"] = 1.0

    # Update overall confidence + status
    all_tags  = set(spec_d.get("tag_specs", {}).keys())
    all_loops = set(spec_d.get("loop_specs", {}).keys())
    fraction  = (len(reviewed_tags & all_tags) + len(reviewed_loops & all_loops)) / \
                max(len(all_tags) + len(all_loops), 1)
    spec_d["extraction_confidence"] = round(
        0.85 + 0.15 * fraction, 3
    )   # 0.85 → 1.0 as review progresses

    if reviewed_tags >= all_tags and reviewed_loops >= all_loops:
        meta["review_status"] = "approved"
        meta["approved_at"]   = datetime.now(timezone.utc).isoformat()
    elif reviewed_tags or reviewed_loops:
        meta["review_status"] = "reviewed"

    entry["meta"]  = meta
    entry["spec"]  = spec_d
    _save_entry(name, entry)
    return review_report(name)
