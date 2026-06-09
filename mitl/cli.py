"""
mitl.cli — Command-line interface.

Entry point: `mitl` (registered via pyproject.toml console_scripts).

Subcommands
-----------
  mitl ingest   <pdf>  --name <name>     Extract spec from PDF via Bedrock
  mitl load     <json> --name <name>     Load a hand-coded spec JSON
  mitl list                              List all specs in the library
  mitl show     <name>                   Print full spec details
  mitl review   <name>                   Show items needing human verification
  mitl approve  <name> [--tags ...] [--all]  Mark items as reviewed
  mitl evaluate <csv>  --spec <name>     Run constraint projection + eTaPR report
  mitl delete   <name>                   Remove spec from library

Environment
-----------
  MITL_LIBRARY_PATH   Override default ~/.mitl/library
  AWS_DEFAULT_REGION  AWS region for Bedrock (default: us-east-1)
  AWS_PROFILE         Named AWS credentials profile
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Colour helpers (no external deps)
# ─────────────────────────────────────────────────────────────────────────────

_USE_COLOUR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None

def _c(code: str, text: str) -> str:
    if not _USE_COLOUR:
        return text
    codes = {"green": "32", "yellow": "33", "red": "31",
             "cyan": "36", "bold": "1", "dim": "2", "reset": "0"}
    return f"\033[{codes.get(code, '0')}m{text}\033[0m"

def _ok(t):    return _c("green",  "✓ " + t)
def _warn(t):  return _c("yellow", "⚠ " + t)
def _err(t):   return _c("red",    "✗ " + t)
def _head(t):  return _c("bold",   t)
def _dim(t):   return _c("dim",    t)


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand implementations
# ─────────────────────────────────────────────────────────────────────────────

def cmd_ingest(args: argparse.Namespace) -> int:
    """Extract a ConstraintSpec from a PDF via Bedrock and store in the library."""
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(_err(f"PDF not found: {pdf_path}"))
        print()
        print("To get the HAI technical manual:")
        print("  git clone https://github.com/icsdataset/hai")
        print("  # PDF is at: hai/HAI_Dataset_Technical_Details.pdf")
        return 1

    name   = args.name or pdf_path.stem
    region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    model  = args.model or "anthropic.claude-sonnet-4-6"

    print(_head(f"\n  MITL — Ingesting '{name}' from {pdf_path.name}"))
    print(f"  Bedrock region: {region}  |  Model: {model}")
    print(f"  PDF size: {pdf_path.stat().st_size / 1024:.1f} KB\n")

    try:
        import boto3
    except ImportError:
        print(_err("boto3 not installed.  Run: pip install 'mitl[bedrock]'"))
        return 1

    try:
        client = boto3.client("bedrock-runtime", region_name=region)
        from .extract.bedrock import extract_constraint_spec
        print("  Calling Bedrock Claude …", flush=True)
        spec = extract_constraint_spec(str(pdf_path), client, model_id=model)
    except Exception as exc:
        print(_err(f"Bedrock extraction failed: {exc}"))
        print("  Check AWS credentials and that the Bedrock model is enabled in your region.")
        return 1

    from .library import save as lib_save, review_report
    lib_path = lib_save(spec, name=name, system=args.system or "",
                        source_pdf=str(pdf_path), model_id=model)
    print(_ok(f"Spec '{name}' saved → {lib_path}"))
    print()

    rpt = review_report(name)
    _print_review_summary(rpt)
    return 0


def cmd_load(args: argparse.Namespace) -> int:
    """Load a hand-coded ConstraintSpec JSON into the library."""
    json_path = Path(args.json_path)
    if not json_path.exists():
        print(_err(f"File not found: {json_path}"))
        return 1

    name = args.name or json_path.stem
    with open(json_path) as f:
        raw = json.load(f)

    from .library import spec_from_dict, save as lib_save
    # Accept either a bare spec dict or an entry.json-style {"spec": {...}}
    spec_dict = raw.get("spec", raw)
    spec = spec_from_dict(spec_dict)
    # Mark as hand-coded (full confidence, pre-approved)
    for ts in spec.tag_specs.values():
        ts.source.confidence   = 1.0
        ts.source.extracted_by = "manual"
    for ls in spec.loop_specs.values():
        ls.source.confidence   = 1.0
        ls.source.extracted_by = "manual"
    spec.extraction_confidence = 1.0

    lib_path = lib_save(spec, name=name, system=args.system or "",
                        source_pdf=str(json_path), model_id="manual")
    # Auto-approve since it's hand-coded
    from .library import approve
    approve(name, approve_all=True)
    print(_ok(f"Spec '{name}' loaded (hand-coded, auto-approved) → {lib_path}"))
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    """List all specs in the library."""
    from .library import list_specs
    rows = list_specs()
    if not rows:
        from .library import library_root
        print(_dim(f"  Library is empty.  ({library_root()})"))
        print()
        print("  Add a spec:  mitl ingest manual.pdf --name my-system")
        return 0

    col_w = [30, 22, 6, 6, 10, 10, 11]
    header = (f"  {'Name':<{col_w[0]}} {'System':<{col_w[1]}} "
              f"{'Tags':>{col_w[2]}} {'Loops':>{col_w[3]}} "
              f"{'Conf':>{col_w[4]}} {'Status':<{col_w[5]}} {'Ingested':<{col_w[6]}}")
    print()
    print(_head(header))
    print("  " + "─" * sum(col_w + [6]))
    for r in rows:
        status_fmt = {
            "approved": _ok("approved"),
            "reviewed": _warn("reviewed"),
            "pending":  _warn("pending"),
        }.get(r["review_status"], r["review_status"])
        conf_str = f"{r['confidence']:.2f}"
        conf_fmt = _ok(conf_str) if r["confidence"] >= 1.0 else _warn(conf_str)
        print(f"  {r['name']:<{col_w[0]}} {r['system']:<{col_w[1]}} "
              f"{r['n_tags']:>{col_w[2]}} {r['n_loops']:>{col_w[3]}} "
              f"{conf_fmt:>{col_w[4]+10}} {status_fmt:<{col_w[5]+10}} "
              f"{r['ingested_at']:<{col_w[6]}}")
    print()
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    """Print full spec details for one library entry."""
    from .library import load_with_meta
    try:
        spec, meta = load_with_meta(args.name)
    except KeyError as e:
        print(_err(str(e)))
        return 1

    print()
    print(_head(f"  ── Spec: {args.name} ──────────────────────────────────────"))
    print(f"  System:    {meta.get('system', '—')}")
    print(f"  Manual:    {spec.manual_version}")
    print(f"  Source:    {Path(meta.get('source_pdf', '')).name}")
    print(f"  Ingested:  {meta.get('ingested_at','')[:19]}")
    print(f"  Model:     {meta.get('model_id','—')}")
    print(f"  Status:    {meta.get('review_status','—')}")
    print(f"  Confidence:{spec.extraction_confidence:.3f}")
    print()

    print(_head("  Tag Bounds (C1 — saturation constraints):"))
    print(f"  {'Tag':<22} {'Min':>10} {'Max':>10}  {'Unit':<8} {'Conf':>6}  Source")
    print("  " + "─" * 72)
    for tag, ts in sorted(spec.tag_specs.items()):
        conf_s = f"{ts.source.confidence:.2f}"
        conf_f = _ok(conf_s) if ts.source.confidence >= 1.0 else _warn(conf_s)
        src_s  = f"p{ts.source.page_number}" + (f" {ts.source.figure_id}" if ts.source.figure_id else "")
        print(f"  {tag:<22} {ts.min_val:>10.3g} {ts.max_val:>10.3g}  {ts.unit:<8} "
              f"{conf_f:>{6+10}}  {_dim(src_s)}")
    print()

    if spec.loop_specs:
        print(_head("  Control Loops (C2/C3/C4 — topology):"))
        print(f"  {'Loop':<12} {'SP':<18} {'PV':<18} {'CV':<18} {'RL':>4} {'Sat':>4} {'Conf':>6}")
        print("  " + "─" * 80)
        for lid, ls in sorted(spec.loop_specs.items()):
            rl  = _ok("yes") if ls.has_rate_limiter else _dim("no")
            sat = _ok("yes") if ls.has_saturation   else _dim("no")
            conf_s = f"{ls.source.confidence:.2f}"
            conf_f = _ok(conf_s) if ls.source.confidence >= 1.0 else _warn(conf_s)
            print(f"  {lid:<12} {ls.setpoint_tag:<18} {ls.process_var_tag:<18} "
                  f"{ls.control_var_tag:<18} {rl:>{4+10}} {sat:>{4+10}} {conf_f:>{6+10}}")
            if ls.cross_layer_inputs:
                print(f"  {'':12} Cross-layer inputs: {', '.join(ls.cross_layer_inputs)}")
    print()
    return 0


def cmd_review(args: argparse.Namespace) -> int:
    """Show items needing human verification for a library entry."""
    from .library import review_report
    try:
        rpt = review_report(args.name)
    except KeyError as e:
        print(_err(str(e)))
        return 1
    _print_review_summary(rpt, verbose=True)
    return 0


def _print_review_summary(rpt: dict, verbose: bool = False) -> None:
    name   = rpt["name"]
    status = rpt["review_status"]
    status_f = {"approved": _ok("approved"),
                "reviewed": _warn("reviewed"),
                "pending":  _warn("pending")}.get(status, status)

    print(_head(f"  ── Review: {name} {'─'*(40-len(name))}"))
    print(f"  Status: {status_f}  |  "
          f"Tags: {rpt['n_tags']}  |  Loops: {rpt['n_loops']}  |  "
          f"Confidence: {rpt['confidence']:.2f}")
    print()

    needs = rpt["tags_needing_review"]
    if not needs:
        print(_ok("  All tag bounds verified (confidence = 1.0 or already reviewed)."))
    else:
        print(_warn(f"  {len(needs)} tag(s) need review:"))
        for t in needs:
            print(f"    {_warn(t['tag']):<30}  bounds=[{t['bounds'][0]}, {t['bounds'][1]}] "
                  f"{t.get('unit',''):<6}  conf={t['confidence']:.2f}  p{t['page']}")

    needs_l = rpt["loops_needing_review"]
    if needs_l and verbose:
        print()
        print(_warn(f"  {len(needs_l)} loop(s) need review:"))
        for l in needs_l:
            rl  = "RL=yes" if l["has_rate_limiter"] else "RL=no"
            sat = "Sat=yes" if l["has_saturation"]  else "Sat=no"
            print(f"    {_warn(l['loop_id']):<14}  {rl}  {sat}  "
                  f"conf={l['confidence']:.2f}  {l.get('figure','')}")

    suspicious = rpt["suspicious_bounds"]
    if suspicious:
        print()
        print(_warn(f"  {len(suspicious)} tag(s) with suspicious bounds:"))
        for s in suspicious:
            print(f"    {s['tag']:<22}  bounds={s['bounds']}")

    reviewed = rpt["reviewed_tags"]
    if reviewed and verbose:
        print()
        print(_ok(f"  Already reviewed: {', '.join(reviewed)}"))

    print()
    if needs or needs_l:
        tag_list = " ".join(t["tag"] for t in needs[:6])
        loop_list = " ".join(l["loop_id"] for l in needs_l[:4])
        approve_cmd = f"  mitl approve {name}"
        if tag_list:
            approve_cmd += f" --tags {tag_list}"
        if loop_list:
            approve_cmd += f" --loops {loop_list}"
        print(f"  Next step: {_c('cyan', approve_cmd)}")
        print(f"  Or approve all:  {_c('cyan', f'mitl approve {name} --all')}")
    else:
        print(f"  Run evaluation:  {_c('cyan', f'mitl evaluate data.csv --spec {name}')}")
    print()


def cmd_approve(args: argparse.Namespace) -> int:
    """Mark tags/loops as human-reviewed, promoting their confidence to 1.0."""
    from .library import approve
    try:
        rpt = approve(
            args.name,
            tags=args.tags or None,
            loops=args.loops or None,
            approve_all=args.all,
        )
    except KeyError as e:
        print(_err(str(e)))
        return 1

    n_reviewed = len(rpt["reviewed_tags"]) + len(rpt["reviewed_loops"])
    n_total    = rpt["n_tags"] + rpt["n_loops"]
    print(_ok(f"  {args.name}: {n_reviewed}/{n_total} items reviewed, "
              f"status={rpt['review_status']}, confidence={rpt['confidence']:.3f}"))
    remaining = rpt["tags_needing_review"]
    if remaining:
        print(_warn(f"  {len(remaining)} tag(s) still pending review:"))
        for t in remaining[:5]:
            print(f"    {t['tag']:<22}  bounds=[{t['bounds'][0]}, {t['bounds'][1]}]")
        if len(remaining) > 5:
            print(f"    … and {len(remaining)-5} more.")
    print()
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Run MITL constraint projection against a CSV file."""
    import numpy as np
    import pandas as pd

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(_err(f"CSV not found: {csv_path}"))
        return 1

    from .library import load as lib_load
    try:
        spec = lib_load(args.spec)
    except KeyError as e:
        print(_err(str(e)))
        print(f"  Available specs: mitl list")
        return 1

    print(_head(f"\n  MITL Evaluate — {csv_path.name}  ×  {args.spec}"))
    print(f"  Window: {args.window}s  |  Warm-up: {args.warmup*100:.0f}%  |  "
          f"eTaPR buffer: {args.buffer}s\n")

    # Load CSV
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = df.columns.str.strip()

    # Timestamp / bucket
    ts_col = next((c for c in ["timestamp","time","Timestamp","Time"] if c in df.columns), None)
    if ts_col:
        df["_ts"] = pd.to_datetime(df[ts_col], errors="coerce")
        df = df.dropna(subset=["_ts"]).sort_values("_ts")
        df["unix_ts"] = df["_ts"].astype("int64") // 10**9
    else:
        df["unix_ts"] = np.arange(len(df))
    df["bucket"] = (df["unix_ts"] // args.window) * args.window

    # Label column
    label_col = next((c for c in ["attack","Attack","label","Label","anomaly"]
                      if c in df.columns), None)
    has_labels = label_col is not None
    if has_labels:
        df["attack"] = df[label_col].fillna(0).astype(int)
    else:
        print(_warn("  No label column found — evaluation will report flag rate only."))
        df["attack"] = 0

    # Split train (warm-up) / test
    n_warmup = max(10, int(len(df) * args.warmup))
    train_df = df.iloc[:n_warmup]
    test_df  = df.iloc[n_warmup:].copy()
    print(f"  Rows: total={len(df)}  warm-up={n_warmup}  test={len(test_df)}")

    # Calibrate
    from .calibrate import BaselineCalibrator
    baseline = BaselineCalibrator(warmup_fraction=1.0).fit(train_df, spec)

    # Choose constraints: dataset-specific if known, else generic
    dataset = spec.dataset_name.lower()
    if "hai" in dataset:
        from .datasets.hai import hai_constraints
        constraints = hai_constraints()
        constraint_set = "HAI-specific"
    elif "icssim" in dataset:
        from .datasets.icssim import icssim_constraints
        # ICSSim needs network features — fall back to generic if absent
        if "net_flow_count" not in test_df.columns:
            from .generic import generic_constraints
            constraints = generic_constraints()
            constraint_set = "generic (net_flow_count not in CSV)"
        else:
            constraints = icssim_constraints()
            constraint_set = "ICSSim-specific"
    else:
        from .generic import generic_constraints
        constraints = generic_constraints()
        constraint_set = "generic"

    print(f"  Constraint set: {constraint_set}  ({len(constraints)} constraints)")

    # Project
    from .project import ConstraintProjector
    projector = ConstraintProjector(spec=spec, baseline=baseline,
                                    constraints=constraints, window_col="bucket")
    results = projector.evaluate(test_df)

    y_pred = [int(r.flagged) for r in results]
    y_true = [
        int((test_df[test_df["bucket"] == r.bucket_ts]["attack"] > 0).any())
        for r in results
    ]

    # Report
    import numpy as np
    flagged = sum(y_pred)
    n_w     = len(results)
    print()
    print(_head(f"  ── Results ──────────────────────────────────────────"))
    print(f"  Windows evaluated:  {n_w}")
    print(f"  Windows flagged:    {flagged}  ({100*flagged/max(n_w,1):.1f}%)")

    if has_labels:
        from .metrics import etapr_report
        rpt = etapr_report(np.array(y_true), np.array(y_pred),
                           label=args.spec, buffer_steps=args.buffer)
        print(f"  Attack events:      {rpt['n_true_events']}")
        print()
        print(f"  {'Metric':<20}  {'Value':>8}")
        print("  " + "─" * 30)
        for k, v in [("eTaP (event prec)", rpt["etap"]),
                     ("eTaR (event recall)", rpt["etar"]),
                     ("eTaPR F1", rpt["etapr_f1"]),
                     ("Std F1 (ref)", rpt["std_f1"])]:
            val_f = _ok(f"{v:.3f}") if v >= 0.7 else (_warn(f"{v:.3f}") if v >= 0.4 else _err(f"{v:.3f}"))
            print(f"  {k:<20}  {val_f:>8}")
        print()

        # Per-constraint breakdown
        constraint_ids = sorted({v.constraint_id for r in results for v in r.violations})
        if constraint_ids:
            print(_head("  Per-constraint contribution:"))
            for cid in constraint_ids:
                c_pred = [int(any(v.constraint_id == cid for v in r.violations)) for r in results]
                c_rpt  = etapr_report(np.array(y_true), np.array(c_pred),
                                      label=cid, buffer_steps=args.buffer)
                fires  = sum(c_pred)
                print(f"    {cid:<6}  fires={fires:<5}  eTaR={c_rpt['etar']:.3f}  "
                      f"eTaP={c_rpt['etap']:.3f}  eTaF1={c_rpt['etapr_f1']:.3f}")
            print()

    # Violations sample
    all_viols = [v for r in results for v in r.violations]
    if all_viols:
        print(_head(f"  Sample violations (first 5 of {len(all_viols)}):"))
        for v in all_viols[:5]:
            print(f"    [{v.constraint_id}] p{v.spec_page} {v.spec_figure}")
            print(f"      {_dim(v.evidence[:80])}")
        print()

    # Output file
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_data = {
            "spec":    args.spec,
            "csv":     str(csv_path),
            "windows": n_w,
            "flagged": flagged,
            "results": [
                {
                    "window_id": r.window_id,
                    "bucket_ts": r.bucket_ts,
                    "flagged":   r.flagged,
                    "flag_reason": r.flag_reason(),
                    "violations": [
                        {"constraint_id": v.constraint_id,
                         "severity": v.severity,
                         "evidence": v.evidence,
                         "spec_page": v.spec_page,
                         "spec_figure": v.spec_figure}
                        for v in r.violations
                    ],
                }
                for r in results
            ],
        }
        if has_labels:
            from .metrics import etapr_report
            out_data["etapr"] = etapr_report(
                np.array(y_true), np.array(y_pred),
                label=args.spec, buffer_steps=args.buffer
            )
        with open(out_path, "w") as f:
            json.dump(out_data, f, indent=2)
        print(_ok(f"  Report → {out_path}"))
    print()
    return 0


def cmd_delete(args: argparse.Namespace) -> int:
    """Remove a spec from the library."""
    from .library import delete
    try:
        delete(args.name)
        print(_ok(f"  Deleted '{args.name}' from library."))
    except KeyError as e:
        print(_err(str(e)))
        return 1
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mitl",
        description=(
            "Manual-in-the-Loop (MITL) — specification-derived constraint\n"
            "projection for ICS anomaly detection.\n\n"
            "Quickstart:\n"
            "  mitl ingest manual.pdf --name my-system\n"
            "  mitl review  my-system\n"
            "  mitl approve my-system --all\n"
            "  mitl evaluate sensor_data.csv --spec my-system\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", metavar="<command>")

    # ingest
    pi = sub.add_parser("ingest", help="Extract spec from a PDF manual via Bedrock")
    pi.add_argument("pdf",    help="Path to the ICS engineering manual PDF")
    pi.add_argument("--name", help="Library name (default: PDF filename stem)")
    pi.add_argument("--system", default="", help="Short description of the system")
    pi.add_argument("--model", default="", help="Bedrock model ID")

    # load (hand-coded JSON)
    pl = sub.add_parser("load", help="Load a hand-coded ConstraintSpec JSON")
    pl.add_argument("json_path", metavar="json", help="Path to spec JSON file")
    pl.add_argument("--name",   help="Library name (default: filename stem)")
    pl.add_argument("--system", default="", help="Short description of the system")

    # list
    sub.add_parser("list", help="List all specs in the library")

    # show
    ps = sub.add_parser("show", help="Print full spec details")
    ps.add_argument("name", help="Library spec name")

    # review
    pr = sub.add_parser("review", help="Show items needing human verification")
    pr.add_argument("name", help="Library spec name")

    # approve
    pa = sub.add_parser("approve", help="Mark items as reviewed (confidence → 1.0)")
    pa.add_argument("name",   help="Library spec name")
    pa.add_argument("--tags",  nargs="+", help="Tag names to approve")
    pa.add_argument("--loops", nargs="+", help="Loop IDs to approve")
    pa.add_argument("--all",   action="store_true", help="Approve all items")

    # evaluate
    pe = sub.add_parser("evaluate", help="Run constraint projection against a CSV")
    pe.add_argument("csv",       help="Path to sensor/network CSV file")
    pe.add_argument("--spec",    required=True, help="Library spec name")
    pe.add_argument("--output",  help="Write JSON report to this path")
    pe.add_argument("--window",  type=int, default=60, help="Window size in seconds (default: 60)")
    pe.add_argument("--warmup",  type=float, default=0.15,
                    help="Warm-up fraction of data (default: 0.15)")
    pe.add_argument("--buffer",  type=int, default=60,
                    help="eTaPR buffer in seconds (default: 60)")

    # delete
    pd_ = sub.add_parser("delete", help="Remove spec from library")
    pd_.add_argument("name", help="Library spec name")

    return p


def main(argv=None) -> int:
    parser = build_parser()
    args   = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    dispatch = {
        "ingest":   cmd_ingest,
        "load":     cmd_load,
        "list":     cmd_list,
        "show":     cmd_show,
        "review":   cmd_review,
        "approve":  cmd_approve,
        "evaluate": cmd_evaluate,
        "delete":   cmd_delete,
    }
    fn = dispatch.get(args.command)
    if fn is None:
        parser.print_help()
        return 1
    return fn(args)


if __name__ == "__main__":
    sys.exit(main())
