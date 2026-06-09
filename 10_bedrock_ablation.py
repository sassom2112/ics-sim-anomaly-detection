"""
10_bedrock_ablation.py
======================
Sprint 11 — Bedrock LLM extraction ablation (Table 3 in the MITL paper).

Answers: how much eTaPR do you lose by using LLM-extracted constraints
instead of hand-coded ones?  This is the "any manual" generalisation claim
quantified: a domain engineer runs mitl.extract.bedrock, gets a ConstraintSpec
from the PDF, runs Sprint 9 with it.  How close does that get to hand-coded?

Process
-------
1. Find the HAI technical manual PDF (see PDF_PATHS below).
2. Call mitl.extract.bedrock.extract_constraint_spec() with Bedrock Claude.
3. Compare extracted ConstraintSpec to hand-coded build_hai_spec():
     - n_tags found, bounds accuracy (|Δmin| + |Δmax| per tag)
     - n_loops found, Rate Limiter / Saturation flags correct?
4. Run the Sprint 9 evaluation twice on the same data:
     Run A: hand-coded hai_constraints() + build_hai_spec()
     Run B: c1_from_spec() + llm_spec (LLM bounds for C1; C2-C4 unchanged)
5. Compare eTaPR → fill Table 3 in mitl_paper.tex.

Getting the PDF
---------------
The HAI technical manual is bundled with the HAI dataset GitHub repo:

    git clone https://github.com/icsdataset/hai
    # PDF is at:  hai/HAI_Dataset_Technical_Details.pdf

Or save the PDF you already have to one of the paths in PDF_PATHS below.

AWS Bedrock credentials
-----------------------
boto3 uses the standard AWS credential chain (environment variables,
~/.aws/credentials, instance profile).  Region must be one where
Bedrock Anthropic models are available (us-east-1, us-west-2).

    export AWS_DEFAULT_REGION=us-east-1
    # or: aws configure
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings_available = True
try:
    import warnings
    warnings.filterwarnings("ignore")
except Exception:
    pass

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))

from mitl import (BaselineCalibrator, ConstraintProjector, WindowConstraintResult,
                  etapr_report)
from mitl.datasets.hai import (build_hai_spec, hai_constraints,
                                hai_constraints_from_spec, HAI_TAG_BOUNDS)
from mitl.metrics import etapr_f1

OUTPUT_DIR   = "outputs/bedrock_ablation"
WARMUP_FRAC  = 0.15
WINDOW_S     = 60
ETAPR_BUFFER = 60

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Where to look for the PDF ─────────────────────────────────────────────────
PDF_PATHS = [
    "hai/HAI_Dataset_Technical_Details.pdf",
    "data/hai/HAI_Dataset_Technical_Details.pdf",
    "/kaggle/input/hai-security-dataset/HAI_Dataset_Technical_Details.pdf",
    "hai_dataset_technical_details.pdf",
    "HAI_Dataset_Technical_Details.pdf",
    str(Path.home() / "Downloads" / "HAI_Dataset_Technical_Details.pdf"),
]


# ══════════════════════════════════════════════════════════════════════════════
# §1  FIND PDF
# ══════════════════════════════════════════════════════════════════════════════

pdf_path = next((p for p in PDF_PATHS if os.path.exists(p)), None)

if pdf_path is None:
    log.error("=" * 60)
    log.error("  HAI technical manual PDF not found.")
    log.error("")
    log.error("  Step 1 — get the PDF (one of):")
    log.error("    git clone https://github.com/icsdataset/hai")
    log.error("    # file is at: hai/HAI_Dataset_Technical_Details.pdf")
    log.error("")
    log.error("  Step 2 — or copy your existing PDF to:")
    log.error("    %s/hai_dataset_technical_details.pdf",
              Path(__file__).parent)
    log.error("=" * 60)
    log.error("Running in MOCK mode (simulated LLM extraction).")
    MOCK_MODE = True
else:
    log.info("PDF found at: %s (%.1f KB)", pdf_path, os.path.getsize(pdf_path) / 1024)
    MOCK_MODE = False


# ══════════════════════════════════════════════════════════════════════════════
# §2  LLM EXTRACTION (real or mock)
# ══════════════════════════════════════════════════════════════════════════════

def _mock_llm_spec():
    """
    Simulate imperfect LLM extraction for testing without Bedrock.

    Intentionally introduces:
      - A few missing tags (LLM didn't read every table row)
      - Some slightly wrong bounds (rounding or unit confusion)
      - All 5 loops found, all Rate Limiter flags correct
    """
    from mitl.spec import ConstraintSource, ConstraintSpec, ControlLoopSpec, TagSpec

    def _src(page, fig, quote):
        return ConstraintSource(
            document="HAI_Dataset_Technical_Details.pdf",
            page_number=page, figure_id=fig, quote=quote,
            extracted_by="llm", confidence=0.85,
        )

    # Simulate: LLM found 14 of 19 tags; two bounds slightly off
    partial_bounds = {
        "P1_B2016":  (0.0,   10.0,  "bar", 12),
        "P1_PIT01":  (0.0,   10.0,  "bar", 13),
        "P1_PCV01D": (0.0,  100.0,  "%",   13),
        "P1_B3004":  (0.0,  720.0,  "mm",  12),
        "P1_LIT01":  (0.0,  720.0,  "mm",  13),
        "P1_LCV01D": (0.0,  100.0,  "%",   13),
        "P1_FIT01":  (0.0,   12.0,  "m³/h",13),   # slightly wrong: 12 instead of 10
        "P2_AutoSD": (0.0, 3200.0,  "RPM", 13),
        "P2_SIT01":  (0.0, 3200.0,  "RPM", 14),
        "P2_RTR":    (0.0, 3000.0,  "RPM", 14),   # slightly wrong: 3000 instead of 2880
        "P3_LIT01":  (0.0, 1000.0,  "mm",  14),
        "P4_ST_PS":  (0.0,  100.0,  "%",   15),
        "P4_HT_PS":  (0.0,  100.0,  "%",   15),
        # Missing: P1_B3005, P1_FCV03D, P1_B3003, P1_TIT01, P2_SCO, P3_LCV01
    }

    tag_specs = {}
    for tag, (lo, hi, unit, page) in partial_bounds.items():
        tag_specs[tag] = TagSpec(
            name=tag, min_val=lo, max_val=hi, unit=unit,
            description=f"Extracted by LLM: {tag}",
            source=_src(page, "Table 1", f"{tag}: [{lo}, {hi}] {unit}"),
        )

    loop_specs = {
        "P1-PC": ControlLoopSpec(
            loop_id="P1-PC", setpoint_tag="P1_B2016",
            process_var_tag="P1_PIT01", control_var_tag="P1_PCV01D",
            has_saturation=True, has_rate_limiter=True, cross_layer_inputs=[],
            source=_src(7, "Figure 4", "P1-PC loop with Rate Limiter"),
        ),
        "P2-SC": ControlLoopSpec(
            loop_id="P2-SC", setpoint_tag="P2_AutoSD",
            process_var_tag="P2_SIT01", control_var_tag="P2_SCO",
            has_saturation=True, has_rate_limiter=True,
            cross_layer_inputs=["P4_ST_PS"],
            source=_src(10, "Figure 11",
                        "P2-SC: AutoSD → Ramp → PID → SIT01. SIT01 must track AutoSD."),
        ),
        # LLM found only 2 of 5 loops (P1-LC, P1-FC, P3-LC missed)
    }

    return ConstraintSpec(
        dataset_name="HAI-22.04",
        manual_version="HAI_Dataset_Technical_Details.pdf (LLM extracted)",
        tag_specs=tag_specs,
        loop_specs=loop_specs,
        extraction_confidence=round(len(tag_specs) / 19, 3),  # 14/19 = 0.74
    )


def run_bedrock_extraction(pdf_path: str) -> "ConstraintSpec":
    """Call Bedrock Claude to extract ConstraintSpec from the PDF."""
    try:
        import boto3
    except ImportError:
        raise RuntimeError("boto3 not installed — run: pip install 'mitl[bedrock]'")

    region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    from botocore.config import Config
    client = boto3.client(
        "bedrock-runtime",
        region_name=region,
        config=Config(read_timeout=300, connect_timeout=10),
    )

    log.info("Calling Bedrock Claude for PDF extraction …")
    log.info("  PDF: %s", pdf_path)
    log.info("  Region: %s", region)

    from mitl.extract.bedrock import extract_constraint_spec
    spec = extract_constraint_spec(pdf_path, client,
                                   model_id="us.anthropic.claude-sonnet-4-6",
                                   max_tokens=65536)
    return spec


if MOCK_MODE:
    log.info("Using MOCK LLM extraction (no PDF / no Bedrock call).")
    llm_spec = _mock_llm_spec()
else:
    llm_spec = run_bedrock_extraction(pdf_path)

log.info("LLM spec: %s", llm_spec.describe())


# ══════════════════════════════════════════════════════════════════════════════
# §3  COMPARE: LLM-EXTRACTED vs HAND-CODED
# ══════════════════════════════════════════════════════════════════════════════

log.info("[2/5] Comparing LLM-extracted vs hand-coded ConstraintSpec …")

hand_spec = build_hai_spec()

# Tag coverage
llm_tags  = set(llm_spec.tag_specs.keys())
hand_tags = set(hand_spec.tag_specs.keys())
found_pct = len(llm_tags & hand_tags) / max(len(hand_tags), 1)
missing   = hand_tags - llm_tags
extra     = llm_tags - hand_tags

# Bounds accuracy for matched tags
bounds_errors: Dict[str, Dict] = {}
for tag in llm_tags & hand_tags:
    lt = llm_spec.tag_specs[tag]
    ht = hand_spec.tag_specs[tag]
    err_lo = abs(lt.min_val - ht.min_val)
    err_hi = abs(lt.max_val - ht.max_val)
    if err_lo > 0 or err_hi > 0:
        bounds_errors[tag] = {
            "hand_bounds": [ht.min_val, ht.max_val],
            "llm_bounds":  [lt.min_val, lt.max_val],
            "err_lo": err_lo, "err_hi": err_hi,
        }

# Loop topology
llm_loops  = set(llm_spec.loop_specs.keys())
hand_loops = set(hand_spec.loop_specs.keys())
rl_correct = sum(
    1 for lid in llm_loops & hand_loops
    if llm_spec.loop_specs[lid].has_rate_limiter ==
       hand_spec.loop_specs[lid].has_rate_limiter
)

log.info("  Tags:  LLM found %d / %d hand-coded (%.0f%% coverage)",
         len(llm_tags & hand_tags), len(hand_tags), found_pct * 100)
if missing:
    log.info("  Missing tags: %s", sorted(missing))
if extra:
    log.info("  Extra tags (not in hand spec): %s", sorted(extra))
if bounds_errors:
    log.info("  Bounds errors (%d tags):", len(bounds_errors))
    for tag, e in bounds_errors.items():
        log.info("    %s: hand %s vs LLM %s",
                 tag, e["hand_bounds"], e["llm_bounds"])
else:
    log.info("  All matched tag bounds are exact.")

log.info("  Loops: LLM found %d / %d hand-coded",
         len(llm_loops & hand_loops), len(hand_loops))
if llm_loops & hand_loops:
    log.info("  Rate Limiter flags: %d / %d correct",
             rl_correct, len(llm_loops & hand_loops))


# ══════════════════════════════════════════════════════════════════════════════
# §4  SHARED DATA — load or generate demo data
# ══════════════════════════════════════════════════════════════════════════════

log.info("[3/5] Preparing evaluation data …")

# Reuse the demo data generator from Sprint 9
def _make_demo_data(n_train=3600, n_test=3600, seed=42):
    rng = np.random.default_rng(seed)
    t_train = np.arange(n_train)
    t_test  = np.arange(n_test)

    def _spd(t):
        return 800 + 400 * np.sin(2 * np.pi * t / 1800) + rng.normal(0, 5, len(t))

    train_df = pd.DataFrame({
        "unix_ts": t_train, "P2_AutoSD": _spd(t_train),
        "P2_SIT01": _spd(t_train) + rng.normal(0, 10, n_train),
        "P4_ST_PS": 50 + 20 * np.sin(2 * np.pi * t_train / 3600) + rng.normal(0, 2, n_train),
        "P1_PIT01": 5 + rng.normal(0, 0.1, n_train),
        "P1_LIT01": 360 + rng.normal(0, 10, n_train),
        "P1_FIT01": 5 + rng.normal(0, 0.5, n_train),
        "attack": 0,
    })
    sit01 = _spd(t_test) + rng.normal(0, 10, n_test)
    autosd = _spd(t_test)
    attack = np.zeros(n_test, dtype=int)
    for s, e in [(1200, 1500), (2700, 3000)]:
        sit01[s:e] = sit01[s - 1]
        autosd[s:e] += np.linspace(0, 200, e - s)
        attack[s:e] = 1
    test_df = pd.DataFrame({
        "unix_ts": t_test, "P2_AutoSD": autosd, "P2_SIT01": sit01,
        "P4_ST_PS": 50 + 20 * np.sin(2 * np.pi * t_test / 3600) + rng.normal(0, 2, n_test),
        "P1_PIT01": 5 + rng.normal(0, 0.1, n_test),
        "P1_LIT01": 360 + rng.normal(0, 10, n_test),
        "P1_FIT01": 5 + rng.normal(0, 0.5, n_test),
        "attack": attack,
    })
    for df in (train_df, test_df):
        df["bucket"] = (df["unix_ts"] // WINDOW_S) * WINDOW_S
    return train_df, test_df

# Try to load real HAI data first; fall back to demo
HAI_CANDIDATES = [
    "/kaggle/input/hai-security-dataset", "hai", "data/hai",
]
HAI_DIR = next((p for p in HAI_CANDIDATES if os.path.exists(p)), None)

if HAI_DIR:
    log.info("  Loading real HAI data from %s", HAI_DIR)
    from pathlib import Path as _P
    csvs   = sorted(_P(HAI_DIR).glob("**/*.csv"))
    trains = [str(p) for p in csvs if "train" in p.stem.lower()]
    tests  = [str(p) for p in csvs if "test"  in p.stem.lower()]
    hai_train = pd.concat([pd.read_csv(f, low_memory=False) for f in trains[:1]])
    hai_test  = pd.concat([pd.read_csv(f, low_memory=False) for f in tests[:1]])
    for df in (hai_train, hai_test):
        df.columns = df.columns.str.strip()
        df["unix_ts"] = np.arange(len(df))
        df["bucket"]  = (df["unix_ts"] // WINDOW_S) * WINDOW_S
        if "attack" not in df.columns:
            df["attack"] = 0
    EVAL_SOURCE = "HAI-22.04"
else:
    log.info("  HAI data not found — using demo (synthetic AP27) data.")
    hai_train, hai_test = _make_demo_data()
    EVAL_SOURCE = "DEMO"

calibrator = BaselineCalibrator(warmup_fraction=WARMUP_FRAC)
baseline   = calibrator.fit(hai_train, hand_spec)   # same baseline for both runs


# ══════════════════════════════════════════════════════════════════════════════
# §5  RUN A: HAND-CODED | RUN B: LLM-EXTRACTED
# ══════════════════════════════════════════════════════════════════════════════

log.info("[4/5] Running constraint projection — hand-coded vs. LLM-extracted …")

def _evaluate(spec, constraints, label):
    proj = ConstraintProjector(spec=spec, baseline=baseline,
                               constraints=constraints, window_col="bucket")
    results = proj.evaluate(hai_test)
    y_true  = np.array([
        int((hai_test[hai_test["bucket"] == r.bucket_ts]["attack"].fillna(0) > 0).any())
        for r in results
    ])
    y_pred = np.array([int(r.flagged) for r in results])
    rpt    = etapr_report(y_true, y_pred, label=label, buffer_steps=ETAPR_BUFFER)
    rpt["n_flagged"]   = int(y_pred.sum())
    rpt["flag_rate"]   = round(float(y_pred.mean()), 4)
    return rpt, results

run_A, res_A = _evaluate(hand_spec, hai_constraints(),           "Hand-coded")
run_B, res_B = _evaluate(llm_spec,  hai_constraints_from_spec(), "LLM-extracted (Bedrock)")

# ══════════════════════════════════════════════════════════════════════════════
# §6  TABLE 3 OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

log.info("[5/5] Table 3 — Bedrock ablation results")
log.info("")
log.info("=" * 75)
log.info("  Table 3: eTaPR — Hand-coded vs LLM-extracted constraints (HAI 22.04)")
log.info("  eTaPR buffer = %ds | Window = %ds | Data = %s",
         ETAPR_BUFFER, WINDOW_S, EVAL_SOURCE)
log.info("=" * 75)

hdr = f"  {'Extraction':<30} {'n_tags':>7} {'loop_cov':>9} {'eTaP':>7} {'eTaR':>7} {'eTaF1':>7}"
log.info(hdr)
log.info("  " + "-" * 73)

for rpt, n_tags, n_loops, n_hand_tags, n_hand_loops in [
    (run_A, len(hand_spec.tag_specs), len(hand_spec.loop_specs),
     len(hand_spec.tag_specs), len(hand_spec.loop_specs)),
    (run_B, len(llm_spec.tag_specs & hand_spec.tag_specs
                if hasattr(llm_spec.tag_specs, '__and__')
                else set(llm_spec.tag_specs) & set(hand_spec.tag_specs)),
     len(set(llm_spec.loop_specs) & set(hand_spec.loop_specs)),
     len(hand_spec.tag_specs), len(hand_spec.loop_specs)),
]:
    tag_cov  = f"{n_tags}/{n_hand_tags}"
    loop_cov = f"{n_loops}/{n_hand_loops}"
    log.info("  %-30s %7s %9s %7.3f %7.3f %7.3f",
             rpt["label"], tag_cov, loop_cov,
             rpt["etap"], rpt["etar"], rpt["etapr_f1"])

log.info("  " + "-" * 73)
delta_f1 = run_A["etapr_f1"] - run_B["etapr_f1"]
log.info("  eTaPR-F1 delta (hand vs LLM): %.3f", delta_f1)
if MOCK_MODE:
    log.info("  [MOCK MODE — run with real PDF + Bedrock for paper numbers]")
log.info("=" * 75)

# ── Bounds accuracy table ─────────────────────────────────────────────────────
if bounds_errors:
    log.info("")
    log.info("  Bounds extraction errors (LLM vs hand-coded):")
    for tag, e in bounds_errors.items():
        log.info("    %-18s  hand=[%.1f, %.1f]  llm=[%.1f, %.1f]  Δ=%.1f",
                 tag, *e["hand_bounds"], *e["llm_bounds"],
                 e["err_lo"] + e["err_hi"])

# ── Save results ─────────────────────────────────────────────────────────────
ablation_card = {
    "version":    "1.0",
    "script":     "10_bedrock_ablation.py",
    "eval_source": EVAL_SOURCE,
    "mock_mode":  MOCK_MODE,
    "spec_comparison": {
        "hand_n_tags":   len(hand_spec.tag_specs),
        "hand_n_loops":  len(hand_spec.loop_specs),
        "llm_n_tags":    len(llm_spec.tag_specs),
        "llm_n_loops":   len(llm_spec.loop_specs),
        "tag_coverage":  round(found_pct, 3),
        "n_bounds_errors": len(bounds_errors),
        "bounds_errors":  bounds_errors,
        "rate_limiter_flags_correct": rl_correct,
    },
    "run_A_hand_coded":     run_A,
    "run_B_llm_extracted":  run_B,
    "etapr_f1_delta":       round(delta_f1, 4),
    "table3_latex": (
        "\\midrule\n"
        f"Hand-coded & {len(hand_spec.tag_specs)}/{len(hand_spec.tag_specs)} & "
        f"1.00 & {run_A['etap']:.3f} & {run_A['etar']:.3f} & "
        f"\\textbf{{{run_A['etapr_f1']:.3f}}} \\\\\n"
        f"Bedrock LLM (auto) & "
        f"{len(set(llm_spec.tag_specs) & set(hand_spec.tag_specs))}/{len(hand_spec.tag_specs)} & "
        f"{found_pct:.2f} & {run_B['etap']:.3f} & {run_B['etar']:.3f} & "
        f"{run_B['etapr_f1']:.3f} \\\\\n"
    ),
}

card_path = os.path.join(OUTPUT_DIR, "ablation_card.json")
with open(card_path, "w") as f:
    json.dump(ablation_card, f, indent=4)
log.info("")
log.info("  Ablation card → %s", card_path)
log.info("  Copy ablation_card['table3_latex'] into mitl_paper.tex Table 3.")
