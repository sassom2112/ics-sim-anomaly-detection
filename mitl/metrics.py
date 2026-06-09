"""
mitl.metrics — eTaPR (Enhanced Time-series Aware Precision and Recall).

HAI 22.04 mandates eTaPR rather than point-wise F1.  Standard F1 inflates
recall for detectors that flag entire attack durations as a single blob,
and penalises early-warning detectors that fire slightly before the event.
eTaPR scores at the event level with a configurable lead-time buffer.

Reference:
  Hwang et al., "Do You Know What You Are Protecting? An Enhanced Metric
  for Intrusion Detection System Evaluation in Industrial Control System,"
  HAICon 2021.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


def extract_segments(
    binary_labels: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
) -> List[Tuple[int, int]]:
    """Convert a binary array to (start_idx, end_idx) segment pairs (inclusive)."""
    segments: List[Tuple[int, int]] = []
    in_seg = False
    start  = 0
    for i, v in enumerate(binary_labels):
        if v and not in_seg:
            start  = i
            in_seg = True
        elif not v and in_seg:
            segments.append((start, i - 1))
            in_seg = False
    if in_seg:
        segments.append((start, len(binary_labels) - 1))
    return segments


def time_aware_recall(
    true_segments: List[Tuple[int, int]],
    pred_segments: List[Tuple[int, int]],
    buffer_steps:  int = 60,
) -> float:
    """
    eTaR — fraction of true attack events that are detected.

    An event is detected if any predicted segment overlaps
    [event_start − buffer, event_end].
    """
    if not true_segments:
        return 1.0
    detected = 0
    for t_start, t_end in true_segments:
        window_start = max(0, t_start - buffer_steps)
        for p_start, p_end in pred_segments:
            if p_end >= window_start and p_start <= t_end:
                detected += 1
                break
    return detected / len(true_segments)


def time_aware_precision(
    true_segments: List[Tuple[int, int]],
    pred_segments: List[Tuple[int, int]],
    buffer_steps:  int = 60,
) -> float:
    """
    eTaP — fraction of predicted clusters that overlap a real attack event.

    A prediction is valid if it overlaps
    [event_start − buffer, event_end + buffer].
    """
    if not pred_segments:
        return 1.0
    valid = 0
    for p_start, p_end in pred_segments:
        for t_start, t_end in true_segments:
            window_start = max(0, t_start - buffer_steps)
            window_end   = t_end + buffer_steps
            if p_end >= window_start and p_start <= window_end:
                valid += 1
                break
    return valid / len(pred_segments)


def etapr_f1(
    true_labels:   np.ndarray,
    pred_labels:   np.ndarray,
    buffer_steps:  int = 60,
) -> Tuple[float, float, float]:
    """
    Compute eTaPR F1 (harmonic mean of eTaP and eTaR).

    Parameters
    ----------
    true_labels   : binary array of ground-truth attack labels (1 = attack)
    pred_labels   : binary array of predicted labels
    buffer_steps  : lead-time buffer in time-steps (60 = 60 seconds at 1 Hz)

    Returns
    -------
    (eTaP, eTaR, eTaPR-F1)
    """
    true_segs = extract_segments(np.asarray(true_labels, dtype=int))
    pred_segs = extract_segments(np.asarray(pred_labels, dtype=int))

    if not true_segs and not pred_segs:
        return 1.0, 1.0, 1.0

    eta_r = time_aware_recall(true_segs, pred_segs, buffer_steps)
    eta_p = time_aware_precision(true_segs, pred_segs, buffer_steps)
    f1    = (2 * eta_p * eta_r / (eta_p + eta_r)) if (eta_p + eta_r) > 0 else 0.0
    return eta_p, eta_r, f1


def etapr_report(
    true_labels:  np.ndarray,
    pred_labels:  np.ndarray,
    label:        str = "",
    buffer_steps: int = 60,
) -> dict:
    """
    Full eTaPR report dict suitable for JSON serialisation.

    Includes both standard (point-wise) and eTaPR metrics so the gap is
    visible — this is Table 2 in the MITL paper.
    """
    from sklearn.metrics import f1_score, precision_score, recall_score

    y_true = np.asarray(true_labels, dtype=int)
    y_pred = np.asarray(pred_labels, dtype=int)

    std_p  = float(precision_score(y_true, y_pred, zero_division=0))
    std_r  = float(recall_score(y_true, y_pred, zero_division=0))
    std_f  = float(f1_score(y_true, y_pred, zero_division=0))

    eta_p, eta_r, eta_f = etapr_f1(y_true, y_pred, buffer_steps)

    true_segs = extract_segments(y_true)
    pred_segs = extract_segments(y_pred)

    return {
        "label":            label,
        "n_true_events":    len(true_segs),
        "n_pred_clusters":  len(pred_segs),
        "std_precision":    round(std_p,  4),
        "std_recall":       round(std_r,  4),
        "std_f1":           round(std_f,  4),
        "etap":             round(eta_p,  4),
        "etar":             round(eta_r,  4),
        "etapr_f1":         round(eta_f,  4),
        "buffer_steps":     buffer_steps,
    }
