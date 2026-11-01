from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import tempfile
import uuid
import zlib
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components


APP_TITLE = "Chest X-Ray Triage for Pneumonia — Advanced Synthetic Demo"

ACTION_PRIORITIZE = "PRIORITIZE"
ACTION_REVIEW = "REVIEW"
ACTION_ROUTINE = "ROUTINE"
ACTION_HOLD = "HOLD"
ALLOWED_ACTIONS = {ACTION_PRIORITIZE, ACTION_REVIEW, ACTION_ROUTINE, ACTION_HOLD}


# -----------------------------------------------------------------------------
# Configuration helpers
# -----------------------------------------------------------------------------
def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def clamp_float(value: Any, low: float, high: float, default: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(v):
        return default
    return min(high, max(low, v))


def env_choice_int(name: str, default: int, choices: Tuple[int, ...]) -> int:
    try:
        v = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        v = default
    return v if v in choices else default


# -----------------------------------------------------------------------------
# Operational configuration. No credentials, no hard-coded secrets.
# -----------------------------------------------------------------------------
SEED = _env_int("DEMO_RANDOM_SEED", 7)
N_STUDIES = max(30, _env_int("DEMO_N_STUDIES", 140))

TRIAGE_THRESHOLD = clamp_float(_env_float("TRIAGE_THRESHOLD", 0.75), 0.50, 0.95, 0.75)
REVIEW_BAND = clamp_float(_env_float("TRIAGE_REVIEW_BAND", 0.15), 0.00, 0.30, 0.15)
UNCERTAINTY_DEFER_THRESHOLD = clamp_float(
    _env_float("UNCERTAINTY_DEFER_THRESHOLD", 0.65), 0.00, 1.00, 0.65
)
OOD_REJECT_THRESHOLD = clamp_float(
    _env_float("OOD_REJECT_THRESHOLD", 0.70), 0.00, 1.00, 0.70
)
TARGET_SENSITIVITY = clamp_float(
    _env_float("TARGET_SENSITIVITY", 0.95), 0.80, 0.999, 0.95
)

SLA_MINUTES = max(1, _env_int("SLA_MINUTES", 30))
WAIT_REDUCTION_FACTOR = clamp_float(
    _env_float("WAIT_REDUCTION_FACTOR", 0.45), 0.10, 1.00, 0.45
)

PSI_WARN = clamp_float(_env_float("PSI_WARN", 0.20), 0.05, 1.00, 0.20)
PSI_ALERT = clamp_float(_env_float("PSI_ALERT", 0.25), 0.05, 1.00, 0.25)

MAX_PUSHES_PER_MINUTE = max(1, _env_int("MAX_PUSHES_PER_MINUTE", 30))
RETENTION_DAYS = max(1, _env_int("RETENTION_DAYS", 90))
AUDIT_LOG_MAX = max(100, _env_int("AUDIT_LOG_MAX", 10000))

MODEL_VERSION = os.getenv("MODEL_VERSION", "synthetic-triage-advanced-0.5.1")
SCHEMA_VERSION = os.getenv("OUTPUT_SCHEMA_VERSION", "1.1.1")

VIDEO_FPS = env_choice_int("DEMO_VIDEO_FPS", 10, (8, 10, 12, 15))
VIDEO_SECONDS = clamp_float(_env_float("DEMO_VIDEO_SECONDS", 6.0), 3.0, 10.0, 6.0)
VIDEO_CASES = env_choice_int("DEMO_VIDEO_CASES", 6, (3, 4, 6, 8))

VIDEO_WIDTH = max(720, _env_int("DEMO_VIDEO_WIDTH", 1024))
VIDEO_HEIGHT = max(420, _env_int("DEMO_VIDEO_HEIGHT", 576))
XRAY_SIZE = max(240, _env_int("DEMO_XRAY_SIZE", 460))

if VIDEO_WIDTH % 2:
    VIDEO_WIDTH += 1
if VIDEO_HEIGHT % 2:
    VIDEO_HEIGHT += 1


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------
def stable_seed(value: str) -> int:
    return zlib.crc32(str(value).encode("utf-8")) & 0xFFFFFFFF


def hash_obj(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def utcnow_iso() -> str:
    return utcnow().replace(microsecond=0).isoformat()


def _is_missing(value: Any) -> bool:
    try:
        return value is None or pd.isna(value)
    except (TypeError, ValueError):
        return False


def _timestamp_to_iso_or_none(value: Any) -> Optional[str]:
    if _is_missing(value):
        return None
    if isinstance(value, str):
        return value
    try:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts.replace(microsecond=0).isoformat()
    except Exception:
        return str(value)


def sigmoid(x: Any) -> Any:
    x_arr = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x_arr))


def logit(p: Any) -> Any:
    p_arr = np.asarray(p, dtype=float)
    p_arr = np.clip(p_arr, 1e-6, 1.0 - 1e-6)
    return np.log(p_arr / (1.0 - p_arr))


def _format_percent(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        if pd.isna(value):
            return "n/a"
    except (TypeError, ValueError):
        pass
    return f"{float(value):.1%}"


def _format_float(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        if pd.isna(value):
            return "n/a"
    except (TypeError, ValueError):
        pass
    return f"{float(value):.{digits}f}"


def _format_datetime(value: Any) -> str:
    if _is_missing(value):
        return "n/a"
    try:
        return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return str(value)


def _put_text(
    canvas: np.ndarray,
    text: str,
    org: Tuple[int, int],
    scale: float = 0.55,
    color: Tuple[int, int, int] = (230, 230, 230),
    thickness: int = 1,
    max_chars: int = 56,
) -> None:
    text = str(text)
    if len(text) > max_chars:
        text = text[: max_chars - 3] + "..."
    cv2.putText(
        canvas,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _action_color(action: str) -> Tuple[int, int, int]:
    # BGR colors for OpenCV.
    if action == ACTION_PRIORITIZE:
        return (48, 48, 220)
    if action == ACTION_REVIEW:
        return (48, 160, 230)
    if action == ACTION_HOLD:
        return (180, 80, 180)
    return (80, 190, 90)


# -----------------------------------------------------------------------------
# Decision logic
# -----------------------------------------------------------------------------
def decide_action(
    probability: float,
    uncertainty: float,
    ood_score: float,
    threshold: float,
    review_band: float,
    uncertainty_defer: float,
    ood_reject: float,
) -> Tuple[str, str]:
    """
    Deterministic triage rule for the synthetic demo.

    Returns:
        (action, reason)
    """
    if ood_score >= ood_reject:
        return ACTION_HOLD, "Image quality / out-of-distribution reject"

    if probability >= threshold:
        if uncertainty >= uncertainty_defer:
            return ACTION_REVIEW, "High probability but uncertain; human review advised"
        return ACTION_PRIORITIZE, "Above triage threshold"

    if probability >= max(0.0, threshold - review_band):
        return ACTION_REVIEW, "Within review band"

    return ACTION_ROUTINE, "Below review band"


def effective_action(
    row: Any,
    threshold: float,
    review_band: float,
    uncertainty_defer: float,
    ood_reject: float,
) -> Tuple[str, str]:
    override = str(row.get("manual_override", "") or "").strip().upper()
    if override in ALLOWED_ACTIONS:
        return override, "Manual clinician override"

    return decide_action(
        probability=float(row["probability"]),
        uncertainty=float(row["uncertainty"]),
        ood_score=float(row["ood_score"]),
        threshold=float(threshold),
        review_band=float(review_band),
        uncertainty_defer=float(uncertainty_defer),
        ood_reject=float(ood_reject),
    )


def apply_decisions(
    df: pd.DataFrame,
    threshold: float,
    review_band: float,
    uncertainty_defer: float,
    ood_reject: float,
) -> pd.DataFrame:
    out = df.copy()

    if "manual_override" not in out.columns:
        out["manual_override"] = ""

    actions: List[str] = []
    reasons: List[str] = []

    for _, row in out.iterrows():
        action, reason = effective_action(
            row=row,
            threshold=float(threshold),
            review_band=float(review_band),
            uncertainty_defer=float(uncertainty_defer),
            ood_reject=float(ood_reject),
        )
        actions.append(action)
        reasons.append(reason)

    out["action"] = actions
    out["decision_reason"] = reasons
    out["triage_flag"] = out["action"].eq(ACTION_PRIORITIZE)
    out["priority_score"] = (
        out["probability"] * (1.0 - 0.22 * out["uncertainty"])
    ).clip(0.0, 1.0)

    out = out.sort_values(
        ["triage_flag", "priority_score", "wait_minutes"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    out["queue_rank"] = np.arange(1, len(out) + 1, dtype=int)
    out["cohort_size"] = len(out)
    return out


# -----------------------------------------------------------------------------
# Synthetic dataset generation
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def generate_dataset(n_studies: int, seed: int, sla_minutes: int) -> pd.DataFrame:
    """
    Generate a reproducible synthetic chest X-ray triage cohort.

    Important:
    - This is not real patient data.
    - It is intended only to demonstrate workflow, UI, and evaluation logic.
    - It must not be used for clinical decisions or regulatory claims.
    """
    n = max(30, int(n_studies))
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2026-11-01 00:00:00", tz="UTC")

    # Patients may have multiple studies. Split by pseudo-patient cluster to
    # avoid near-duplicate leakage across train/validation/test.
    n_patients = max(12, int(n * 0.82))
    patient_hashes = [
        hashlib.sha256(f"patient-{seed}-{i}".encode("utf-8")).hexdigest()[:16]
        for i in range(n_patients)
    ]
    patient_clusters = np.array(
        [zlib.crc32(ph.encode("utf-8")) % 10 for ph in patient_hashes]
    )
    patient_sites = rng.choice(
        ["Main-Radiology", "Satellite-North", "Mobile-Unit"],
        size=n_patients,
        p=[0.58, 0.27, 0.15],
    )
    patient_idx = rng.integers(0, n_patients, size=n)

    site = patient_sites[patient_idx]

    scanner_by_site = {
        "Main-Radiology": ["DR-VendorA", "DR-VendorB"],
        "Satellite-North": ["DR-VendorB", "DR-VendorC"],
        "Mobile-Unit": ["Portable-C", "Portable-D"],
    }
    scanner = np.array([rng.choice(scanner_by_site[s]) for s in site])

    protocol_list: List[str] = []
    for s, sc in zip(site, scanner):
        if s == "Mobile-Unit":
            protocol_list.append(
                rng.choice(["AP-portable", "AP-semierect"], p=[0.82, 0.18])
            )
        elif str(sc).startswith("Portable"):
            protocol_list.append(
                rng.choice(["AP-portable", "AP-semierect"], p=[0.75, 0.25])
            )
        else:
            protocol_list.append(
                rng.choice(["PA-erect", "AP-erect", "LAT"], p=[0.74, 0.16, 0.10])
            )
    protocol = np.array(protocol_list)

    view = np.where(
        protocol == "LAT",
        "LAT",
        np.where(protocol == "PA-erect", "PA", "AP"),
    )
    laterality = np.where(view == "LAT", rng.choice(["R", "L"], size=n), "N/A")

    age = rng.integers(18, 93, size=n)
    sex = rng.choice(["F", "M", "X"], size=n, p=[0.49, 0.49, 0.02])
    body_habitus = rng.choice(
        ["thin", "average", "obese"], size=n, p=[0.20, 0.62, 0.18]
    )

    software_version = rng.choice(
        ["v2022.1", "v2023.2", "v2024.0"], size=n, p=[0.25, 0.40, 0.35]
    )
    software_version = np.where(
        scanner == "Portable-C", "v2019.4", software_version
    )

    offsets = np.sort(rng.integers(0, 12 * 3600, size=n))
    acquired_at = base + pd.to_timedelta(offsets, unit="s")
    accessed_at = acquired_at + pd.to_timedelta(rng.integers(1, 25, size=n), unit="m")
    sla_due_at = accessed_at + pd.Timedelta(minutes=int(sla_minutes))

    site_effect = (
        pd.Series(site)
        .map(
            {
                "Main-Radiology": 0.00,
                "Satellite-North": 0.02,
                "Mobile-Unit": 0.07,
            }
        )
        .fillna(0.0)
        .to_numpy()
    )
    scanner_effect = (
        pd.Series(scanner)
        .map(
            {
                "DR-VendorA": 0.00,
                "DR-VendorB": 0.01,
                "DR-VendorC": 0.03,
                "Portable-C": -0.02,
                "Portable-D": 0.02,
            }
        )
        .fillna(0.0)
        .to_numpy()
    )
    body_effect = (
        pd.Series(body_habitus)
        .map({"thin": 0.00, "average": 0.00, "obese": 0.04})
        .fillna(0.0)
        .to_numpy()
    )

    severity = rng.beta(2.3, 3.1, size=n) + site_effect + scanner_effect + body_effect
    severity = np.clip(severity, 0.0, 1.0)

    exposure = np.clip(rng.normal(0.84, 0.10, size=n), 0.25, 1.0)
    rotation_deg = np.abs(rng.normal(0.0, 2.2, size=n)) + 1.8 * (
        site == "Mobile-Unit"
    )
    collimation_ok = rng.random(n) > (0.04 + 0.14 * (site == "Mobile-Unit"))
    burned_in_marker_present = rng.random(n) < (
        0.06 + 0.08 * (software_version == "v2019.4")
    )
    annotation_removed = burned_in_marker_present.copy()

    quality_score = (
        0.45 * exposure
        + 0.25 * collimation_ok.astype(float)
        + 0.30 * (rotation_deg < 6.0).astype(float)
    )
    quality_score = np.clip(quality_score, 0.0, 1.0)

    ood_score = (
        0.07
        + 0.24 * (site == "Mobile-Unit")
        + 0.12 * (software_version == "v2019.4")
        + 0.22 * (1.0 - quality_score)
        + rng.normal(0.0, 0.045, size=n)
    )
    ood_score = np.clip(ood_score, 0.0, 1.0)

    noise1 = rng.normal(0.0, 0.10, size=n)
    noise2 = rng.normal(0.0, 0.09, size=n)
    noise3 = rng.normal(0.0, 0.08, size=n)

    p_base = sigmoid(4.7 * (severity - 0.42) + noise1)
    p_dense = sigmoid(
        5.5 * (severity - 0.41)
        + 0.10 * (site == "Satellite-North")
        - 0.05 * (scanner == "Portable-C")
        + noise2
    )
    p_vit = sigmoid(
        5.9 * (severity - 0.40)
        - 0.08 * (body_habitus == "obese")
        + 0.06 * (software_version == "v2024.0")
        + noise3
    )

    raw_ensemble = 0.20 * p_base + 0.35 * p_dense + 0.45 * p_vit
    probability = sigmoid((logit(raw_ensemble) - 0.04) / 1.07)

    model_disagreement = np.std(np.vstack([p_base, p_dense, p_vit]), axis=0)
    borderline = np.exp(-8.0 * np.square(probability - 0.5))

    uncertainty = np.clip(
        0.09
        + 0.42 * model_disagreement
        + 0.34 * ood_score
        + 0.24 * borderline
        + rng.normal(0.0, 0.025, size=n),
        0.0,
        1.0,
    )

    latent_label = 5.1 * (severity - 0.43)
    reader1_label = (
        rng.random(n) < sigmoid(latent_label + rng.normal(0.0, 0.18, size=n))
    ).astype(int)
    reader2_label = (
        rng.random(n) < sigmoid(latent_label + rng.normal(0.0, 0.20, size=n))
    ).astype(int)
    reference_label = (
        rng.random(n) < sigmoid(latent_label + rng.normal(0.0, 0.09, size=n))
    ).astype(int)
    reader_disagreement = (reader1_label != reader2_label).astype(int)

    positive_findings = rng.choice(
        ["consolidation", "multifocal", "pleural_effusion"], size=n
    )
    negative_findings = rng.choice(
        ["normal", "atelectasis", "cardiomegaly", "no_acute_finding"], size=n
    )
    finding_type = np.where(reference_label == 1, positive_findings, negative_findings)

    wait_minutes = np.round(rng.gamma(shape=2.0, scale=7.5, size=n), 1)

    split = np.where(
        patient_clusters[patient_idx] < 6,
        "train",
        np.where(patient_clusters[patient_idx] < 8, "validation", "test"),
    )

    df = pd.DataFrame(
        {
            "study_id": [f"SYN-CXR-{i:05d}" for i in range(1, n + 1)],
            "accession_number": [f"ACC{seed}{i:05d}" for i in range(1, n + 1)],
            "patient_hash": [patient_hashes[i] for i in patient_idx],
            "patient_cluster": patient_clusters[patient_idx],
            "split": split,
            "site": site,
            "scanner": scanner,
            "protocol": protocol,
            "view": view,
            "laterality": laterality,
            "age": age,
            "sex": sex,
            "body_habitus": body_habitus,
            "software_version": software_version,
            "acquired_at": acquired_at,
            "accessed_at": accessed_at,
            "sla_due_at": sla_due_at,
            "severity_latent": severity,
            "exposure": exposure,
            "rotation_deg": rotation_deg,
            "collimation_ok": collimation_ok,
            "burned_in_marker_present": burned_in_marker_present,
            "annotation_removed": annotation_removed,
            "quality_score": quality_score,
            "ood_score": ood_score,
            "prob_baseline": p_base,
            "prob_densenet": p_dense,
            "prob_vit": p_vit,
            "prob_ensemble": raw_ensemble,
            "probability": probability,
            "model_disagreement": model_disagreement,
            "borderline": borderline,
            "uncertainty": uncertainty,
            "reader1_label": reader1_label,
            "reader2_label": reader2_label,
            "reference_label": reference_label,
            "reader_disagreement": reader_disagreement,
            "finding_type": finding_type,
            "wait_minutes": wait_minutes,
            "status": "NEW",
            "clinician_feedback": np.full(n, np.nan, dtype=float),
            "manual_override": [""] * n,
            # Store mutable event timestamps as ISO strings to avoid pandas
            # datetime dtype coercion errors across pandas versions.
            "first_flag_at": [None] * n,
            "report_at": [None] * n,
        }
    )

    # Normalize immutable acquisition timestamps to second resolution.
    for col in ["acquired_at", "accessed_at", "sla_due_at"]:
        df[col] = pd.to_datetime(df[col], utc=True).dt.floor("s")

    return df


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def wilson_interval(successes: int, trials: int, z: float = 1.96) -> Tuple[float, float]:
    if trials <= 0:
        return (float("nan"), float("nan"))

    phat = float(successes) / float(trials)
    denom = 1.0 + (z * z) / float(trials)
    centre = (phat + (z * z) / (2.0 * float(trials))) / denom
    half = (
        z
        / denom
        * math.sqrt(
            (phat * (1.0 - phat)) / float(trials)
            + (z * z) / (4.0 * float(trials) * float(trials))
        )
    )
    low = max(0.0, centre - half)
    high = min(1.0, centre + half)
    return (low, high)


def compute_operating_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {
            "n": 0,
            "positives": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "prioritized": 0,
            "alert_rate": float("nan"),
            "hold_rate": float("nan"),
            "review_rate": float("nan"),
            "sensitivity": float("nan"),
            "specificity": float("nan"),
            "sens_low": float("nan"),
            "sens_high": float("nan"),
            "spec_low": float("nan"),
            "spec_high": float("nan"),
        }

    labels = df["reference_label"].astype(int).to_numpy()
    flags = df["triage_flag"].astype(bool).to_numpy()
    actions = df["action"].astype(str).to_numpy()

    tp = int(((flags) & (labels == 1)).sum())
    fn = int(((~flags) & (labels == 1)).sum())
    fp = int(((flags) & (labels == 0)).sum())
    tn = int(((~flags) & (labels == 0)).sum())

    positives = tp + fn
    negatives = fp + tn
    n = len(df)

    sensitivity = tp / positives if positives > 0 else float("nan")
    specificity = tn / negatives if negatives > 0 else float("nan")
    alert_rate = float(flags.mean()) if n > 0 else float("nan")
    hold_rate = float(np.mean(actions == ACTION_HOLD)) if n > 0 else float("nan")
    review_rate = float(np.mean(actions == ACTION_REVIEW)) if n > 0 else float("nan")

    ci_sens = wilson_interval(tp, tp + fn)
    ci_spec = wilson_interval(tn, tn + fp)

    return {
        "n": n,
        "positives": positives,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "prioritized": int(flags.sum()),
        "alert_rate": alert_rate,
        "hold_rate": hold_rate,
        "review_rate": review_rate,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "sens_low": ci_sens[0],
        "sens_high": ci_sens[1],
        "spec_low": ci_spec[0],
        "spec_high": ci_spec[1],
    }


def subgroup_metrics(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for name, g in df.groupby(group_col, dropna=False):
        m = compute_operating_metrics(g)
        rows.append(
            {
                "slice": str(name),
                "n": m["n"],
                "positives": m["positives"],
                "sensitivity": m["sensitivity"],
                "sens_low": m["sens_low"],
                "sens_high": m["sens_high"],
                "specificity": m["specificity"],
                "spec_low": m["spec_low"],
                "spec_high": m["spec_high"],
                "alert_rate": m["alert_rate"],
                "hold_rate": m["hold_rate"],
                "missed_positives": m["fn"],
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["missed_positives", "n"], ascending=[False, False])
    return out


def split_summary(df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    rows: List[Dict[str, Any]] = []

    for split_name, g in df.groupby("split"):
        rows.append(
            {
                "split": str(split_name),
                "studies": len(g),
                "patients": int(g["patient_hash"].nunique()),
                "positives": int(g["reference_label"].sum()),
                "prevalence": float(g["reference_label"].mean())
                if len(g) > 0
                else float("nan"),
            }
        )

    summary = pd.DataFrame(rows)
    if df.empty:
        overlap_max = float("nan")
    else:
        overlap_max = float(df.groupby("patient_hash")["split"].nunique().max())

    return summary, overlap_max


def roc_auc_and_curve(
    labels: Sequence[int], scores: Sequence[float]
) -> Tuple[float, np.ndarray, np.ndarray]:
    labels_arr = np.asarray(labels, dtype=int)
    scores_arr = np.asarray(scores, dtype=float)

    mask = np.isfinite(scores_arr)
    labels_arr = labels_arr[mask]
    scores_arr = scores_arr[mask]

    pos = labels_arr == 1
    neg = ~pos

    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan"), np.array([0.0, 1.0]), np.array([0.0, 1.0])

    order = np.argsort(-scores_arr)
    y = labels_arr[order]

    tps = np.cumsum(y == 1)
    fps = np.cumsum(y == 0)

    tpr = tps / float(pos.sum())
    fpr = fps / float(neg.sum())

    tpr = np.insert(tpr, 0, 0.0)
    fpr = np.insert(fpr, 0, 0.0)

    try:
        auc = float(np.trapezoid(tpr, fpr))
    except AttributeError:
        auc = float(np.trapz(tpr, fpr))

    return auc, fpr, tpr


def calibration_curve(
    labels: Sequence[int], scores: Sequence[float], bins: int = 10
) -> pd.DataFrame:
    labels_arr = np.asarray(labels, dtype=int)
    scores_arr = np.asarray(scores, dtype=float)

    mask = np.isfinite(scores_arr)
    labels_arr = labels_arr[mask]
    scores_arr = scores_arr[mask]

    if len(labels_arr) == 0:
        return pd.DataFrame(columns=["bin", "n", "predicted", "observed"])

    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_index = np.clip(
        np.digitize(scores_arr, edges[1:-1], right=False), 0, bins - 1
    )

    tmp = pd.DataFrame(
        {"bin_index": bin_index, "label": labels_arr, "score": scores_arr}
    )

    cal = (
        tmp.groupby("bin_index")
        .agg(
            n=("label", "size"),
            predicted=("score", "mean"),
            observed=("label", "mean"),
        )
        .reset_index()
    )

    cal["bin"] = cal["bin_index"].map(
        lambda i: f"{edges[i]:.1f}-{edges[i + 1]:.1f}"
    )
    return cal[["bin", "n", "predicted", "observed"]]


def expected_calibration_error(
    labels: Sequence[int], scores: Sequence[float], bins: int = 10
) -> float:
    cal = calibration_curve(labels, scores, bins=bins)
    if cal.empty:
        return float("nan")
    total = float(cal["n"].sum())
    if total <= 0:
        return float("nan")
    return float(
        ((cal["n"] / total) * (cal["observed"] - cal["predicted"]).abs()).sum()
    )


def optimize_threshold_for_sensitivity(
    df: pd.DataFrame, target_sensitivity: float
) -> Tuple[float, float, float]:
    val = df[df["split"] == "validation"]
    if val.empty:
        val = df

    if val.empty:
        return float("nan"), float("nan"), float("nan")

    labels = val["reference_label"].astype(int).to_numpy()
    positives = int(labels.sum())
    scores = val["probability"].astype(float).to_numpy()

    candidates = np.unique(np.concatenate([scores, np.array([0.0, 1.0])]))

    for thr in sorted(candidates, reverse=True):
        flags = scores >= float(thr)
        tp = int(((flags) & (labels == 1)).sum())
        sens = tp / positives if positives > 0 else float("nan")
        alert = float(flags.mean())

        if positives == 0:
            return float(thr), float("nan"), alert

        if sens >= target_sensitivity:
            return float(thr), float(sens), alert

    flags = scores >= 0.0
    tp = int(((flags) & (labels == 1)).sum())
    sens = tp / positives if positives > 0 else float("nan")
    alert = float(flags.mean())
    return 0.0, float(sens), float(alert)


def stressed_auc(
    df: pd.DataFrame, score_col: str, kind: str, seed: int = SEED
) -> float:
    if df.empty:
        return float("nan")

    labels = df["reference_label"].astype(int).to_numpy()
    scores = df[score_col].astype(float).to_numpy().copy()

    if kind == "acquisition_noise":
        rng = np.random.default_rng(stable_seed(kind + str(seed)))
        scores = scores + rng.normal(0.0, 0.08, size=len(scores))
    elif kind == "mobile_shift":
        scores = np.where(df["site"].to_numpy() == "Mobile-Unit", scores - 0.08, scores)
    elif kind == "low_quality_degradation":
        scores = np.where(
            df["quality_score"].to_numpy() < 0.65, scores * 0.90, scores
        )
    elif kind == "old_software_shift":
        scores = np.where(
            df["software_version"].to_numpy() == "v2019.4", scores - 0.06, scores
        )
    elif kind == "obese_habitus_shift":
        scores = np.where(
            df["body_habitus"].to_numpy() == "obese", scores - 0.05, scores
        )

    scores = np.clip(scores, 0.0, 1.0)
    auc, _, _ = roc_auc_and_curve(labels, scores)
    return auc


def continuous_psi(expected: Sequence[float], actual: Sequence[float], bins: int = 10) -> float:
    exp_arr = np.asarray(expected, dtype=float)
    act_arr = np.asarray(actual, dtype=float)

    exp_arr = exp_arr[np.isfinite(exp_arr)]
    act_arr = act_arr[np.isfinite(act_arr)]

    if len(exp_arr) < 2 or len(act_arr) < 2:
        return float("nan")

    quantiles = np.quantile(exp_arr, np.linspace(0.0, 1.0, bins + 1))
    edges = np.unique(quantiles)

    if len(edges) < 3:
        return 0.0

    edges = edges.copy()
    edges[0] = -np.inf
    edges[-1] = np.inf

    exp_counts = np.histogram(exp_arr, bins=edges)[0].astype(float)
    act_counts = np.histogram(act_arr, bins=edges)[0].astype(float)

    exp_pct = np.clip(exp_counts / float(len(exp_arr)), 1e-6, None)
    act_pct = np.clip(act_counts / float(len(act_arr)), 1e-6, None)

    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def categorical_psi(expected: Sequence[Any], actual: Sequence[Any]) -> float:
    exp_series = pd.Series(expected).astype(str)
    act_series = pd.Series(actual).astype(str)

    if exp_series.empty or act_series.empty:
        return float("nan")

    exp_counts = exp_series.value_counts(normalize=True)
    act_counts = act_series.value_counts(normalize=True)

    categories = sorted(set(exp_counts.index.to_list()) | set(act_counts.index.to_list()))
    e = np.array([float(exp_counts.get(c, 0.0)) for c in categories], dtype=float)
    a = np.array([float(act_counts.get(c, 0.0)) for c in categories], dtype=float)

    e = np.clip(e, 1e-6, None)
    a = np.clip(a, 1e-6, None)

    return float(np.sum((a - e) * np.log(a / e)))


def add_derived_eval_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["age_band"] = (
        pd.cut(
            out["age"],
            bins=[0, 18, 45, 65, 130],
            labels=["<18", "18-44", "45-64", "65+"],
            right=False,
        )
        .astype(str)
    )
    out["agreement_group"] = np.where(
        out["reader_disagreement"].astype(int) == 1, "disputed", "agreed"
    )
    return out


def clinical_utility_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "group",
                "n",
                "baseline_mean_wait_min",
                "triaged_mean_wait_min",
                "mean_reduction_pct",
                "baseline_median_wait_min",
                "triaged_median_wait_min",
            ]
        )

    base = df["wait_minutes"].astype(float).to_numpy()
    flags = df["triage_flag"].astype(bool).to_numpy()
    triaged = np.where(flags, base * WAIT_REDUCTION_FACTOR, base)

    groups = [
        ("All studies", np.ones(len(df), dtype=bool)),
        ("Synthetic positives", df["reference_label"].astype(int).to_numpy() == 1),
        ("Prioritized studies", flags),
    ]

    rows: List[Dict[str, Any]] = []
    for name, mask in groups:
        b = base[mask]
        t = triaged[mask]
        if len(b) == 0:
            continue
        baseline_mean = float(np.mean(b))
        triaged_mean = float(np.mean(t))
        reduction = (
            (baseline_mean - triaged_mean) / baseline_mean
            if baseline_mean > 0
            else float("nan")
        )
        rows.append(
            {
                "group": name,
                "n": int(mask.sum()),
                "baseline_mean_wait_min": baseline_mean,
                "triaged_mean_wait_min": triaged_mean,
                "mean_reduction_pct": reduction,
                "baseline_median_wait_min": float(np.median(b)),
                "triaged_median_wait_min": float(np.median(t)),
            }
        )

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Synthetic image and video rendering
# -----------------------------------------------------------------------------
def synthesize_xray(
    case: pd.Series,
    size: int,
    frame_index: int = 0,
    frames_per_case: int = 1,
    show_heatmap: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[Tuple[int, int, int, int]]]:
    """
    Create a synthetic chest X-ray-like image and optional attention overlay.

    This is illustrative only. It is not a validated image simulator and not a
    real Grad-CAM explanation.
    """
    rng = np.random.default_rng(stable_seed(str(case["study_id"])))

    yy, xx = np.mgrid[0:size, 0:size]
    xn = xx / float(max(1, size - 1))
    yn = yy / float(max(1, size - 1))

    img = np.full((size, size), 0.07, dtype=np.float32)
    img += rng.normal(0.0, 0.012, size=(size, size)).astype(np.float32)

    lung_params = [
        (0.31, 0.47, 0.17, 0.29),
        (0.69, 0.47, 0.17, 0.29),
    ]

    lung_mask = np.zeros((size, size), dtype=bool)
    for cx, cy, rx, ry in lung_params:
        m = (((xn - cx) / rx) ** 2 + ((yn - cy) / ry) ** 2) <= 1.0
        lung_mask |= m

    vascular = (
        0.28
        + 0.035 * np.sin(yn * 55.0 + xn * 8.0)
        + 0.025 * np.cos(xn * 42.0 - yn * 5.0)
    )
    img[lung_mask] = np.maximum(img[lung_mask], vascular[lung_mask])

    ribs = 0.055 * np.sin(yn * 70.0 + 0.35 * np.sin(xn * 9.0))
    img = np.clip(img + ribs * lung_mask, 0.0, 1.0)

    heart_mask = (
        ((xn - 0.50) / 0.15) ** 2 + ((yn - 0.59) / 0.21) ** 2
    ) <= 1.0
    heart_val = 0.50 + 0.04 * np.sin(yn * 20.0)
    img[heart_mask] = np.maximum(img[heart_mask], heart_val[heart_mask])

    diaphragm_curve = 0.38 + 0.08 * np.sin((xn - 0.5) * np.pi * 1.2)
    below_diaphragm = yn > diaphragm_curve
    soft_tissue = np.zeros_like(img, dtype=np.float32)
    soft_tissue[below_diaphragm & ~lung_mask] = 0.22
    img = np.maximum(img, soft_tissue)

    heat = np.zeros((size, size), dtype=np.float32)
    boxes: List[Tuple[int, int, int, int]] = []

    probability = float(case["probability"])
    label = int(case["reference_label"])
    severity = max(probability, 0.58 if label == 1 else 0.0)

    if severity > 0.34:
        n_lesions = 1 if severity < 0.76 else 2
        progress = float(frame_index) / float(max(1, frames_per_case - 1))

        for _ in range(n_lesions):
            side = 0 if rng.random() < 0.55 else 1
            cx, cy, rx, ry = lung_params[side]

            lx = float(
                np.clip(cx + rng.normal(0.0, rx * 0.35), cx - rx * 0.75, cx + rx * 0.75)
            )
            ly = float(
                np.clip(cy + rng.normal(0.0, ry * 0.35), cy - ry * 0.75, cy + ry * 0.75)
            )

            sigma = 0.035 + 0.045 * severity * (0.55 + 0.45 * progress)
            opacity = 0.18 + 0.42 * severity * (0.45 + 0.55 * progress)

            d2 = (xn - lx) ** 2 + (yn - ly) ** 2
            blob = opacity * np.exp(-d2 / (2.0 * sigma * sigma))

            lesion_zone = (
                ((xn - cx) / rx) ** 2 + ((yn - cy) / ry) ** 2
            ) <= 1.18

            img = np.clip(img + blob * lesion_zone, 0.0, 1.0)
            heat = np.clip(heat + blob * lesion_zone, 0.0, 1.0)

            r_px = int(max(10, 3.0 * sigma * size))
            x0 = int(np.clip(lx * size - r_px, 0, size - 1))
            x1 = int(np.clip(lx * size + r_px, 0, size - 1))
            y0 = int(np.clip(ly * size - r_px, 0, size - 1))
            y1 = int(np.clip(ly * size + r_px, 0, size - 1))

            if x1 > x0 and y1 > y0:
                boxes.append((x0, y0, x1, y1))

    img = np.clip(img, 0.0, 1.0)
    img = np.power(img, 0.92)
    gray = (img * 255.0).astype(np.uint8)

    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    overlay = bgr.copy()

    if show_heatmap and heat.max() > 0:
        heat_norm = (heat / max(float(heat.max()), 1e-6) * 255.0).astype(np.uint8)
        heat_color = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(bgr, 0.62, heat_color, 0.38, 0.0)

    best_box: Optional[Tuple[int, int, int, int]] = None
    if boxes:
        best_box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))

    return gray, overlay, best_box


def _draw_video_frame(
    case: pd.Series,
    frame_index: int,
    frames_per_case: int,
    threshold: float,
    review_band: float,
    uncertainty_defer: float,
    ood_reject: float,
    queue_rank: int,
    cohort_size: int,
) -> np.ndarray:
    size = min(XRAY_SIZE, VIDEO_HEIGHT - 110, VIDEO_WIDTH // 2 - 80)
    size = max(220, size)

    gray, overlay, box = synthesize_xray(
        case=case,
        size=size,
        frame_index=frame_index,
        frames_per_case=frames_per_case,
        show_heatmap=True,
    )

    canvas = np.full((VIDEO_HEIGHT, VIDEO_WIDTH, 3), (18, 20, 24), dtype=np.uint8)

    _put_text(
        canvas,
        "CHEST X-RAY TRIAGE - ADVANCED SYNTHETIC LIVE DEMO",
        (24, 30),
        scale=0.68,
        color=(245, 245, 245),
        thickness=2,
        max_chars=64,
    )
    _put_text(
        canvas,
        "Research demonstration only | No real patient data | Clinician remains responsible",
        (24, 50),
        scale=0.45,
        color=(150, 170, 190),
        thickness=1,
        max_chars=88,
    )

    x0 = 26
    y0 = 68
    canvas[y0 : y0 + size, x0 : x0 + size] = overlay
    cv2.rectangle(
        canvas,
        (x0 - 2, y0 - 2),
        (x0 + size + 1, y0 + size + 1),
        (85, 90, 100),
        2,
    )

    probability = float(case["probability"])
    uncertainty = float(case["uncertainty"])
    ood = float(case["ood_score"])

    action, reason = effective_action(
        row=case,
        threshold=threshold,
        review_band=review_band,
        uncertainty_defer=uncertainty_defer,
        ood_reject=ood_reject,
    )
    color = _action_color(action)

    if box is not None and action in {ACTION_PRIORITIZE, ACTION_REVIEW}:
        bx0, by0, bx1, by1 = box
        cv2.rectangle(canvas, (x0 + bx0, y0 + by0), (x0 + bx1, y0 + by1), color, 2)
        _put_text(
            canvas,
            "synthetic attention region",
            (x0 + bx0, max(y0 + by0 - 7, y0 + 12)),
            scale=0.40,
            color=color,
            thickness=1,
            max_chars=34,
        )

    px = x0 + size + 34
    py = y0 + 4

    def line(
        text: str,
        dy: int = 28,
        scale: float = 0.54,
        col: Tuple[int, int, int] = (230, 230, 230),
        thick: int = 1,
        chars: int = 46,
    ) -> None:
        nonlocal py
        _put_text(canvas, text, (px, py), scale, col, thick, chars)
        py += dy

    line(f"Study: {case['study_id']}", 31, 0.62, (255, 255, 255), 2, 40)
    line(
        f"Site: {case['site']} | Scanner: {case['scanner']}",
        26,
        0.50,
        (210, 210, 215),
        1,
        52,
    )
    line(
        f"Age/sex: {case['age']} / {case['sex']} | {case['protocol']}",
        26,
        0.50,
        (210, 210, 215),
        1,
        52,
    )

    acquired = _format_datetime(case["acquired_at"])
    line(f"Acquired: {acquired}", 26, 0.50, (210, 210, 215), 1, 52)

    py += 8
    line(
        f"Pneumonia probability: {probability:.1%}",
        34,
        0.66,
        color,
        2,
        44,
    )
    line(
        f"Uncertainty: {uncertainty:.2f} | OOD: {ood:.2f}",
        26,
        0.50,
        (210, 210, 215),
        1,
        52,
    )
    line(
        f"Threshold: {threshold:.2f} | Review band: {review_band:.2f}",
        26,
        0.50,
        (200, 200, 210),
        1,
        52,
    )
    line(f"Triage decision: {action}", 34, 0.70, color, 2, 40)
    line(f"Reason: {reason}", 26, 0.48, (200, 200, 210), 1, 58)

    ref = "positive" if int(case["reference_label"]) == 1 else "negative"
    line(f"Synthetic reference label: {ref}", 26, 0.50, (200, 200, 210), 1, 52)
    line(
        f"Queue rank: {int(queue_rank)} / {int(cohort_size)}",
        26,
        0.50,
        (200, 200, 210),
        1,
        52,
    )
    line(
        f"Current wait: {float(case['wait_minutes']):.1f} min",
        26,
        0.50,
        (200, 200, 210),
        1,
        52,
    )

    bar_x = px
    bar_y = py + 12
    bar_w = min(410, VIDEO_WIDTH - px - 28)
    bar_h = 14

    cv2.rectangle(
        canvas,
        (bar_x, bar_y),
        (bar_x + bar_w, bar_y + bar_h),
        (95, 100, 110),
        1,
    )
    frac = float(frame_index + 1) / float(max(1, frames_per_case))
    cv2.rectangle(
        canvas,
        (bar_x + 1, bar_y + 1),
        (bar_x + 1 + int((bar_w - 2) * frac), bar_y + bar_h - 1),
        color,
        -1,
    )
    _put_text(
        canvas,
        "case playback progress",
        (bar_x, bar_y + 32),
        scale=0.42,
        color=(150, 155, 165),
        thickness=1,
        max_chars=40,
    )

    _put_text(
        canvas,
        "SYNTHETIC DEMO - NOT FOR CLINICAL USE - NO DICOM/PACS CONNECTION",
        (24, VIDEO_HEIGHT - 16),
        scale=0.44,
        color=(120, 180, 220),
        thickness=1,
        max_chars=82,
    )

    return canvas


@st.cache_data(show_spinner=False)
def build_demo_video(
    threshold: float,
    review_band: float,
    uncertainty_defer: float,
    ood_reject: float,
    fps: int,
    seconds: float,
    n_cases: int,
    seed: int,
    n_studies: int,
    sla_minutes: int,
) -> Tuple[bytes, str]:
    """
    Render a short synthetic triage video.

    Primary output: browser-friendly H.264 MP4 via imageio/ffmpeg if available.
    Fallback output: animated GIF via Pillow.
    """
    df = generate_dataset(n_studies, seed, sla_minutes)
    q = apply_decisions(df, threshold, review_band, uncertainty_defer, ood_reject)

    cases = q.head(max(1, min(int(n_cases), len(q)))).copy()
    if cases.empty:
        raise ValueError("No cases available for video rendering.")

    total_frames = max(len(cases) * 3, int(round(float(fps) * float(seconds))))
    frames_per_case = max(3, total_frames // max(1, len(cases)))

    frames: List[np.ndarray] = []
    for rank, (_, case) in enumerate(cases.iterrows(), start=1):
        for f in range(frames_per_case):
            frames.append(
                _draw_video_frame(
                    case=case,
                    frame_index=f,
                    frames_per_case=frames_per_case,
                    threshold=threshold,
                    review_band=review_band,
                    uncertainty_defer=uncertainty_defer,
                    ood_reject=ood_reject,
                    queue_rank=rank,
                    cohort_size=len(q),
                )
            )

    rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()

    try:
        try:
            import imageio.v2 as imageio

            writer = imageio.get_writer(
                tmp.name,
                fps=float(fps),
                codec="libx264",
                pixelformat="yuv420p",
                quality=8,
            )
            for frame in rgb_frames:
                writer.append_data(frame)
            writer.close()

            if os.path.exists(tmp.name) and os.path.getsize(tmp.name) > 0:
                with open(tmp.name, "rb") as fh:
                    data = fh.read()
                return data, "video/mp4"

            raise RuntimeError("MP4 renderer produced an empty file.")

        except Exception:
            from PIL import Image

            # Smaller/faster GIF fallback.
            gif_frames = []
            for frame in rgb_frames[::2]:
                resized = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
                gif_frames.append(Image.fromarray(resized))

            if not gif_frames:
                gif_frames = [Image.fromarray(cv2.resize(rgb_frames[0], (640, 360)))]

            buf = io.BytesIO()
            duration = int(max(60, 2 * 1000.0 / max(1.0, float(fps))))
            gif_frames[0].save(
                buf,
                format="GIF",
                save_all=True,
                append_images=gif_frames[1:],
                duration=duration,
                loop=0,
            )
            return buf.getvalue(), "image/gif"

    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


# -----------------------------------------------------------------------------
# Worklist simulation
# -----------------------------------------------------------------------------
PRIORITY_MAP = {
    ACTION_PRIORITIZE: "URGENT",
    ACTION_REVIEW: "SOON",
    ACTION_ROUTINE: "ROUTINE",
    ACTION_HOLD: "HOLD",
}

REQUIRED_MESSAGE_FIELDS: Dict[str, Any] = {
    "schema_version": str,
    "message_id": str,
    "idempotency_key": str,
    "study_id": str,
    "accession_number": str,
    "patient_ref": str,
    "priority": str,
    "triage_action": str,
    "reason": str,
    "probability": float,
    "uncertainty": float,
    "ood_score": float,
    "threshold": float,
    "sla_due_at": str,
    "generated_at": str,
    "model_version": str,
    "site": str,
    "scanner": str,
}


def REQUIRED_MESSAGE_FIELDS_KEYS_ONLY() -> Dict[str, str]:
    return {k: type(v).__name__ for k, v in REQUIRED_MESSAGE_FIELDS.items()}


def _is_type(value: Any, typ: Any) -> bool:
    if typ is float:
        return isinstance(value, (int, float, np.floating)) and not isinstance(value, bool)
    if typ is int:
        return isinstance(value, (int, np.integer)) and not isinstance(value, bool)
    if typ is str:
        return isinstance(value, str)
    return isinstance(value, typ)


def validate_message(msg: Dict[str, Any]) -> List[str]:
    errors: List[str] = []

    for field, typ in REQUIRED_MESSAGE_FIELDS.items():
        if field not in msg:
            errors.append(f"missing field: {field}")
        elif not _is_type(msg[field], typ):
            errors.append(f"wrong type for {field}: expected {typ.__name__}")

    for numeric_field in ["probability", "uncertainty", "ood_score", "threshold"]:
        if numeric_field in msg and _is_type(msg[numeric_field], float):
            v = float(msg[numeric_field])
            if not (0.0 <= v <= 1.0):
                errors.append(f"{numeric_field} out of range: {v}")

    if msg.get("triage_action") not in ALLOWED_ACTIONS:
        errors.append(f"invalid triage_action: {msg.get('triage_action')}")

    if msg.get("priority") not in set(PRIORITY_MAP.values()):
        errors.append(f"invalid priority: {msg.get('priority')}")

    action = msg.get("triage_action")
    priority = msg.get("priority")
    if action in PRIORITY_MAP and priority != PRIORITY_MAP[action]:
        errors.append(
            f"priority/action mismatch: {priority} != {PRIORITY_MAP.get(action)}"
        )

    return errors


def build_worklist_message(
    row: pd.Series,
    threshold: float,
    review_band: float,
    uncertainty_defer: float,
    ood_reject: float,
) -> Dict[str, Any]:
    action, reason = effective_action(
        row=row,
        threshold=float(threshold),
        review_band=float(review_band),
        uncertainty_defer=float(uncertainty_defer),
        ood_reject=float(ood_reject),
    )

    idempotency_key = hashlib.sha256(
        f"{row['study_id']}|{threshold}|{MODEL_VERSION}|{SCHEMA_VERSION}".encode(
            "utf-8"
        )
    ).hexdigest()[:24]

    msg: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "message_id": str(uuid.uuid4()),
        "idempotency_key": idempotency_key,
        "study_id": str(row["study_id"]),
        "accession_number": str(row["accession_number"]),
        "patient_ref": str(row["patient_hash"]),
        "priority": PRIORITY_MAP[action],
        "triage_action": action,
        "reason": reason,
        "probability": float(row["probability"]),
        "uncertainty": float(row["uncertainty"]),
        "ood_score": float(row["ood_score"]),
        "threshold": float(threshold),
        "sla_due_at": pd.Timestamp(row["sla_due_at"]).isoformat(),
        "generated_at": utcnow_iso(),
        "model_version": MODEL_VERSION,
        "site": str(row["site"]),
        "scanner": str(row["scanner"]),
        "evidence": {
            "type": "synthetic_thumbnail",
            "uri": f"demo://{row['study_id']}/thumbnail",
        },
        "links": {
            "case_detail": f"/queue?study={row['study_id']}",
            "audit_trail": f"/audit?study={row['study_id']}",
        },
    }

    return msg


def simulate_delivery(
    msg: Dict[str, Any], max_attempts: int = 3, attempt_offset: int = 0
) -> Dict[str, Any]:
    attempts: List[Dict[str, Any]] = []
    success = False
    key = msg["idempotency_key"]

    for attempt in range(1, max_attempts + 1):
        global_attempt = attempt + attempt_offset
        delay_s = float((2 ** (attempt - 1)) * 0.1)

        # Deterministic simulated transient failure.
        fail = stable_seed(f"{key}|{global_attempt}") % 9 == 0

        attempts.append(
            {
                "attempt": attempt,
                "global_attempt": global_attempt,
                "delay_s": delay_s,
                "ok": (not fail),
                "timestamp": utcnow_iso(),
            }
        )

        if not fail:
            success = True
            break

    status = "DELIVERED" if success else "DEAD_LETTER"
    delivered_at = utcnow_iso() if success else None

    return {
        "message": msg,
        "status": status,
        "attempts": attempts,
        "delivered_at": delivered_at,
    }


# -----------------------------------------------------------------------------
# Session state, audit, events, flash messages
# -----------------------------------------------------------------------------
def append_audit(
    event_type: str,
    study_id: Optional[str] = None,
    actor: str = "system",
    detail: str = "",
) -> None:
    entries = st.session_state.setdefault("audit", [])
    prev_hash = entries[-1]["entry_hash"] if entries else "GENESIS"

    entry: Dict[str, Any] = {
        "timestamp": utcnow_iso(),
        "event_type": event_type,
        "study_id": study_id,
        "actor": actor,
        "detail": str(detail),
        "prev_hash": prev_hash,
    }
    entry["entry_hash"] = hash_obj(entry)
    entries.append(entry)

    if len(entries) > AUDIT_LOG_MAX:
        st.session_state.audit = entries[-AUDIT_LOG_MAX:]


def add_event(text: str, level: str = "info") -> None:
    events = st.session_state.setdefault("events", [])
    events.append(
        {
            "time": utcnow().strftime("%H:%M:%S"),
            "level": level,
            "event": str(text),
        }
    )
    if len(events) > 300:
        st.session_state.events = events[-300:]


def set_flash(message: str, kind: str = "success") -> None:
    st.session_state.flash = {"message": str(message), "kind": kind}


def render_flash() -> None:
    flash = st.session_state.pop("flash", None)
    if not flash:
        return
    msg = flash.get("message", "")
    kind = flash.get("kind", "success")
    if kind == "error":
        st.error(msg)
    elif kind == "warning":
        st.warning(msg)
    else:
        st.success(msg)


def events_dataframe() -> pd.DataFrame:
    events = st.session_state.get("events", [])
    if not events:
        return pd.DataFrame(columns=["time", "level", "event"])
    return pd.DataFrame(events)


def update_study(
    study_id: str,
    status: Optional[str] = None,
    feedback: Optional[int] = None,
    override: Optional[str] = None,
    event_detail: str = "",
) -> bool:
    df = st.session_state.df_raw

    if "first_flag_at" not in df.columns:
        df["first_flag_at"] = None
    if "report_at" not in df.columns:
        df["report_at"] = None
    if "manual_override" not in df.columns:
        df["manual_override"] = ""

    idx = df.index[df["study_id"] == study_id]
    if len(idx) == 0:
        return False

    i = idx[0]
    now_iso = utcnow_iso()

    if status:
        df.at[i, "status"] = str(status)
    if feedback is not None:
        df.at[i, "clinician_feedback"] = int(feedback)
    if override:
        df.at[i, "manual_override"] = str(override)

    if status in {"FLAGGED", "WORKLIST_SENT"} and _is_missing(df.at[i, "first_flag_at"]):
        df.at[i, "first_flag_at"] = now_iso
    if status == "REPORTED" and _is_missing(df.at[i, "report_at"]):
        df.at[i, "report_at"] = now_iso

    append_audit(
        "QUEUE_UPDATE",
        study_id=study_id,
        actor="user",
        detail=event_detail or status or override or "",
    )
    add_event(f"Updated {study_id}: {event_detail or status or override}")

    st.session_state.df_raw = df
    return True


def push_to_worklist(study_id: str) -> Tuple[bool, str]:
    now = utcnow()

    recent: List[datetime] = []
    for t in st.session_state.get("delivery_history", []):
        try:
            ts = pd.Timestamp(t)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            dt = ts.to_pydatetime()
            if (now - dt).total_seconds() < 60:
                recent.append(dt)
        except Exception:
            continue

    st.session_state.delivery_history = recent

    if len(recent) >= MAX_PUSHES_PER_MINUTE:
        append_audit("RATE_LIMIT", study_id=study_id, detail="push suppressed")
        return False, "Rate limit reached; push suppressed."

    df = st.session_state.df_raw
    idx = df.index[df["study_id"] == study_id]
    if len(idx) == 0:
        return False, "Study not found."

    row = df.loc[idx[0]]
    msg = build_worklist_message(
        row=row,
        threshold=float(st.session_state.threshold),
        review_band=float(st.session_state.review_band),
        uncertainty_defer=float(st.session_state.uncertainty_defer),
        ood_reject=float(st.session_state.ood_reject),
    )

    errors = validate_message(msg)
    if errors:
        append_audit("SCHEMA_VALIDATION_FAILED", study_id=study_id, detail="; ".join(errors))
        return False, "Schema validation failed: " + "; ".join(errors)

    # Duplicate suppression for already-delivered idempotency keys.
    duplicate = next(
        (
            d
            for d in st.session_state.messages
            if d["message"]["idempotency_key"] == msg["idempotency_key"]
            and d["status"] == "DELIVERED"
        ),
        None,
    )
    if duplicate is not None:
        append_audit("DUPLICATE_SUPPRESSED", study_id=study_id, detail=msg["idempotency_key"])
        return False, "Duplicate suppressed; message already delivered."

    # If a prior dead-letter exists for the same key, remove it before retry.
    st.session_state.dlq = [
        d
        for d in st.session_state.dlq
        if d["message"]["idempotency_key"] != msg["idempotency_key"]
    ]

    delivery = simulate_delivery(msg, max_attempts=3, attempt_offset=0)
    st.session_state.delivery_history.append(now)

    if delivery["status"] == "DELIVERED":
        st.session_state.messages.append(delivery)
        update_study(
            study_id=study_id,
            status="WORKLIST_SENT",
            event_detail="priority flag pushed to simulated worklist",
        )
        append_audit("WORKLIST_DELIVERED", study_id=study_id, detail=msg["idempotency_key"])
        return True, "Delivered to simulated worklist."

    st.session_state.dlq.append(delivery)
    append_audit("WORKLIST_DEAD_LETTER", study_id=study_id, detail=msg["idempotency_key"])
    return False, "Delivery failed; routed to dead-letter queue."


def replay_dead_letters() -> int:
    remaining: List[Dict[str, Any]] = []
    recovered = 0

    for d in st.session_state.dlq:
        prior_attempts = len(d.get("attempts", []))
        new_delivery = simulate_delivery(
            d["message"], max_attempts=3, attempt_offset=prior_attempts
        )
        if new_delivery["status"] == "DELIVERED":
            st.session_state.messages.append(new_delivery)
            append_audit(
                "WORKLIST_REPLAY_DELIVERED",
                study_id=d["message"]["study_id"],
                detail=d["message"]["idempotency_key"],
            )
            recovered += 1
        else:
            remaining.append(new_delivery)

    st.session_state.dlq = remaining
    add_event(f"Dead-letter replay complete: {recovered} recovered.")
    return recovered


def run_initial_simulation() -> None:
    df = st.session_state.df_raw.copy()

    if "first_flag_at" not in df.columns:
        df["first_flag_at"] = None
    if "report_at" not in df.columns:
        df["report_at"] = None

    q = apply_decisions(
        df,
        threshold=float(st.session_state.threshold),
        review_band=float(st.session_state.review_band),
        uncertainty_defer=float(st.session_state.uncertainty_defer),
        ood_reject=float(st.session_state.ood_reject),
    )

    top = q.head(5)
    now = utcnow()
    now_iso = utcnow_iso()

    for _, row in top.iterrows():
        idx = df.index[df["study_id"] == row["study_id"]]
        if len(idx) == 0:
            continue
        i = idx[0]

        df.at[i, "status"] = "FLAGGED"
        df.at[i, "first_flag_at"] = now_iso

        append_audit(
            "TRIAGE_FLAG",
            study_id=str(row["study_id"]),
            detail=str(row["action"]),
        )

        msg = build_worklist_message(
            row=df.loc[i],
            threshold=float(st.session_state.threshold),
            review_band=float(st.session_state.review_band),
            uncertainty_defer=float(st.session_state.uncertainty_defer),
            ood_reject=float(st.session_state.ood_reject),
        )
        delivery = simulate_delivery(msg, max_attempts=3, attempt_offset=0)

        if delivery["status"] == "DELIVERED":
            st.session_state.messages.append(delivery)
            append_audit(
                "WORKLIST_DELIVERED",
                study_id=str(row["study_id"]),
                detail=msg["idempotency_key"],
            )
        else:
            st.session_state.dlq.append(delivery)
            append_audit(
                "WORKLIST_DEAD_LETTER",
                study_id=str(row["study_id"]),
                detail=msg["idempotency_key"],
            )

    for j, (_, row) in enumerate(top.head(2).iterrows()):
        idx = df.index[df["study_id"] == row["study_id"]]
        if len(idx) == 0:
            continue
        i = idx[0]

        feedback = 1 if j == 0 else 0
        df.at[i, "status"] = "REPORTED"
        df.at[i, "report_at"] = (
            now + timedelta(minutes=2 + j)
        ).replace(microsecond=0).isoformat()
        df.at[i, "clinician_feedback"] = feedback

        append_audit(
            "CLINICIAN_REPORT",
            study_id=str(row["study_id"]),
            actor="radiologist-sim",
            detail=f"feedback={feedback}",
        )

    st.session_state.df_raw = df
    add_event(
        "Initial synthetic workload complete: 5 flagged, 2 reported, "
        "worklist messages simulated."
    )


def simulate_live_events(steps: int = 8) -> None:
    q = apply_decisions(
        st.session_state.df_raw,
        threshold=float(st.session_state.threshold),
        review_band=float(st.session_state.review_band),
        uncertainty_defer=float(st.session_state.uncertainty_defer),
        ood_reject=float(st.session_state.ood_reject),
    )

    candidates = q[q["status"].isin(["NEW", "FLAGGED"])].head(max(1, int(steps)))

    for _, row in candidates.iterrows():
        study_id = str(row["study_id"])

        if row["status"] == "NEW":
            update_study(
                study_id=study_id,
                status="FLAGGED",
                event_detail="synthetic acquisition flagged",
            )
            ok, msg = push_to_worklist(study_id)
            add_event(f"{study_id}: worklist push -> {msg}", level="info" if ok else "warning")
        else:
            # Simulated radiologist feedback, occasionally disagreeing with
            # synthetic reference label to mimic noisy ground truth.
            sid_seed = stable_seed(study_id)
            feedback = int(row["reference_label"])
            if sid_seed % 7 == 0:
                feedback = 1 - feedback

            update_study(
                study_id=study_id,
                status="REPORTED",
                feedback=feedback,
                event_detail="simulated radiologist report",
            )

    add_event(f"Live simulation advanced by {len(candidates)} events.")


def init_session_state() -> None:
    required_cols = {
        "study_id",
        "accession_number",
        "patient_hash",
        "patient_cluster",
        "split",
        "site",
        "scanner",
        "protocol",
        "view",
        "laterality",
        "age",
        "sex",
        "body_habitus",
        "software_version",
        "acquired_at",
        "accessed_at",
        "sla_due_at",
        "severity_latent",
        "exposure",
        "rotation_deg",
        "collimation_ok",
        "burned_in_marker_present",
        "annotation_removed",
        "quality_score",
        "ood_score",
        "prob_baseline",
        "prob_densenet",
        "prob_vit",
        "prob_ensemble",
        "probability",
        "model_disagreement",
        "borderline",
        "uncertainty",
        "reader1_label",
        "reader2_label",
        "reference_label",
        "reader_disagreement",
        "finding_type",
        "wait_minutes",
        "status",
        "clinician_feedback",
        "manual_override",
        "first_flag_at",
        "report_at",
    }

    if (
        "df_raw" not in st.session_state
        or not all(c in st.session_state.df_raw.columns for c in required_cols)
    ):
        st.session_state.df_raw = generate_dataset(N_STUDIES, SEED, SLA_MINUTES).copy()
        for key in ["audit", "messages", "dlq", "delivery_history", "events"]:
            st.session_state[key] = []
        st.session_state["initial_sim_done"] = False
    else:
        df = st.session_state.df_raw.copy()

        # Migrate old datetime columns to ISO strings if present.
        for col in ["first_flag_at", "report_at"]:
            if col not in df.columns:
                df[col] = None
            else:
                df[col] = df[col].map(_timestamp_to_iso_or_none)

        if "manual_override" not in df.columns:
            df["manual_override"] = ""
        else:
            df["manual_override"] = df["manual_override"].fillna("").astype(str)

        st.session_state.df_raw = df

        # Migrate delivery history to datetime objects.
        hist: List[datetime] = []
        for t in st.session_state.get("delivery_history", []):
            try:
                ts = pd.Timestamp(t)
                if ts.tzinfo is None:
                    ts = ts.tz_localize("UTC")
                else:
                    ts = ts.tz_convert("UTC")
                hist.append(ts.to_pydatetime())
            except Exception:
                continue
        st.session_state.delivery_history = hist

    defaults = {
        "threshold": TRIAGE_THRESHOLD,
        "review_band": REVIEW_BAND,
        "uncertainty_defer": UNCERTAINTY_DEFER_THRESHOLD,
        "ood_reject": OOD_REJECT_THRESHOLD,
        "target_sensitivity": TARGET_SENSITIVITY,
        "video_seconds": VIDEO_SECONDS,
        "fps": VIDEO_FPS,
        "video_cases": VIDEO_CASES,
        "applied_threshold": TRIAGE_THRESHOLD,
        "applied_review_band": REVIEW_BAND,
        "applied_uncertainty_defer": UNCERTAINTY_DEFER_THRESHOLD,
        "applied_ood_reject": OOD_REJECT_THRESHOLD,
        "applied_video_seconds": VIDEO_SECONDS,
        "applied_fps": VIDEO_FPS,
        "applied_video_cases": VIDEO_CASES,
        "show_heatmap": True,
    }

    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

    for key in ["audit", "messages", "dlq", "delivery_history", "events"]:
        st.session_state.setdefault(key, [])

    st.session_state.threshold = clamp_float(
        st.session_state.threshold, 0.50, 0.95, TRIAGE_THRESHOLD
    )
    st.session_state.review_band = clamp_float(
        st.session_state.review_band, 0.00, 0.30, REVIEW_BAND
    )
    st.session_state.uncertainty_defer = clamp_float(
        st.session_state.uncertainty_defer, 0.00, 1.00, UNCERTAINTY_DEFER_THRESHOLD
    )
    st.session_state.ood_reject = clamp_float(
        st.session_state.ood_reject, 0.00, 1.00, OOD_REJECT_THRESHOLD
    )
    st.session_state.target_sensitivity = clamp_float(
        st.session_state.target_sensitivity, 0.80, 0.999, TARGET_SENSITIVITY
    )
    st.session_state.video_seconds = clamp_float(
        st.session_state.video_seconds, 3.0, 10.0, VIDEO_SECONDS
    )

    if st.session_state.fps not in (8, 10, 12, 15):
        st.session_state.fps = VIDEO_FPS
    if st.session_state.video_cases not in (3, 4, 6, 8):
        st.session_state.video_cases = VIDEO_CASES

    if not st.session_state.get("initial_sim_done", False):
        run_initial_simulation()
        st.session_state.initial_sim_done = True


def reset_session_state() -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]


# -----------------------------------------------------------------------------
# Plotly helpers
# -----------------------------------------------------------------------------
def make_roc_figure(fpr: np.ndarray, tpr: np.ndarray, auc: float, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"AUC={_format_float(auc, 3)}",
            line=dict(width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0.0, 1.0],
            y=[0.0, 1.0],
            mode="lines",
            name="Chance",
            line=dict(dash="dot", color="gray"),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="False positive rate",
        yaxis_title="True positive rate",
        template="plotly_white",
        height=380,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def make_calibration_figure(cal: pd.DataFrame, title: str) -> Optional[go.Figure]:
    if cal.empty:
        return None

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=cal["predicted"],
            y=cal["observed"],
            mode="lines+markers",
            name="Model",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0.0, 1.0],
            y=[0.0, 1.0],
            mode="lines",
            name="Perfect calibration",
            line=dict(dash="dot", color="gray"),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Mean predicted probability",
        yaxis_title="Observed positive rate",
        template="plotly_white",
        height=380,
        margin=dict(l=40, r=20, t=50, b=40),
        xaxis=dict(range=[0.0, 1.0]),
        yaxis=dict(range=[0.0, 1.0]),
    )
    return fig


# -----------------------------------------------------------------------------
# Display formatting
# -----------------------------------------------------------------------------
def format_queue_for_display(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["probability"] = out["probability"].map(lambda x: f"{x:.1%}")
    out["uncertainty"] = out["uncertainty"].map(lambda x: f"{x:.2f}")
    out["ood_score"] = out["ood_score"].map(lambda x: f"{x:.2f}")
    out["priority_score"] = out["priority_score"].map(lambda x: f"{x:.3f}")
    out["acquired_at"] = out["acquired_at"].map(_format_datetime)

    cols = [
        "queue_rank",
        "study_id",
        "site",
        "scanner",
        "protocol",
        "age",
        "sex",
        "acquired_at",
        "probability",
        "uncertainty",
        "ood_score",
        "priority_score",
        "action",
        "decision_reason",
        "status",
        "wait_minutes",
    ]
    return out[[c for c in cols if c in out.columns]]


def format_subgroup_table(tbl: pd.DataFrame) -> pd.DataFrame:
    out = tbl.copy()
    for col in [
        "sensitivity",
        "specificity",
        "alert_rate",
        "hold_rate",
        "sens_low",
        "sens_high",
        "spec_low",
        "spec_high",
    ]:
        if col in out.columns:
            out[col] = out[col].map(_format_percent)
    return out


# -----------------------------------------------------------------------------
# Self-checks and integration tests
# -----------------------------------------------------------------------------
def run_self_checks() -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": str(detail)})

    add(
        "Prioritize above threshold",
        decide_action(0.90, 0.10, 0.10, 0.75, 0.15, 0.65, 0.70)[0] == ACTION_PRIORITIZE,
    )
    add(
        "Hold on OOD",
        decide_action(0.90, 0.10, 0.85, 0.75, 0.15, 0.65, 0.70)[0] == ACTION_HOLD,
    )
    add(
        "Review borderline",
        decide_action(0.70, 0.10, 0.10, 0.75, 0.15, 0.65, 0.70)[0] == ACTION_REVIEW,
    )

    auc, _, _ = roc_auc_and_curve([1, 0, 1, 0], [0.9, 0.2, 0.8, 0.1])
    add("Perfect AUC near 1.0", (not math.isnan(auc)) and auc > 0.99, f"auc={auc:.3f}")

    low, high = wilson_interval(9, 10)
    add("Wilson CI ordered", 0.0 <= low < high <= 1.0, f"[{low:.3f}, {high:.3f}]")

    sample_df = st.session_state.get("df_raw", None)
    if sample_df is None or sample_df.empty:
        sample_df = generate_dataset(30, SEED, SLA_MINUTES)
    sample = sample_df.iloc[0]

    msg = build_worklist_message(sample, 0.75, 0.15, 0.65, 0.70)
    errs = validate_message(msg)
    add("Worklist schema valid", len(errs) == 0, "; ".join(errs))

    bad = dict(msg)
    bad.pop("study_id", None)
    add("Schema rejects missing field", len(validate_message(bad)) > 0)

    small = generate_dataset(40, SEED + 1, SLA_MINUTES)
    overlap = small.groupby("patient_hash")["split"].nunique().max()
    add("No patient split overlap", overlap == 1, f"max_splits_per_patient={overlap}")

    frame = _draw_video_frame(sample, 0, 5, 0.75, 0.15, 0.65, 0.70, 1, 10)
    add(
        "Video frame dimensions",
        frame.shape == (VIDEO_HEIGHT, VIDEO_WIDTH, 3) and frame.dtype == np.uint8,
        str(frame.shape),
    )

    audit = st.session_state.get("audit", [])
    chain_ok = True
    prev = "GENESIS"
    for entry in audit:
        if entry.get("prev_hash") != prev:
            chain_ok = False
            break
        payload = {k: v for k, v in entry.items() if k != "entry_hash"}
        if hash_obj(payload) != entry.get("entry_hash"):
            chain_ok = False
            break
        prev = entry.get("entry_hash")

    add("Audit hash chain", chain_ok, f"entries={len(audit)}")

    return checks


def run_worklist_e2e_test() -> List[Tuple[str, bool, str]]:
    results: List[Tuple[str, bool, str]] = []

    df = st.session_state.get("df_raw", None)
    if df is None or df.empty:
        df = generate_dataset(30, SEED, SLA_MINUTES)

    row = df.iloc[0]

    msg = build_worklist_message(
        row=row,
        threshold=float(st.session_state.get("threshold", TRIAGE_THRESHOLD)),
        review_band=float(st.session_state.get("review_band", REVIEW_BAND)),
        uncertainty_defer=float(
            st.session_state.get("uncertainty_defer", UNCERTAINTY_DEFER_THRESHOLD)
        ),
        ood_reject=float(st.session_state.get("ood_reject", OOD_REJECT_THRESHOLD)),
    )

    errs = validate_message(msg)
    results.append(("Schema validation", len(errs) == 0, str(errs)))

    msg2 = build_worklist_message(
        row=row,
        threshold=float(st.session_state.get("threshold", TRIAGE_THRESHOLD)),
        review_band=float(st.session_state.get("review_band", REVIEW_BAND)),
        uncertainty_defer=float(
            st.session_state.get("uncertainty_defer", UNCERTAINTY_DEFER_THRESHOLD)
        ),
        ood_reject=float(st.session_state.get("ood_reject", OOD_REJECT_THRESHOLD)),
    )
    results.append(
        (
            "Idempotency stable",
            msg["idempotency_key"] == msg2["idempotency_key"],
            msg["idempotency_key"],
        )
    )

    delivery = simulate_delivery(msg, max_attempts=3, attempt_offset=0)
    results.append(
        (
            "Terminal delivery state",
            delivery["status"] in {"DELIVERED", "DEAD_LETTER"},
            delivery["status"],
        )
    )

    fake_delivered = [{"message": msg, "status": "DELIVERED"}]
    dup = next(
        (
            d
            for d in fake_delivered
            if d["message"]["idempotency_key"] == msg2["idempotency_key"]
            and d["status"] == "DELIVERED"
        ),
        None,
    )
    results.append(("Duplicate detectable", dup is not None, ""))

    return results


# -----------------------------------------------------------------------------
# UI sections
# -----------------------------------------------------------------------------
def render_sidebar() -> None:
    with st.sidebar:
        st.header("Operating point")

        st.slider(
            "Triage threshold",
            min_value=0.50,
            max_value=0.95,
            step=0.01,
            key="threshold",
            format="%.2f",
            help="Updates live queue, metrics, and tables.",
        )
        st.slider(
            "Review band",
            min_value=0.00,
            max_value=0.30,
            step=0.01,
            key="review_band",
            format="%.2f",
        )
        st.slider(
            "Uncertainty defer threshold",
            min_value=0.00,
            max_value=1.00,
            step=0.01,
            key="uncertainty_defer",
            format="%.2f",
        )
        st.slider(
            "OOD hold threshold",
            min_value=0.00,
            max_value=1.00,
            step=0.01,
            key="ood_reject",
            format="%.2f",
        )
        st.slider(
            "Target sensitivity",
            min_value=0.80,
            max_value=0.999,
            step=0.005,
            key="target_sensitivity",
            format="%.3f",
        )

        st.divider()
        st.header("Video renderer")

        st.slider(
            "Video length (seconds)",
            min_value=3.0,
            max_value=10.0,
            step=0.5,
            key="video_seconds",
            format="%.1f",
        )
        st.selectbox("Frame rate (fps)", options=[8, 10, 12, 15], key="fps")
        st.selectbox("Cases in video", options=[3, 4, 6, 8], key="video_cases")

        if st.button("Apply video settings and render", type="primary", key="apply_video_settings"):
            st.session_state.applied_threshold = float(st.session_state.threshold)
            st.session_state.applied_review_band = float(st.session_state.review_band)
            st.session_state.applied_uncertainty_defer = float(
                st.session_state.uncertainty_defer
            )
            st.session_state.applied_ood_reject = float(st.session_state.ood_reject)
            st.session_state.applied_video_seconds = float(st.session_state.video_seconds)
            st.session_state.applied_fps = int(st.session_state.fps)
            st.session_state.applied_video_cases = int(st.session_state.video_cases)
            st.rerun()

        st.caption(
            f"Video currently uses threshold {_format_percent(st.session_state.applied_threshold)} "
            f"and review band {_format_percent(st.session_state.applied_review_band)}. "
            "Queue metrics use the live slider."
        )

        st.divider()
        st.header("Live simulation")

        c1, c2 = st.columns(2)
        if c1.button("Run 8 events", key="run_8_events"):
            simulate_live_events(8)
            st.rerun()
        if c2.button("Reset session", key="reset_session"):
            reset_session_state()
            st.rerun()

        st.divider()
        st.markdown(
            "**Safety note**\n\n"
            "- Synthetic data only\n"
            "- No DICOM/PACS connection\n"
            "- No clinical use\n"
            "- Clinician remains responsible"
        )


def render_live_tab(q: pd.DataFrame) -> None:
    st.subheader("Automatic synthetic triage simulation")
    st.write(
        "The demo starts automatically by rendering a moving triage sequence and "
        "loading a synthetic workload. This is not a static mockup."
    )

    try:
        with st.spinner("Rendering synthetic triage video..."):
            video_bytes, video_mime = build_demo_video(
                threshold=float(st.session_state.applied_threshold),
                review_band=float(st.session_state.applied_review_band),
                uncertainty_defer=float(st.session_state.applied_uncertainty_defer),
                ood_reject=float(st.session_state.applied_ood_reject),
                fps=int(st.session_state.applied_fps),
                seconds=float(st.session_state.applied_video_seconds),
                n_cases=int(st.session_state.applied_video_cases),
                seed=SEED,
                n_studies=N_STUDIES,
                sla_minutes=SLA_MINUTES,
            )

        if video_mime == "video/mp4":
            video_b64 = base64.b64encode(video_bytes).decode("ascii")
            html = (
                '<div style="text-align:center;">'
                f'<video width="100%" style="max-width:{VIDEO_WIDTH}px;" '
                'controls autoplay muted loop playsinline>'
                f'<source src="data:video/mp4;base64,{video_b64}" type="video/mp4">'
                "Your browser does not support embedded video."
                "</video>"
                "</div>"
            )
            components.html(html, height=VIDEO_HEIGHT + 40)

            st.download_button(
                "Download MP4 demo",
                data=video_bytes,
                file_name="chest_xray_triage_advanced_demo.mp4",
                mime="video/mp4",
                key="download_mp4_demo",
            )
        else:
            st.image(video_bytes, caption="Animated synthetic demo (GIF fallback)")
            st.download_button(
                "Download GIF demo",
                data=video_bytes,
                file_name="chest_xray_triage_advanced_demo.gif",
                mime="image/gif",
                key="download_gif_demo",
            )

    except Exception as exc:
        st.error(f"Video rendering failed: {exc}")

    st.info(
        "This video is generated from synthetic data. It is a UI and workflow "
        "demonstration, not evidence of clinical performance."
    )

    test_q = q[q["split"] == "test"]
    if test_q.empty:
        test_q = q

    m = compute_operating_metrics(test_q)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Test studies", m["n"])
    c2.metric("Prioritized", m["prioritized"], _format_percent(m["alert_rate"]))
    c3.metric(
        "Synthetic sensitivity",
        _format_percent(m["sensitivity"]),
        f"95% CI {_format_percent(m['sens_low'])} - {_format_percent(m['sens_high'])}",
    )
    c4.metric("Missed synthetic positives", m["fn"])
    c5.metric("Hold rate", _format_percent(m["hold_rate"]))

    st.divider()
    st.subheader("Recent simulation events")
    st.dataframe(events_dataframe().tail(30), hide_index=True)

    st.divider()
    st.subheader("Top of queue")
    st.dataframe(format_queue_for_display(q.head(12)), hide_index=True)


def render_case_inspector(row: pd.Series, show_heatmap: bool) -> None:
    left, right = st.columns([1, 1])

    with left:
        gray, overlay, _ = synthesize_xray(
            case=row,
            size=360,
            frame_index=5,
            frames_per_case=10,
            show_heatmap=show_heatmap,
        )

        if show_heatmap:
            display_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        else:
            display_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        st.image(
            display_rgb,
            caption="Synthetic preview with optional attention overlay",
        )

    with right:
        st.metric("Study", row["study_id"])
        st.metric("Probability", f"{float(row['probability']):.1%}")
        st.metric("Triage action", row["action"])
        st.metric("Queue rank", f"{int(row['queue_rank'])} / {int(row['cohort_size'])}")

        st.markdown(
            f"""
            - **Site:** {row['site']}
            - **Scanner:** {row['scanner']}
            - **Protocol:** {row['protocol']}
            - **Age / sex:** {row['age']} / {row['sex']}
            - **Acquired:** {_format_datetime(row['acquired_at'])}
            - **SLA due:** {_format_datetime(row['sla_due_at'])}
            - **Status:** {row['status']}
            - **Uncertainty:** {float(row['uncertainty']):.2f}
            - **OOD score:** {float(row['ood_score']):.2f}
            - **Decision reason:** {row['decision_reason']}
            """
        )

        model_df = pd.DataFrame(
            {
                "model": ["Baseline", "DenseNet-sim", "ViT-sim", "Ensemble-cal"],
                "probability": [
                    float(row["prob_baseline"]),
                    float(row["prob_densenet"]),
                    float(row["prob_vit"]),
                    float(row["probability"]),
                ],
            }
        )
        st.bar_chart(model_df.set_index("model"))

        st.caption(
            "The overlay is a synthetic illustration, not a validated segmentation "
            "or real Grad-CAM explanation."
        )


def render_case_actions(row: pd.Series) -> None:
    st.divider()
    st.subheader("Clinician actions")

    study_id = str(row["study_id"])

    c1, c2, c3, c4, c5, c6 = st.columns(6)

    if c1.button("Confirm pneumonia", key=f"confirm_{study_id}"):
        update_study(study_id, status="REPORTED", feedback=1, event_detail="confirmed pneumonia")
        set_flash(f"{study_id}: confirmed pneumonia.")
        st.rerun()

    if c2.button("Rule out", key=f"ruleout_{study_id}"):
        update_study(study_id, status="REPORTED", feedback=0, event_detail="ruled out pneumonia")
        set_flash(f"{study_id}: ruled out pneumonia.")
        st.rerun()

    if c3.button("Escalate", key=f"escalate_{study_id}"):
        update_study(study_id, status="FLAGGED", override=ACTION_PRIORITIZE, event_detail="manual escalation")
        set_flash(f"{study_id}: manually escalated.")
        st.rerun()

    if c4.button("Defer to human", key=f"defer_{study_id}"):
        update_study(study_id, status="IN_REVIEW", override=ACTION_REVIEW, event_detail="manual deferral")
        set_flash(f"{study_id}: deferred to human review.")
        st.rerun()

    if c5.button("Hold for repeat", key=f"hold_{study_id}"):
        update_study(study_id, status="HOLD", override=ACTION_HOLD, event_detail="manual hold")
        set_flash(f"{study_id}: held for repeat / quality review.")
        st.rerun()

    if c6.button("Push worklist", key=f"push_{study_id}"):
        ok, msg = push_to_worklist(study_id)
        set_flash(f"{study_id}: {msg}", kind="success" if ok else "warning")
        st.rerun()


def render_queue_tab(q: pd.DataFrame) -> None:
    st.subheader("Interactive reading queue")

    split_options = sorted(q["split"].unique().tolist())
    action_options = sorted(q["action"].unique().tolist())
    site_options = sorted(q["site"].unique().tolist())

    f1, f2, f3, f4 = st.columns([1.2, 1.2, 1.2, 1.4])

    with f1:
        split_filter = st.multiselect(
            "Split",
            options=split_options,
            default=split_options,
            key="queue_split_filter",
        )
    with f2:
        action_filter = st.multiselect(
            "Action",
            options=action_options,
            default=action_options,
            key="queue_action_filter",
        )
    with f3:
        site_filter = st.multiselect(
            "Site",
            options=site_options,
            default=site_options,
            key="queue_site_filter",
        )
    with f4:
        search = st.text_input("Search study / scanner / protocol", "", key="queue_search")

    filtered = q[
        q["split"].isin(split_filter)
        & q["action"].isin(action_filter)
        & q["site"].isin(site_filter)
    ].copy()

    if search.strip():
        s = search.strip().lower()
        filtered = filtered[
            filtered["study_id"].str.lower().str.contains(s, na=False)
            | filtered["scanner"].str.lower().str.contains(s, na=False)
            | filtered["protocol"].str.lower().str.contains(s, na=False)
        ]

    m = compute_operating_metrics(filtered)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Filtered studies", m["n"])
    c2.metric("Prioritized", m["prioritized"], _format_percent(m["alert_rate"]))
    c3.metric("Sensitivity", _format_percent(m["sensitivity"]))
    c4.metric("Specificity", _format_percent(m["specificity"]))
    c5.metric("Missed positives", m["fn"])

    st.dataframe(format_queue_for_display(filtered), hide_index=True)

    if filtered.empty:
        st.warning("No studies match the current filters.")
        return

    selected_id = st.selectbox("Inspect a study", filtered["study_id"].tolist(), key="queue_selected_study")
    row = filtered[filtered["study_id"] == selected_id].iloc[0]

    show_heatmap = st.checkbox(
        "Show synthetic attention overlay",
        value=bool(st.session_state.get("show_heatmap", True)),
        key="queue_show_heatmap",
    )
    st.session_state.show_heatmap = show_heatmap

    render_case_inspector(row, show_heatmap=show_heatmap)
    render_case_actions(row)

    st.divider()
    st.subheader("Case audit trail")
    case_audit = [
        e
        for e in st.session_state.get("audit", [])
        if e.get("study_id") == selected_id
    ]
    if case_audit:
        st.dataframe(pd.DataFrame(case_audit[-30:]), hide_index=True)
    else:
        st.info("No audit entries for this case yet.")


def render_model_lab_tab(q: pd.DataFrame) -> None:
    st.subheader("Model comparison (synthetic)")

    split_choice = st.radio(
        "Evaluation split",
        options=["validation", "test", "all"],
        horizontal=True,
        key="model_lab_split",
    )

    if split_choice == "all":
        sub = q
    else:
        sub = q[q["split"] == split_choice]

    if len(sub) < 4:
        st.warning("Not enough studies in this split for model comparison.")
        return

    model_options = {
        "Baseline logistic": "prob_baseline",
        "DenseNet-sim": "prob_densenet",
        "ViT-sim": "prob_vit",
        "Ensemble-calibrated": "probability",
    }

    selected_model_name = st.selectbox(
        "Selected model",
        options=list(model_options.keys()),
        index=3,
        key="model_lab_selected_model",
    )
    selected_model_col = model_options[selected_model_name]

    rows: List[Dict[str, Any]] = []
    for name, col in model_options.items():
        auc, _, _ = roc_auc_and_curve(sub["reference_label"], sub[col])
        ece = expected_calibration_error(sub["reference_label"], sub[col])
        rows.append(
            {
                "model": name,
                "AUROC": auc,
                "ECE": ece,
                "n": len(sub),
                "positives": int(sub["reference_label"].sum()),
            }
        )

    model_tbl = pd.DataFrame(rows)
    model_tbl["AUROC"] = model_tbl["AUROC"].map(lambda x: _format_float(x, 3))
    model_tbl["ECE"] = model_tbl["ECE"].map(lambda x: _format_float(x, 3))
    st.dataframe(model_tbl, hide_index=True)

    auc, fpr, tpr = roc_auc_and_curve(sub["reference_label"], sub[selected_model_col])
    st.plotly_chart(
        make_roc_figure(fpr, tpr, auc, f"ROC — {selected_model_name} ({split_choice})"),
        use_container_width=True,
    )

    cal = calibration_curve(sub["reference_label"], sub[selected_model_col])
    fig_cal = make_calibration_figure(
        cal, f"Calibration — {selected_model_name} ({split_choice})"
    )
    if fig_cal is not None:
        st.plotly_chart(fig_cal, use_container_width=True)
    else:
        st.info("Calibration data unavailable.")

    st.divider()
    st.subheader("Threshold optimizer")

    target = float(st.session_state.target_sensitivity)
    rec_thr, rec_val_sens, rec_val_alert = optimize_threshold_for_sensitivity(
        q, target_sensitivity=target
    )

    if _is_missing(rec_thr):
        st.warning("Could not compute a recommended threshold.")
    else:
        st.metric(
            "Recommended threshold",
            f"{rec_thr:.2f}",
            f"validation sensitivity {_format_percent(rec_val_sens)}, alert rate {_format_percent(rec_val_alert)}",
        )

        if st.button("Apply recommended threshold to live queue", key="apply_recommended_threshold") and not _is_missing(rec_thr):
            st.session_state.threshold = clamp_float(rec_thr, 0.50, 0.95, TRIAGE_THRESHOLD)
            set_flash(f"Applied recommended threshold {rec_thr:.2f}.")
            st.rerun()

        q_rec = apply_decisions(
            st.session_state.df_raw,
            threshold=float(rec_thr),
            review_band=float(st.session_state.review_band),
            uncertainty_defer=float(st.session_state.uncertainty_defer),
            ood_reject=float(st.session_state.ood_reject),
        )
        test_rec = q_rec[q_rec["split"] == "test"]
        if test_rec.empty:
            test_rec = q_rec

        m_rec = compute_operating_metrics(test_rec)
        st.markdown("**Projected test metrics at recommended threshold**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sensitivity", _format_percent(m_rec["sensitivity"]))
        c2.metric("Specificity", _format_percent(m_rec["specificity"]))
        c3.metric("Alert rate", _format_percent(m_rec["alert_rate"]))
        c4.metric("Missed positives", m_rec["fn"])

    st.divider()
    st.subheader("Synthetic stress tests")
    st.caption(
        "These are simulated degradations applied to synthetic scores to demonstrate "
        "sensitivity analysis. They are not real model retraining results."
    )

    stress_kinds = [
        "baseline",
        "acquisition_noise",
        "mobile_shift",
        "low_quality_degradation",
        "old_software_shift",
        "obese_habitus_shift",
    ]

    stress_rows: List[Dict[str, Any]] = []
    baseline_auc = stressed_auc(sub, selected_model_col, "baseline")
    for kind in stress_kinds:
        a = stressed_auc(sub, selected_model_col, kind)
        delta = a - baseline_auc if (not math.isnan(a) and not math.isnan(baseline_auc)) else float("nan")
        stress_rows.append(
            {
                "stress": kind,
                "AUROC": _format_float(a, 3),
                "delta_vs_baseline": _format_float(delta, 3),
            }
        )

    st.dataframe(pd.DataFrame(stress_rows), hide_index=True)


def render_explain_tab(q: pd.DataFrame) -> None:
    st.subheader("Explainability, uncertainty, and quality gating")

    if q.empty:
        st.warning("No studies available.")
        return

    selected_id = st.selectbox(
        "Select study",
        q["study_id"].tolist(),
        index=0,
        key="explain_selected_study",
    )
    row = q[q["study_id"] == selected_id].iloc[0]

    show_heatmap = st.checkbox(
        "Show synthetic attention overlay",
        value=True,
        key="explain_show_heatmap",
    )

    render_case_inspector(row, show_heatmap=show_heatmap)

    st.divider()
    st.subheader("Uncertainty decomposition")

    dis_norm = min(1.0, float(row["model_disagreement"]) / 0.25)
    epistemic = clamp_float(0.55 * dis_norm + 0.45 * float(row["ood_score"]), 0.0, 1.0, 0.5)
    aleatoric = clamp_float(
        0.50 * float(row["borderline"]) + 0.50 * (1.0 - float(row["quality_score"])),
        0.0,
        1.0,
        0.5,
    )

    unc_df = pd.DataFrame(
        {
            "component": ["Model disagreement", "OOD proxy", "Epistemic estimate", "Aleatoric estimate"],
            "score": [
                float(row["model_disagreement"]),
                float(row["ood_score"]),
                epistemic,
                aleatoric,
            ],
        }
    )
    st.bar_chart(unc_df.set_index("component"))

    if float(row["ood_score"]) >= float(st.session_state.ood_reject):
        recommendation = "HOLD: repeat or acquire better image before AI triage."
    elif float(row["uncertainty"]) >= float(st.session_state.uncertainty_defer):
        recommendation = "DEFER: route to human review with evidence overlay."
    elif float(row["probability"]) >= float(st.session_state.threshold):
        recommendation = "PRIORITIZE: above threshold and acceptable uncertainty."
    else:
        recommendation = "ROUTINE or REVIEW according to threshold band."

    st.info(recommendation)

    st.divider()
    st.subheader("Acquisition quality checks")

    checks = [
        ("Exposure adequate", float(row["exposure"]) >= 0.55, f"{float(row['exposure']):.2f}"),
        ("Rotation acceptable", float(row["rotation_deg"]) <= 6.0, f"{float(row['rotation_deg']):.1f} deg"),
        ("Collimation acceptable", bool(row["collimation_ok"]), ""),
        (
            "Burned-in marker removed",
            (not bool(row["burned_in_marker_present"])) or bool(row["annotation_removed"]),
            "",
        ),
        (
            "OOD below hold threshold",
            float(row["ood_score"]) < float(st.session_state.ood_reject),
            f"{float(row['ood_score']):.2f} < {float(st.session_state.ood_reject):.2f}",
        ),
    ]

    check_df = pd.DataFrame(
        [
            {
                "check": name,
                "status": "PASS" if passed else "FAIL",
                "detail": detail,
            }
            for name, passed, detail in checks
        ]
    )
    st.dataframe(check_df, hide_index=True)

    st.caption(
        "These checks are synthetic demonstrations of the kind of gating required "
        "before trusting a model output in a real imaging pipeline."
    )


def render_worklist_tab(q: pd.DataFrame) -> None:
    st.subheader("Simulated PACS/RIS worklist delivery")
    st.caption(
        "This tab simulates a reliable output contract: schema validation, idempotency, "
        "retries, dead-letter queue, rate limiting, and duplicate suppression. "
        "No real network or PACS system is touched."
    )

    if q.empty:
        st.warning("No studies available.")
        return

    selected_id = st.selectbox(
        "Select study for worklist push",
        q["study_id"].tolist(),
        index=0,
        key="worklist_selected_study",
    )
    row = q[q["study_id"] == selected_id].iloc[0]

    msg_preview = build_worklist_message(
        row=row,
        threshold=float(st.session_state.threshold),
        review_band=float(st.session_state.review_band),
        uncertainty_defer=float(st.session_state.uncertainty_defer),
        ood_reject=float(st.session_state.ood_reject),
    )

    c1, c2 = st.columns([2, 1])
    with c1:
        st.json(msg_preview)
    with c2:
        if st.button("Validate and push selected case", type="primary", key="push_selected_worklist"):
            ok, msg = push_to_worklist(selected_id)
            set_flash(f"{selected_id}: {msg}", kind="success" if ok else "warning")
            st.rerun()

        st.markdown("**Controls**")
        if st.button("Replay dead-letter queue", key="replay_dlq"):
            recovered = replay_dead_letters()
            set_flash(f"Dead-letter replay complete: {recovered} recovered.")
            st.rerun()

        if st.button("Run end-to-end delivery test", key="run_e2e"):
            results = run_worklist_e2e_test()
            for name, passed, detail in results:
                icon = "PASS" if passed else "FAIL"
                st.markdown(f"**{icon}** — {name}: {detail}")

    st.divider()
    st.subheader("Delivered messages")

    if st.session_state.messages:
        delivered_df = pd.DataFrame(
            [
                {
                    "study_id": d["message"]["study_id"],
                    "priority": d["message"]["priority"],
                    "action": d["message"]["triage_action"],
                    "idempotency_key": d["message"]["idempotency_key"],
                    "generated_at": d["message"]["generated_at"],
                    "attempts": len(d["attempts"]),
                    "status": d["status"],
                }
                for d in st.session_state.messages
            ]
        )
        st.dataframe(delivered_df, hide_index=True)
    else:
        st.info("No delivered messages yet.")

    st.divider()
    st.subheader("Dead-letter queue")

    if st.session_state.dlq:
        dlq_df = pd.DataFrame(
            [
                {
                    "study_id": d["message"]["study_id"],
                    "priority": d["message"]["priority"],
                    "action": d["message"]["triage_action"],
                    "idempotency_key": d["message"]["idempotency_key"],
                    "generated_at": d["message"]["generated_at"],
                    "attempts": len(d["attempts"]),
                    "status": d["status"],
                }
                for d in st.session_state.dlq
            ]
        )
        st.dataframe(dlq_df, hide_index=True)
    else:
        st.success("Dead-letter queue is empty.")

    st.divider()
    st.subheader("Output schema")
    st.code(json.dumps(REQUIRED_MESSAGE_FIELDS_KEYS_ONLY(), indent=2), language="json")


def render_evaluation_tab(q: pd.DataFrame) -> None:
    st.subheader("Evaluation, slicing, drift, and safety review")

    if q.empty:
        st.warning("No studies available.")
        return

    st.markdown("**Split integrity**")
    summary, overlap_max = split_summary(q)
    st.dataframe(summary, hide_index=True)

    if _is_missing(overlap_max):
        st.info("Could not compute split overlap.")
    elif overlap_max <= 1.0:
        st.success("No patient appears in more than one split.")
    else:
        st.error(f"Split leakage detected: max splits per patient = {overlap_max}")

    st.divider()
    st.markdown("**Operating point on test split**")

    test_q = q[q["split"] == "test"]
    if test_q.empty:
        test_q = q

    m = compute_operating_metrics(test_q)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Test studies", m["n"])
    c2.metric("Prioritized", m["prioritized"], _format_percent(m["alert_rate"]))
    c3.metric(
        "Sensitivity",
        _format_percent(m["sensitivity"]),
        f"95% CI {_format_percent(m['sens_low'])} - {_format_percent(m['sens_high'])}",
    )
    c4.metric(
        "Specificity",
        _format_percent(m["specificity"]),
        f"95% CI {_format_percent(m['spec_low'])} - {_format_percent(m['spec_high'])}",
    )
    c5.metric("Missed positives", m["fn"])

    st.divider()
    st.subheader("Subgroup performance")

    qq = add_derived_eval_columns(q)
    dimension = st.selectbox(
        "Slice dimension",
        options=[
            "site",
            "scanner",
            "sex",
            "age_band",
            "protocol",
            "body_habitus",
            "software_version",
            "agreement_group",
        ],
        index=0,
        key="eval_slice_dimension",
    )

    tbl = subgroup_metrics(qq, dimension)
    st.dataframe(format_subgroup_table(tbl), hide_index=True)

    st.divider()
    st.subheader("Safety review")

    fn_cases = q[(q["reference_label"] == 1) & (~q["triage_flag"])].copy()
    fp_cases = q[(q["reference_label"] == 0) & (q["triage_flag"])].copy()
    hold_cases = q[q["action"] == ACTION_HOLD].copy()

    s1, s2, s3 = st.columns(3)
    s1.metric("Missed synthetic positives", len(fn_cases))
    s2.metric("False alarm prioritizations", len(fp_cases))
    s3.metric("Held for quality/OOD", len(hold_cases))

    st.markdown("**Missed synthetic positives**")
    if fn_cases.empty:
        st.success("No missed synthetic positives at current operating point.")
    else:
        st.dataframe(format_queue_for_display(fn_cases.head(30)), hide_index=True)

    st.markdown("**False alarm prioritizations**")
    if fp_cases.empty:
        st.success("No false alarm prioritizations at current operating point.")
    else:
        st.dataframe(format_queue_for_display(fp_cases.head(30)), hide_index=True)

    st.divider()
    st.subheader("Drift monitor")

    validation = q[q["split"] == "validation"]
    monitoring = q.sort_values("acquired_at").tail(max(10, int(0.25 * len(q))))

    if len(validation) >= 5 and len(monitoring) >= 5:
        drift_rows = [
            {
                "feature": "probability",
                "psi": continuous_psi(validation["probability"], monitoring["probability"]),
            },
            {
                "feature": "uncertainty",
                "psi": continuous_psi(validation["uncertainty"], monitoring["uncertainty"]),
            },
            {
                "feature": "site",
                "psi": categorical_psi(validation["site"], monitoring["site"]),
            },
            {
                "feature": "scanner",
                "psi": categorical_psi(validation["scanner"], monitoring["scanner"]),
            },
        ]

        drift_df = pd.DataFrame(drift_rows)

        def psi_status(value: float) -> str:
            if pd.isna(value):
                return "n/a"
            if value >= PSI_ALERT:
                return "ALERT"
            if value >= PSI_WARN:
                return "WARN"
            return "OK"

        drift_df["status"] = drift_df["psi"].map(psi_status)
        drift_df["psi"] = drift_df["psi"].map(lambda x: _format_float(x, 3))

        st.dataframe(drift_df, hide_index=True)
        st.caption(
            f"PSI warn >= {PSI_WARN:.2f}, alert >= {PSI_ALERT:.2f}. "
            "This is a synthetic demonstration of drift monitoring."
        )
    else:
        st.info("Not enough data for drift monitoring.")

    st.divider()
    st.subheader("Simulated clinical utility")
    st.caption(
        "This is a simulated operational effect, not a measured reader study. "
        "It assumes prioritized studies are reviewed faster by a factor configured "
        "in WAIT_REDUCTION_FACTOR."
    )

    util = clinical_utility_table(q)
    if util.empty:
        st.info("No utility data available.")
    else:
        util_display = util.copy()
        util_display["mean_reduction_pct"] = util_display["mean_reduction_pct"].map(
            _format_percent
        )
        for col in [
            "baseline_mean_wait_min",
            "triaged_mean_wait_min",
            "baseline_median_wait_min",
            "triaged_median_wait_min",
        ]:
            util_display[col] = util_display[col].map(lambda x: _format_float(x, 1))
        st.dataframe(util_display, hide_index=True)


def render_governance_tab(q: pd.DataFrame) -> None:
    st.subheader("Governance, QA, and exports")

    st.markdown(
        """
        ### Intended use in this demo
        This application demonstrates a possible human-in-the-loop triage interface for
        chest X-ray prioritization. It is a **synthetic proof-of-concept** for workflow,
        visualization, threshold exploration, delivery reliability, and evaluation reporting.

        ### Explicit non-goals
        - It is **not** a diagnostic device.
        - It does **not** connect to DICOM, PACS, RIS, or EHR systems.
        - It does **not** use real patient images or reports.
        - It does **not** provide validated sensitivity, specificity, AUROC, or clinical utility.
        - It must **not** be used to prioritize real patient care.
        """
    )

    st.divider()
    st.subheader("Architecture decision records (demo status)")

    adr = pd.DataFrame(
        [
            {
                "component": "Classifier family",
                "decision": "Use simulated ensemble in demo; compare DenseNet vs ViT on real benchmark before selection.",
                "status": "Open for real data",
                "deciding_factor": "Calibration, subgroup robustness, inference cost, and validated performance.",
            },
            {
                "component": "DICOM handling",
                "decision": "No real DICOM I/O in demo. Future implementation should wrap pydicom behind an interface.",
                "status": "Deferred",
                "deciding_factor": "Replaceability and testability without production data.",
            },
            {
                "component": "PyTorch",
                "decision": "Not used in this synthetic UI demo. Future training module should document failure modes and monitoring.",
                "status": "Deferred",
                "deciding_factor": "Keep demo dependency-light and auditable.",
            },
            {
                "component": "PACS integration",
                "decision": "Simulated worklist delivery only. Real integration requires DICOMweb/HL7/FHIR review and operational sign-off.",
                "status": "Simulated",
                "deciding_factor": "Avoid irreversible or production-affecting actions.",
            },
        ]
    )
    st.dataframe(adr, hide_index=True)

    st.divider()
    st.subheader("Risk register")

    risks = pd.DataFrame(
        [
            {
                "risk": "Patient safety",
                "description": "False negatives could delay care in a real system.",
                "mitigation": "Clinician remains responsible; conservative thresholds; deferral; audit trail.",
            },
            {
                "risk": "Regulatory",
                "description": "Medical imaging triage software may be regulated.",
                "mitigation": "QMS, risk management, clinical evaluation, and validation before deployment.",
            },
            {
                "risk": "Bias / generalization",
                "description": "Performance may vary by site, scanner, protocol, demographics.",
                "mitigation": "Pre-agreed split policy, subgroup evaluation, external validation, drift monitoring.",
            },
            {
                "risk": "Automation bias",
                "description": "Users may over-trust synthetic overlays or scores.",
                "mitigation": "Show uncertainty, avoid overconfident displays, training and clear non-clinical labels.",
            },
            {
                "risk": "Privacy",
                "description": "Real DICOM data contains sensitive metadata and pixel-burned content.",
                "mitigation": "Ethics approval, DUA, de-identification, independent audit, access controls.",
            },
        ]
    )
    st.dataframe(risks, hide_index=True)

    st.divider()
    st.subheader("Configuration")

    config_rows = [
        ("SEED", SEED),
        ("N_STUDIES", N_STUDIES),
        ("TRIAGE_THRESHOLD", TRIAGE_THRESHOLD),
        ("REVIEW_BAND", REVIEW_BAND),
        ("UNCERTAINTY_DEFER_THRESHOLD", UNCERTAINTY_DEFER_THRESHOLD),
        ("OOD_REJECT_THRESHOLD", OOD_REJECT_THRESHOLD),
        ("TARGET_SENSITIVITY", TARGET_SENSITIVITY),
        ("SLA_MINUTES", SLA_MINUTES),
        ("WAIT_REDUCTION_FACTOR", WAIT_REDUCTION_FACTOR),
        ("PSI_WARN", PSI_WARN),
        ("PSI_ALERT", PSI_ALERT),
        ("MAX_PUSHES_PER_MINUTE", MAX_PUSHES_PER_MINUTE),
        ("RETENTION_DAYS", RETENTION_DAYS),
        ("MODEL_VERSION", MODEL_VERSION),
        ("SCHEMA_VERSION", SCHEMA_VERSION),
    ]
    st.dataframe(
        pd.DataFrame(config_rows, columns=["key", "default_or_env_value"]),
        hide_index=True,
    )

    st.divider()
    st.subheader("Self-checks")

    checks = run_self_checks()
    checks_df = pd.DataFrame(checks)
    st.dataframe(checks_df, hide_index=True)

    passed = int(checks_df["passed"].sum()) if not checks_df.empty else 0
    total = len(checks_df)
    if total > 0 and passed == total:
        st.success(f"All {total} self-checks passed.")
    else:
        st.warning(f"{passed}/{total} self-checks passed.")

    st.divider()
    st.subheader("Exports")

    if q.empty:
        st.warning("No data available for exports.")
        return

    manifest_cols = [
        "study_id",
        "accession_number",
        "patient_hash",
        "patient_cluster",
        "split",
        "site",
        "scanner",
        "protocol",
        "age",
        "sex",
        "reference_label",
        "reader1_label",
        "reader2_label",
        "reader_disagreement",
        "probability",
        "uncertainty",
        "ood_score",
        "action",
        "decision_reason",
        "status",
    ]
    manifest = q[[c for c in manifest_cols if c in q.columns]].copy()
    manifest["record_hash"] = manifest.apply(
        lambda r: hash_obj(r.to_dict()), axis=1
    )

    st.download_button(
        "Download synthetic manifest CSV",
        data=manifest.to_csv(index=False).encode("utf-8"),
        file_name="synthetic_chest_xray_manifest.csv",
        mime="text/csv",
        key="download_manifest",
    )

    st.download_button(
        "Download audit log JSON",
        data=json.dumps(st.session_state.get("audit", []), default=str, indent=2).encode(
            "utf-8"
        ),
        file_name="synthetic_audit_log.json",
        mime="application/json",
        key="download_audit",
    )

    st.download_button(
        "Download worklist messages JSON",
        data=json.dumps(st.session_state.get("messages", []), default=str, indent=2).encode(
            "utf-8"
        ),
        file_name="synthetic_worklist_messages.json",
        mime="application/json",
        key="download_messages",
    )

    test_q = q[q["split"] == "test"]
    if test_q.empty:
        test_q = q

    eval_summary = {
        "generated_at": utcnow_iso(),
        "model_version": MODEL_VERSION,
        "schema_version": SCHEMA_VERSION,
        "config": {
            "threshold": float(st.session_state.threshold),
            "review_band": float(st.session_state.review_band),
            "uncertainty_defer": float(st.session_state.uncertainty_defer),
            "ood_reject": float(st.session_state.ood_reject),
            "target_sensitivity": float(st.session_state.target_sensitivity),
        },
        "test_metrics": compute_operating_metrics(test_q),
        "split_counts": q["split"].value_counts().to_dict(),
    }

    st.download_button(
        "Download evaluation summary JSON",
        data=json.dumps(eval_summary, default=str, indent=2).encode("utf-8"),
        file_name="synthetic_evaluation_summary.json",
        mime="application/json",
        key="download_eval_summary",
    )


# -----------------------------------------------------------------------------
# Main app
# -----------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🫁",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()

    st.title(APP_TITLE)
    st.caption(
        "Synthetic research demo only. This application does not use real patient data, "
        "is not a medical device, and must not be used for clinical decisions."
    )

    render_sidebar()
    render_flash()

    q = apply_decisions(
        st.session_state.df_raw,
        threshold=float(st.session_state.threshold),
        review_band=float(st.session_state.review_band),
        uncertainty_defer=float(st.session_state.uncertainty_defer),
        ood_reject=float(st.session_state.ood_reject),
    )
    st.session_state.current_queue = q

    tabs = st.tabs(
        [
            "🎥 Live demo",
            "📋 Triage queue",
            "🧠 Model lab",
            "🔍 Explain & uncertainty",
            "🔄 Worklist simulator",
            "📊 Evaluation & safety",
            "🛡️ Governance & QA",
        ]
    )

    with tabs[0]:
        render_live_tab(q)

    with tabs[1]:
        render_queue_tab(q)

    with tabs[2]:
        render_model_lab_tab(q)

    with tabs[3]:
        render_explain_tab(q)

    with tabs[4]:
        render_worklist_tab(q)

    with tabs[5]:
        render_evaluation_tab(q)

    with tabs[6]:
        render_governance_tab(q)


if __name__ == "__main__":
    main()