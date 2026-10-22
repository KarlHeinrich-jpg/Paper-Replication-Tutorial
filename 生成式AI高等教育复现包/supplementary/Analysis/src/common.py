from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
ROLES = ("student", "teacher")
KEYS = ["cohort", "respondent_id"]
FOCAL = ["literacy", "use", "trust", "habit", "risk", "capability", "dependence"]


def config(path: str | Path | None = None) -> dict:
    cfg = json.loads(Path(path or BASE / "config/analysis.json").read_text(encoding="utf-8"))
    for key in ("scales", "seed", "controls", "ancillary", "categorical_levels"):
        if key not in cfg:
            raise ValueError(f"Configuration missing {key}.")
    for name, cols in cfg["scales"].items():
        if not cols or len(cols) != len(set(cols)):
            raise ValueError(f"Invalid/duplicated item set: {name}.")
    if not isinstance(cfg.get("bootstrap_B"), int) or cfg["bootstrap_B"] < 1:
        raise ValueError("bootstrap_B must be a positive integer.")
    cfa = cfg.get("analysis_choices", {}).get("cfa", {})
    expected = {"estimator": "ML", "meanstructure": True,
                "identification": "marker_except_unit_variance_trust_comparisons",
                "residual_covariances": "zero", "cross_loadings": "zero",
                "ordinal_treatment": "seven_point_indicators_treated_as_numeric"}
    for key, value in expected.items():
        if cfa.get(key) != value:
            raise ValueError(f"Unsupported CFA setting {key}: the corresponding estimator must be implemented explicitly.")
    wc = cfg.get("analysis_choices", {}).get("wild_cluster", {})
    for key, value in {"weights": "Webb_six_point", "B": 7776, "impose_null": True,
                       "residual_transform": "none", "statistic": "CR1_t",
                       "tail": "abs_greater_equal", "tie_tolerance": 1e-12}.items():
        if wc.get(key) != value:
            raise ValueError(f"Unsupported wild-cluster setting: {key}.")
    if cfg.get("analysis_choices", {}).get("shap_additivity_tolerance") != 2e-5:
        raise ValueError("SHAP additivity tolerance must be 2e-5 for this implementation.")
    return cfg


def write_json(path: str | Path, data) -> None:
    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple, np.ndarray)):
            return [clean(v) for v in x]
        if isinstance(x, np.generic):
            return clean(x.item())
        if isinstance(x, float) and not np.isfinite(x):
            return None
        if isinstance(x, Path):
            return str(x)
        return x
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(clean(data), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def read_csv(path: str | Path) -> pd.DataFrame:
    import csv
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        header = next(csv.reader(f))
    if len(header) != len(set(header)):
        raise ValueError("Duplicate CSV column names are not allowed.")
    d = pd.read_csv(path, dtype={k: "string" for k in ["respondent_id", "institution", "cohort"] if k in header},
                    keep_default_na=False, na_values=[""], encoding="utf-8-sig")
    return d


def validate_keys(d: pd.DataFrame, require_both: bool = False) -> None:
    for c in ["respondent_id", "cohort", "institution"]:
        if c not in d or d[c].isna().any():
            raise ValueError(f"Missing identifier: {c}.")
        s = d[c].astype(str)
        if (s.str.strip() == "").any() or (s != s.str.strip()).any():
            raise ValueError(f"Blank or whitespace-padded {c}; resolve the source file explicitly.")
    if d.duplicated(KEYS).any():
        raise ValueError("Duplicate (cohort, respondent_id) keys.")
    if not d.cohort.isin(ROLES).all():
        raise ValueError("cohort must be student or teacher.")
    if require_both and set(d.cohort) != set(ROLES):
        raise ValueError("Both student and teacher cohorts are required for a complete run.")


def quantile(x, probs):
    a = np.asarray(x, dtype=float)
    p = np.asarray(probs, dtype=float)
    if a.ndim != 1 or len(a) == 0 or not np.isfinite(a).all():
        raise ValueError("Quantiles require a nonempty finite one-dimensional vector.")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Quantile probabilities must be in [0,1].")
    return np.quantile(a, p, method="linear")


def pooled_metrics(y, pred) -> dict:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if y.ndim != 1 or y.shape != pred.shape or len(y) < 2 or not np.isfinite(np.r_[y, pred]).all():
        raise ValueError("Invalid outcome/prediction vectors.")
    sse = float(np.sum((y - pred) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    if sst <= 0:
        raise ValueError("R-squared is undefined for a constant outcome.")
    r2 = 1 - sse / sst
    sd = float(y.std(ddof=1))
    rmse = float(np.sqrt(sse / len(y)))
    identity_rhs = (1 - r2) * (len(y) - 1) * sd ** 2 / len(y)
    identity_error = float(rmse ** 2 - identity_rhs)
    if not np.isclose(rmse ** 2, identity_rhs, rtol=1e-10, atol=1e-12):
        raise ArithmeticError("RMSE/R2/sample-variance identity failed on the prediction rows.")
    return {"n": len(y), "outcome_mean": float(y.mean()), "outcome_sd": sd, "R2": r2,
            "RMSE": rmse, "MAE": float(np.mean(abs(y - pred))),
            "RMSE_identity_checked": True, "RMSE_identity_error": identity_error}


def validate_design(X, y=None) -> None:
    a = np.asarray(X, float)
    if a.ndim != 2 or not np.isfinite(a).all() or a.shape[0] <= a.shape[1]:
        raise ValueError("OLS design must be finite with n > k.")
    if np.linalg.matrix_rank(a) != a.shape[1]:
        raise ValueError("Rank-deficient OLS design; inspect categories/controls rather than using a pseudoinverse silently.")
    if y is not None:
        y = np.asarray(y, float)
        if y.ndim != 1 or len(y) != len(a) or not np.isfinite(y).all() or y.std() == 0:
            raise ValueError("Outcome must be finite, nonconstant and aligned to the design.")


def validate_controls(d: pd.DataFrame, cfg: dict, role: str, allow_missing=False) -> None:
    num, cat = controls(cfg, role, expanded=True)
    missing = set(num + cat) - set(d.columns)
    if missing:
        raise ValueError(f"Missing control columns: {sorted(missing)}")
    choices = cfg.get("analysis_choices", {})
    code_keys = {"study_year": "study_year_codes", "academic_rank": "academic_rank_codes", "ai_tenure": "ai_tenure_codes", "overall_frequency": "frequency_codes"}
    for c in num:
        values = pd.to_numeric(d[c], errors="raise")
        if not allow_missing and values.isna().any():
            raise ValueError(f"Missing control {c}; no undocumented row deletion.")
        a = values.dropna().to_numpy(float)
        if not np.isfinite(a).all():
            raise ValueError(f"Nonfinite control {c}.")
        if c in code_keys and code_keys[c] in choices:
            allowed = [float(x) for x in choices[code_keys[c]]]
            if not np.isin(a, allowed).all():
                raise ValueError(f"{c} is outside the declared ordinal code set {allowed}.")
        if c in {"age", "teaching_experience"} and np.any(a < 0):
            raise ValueError(f"{c} cannot be negative.")
        if c in cfg["ancillary"] and np.any((a < 1) | (a > 7)):
            raise ValueError(f"{c} composite outside 1--7.")
    for c in cat:
        if not allow_missing and d[c].isna().any():
            raise ValueError(f"Missing category {c}.")
        if not d[c].dropna().isin(cfg["categorical_levels"][c]).all():
            raise ValueError(f"Unknown {c} category; check the administration codebook.")


def read_scored(path: str | Path, role: str, cfg: dict) -> pd.DataFrame:
    if role not in ROLES:
        raise ValueError(f"Unknown role: {role}")
    d = read_csv(path)
    validate_keys(d)
    missing = set(FOCAL) - set(d.columns)
    if missing:
        raise ValueError(f"Missing focal composites: {sorted(missing)}")
    d = d.loc[d.cohort == role].copy().reset_index(drop=True)
    if d.empty:
        raise ValueError(f"No {role} observations.")
    d[FOCAL] = d[FOCAL].apply(pd.to_numeric, errors="raise")
    a = d[FOCAL].to_numpy(float)
    if not np.isfinite(a).all() or not ((a >= 1) & (a <= 7)).all():
        raise ValueError("Focal scores must be complete finite values in 1--7.")
    return d


def controls(cfg, role, expanded=True):
    numeric = list(cfg["controls"][role]["numeric"])
    if expanded:
        numeric += cfg["ancillary"]
    return numeric, list(cfg["controls"][role]["categorical"])


def design(d: pd.DataFrame, cfg: dict, role: str, expanded=True, use_col="use", literacy_col="literacy", centres=None, trust="trust"):
    num, cat = controls(cfg, role, expanded)
    focal = [literacy_col, use_col, "habit", "risk"] + ([] if trust is None else [trust])
    req = set(num + cat + focal + ["institution"])
    if req - set(d):
        raise ValueError(f"Missing regression columns: {sorted(req-set(d))}")
    if d[list(req)].isna().any().any():
        raise ValueError("Resolve missing covariates before OLS; no silent sample change.")
    L0, U0 = centres if centres is not None else (float(d[literacy_col].mean()), float(d[use_col].mean()))
    if not np.isfinite([L0, U0]).all():
        raise ValueError("Invalid centring constants.")
    X = pd.DataFrame({"Intercept": 1., "literacy_c": d[literacy_col] - L0, "use_c": d[use_col] - U0,
                      "habit": d.habit, "risk": d.risk}, index=d.index)
    if trust is not None:
        X["trust"] = d[trust]
    X["use_sq"] = X.use_c ** 2
    X["literacy_use"] = X.literacy_c * X.use_c
    control_centres = {}
    for c in num:
        v = pd.to_numeric(d[c], errors="raise")
        centre = float(v.mean()) if c in cfg.get("analysis_choices", {}).get("continuous_control_centering", []) else 0.
        X[c] = v - centre
        control_centres[c] = centre
    omitted = []
    for c in cat:
        levels = cfg["categorical_levels"][c]
        if not d[c].isin(levels).all():
            raise ValueError(f"Unknown {c} categories. Confirm the codebook.")
        for level in levels[1:]:
            name = c + "__" + level
            if (d[c] == level).any():
                X[name] = (d[c] == level).astype(float)
            else:
                omitted.append(name)
    X = X.astype(float)
    validate_design(X)
    X.attrs["specification"] = {"role": role, "expanded": expanded, "literacy_source": literacy_col,
        "use_source": use_col, "trust_source": trust, "L_mean": L0, "U_mean": U0,
        "control_centres": control_centres, "omitted_unobserved_indicators": omitted,
        "categorical_references": {c: cfg["categorical_levels"][c][0] for c in cat}}
    return X, (L0, U0)
