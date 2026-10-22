from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from common import FOCAL, config, read_csv, validate_controls, validate_keys, write_json


def alpha_raw(x):
    x = np.asarray(x, float)
    if x.ndim != 2 or x.shape[1] < 2 or len(x) < 2 or not np.isfinite(x).all():
        return np.nan
    S = np.cov(x, rowvar=False, ddof=1)
    total = S.sum()
    return x.shape[1] / (x.shape[1] - 1) * (1 - np.trace(S) / total) if total > 0 else np.nan


def score_frame(d: pd.DataFrame, cfg: dict, allow_missing_controls=False):
    d = d.copy().reset_index(drop=True)
    validate_keys(d)
    summaries = []
    for role, g in d.groupby("cohort", sort=False):
        other = "teacher" if role == "student" else "student"
        allitems = list(dict.fromkeys(c for name, cols in cfg["scales"].items()
                                     if not name.endswith("_" + other) for c in cols))
        missing = set(allitems) - set(d.columns)
        if missing:
            raise ValueError(f"Missing indicators: {sorted(missing)}")
        x = g[allitems].apply(pd.to_numeric, errors="raise")
        a = x.to_numpy(float)
        observed = a[~np.isnan(a)]
        core = sum([cfg["scales"][name if name not in {"capability", "dependence"}
                                   else name + "_" + role] for name in FOCAL], [])
        if not np.isfinite(observed).all() or not ((observed >= 1) & (observed <= 7) & (observed == np.round(observed))).all():
            raise ValueError(f"{role}: observed items must be integer categories 1--7.")
        if x[core].isna().any().any() or (not allow_missing_controls and x.isna().any().any()):
            raise ValueError(f"{role}: complete items are required; only prediction-only ancillary controls may be missing.")
        d.loc[g.index, allitems] = x.to_numpy()
        for name, cols in cfg["scales"].items():
            if name.endswith("_" + other):
                continue
            dest = "capability" if name == "capability_" + role else ("dependence" if name == "dependence_" + role else name)
            d.loc[g.index, dest] = x[cols].mean(axis=1, skipna=False)
        for position, col in enumerate(allitems, 1):
            z = x[col].dropna()
            row = {"cohort": role, "item": col, "questionnaire_order": position,
                   "is_focal": col in core, "n_valid": len(z), "n_missing": len(x) - len(z),
                   "mean": z.mean(), "sd": z.std(ddof=1), "minimum": z.min(), "maximum": z.max()}
            for k in range(1, 8):
                row[f"n_{k}"] = int((z == k).sum())
                row[f"p_{k}"] = float((z == k).mean())
            summaries.append(row)
    return d, pd.DataFrame(summaries)


def export_summaries(d, cfg, out):
    rows, correlations = [], []
    for role, g in d.groupby("cohort", sort=False):
        other = "teacher" if role == "student" else "student"
        for key, items in cfg["scales"].items():
            if key.endswith("_" + other):
                continue
            col = "capability" if key == "capability_" + role else ("dependence" if key == "dependence_" + role else key)
            z = g[col].dropna().to_numpy(float)
            rows.append({"cohort": role, "composite": col, "n": len(g), "items": len(items),
                         "n_valid": len(z), "n_missing": len(g) - len(z),
                         "mean": z.mean() if len(z) else np.nan,
                         "sd": z.std(ddof=1) if len(z) > 1 else np.nan,
                         "min": z.min() if len(z) else np.nan, "max": z.max() if len(z) else np.nan,
                         "alpha": alpha_raw(g[items])})
        for a, b in combinations(FOCAL, 2):
            xa, xb = g[a].to_numpy(float), g[b].to_numpy(float)
            if xa.std() == 0 or xb.std() == 0:
                r, p = np.nan, np.nan
            else:
                r, p = pearsonr(xa, xb)
            correlations.append({"cohort": role, "variable_a": a, "variable_b": b, "n": len(g), "r": r, "p": p})
        g[FOCAL].corr().to_csv(out / f"{role}_composite_correlation.csv")
        itemcols=sum([cfg['scales'][c] for c in ['literacy','use','trust','habit','risk','capability_'+role,'dependence_'+role]],[])
        z=g[itemcols].to_numpy(float)
        if np.all(z.std(axis=0)>0):
            eigen=np.linalg.eigvalsh(np.corrcoef(z,rowvar=False))[::-1]
            write_json(out/f'{role}_first_component.json',{'matrix':'Pearson correlation of focal items','items':itemcols,
                'eigenvalues':eigen,'first_component_variance_fraction':eigen[0]/len(itemcols)})
        else:
            write_json(out/f'{role}_first_component.json',{'status':'undefined_constant_indicator'})
    pd.DataFrame(rows).to_csv(out / "composite_summaries.csv", index=False)
    pd.DataFrame(correlations).to_csv(out / "correlations_long.csv", index=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, help="Item-level CSV after the study's sample exclusions")
    p.add_argument("--out", required=True)
    p.add_argument("--config")
    p.add_argument("--allow-missing-controls", action="store_true",
                   help="Prediction-only: retain missing ancillary controls for training-fold imputation; focal items remain complete")
    a = p.parse_args()
    cfg = config(a.config)
    d, stats = score_frame(read_csv(a.data), cfg, allow_missing_controls=a.allow_missing_controls)
    o = Path(a.out)
    o.mkdir(parents=True, exist_ok=True)
    for role, g in d.groupby("cohort"):
        validate_controls(g, cfg, role, allow_missing=a.allow_missing_controls)
    d.to_csv(o / "scored_private.csv", index=False)
    stats.to_csv(o / "item_summaries.csv", index=False)
    for role, g in d.groupby("cohort"):
        cols = stats.loc[stats.cohort == role, "item"].tolist()
        z = g[cols].apply(pd.to_numeric)
        z.cov().to_csv(o / f"{role}_item_covariance.csv")
        z.corr().to_csv(o / f"{role}_item_correlation.csv")
    export_summaries(d, cfg, o)


if __name__ == "__main__":
    main()
