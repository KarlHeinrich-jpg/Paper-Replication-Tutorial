#!/usr/bin/env python
"""Regenerate the manuscript figures and the SHAP display data from a completed output directory.

Usage:
    python src/plot_outputs.py OUT_DIR [--data OUT_DIR/scoring/scored_private.csv] [--figures OUT_DIR/figures]

Produces, under --figures (default OUT_DIR/figures):
    fig_u_students.pdf, fig_u_teachers.pdf
        Literacy-conditioned use--outcome curves at the 25th/50th/75th literacy percentiles with
        pointwise 95% HC3 bands and the model-implied maxima (manuscript Figures 2 and 3), drawn
        from inference/<role>/capability/curve_data.csv and shape_tests.csv.
    fig_literacy_dimensions_students.pdf, fig_literacy_dimensions_teachers.pdf
        Standardised dimension-by-use coefficients with 95% intervals (Figures 4 and 5), drawn from
        inference/<role>/dimension_models/<dim>/coefficients.csv.
and, under OUT_DIR/prediction/<role>/shap_display/:
    beeswarm_<outcome>_private.csv   per-respondent held-out attributions joined with the predictor
                                     values (row level; withheld under the data-access conditions)
    profile_<outcome>.csv            binned attribution profiles for use, literacy and trust (aggregate)
    beeswarm_summary_<outcome>.csv   per-predictor quantiles of the held-out attributions (aggregate)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DIMS = [("AW", "Awareness"), ("PU", "Practical use"), ("CE", "Critical evaluation"), ("EJ", "Ethical judgement")]
OUTCOME_LABEL = {"student": "Perceived learning agency", "teacher": "Teaching innovation competence"}
ROLE_LABEL = {"student": "Students", "teacher": "Teachers"}
PROFILE_PREDICTORS = ["use", "literacy", "trust"]


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def plot_curves(out: Path, role: str, figdir: Path):
    plt = _mpl()
    o = out / "inference" / role / "capability"
    grid = pd.read_csv(o / "curve_data.csv")
    tests = pd.read_csv(o / "shape_tests.csv")
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    styles = {"lower": ("25th percentile", "--", "#1f77b4"), "median": ("Median", "-", "#2ca02c"), "upper": ("75th percentile", "-.", "#9467bd")}
    for level, (label, ls, colour) in styles.items():
        g = grid[grid["literacy_level"] == level]
        ax.plot(g["use"], g["mean"], ls, color=colour, lw=1.6, label=label)
        ax.fill_between(g["use"], g["CI_low"], g["CI_high"], color=colour, alpha=.12, lw=0)
    names = {"low": "lower", "median": "median", "high": "upper"}
    for _, row in tests.iterrows():
        m = row["maximum"]
        if not np.isfinite(m):
            continue
        g = grid[grid["literacy_level"] == names[row["literacy_level"]]]
        y = np.interp(m, g["use"], g["mean"])
        ax.plot([m], [y], "o", ms=4, color="#d62728")
        ax.annotate(f"{m:.3f}", (m, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)
    ax.set_xlim(1, 7); ax.set_ylim(1, 7)
    ax.set_xlabel("GenAI task-use frequency (1--7)"); ax.set_ylabel(OUTCOME_LABEL[role])
    ax.legend(title="AI literacy", frameon=False, fontsize=7, title_fontsize=7, loc="lower left")
    fig.tight_layout(); fig.savefig(figdir / f"fig_u_{role}s.pdf"); plt.close(fig)


def plot_dimensions(out: Path, role: str, figdir: Path):
    plt = _mpl()
    rows = []
    for code, label in DIMS:
        f = out / "inference" / role / "dimension_models" / code / "coefficients.csv"
        c = pd.read_csv(f).set_index("term").loc["literacy_use"]
        rows.append((label, c["beta"], c["beta_CI_low"], c["beta_CI_high"]))
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    y = np.arange(len(rows))[::-1]
    for yi, (label, b, lo, hi) in zip(y, rows):
        ax.plot([lo, hi], [yi, yi], color="#1f77b4", lw=1.4)
        ax.plot([b], [yi], "o", color="#1f77b4", ms=4)
    ax.axvline(0, color="black", lw=.8)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("Standardised literacy dimension x use coefficient")
    ax.set_title(f"{ROLE_LABEL[role]}: dimension-level moderation", fontsize=9)
    ax.grid(axis="x", lw=.4, alpha=.5)
    fig.tight_layout(); fig.savefig(figdir / f"fig_literacy_dimensions_{role}s.pdf"); plt.close(fig)


def shap_display(out: Path, role: str, scored: pd.DataFrame):
    src = out / "prediction" / role / "heldout_shap_private.csv"
    if not src.exists():
        print(f"({role}: no held-out SHAP record; display data skipped)")
        return
    shap = pd.read_csv(src, dtype={"respondent_id": "string"})
    d = scored.loc[scored["cohort"] == role].copy()
    d["respondent_id"] = d["respondent_id"].astype("string")
    shapcols = [c for c in shap.columns if c.startswith("SHAP__")]
    predictors = [c[6:] for c in shapcols]
    present = [p for p in predictors if p in d.columns]
    dest = out / "prediction" / role / "shap_display"
    dest.mkdir(parents=True, exist_ok=True)
    for outcome, g in shap.groupby("outcome"):
        merged = g.merge(d[["respondent_id"] + present], on="respondent_id", how="left", validate="one_to_one")
        if merged[present].isna().any().any():
            raise ValueError("Held-out SHAP rows could not be joined to the scored predictors.")
        bees = merged.drop(columns=["respondent_id"])
        bees.to_csv(dest / f"beeswarm_{outcome}_private.csv", index=False)
        summary = []
        for p in predictors:
            v = merged["SHAP__" + p]
            q = v.quantile([.05, .25, .5, .75, .95])
            summary.append({"cohort": role, "outcome": outcome, "predictor": p, "n": len(v),
                            "mean_absolute_SHAP": float(v.abs().mean()), "mean_SHAP": float(v.mean()),
                            **{f"q{int(k*100):02d}": float(x) for k, x in q.items()}})
        pd.DataFrame(summary).sort_values("mean_absolute_SHAP", ascending=False).to_csv(dest / f"beeswarm_summary_{outcome}.csv", index=False)
        profile = []
        for p in PROFILE_PREDICTORS:
            if p not in present:
                continue
            x = merged[p].astype(float); v = merged["SHAP__" + p].astype(float)
            edges = np.arange(1, 7.01, .5)
            bins = pd.cut(x, edges, include_lowest=True)
            for interval, sub in v.groupby(bins, observed=True):
                if len(sub) < 5:  # no cells describing fewer than five respondents
                    continue
                profile.append({"cohort": role, "outcome": outcome, "predictor": p, "bin_low": float(interval.left),
                                "bin_high": float(interval.right), "n": int(len(sub)), "mean_SHAP": float(sub.mean()),
                                "q25_SHAP": float(sub.quantile(.25)), "q75_SHAP": float(sub.quantile(.75))})
        pd.DataFrame(profile).to_csv(dest / f"profile_{outcome}.csv", index=False)
    print(f"{role}: SHAP display data written to {dest}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out"); ap.add_argument("--data"); ap.add_argument("--figures")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    figdir = Path(a.figures).resolve() if a.figures else out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    data = Path(a.data) if a.data else out / "scoring" / "scored_private.csv"
    scored = pd.read_csv(data, dtype={"respondent_id": "string", "cohort": "string"}) if data.exists() else None
    for role in ("student", "teacher"):
        if (out / "inference" / role / "capability" / "curve_data.csv").exists():
            plot_curves(out, role, figdir)
        if (out / "inference" / role / "dimension_models").is_dir():
            plot_dimensions(out, role, figdir)
        if scored is not None:
            shap_display(out, role, scored)
    print(f"figures written to {figdir}")


if __name__ == "__main__":
    main()
