#!/usr/bin/env python
"""Consistency checks on a completed output directory.

Usage:
    python src/audit_outputs.py OUT_DIR                       # run the checks, print a report
    python src/audit_outputs.py OUT_DIR --export-shareable DEST  # also copy the aggregate (non-participant-level) files to DEST for archiving

The checks reproduce, from the exported files alone, the identities that link the
reported quantities to one another:
  * CFA: degrees of freedom equal sample moments minus effective free parameters;
    every AVE equals the mean squared standardised loading of its items; all models passed.
  * Regression: hierarchical R^2 blocks are additive; model-implied maxima and boundary
    slopes are reproduced from the exported curve parameters and covariance blocks; the
    plotted curve means and bands are reproduced from the same inputs; support strata
    partition the cohort; beyond-maximum counts are consistent with the exported use
    percentiles under the type-7 definition; bootstrap draw accounting closes.
  * Wild cluster: p = tail count / 7776, and tail counts are even (sign-symmetric weights).
  * Prediction: RMSE^2 = (1 - R^2)(n - 1) s_y^2 / n for every pooled record; each
    family/outcome has exactly five outer-fold records with seeds 251102-251106; the
    best-family table names the family with the highest pooled R^2.
Every check prints PASS or FAIL with the numbers involved; the exit status is nonzero
if any check fails.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st

PRIVATE_PATTERNS = ("*_private.*", "fit_private.rds", "*.joblib", "*.rds")
REPORT = []


def check(name, ok, detail=""):
    REPORT.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))


def close(a, b, tol=1e-8):
    return bool(np.all(np.abs(np.asarray(a, float) - np.asarray(b, float)) <= tol * (1 + np.abs(np.asarray(b, float)))))


# ----------------------------------------------------------------------------- measurement
def audit_measurement(out: Path):
    m = out / "measurement"
    if not m.is_dir():
        print("(measurement outputs not present; skipped)")
        return
    status = json.loads((m / "model_status.json").read_text(encoding="utf-8"))
    check("CFA: all models passed", all(s.get("status") == "passed" for s in status),
          f"{sum(s.get('status') == 'passed' for s in status)}/{len(status)} passed")
    for folder in sorted(p for p in m.iterdir() if p.is_dir()):
        inv = folder / "model_inventory.json"
        if not inv.exists():
            continue
        info = json.loads(inv.read_text(encoding="utf-8"))
        df, moments, eff = info.get("df"), info.get("sample_moments"), info.get("effective_free_parameters")
        raw, eq = info.get("raw_free_parameter_count"), info.get("equality_constraint_rows", 0)
        ok = df is not None and eff is not None and close(moments - eff, df)
        check(f"CFA df identity: {folder.name}", ok,
              f"moments {moments}, effective free parameters {eff}, raw free {raw}, equality rows {eq}, df {df}")
        rel = folder / "reliability.csv"
        if rel.exists():
            r = pd.read_csv(rel)
            ok = close(r["AVE"], r["mean_squared_loading"], 1e-6)
            check(f"AVE = mean squared standardised loading: {folder.name}", ok,
                  "; ".join(f"{f}: AVE {a:.4f} / MSL {b:.4f}" for f, a, b in zip(r["factor"], r["AVE"], r["mean_squared_loading"])))
            ok = bool(((r["AVE"] > 0) & (r["AVE"] <= 1) & (r["CR_standardized"] > 0) & (r["CR_standardized"] <= 1)).all())
            check(f"AVE and CR within (0, 1]: {folder.name}", ok)


# ----------------------------------------------------------------------------- inference
def vertex(b, centres, l):
    L0, U0 = centres
    return U0 - (b["use_c"] + b["literacy_use"] * (l - L0)) / (2 * b["use_sq"])


def slope_and_se(b, V, centres, u, l):
    L0, U0 = centres
    c = np.array([0., 0., 1., 2 * (u - U0), l - L0])  # order: adjusted_intercept, literacy_c, use_c, use_sq, literacy_use
    est = float(c @ np.asarray(b))
    se = float(np.sqrt(max(c @ V @ c, 0.)))
    return est, se


def audit_inference(out: Path):
    inf = out / "inference"
    if not inf.is_dir():
        print("(inference outputs not present; skipped)")
        return
    for role in ("student", "teacher"):
        r = inf / role
        if not r.is_dir():
            continue
        shape = json.loads((r / "shape_inputs.json").read_text(encoding="utf-8"))
        centres = (float(shape["L_mean"]), float(shape["U_mean"]))
        n = int(shape["n"])
        check(f"{role}: quantile type recorded as 7", shape.get("quantile_type") == 7)
        for outcome in ("capability", "dependence"):
            o = r / outcome
            if not o.is_dir():
                continue
            blocks = pd.read_csv(o / "hierarchical_blocks.csv")
            ok = close(blocks["R2"].iloc[0] + blocks["delta_R2"].iloc[1:].sum(), blocks["R2"].iloc[-1], 1e-10)
            check(f"{role}/{outcome}: hierarchical R2 blocks additive", ok,
                  f"baseline {blocks['R2'].iloc[0]:.4f} + increments {blocks['delta_R2'].iloc[1:].sum():.4f} = final {blocks['R2'].iloc[-1]:.4f}")
            meta = json.loads((o / "model_metadata.json").read_text(encoding="utf-8"))
            check(f"{role}/{outcome}: adjusted R2 uses design rank", close(meta["adjusted_R2"], 1 - (1 - meta["R2"]) * (meta["n"] - 1) / (meta["n"] - meta["rank"]), 1e-10),
                  f"n {meta['n']}, rank {meta['rank']}, R2 {meta['R2']:.4f}, adjusted {meta['adjusted_R2']:.4f}")
            bs = json.loads((o / "bootstrap_summary.json").read_text(encoding="utf-8"))
            c = bs["counts"]
            ok = c["rank_deficient"] + c["constant_outcome"] + c["valid_delta"] == c["requested"]
            if bs.get("compute_vertices"):
                ok = ok and (c["valid_vertices"] + c["nonconcave"] + c["nonfinite_vertex"] == c["valid_delta"])
            check(f"{role}/{outcome}: bootstrap draw accounting", ok, json.dumps(c))
            if outcome != "capability" or not (o / "curve_parameters.csv").exists():
                continue
            par = pd.read_csv(o / "curve_parameters.csv").set_index("parameter")["estimate"]
            V = pd.read_csv(o / "curve_covariance.csv", index_col=0).to_numpy(float)
            defn = json.loads((o / "curve_definition.json").read_text(encoding="utf-8"))
            crit = st.t.ppf(.975, defn["df_resid"])
            tests = pd.read_csv(o / "shape_tests.csv")
            uq = shape["U_quantiles"]
            for _, row in tests.iterrows():
                l = float(row["literacy"])
                m = vertex(par, centres, l)
                check(f"{role}: maximum at literacy {l:.4f} reproduced from curve parameters", close(m, row["maximum"], 1e-8),
                      f"recomputed {m:.6f} vs shape_tests {row['maximum']:.6f}")
                lo, lo_se = slope_and_se(par.to_numpy(), V, centres, uq[0], l)
                hi, hi_se = slope_and_se(par.to_numpy(), V, centres, uq[-1], l)
                check(f"{role}: boundary slopes and HC3 SEs at literacy {l:.4f} reproduced",
                      close([lo, lo_se, hi, hi_se], [row["lower_slope"], row["lower_se"], row["upper_slope"], row["upper_se"]], 1e-8),
                      f"lower {lo:.4f} (SE {lo_se:.4f}), upper {hi:.4f} (SE {hi_se:.4f})")
                # beyond-maximum count versus exported use percentiles (type 7): k = number of use values <= maximum
                k = n - int(row["n_beyond_pooled"])
                ok = True; notes = []
                for p, q in zip([.05, .25, .5, .75, .95], uq):
                    j = math.floor(1 + (n - 1) * p)
                    cond = (k <= j) if m < q else (k >= j)
                    ok &= cond
                    notes.append(f"Q{int(p*100):02d}={q:.3f}:{'ok' if cond else 'VIOLATED'}")
                check(f"{role}: n beyond maximum {m:.3f} consistent with use percentiles", ok, f"n<=max {k} of {n}; " + ", ".join(notes))
            # plotted curves reproduce from the five parameters and their covariance
            grid = pd.read_csv(o / "curve_data.csv")
            L0, U0 = centres
            X = np.column_stack([np.ones(len(grid)), grid["literacy"] - L0, grid["use"] - U0, (grid["use"] - U0) ** 2, (grid["literacy"] - L0) * (grid["use"] - U0)])
            mu = X @ par.to_numpy()
            se = np.sqrt(np.einsum("ij,jk,ik->i", X, V, X))
            check(f"{role}: curve means and 95% bands reproduced from curve inputs",
                  close(mu, grid["mean"], 1e-8) and close(mu - crit * se, grid["CI_low"], 1e-8) and close(mu + crit * se, grid["CI_high"], 1e-8),
                  f"{len(grid)} grid rows")
            if (o / "joint_support.csv").exists():
                sup = pd.read_csv(o / "joint_support.csv")
                ok = sup["n"].sum() == n and bool(((sup["n_below"] + sup["n_equal"] + sup["n_above"]) == sup["n"]).all())
                check(f"{role}: literacy strata partition the cohort", ok, f"strata n {sup['n'].tolist()} sum {sup['n'].sum()} of {n}")
                check(f"{role}: every stratum has cases on both sides of its maximum",
                      bool(((sup["n_below"] > 0) & (sup["n_above"] > 0)).all()),
                      "; ".join(f"{s}: {b} below / {a} above" for s, b, a in zip(sup['stratum'], sup['n_below'], sup['n_above'])))


# ----------------------------------------------------------------------------- institution
def audit_institution(out: Path):
    f = out / "institution" / "wild_cluster_results.csv"
    if not f.exists():
        print("(institution outputs not present; skipped)")
        return
    w = pd.read_csv(f)
    check("wild cluster: exhaustive Webb enumeration of 6^5 = 7776 combinations", bool((w["enumerations"] == 7776).all()))
    check("wild cluster: p equals tail count / enumerations", close(w["tail_count"] / w["enumerations"], w["p_two_sided"], 1e-12),
          "; ".join(f"{r.cohort}/{r.outcome}/{r.term}: {r.tail_count}/7776 = {r.p_two_sided:.5f}" for r in w.itertuples()))
    check("wild cluster: tail counts even under sign-symmetric weights", bool((w["tail_count"] % 2 == 0).all()))
    check("wild cluster: CR1 correction G/(G-1)(n-1)/(n-k)",
          close(w["small_sample_correction"], w["G"] / (w["G"] - 1) * (w["n"] - 1) / (w["n"] - w["k"]), 1e-12))


# ----------------------------------------------------------------------------- prediction
def audit_prediction(out: Path):
    pred = out / "prediction"
    if not pred.is_dir():
        print("(prediction outputs not present; skipped)")
        return
    for role in ("student", "teacher"):
        r = pred / role
        if not r.is_dir():
            continue
        pm = pd.read_csv(r / "pooled_metrics.csv")
        rhs = (1 - pm["R2"]) * (pm["n"] - 1) * pm["outcome_sd"] ** 2 / pm["n"]
        check(f"{role}: RMSE^2 = (1-R2)(n-1)s_y^2/n for all pooled records", close(pm["RMSE"] ** 2, rhs, 1e-10),
              "; ".join(f"{a}/{b}: RMSE {c:.4f}, R2 {d:.4f}" for a, b, c, d in zip(pm["outcome"], pm["family"], pm["RMSE"], pm["R2"])))
        sel = json.loads((r / "best_parameters.json").read_text(encoding="utf-8"))
        s = pd.DataFrame(sel)
        folds_ok = bool((s.groupby(["outcome", "family"])["fold"].apply(lambda x: sorted(x) == [1, 2, 3, 4, 5])).all())
        seeds_ok = bool((s["seed"] == 251101 + s["fold"]).all())
        check(f"{role}: five outer folds per family/outcome with seeds 251102-251106", folds_ok and seeds_ok,
              f"{len(s)} selection records")
        best = pd.read_csv(r / "best_family_summary.csv")
        ok = True
        for _, b in best.iterrows():
            sub = pm[pm["outcome"] == b["outcome"]]
            ok &= sub.loc[sub["R2"].idxmax(), "family"] == b["family"]
        check(f"{role}: best family is the highest pooled R2", bool(ok),
              "; ".join(f"{b.outcome}: {b.family} (R2 {b.R2:.4f})" for b in best.itertuples()))
        g = r / "global_shap.csv"
        if g.exists():
            gs = pd.read_csv(g)
            check(f"{role}: held-out SHAP summaries cover every respondent once", bool((gs["n"] == pm["n"].iloc[0]).all()),
                  f"n = {gs['n'].unique().tolist()} vs cohort n {pm['n'].iloc[0]}")


# ----------------------------------------------------------------------------- supplementary
def audit_supplementary(out: Path):
    sup = out / "supplementary"
    if not sup.is_dir():
        print("(supplementary outputs not present; skipped)")
        return
    for role in ("student", "teacher"):
        r = sup / role
        if not r.is_dir():
            continue
        h = pd.read_csv(r / "HTMT.csv")
        check(f"{role}: HTMT bootstrap draws all valid", bool((h["valid"] == h["requested"]).all()),
              f"{h['valid'].min()}-{h['valid'].max()} of {h['requested'].iloc[0]}")
        icc = pd.read_csv(r / "institution_ICC.csv")
        check(f"{role}: institution ICCs finite", bool(np.isfinite(icc["ICC"]).all()),
              "; ".join(f"{o}: {v:.3f}" for o, v in zip(icc["outcome"], icc["ICC"])))


# ----------------------------------------------------------------------------- export
def is_private(rel: str) -> bool:
    name = Path(rel).name
    return any(fnmatch.fnmatch(name, pat) for pat in PRIVATE_PATTERNS)


def export_shareable(out: Path, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    copied, withheld = [], []
    for p in sorted(out.rglob("*")):
        if not p.is_file():
            continue
        rel = str(p.relative_to(out))
        (withheld if is_private(rel) else copied).append(rel)
        if not is_private(rel):
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
    (dest / "SHARED_FILES.txt").write_text("\n".join(copied) + "\n", encoding="utf-8")
    (dest / "WITHHELD_FILES.txt").write_text(
        "Participant-level or fitted-model files withheld under the data-access conditions:\n" + "\n".join(withheld) + "\n", encoding="utf-8")
    print(f"exported {len(copied)} shareable files to {dest}; withheld {len(withheld)} private files (listed in WITHHELD_FILES.txt)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", help="Completed output directory written by run.py")
    ap.add_argument("--export-shareable", metavar="DEST", help="Copy the aggregate (non-participant-level) outputs to DEST for archiving")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    if not out.is_dir():
        ap.error(f"Not a directory: {out}")
    audit_measurement(out); audit_inference(out); audit_institution(out); audit_prediction(out); audit_supplementary(out)
    failed = [n for n, ok, _ in REPORT if not ok]
    print(f"\n{len(REPORT) - len(failed)} checks passed, {len(failed)} failed.")
    pd.DataFrame(REPORT, columns=["check", "passed", "detail"]).to_csv(out / "audit_report.csv", index=False)
    if a.export_shareable:
        export_shareable(out, Path(a.export_shareable).resolve())
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
