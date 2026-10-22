from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular

from common import config, design,  read_scored, validate_design

WEBB = np.array([-np.sqrt(1.5), -1., -np.sqrt(.5), np.sqrt(.5), 1., np.sqrt(1.5)])


def _groups(groups, n):
    groups = np.asarray(groups)
    if groups.shape != (n,) or pd.isna(groups).any():
        raise ValueError("Cluster labels must be complete and aligned to rows.")
    codes, labels = pd.factorize(groups, sort=False)
    if len(labels) < 2:
        raise ValueError("At least two clusters required.")
    return codes, labels


def _ols_projection(A):
    Q, R = np.linalg.qr(np.asarray(A, float), mode="reduced")
    return solve_triangular(R, Q.T, lower=False)


def cr1_variance(A, residual, groups):
    A, residual = np.asarray(A, float), np.asarray(residual, float)
    validate_design(A)
    n, k = A.shape; gi, labels = _groups(groups, n); G = len(labels)
    if residual.shape != (n,) or not np.isfinite(residual).all():
        raise ValueError("Invalid residual vector.")
    proj = _ols_projection(A)
    scores = np.column_stack([proj[:, gi == g] @ residual[gi == g] for g in range(G)])
    correction = G / (G - 1) * (n - 1) / (n - k)
    return correction * scores @ scores.T


def enumerate_webb(A, y, groups, j, chunk=256):
    A, y = np.asarray(A, float), np.asarray(y, float)
    validate_design(A, y)
    n, k = A.shape; gi, labels = _groups(groups, n); G = len(labels)
    if G != 5:
        raise ValueError("The manuscript specifies exhaustive Webb enumeration with exactly five clusters.")
    if not isinstance(j, (int, np.integer)) or not 0 <= j < k:
        raise ValueError("Invalid tested coefficient index.")
    if not isinstance(chunk, int) or chunk < 1:
        raise ValueError("chunk must be a positive integer.")
    proj = _ols_projection(A)
    beta = proj @ y; res = y - A @ beta
    V = cr1_variance(A, res, groups); se = np.sqrt(V[j, j])
    if not np.isfinite(se) or se <= 0:
        raise ValueError("Invalid observed CR1 standard error.")
    obs = beta[j] / se
    keep = [x for x in range(k) if x != j]
    fit0 = A[:, keep] @ np.linalg.lstsq(A[:, keep], y, rcond=None)[0]
    res0 = y - fit0
    weights = np.array(list(itertools.product(WEBB, repeat=G)))
    correction = G / (G - 1) * (n - 1) / (n - k)
    h = proj[j]
    tstats = []
    for start in range(0, len(weights), chunk):
        W = weights[start:start + chunk]
        ys = fit0[:, None] + res0[:, None] * W[:, gi].T
        bs = proj @ ys; es = ys - A @ bs
        scores = np.vstack([(h[gi == g, None] * es[gi == g]).sum(axis=0) for g in range(G)])
        ses = np.sqrt(correction * np.sum(scores ** 2, axis=0))
        if not np.isfinite(ses).all() or np.any(ses <= 0):
            raise ValueError("Degenerate bootstrap standard errors; no assignments silently dropped.")
        tstats.extend(bs[j] / ses)
    tstats = np.asarray(tstats)
    tail = int((abs(tstats) >= abs(obs) - 1e-12).sum())
    return {"estimate": float(beta[j]), "CR1_se": float(se), "t_observed": float(obs), "tail_count": tail,
        "enumerations": len(weights), "p_two_sided": tail / len(weights), "G": G, "n": n, "k": k,
        "null_coefficient": 0, "null_imposed": True, "weights": "Webb six-point", "small_sample_correction": correction,
        "tie_tolerance": 1e-12, "interpretation": "cluster sensitivity only"}, tstats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True); p.add_argument("--out", required=True); p.add_argument("--config")
    a = p.parse_args(); cfg = config(a.config); out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True); rows = []
    for role in ["student", "teacher"]:
        d = read_scored(a.data, role, cfg); X, _ = design(d, cfg, role)
        for outcome in ["capability", "dependence"]:
            for term in ["use_sq", "literacy_use"]:
                row, ts = enumerate_webb(X, d[outcome], d.institution, list(X).index(term))
                row.update(cohort=role, outcome=outcome, term=term); rows.append(row)
                np.save(out / f"{role}_{outcome}_{term}_t.npy", ts)
    pd.DataFrame(rows).to_csv(out / "wild_cluster_results.csv", index=False)


if __name__ == "__main__":
    main()
