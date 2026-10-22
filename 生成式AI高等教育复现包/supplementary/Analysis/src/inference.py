from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm

from common import config, design,  quantile, read_scored, validate_design, write_json


def fit_export(X: pd.DataFrame, y: np.ndarray, out: Path):
    validate_design(X, y)
    y = np.asarray(y, float)
    fit = sm.OLS(y, X).fit(cov_type="HC3", use_t=True)
    if not np.isfinite(np.r_[fit.params, fit.bse]).all() or np.any(fit.bse <= 0):
        raise ValueError("Nonfinite/degenerate HC3 estimate. Inspect leverage and model residuals.")
    out.mkdir(parents=True, exist_ok=True)
    sdx, sdy = X.std(ddof=1).to_numpy(), y.std(ddof=1)
    scale = sdx / sdy
    ci = fit.conf_int().to_numpy()
    frame = pd.DataFrame({"term": X.columns, "b": fit.params.to_numpy(), "HC3_SE": fit.bse.to_numpy(),
        "t": fit.tvalues.to_numpy(), "p": fit.pvalues.to_numpy(), "CI_low": ci[:, 0], "CI_high": ci[:, 1],
        "design_sd": sdx, "outcome_sd": sdy, "beta": fit.params.to_numpy() * scale,
        "beta_HC3_SE": fit.bse.to_numpy() * scale, "beta_CI_low": ci[:, 0] * scale, "beta_CI_high": ci[:, 1] * scale})
    intercept = frame.term.eq("Intercept") | frame.term.str.endswith("__Intercept")
    frame.loc[intercept, ["beta", "beta_HC3_SE", "beta_CI_low", "beta_CI_high"]] = np.nan
    frame.to_csv(out / "coefficients.csv", index=False)
    fit.cov_params().to_csv(out / "HC3_covariance.csv")
    pd.DataFrame({"term": X.columns, "mean": X.mean().to_numpy(), "sd": sdx}).to_csv(out / "design_summary.csv", index=False)
    (out / "model_summary.txt").write_text(fit.summary().as_text(), encoding="utf-8")
    write_json(out / "model_metadata.json", {"n": len(y), "k": X.shape[1], "rank": np.linalg.matrix_rank(X),
        "df_resid": fit.df_resid, "R2": fit.rsquared, "adjusted_R2": fit.rsquared_adj,
        "outcome_mean": y.mean(), "outcome_sd": sdy, "design_columns": list(X),
        "condition_number": np.linalg.cond(X), "covariance": "HC3", "reference_distribution": "t(n-k)",
        "specification": X.attrs.get("specification", {})})
    return fit


def _se(contrast, covariance):
    variance = float(contrast @ covariance @ contrast)
    if not np.isfinite(variance) or variance < -1e-12:
        raise ValueError("Invalid contrast variance.")
    return float(np.sqrt(max(variance, 0.)))


def slope(fit, Xcols, centres, u, l):
    L0, U0 = centres
    c = np.zeros(len(Xcols)); ix = {name: i for i, name in enumerate(Xcols)}
    c[ix["use_c"]] = 1.
    c[ix["use_sq"]] = 2 * (u - U0)
    c[ix["literacy_use"]] = l - L0
    b, v = np.asarray(fit.params), np.asarray(fit.cov_params())
    est, se = float(c @ b), _se(c, v)
    if se <= 0:
        raise ValueError("Zero conditional-slope standard error.")
    crit = st.t.ppf(.975, fit.df_resid)
    return {"slope": est, "se": se, "low": est - crit * se, "high": est + crit * se,
            "p_positive": float(st.t.sf(est / se, fit.df_resid)), "p_negative": float(st.t.cdf(est / se, fit.df_resid))}


def vertex(beta, columns, centres, l):
    ix = {name: i for i, name in enumerate(columns)}
    b = np.asarray(beta); L0, U0 = centres
    if not np.isfinite(b).all() or b[ix["use_sq"]] >= 0:
        return np.nan
    return float(U0 - (b[ix["use_c"]] + b[ix["literacy_use"]] * (l - L0)) / (2 * b[ix["use_sq"]]))


def support_frame(use, literacy, maxima, lq):
    u, l, q, m = map(lambda v: np.asarray(v, float), (use, literacy, lq, maxima))
    if u.ndim != 1 or l.shape != u.shape or len(q) != 3 or len(m) != 3 or not np.isfinite(np.r_[u, l, q, m]).all():
        raise ValueError("Invalid support inputs.")
    if not np.all(np.diff(q) > 0):
        raise ValueError("Quartile cutpoints must be distinct; do not split ties to manufacture strata.")
    masks = [l <= q[0], (l > q[0]) & (l < q[2]), l >= q[2]]
    rows = []
    for label, mask, maximum in zip(["lower", "middle", "upper"], masks, m):
        us = u[mask]
        if not len(us):
            raise ValueError("Empty literacy stratum; inspect ties and cutpoints.")
        row = {"stratum": label, "n": len(us), "maximum": maximum}
        row.update(dict(zip(["use_Q05", "use_Q25", "use_Q50", "use_Q75", "use_Q95"], quantile(us, [.05, .25, .5, .75, .95]))))
        row.update({"n_near": int((abs(us - maximum) <= .5).sum()), "n_below": int((us < maximum).sum()),
                    "n_equal": int((us == maximum).sum()), "n_above": int((us > maximum).sum())})
        if row["n_below"] + row["n_equal"] + row["n_above"] != len(us):
            raise AssertionError("Support partition failure.")
        row["p_near"] = row["n_near"] / len(us)
        rows.append(row)
    if sum(r["n"] for r in rows) != len(u):
        raise AssertionError("Literacy strata must partition the entire cohort.")
    return pd.DataFrame(rows)


def paired_bootstrap(X, y, lq, centres, B, seed, compute_vertices=True):
    if not isinstance(B, (int, np.integer)) or isinstance(B, bool) or B < 1:
        raise ValueError("B must be a positive integer.")
    validate_design(X, y)
    A, y = np.asarray(X, float), np.asarray(y, float)
    n, k = A.shape; cols = list(X)
    red = [i for i, name in enumerate(cols) if name not in ["use_sq", "literacy_use"]]
    if len(red) != k - 2:
        raise ValueError("The reduced model must remove use_sq and literacy_use only.")
    if np.shape(lq) != (3,) or not np.isfinite(np.r_[lq, centres]).all():
        raise ValueError("Invalid fixed bootstrap evaluation points.")
    rng = np.random.default_rng(seed)
    deltas, vertices, logs = [], [], []
    counts = {"requested": B, "rank_deficient": 0, "constant_outcome": 0, "nonconcave": 0,
              "nonfinite_vertex": 0, "outside_low": 0, "outside_median": 0, "outside_high": 0}
    for draw in range(1, B + 1):
        ids = rng.integers(0, n, n)
        xb, yb = A[ids], y[ids]
        log = {"draw": draw, "delta_status": "invalid", "vertex_status": "not_requested" if not compute_vertices else "invalid"}
        den = float(np.sum((yb - yb.mean()) ** 2))
        if den <= 0:
            counts["constant_outcome"] += 1; log["reason"] = "constant_outcome"; logs.append(log); continue
        bf, _, rank, _ = np.linalg.lstsq(xb, yb, rcond=None)
        br, _, rankr, _ = np.linalg.lstsq(xb[:, red], yb, rcond=None)
        if rank < k or rankr < len(red):
            counts["rank_deficient"] += 1; log["reason"] = "rank_deficient"; logs.append(log); continue
        ssef, sser = float(np.sum((yb - xb @ bf) ** 2)), float(np.sum((yb - xb[:, red] @ br) ** 2))
        delta = (sser - ssef) / den
        if not np.isfinite(delta) or delta < -1e-10:
            raise ArithmeticError("Nested increment invalid beyond numerical tolerance.")
        delta = max(0., delta)
        deltas.append(delta); log.update(delta_status="valid", delta_R2=delta)
        if compute_vertices:
            if bf[cols.index("use_sq")] >= 0:
                counts["nonconcave"] += 1; log["vertex_status"] = "nonconcave"
            else:
                v = [vertex(bf, cols, centres, float(q)) for q in lq]
                if not np.isfinite(v).all():
                    counts["nonfinite_vertex"] += 1; log["vertex_status"] = "nonfinite"
                else:
                    for name, x in zip(["outside_low", "outside_median", "outside_high"], v):
                        counts[name] += int(x < 1 or x > 7)
                    values = v + [v[2] - v[0]]
                    vertices.append(values)
                    log.update(dict(zip(["low", "median", "high", "displacement"], values)))
                    log["vertex_status"] = "valid"
        logs.append(log)
    ds, arr = np.asarray(deltas), np.asarray(vertices).reshape(-1, 4)
    counts.update(valid_delta=len(ds), valid_vertices=len(arr))
    if counts["rank_deficient"] + counts["constant_outcome"] + counts["valid_delta"] != B:
        raise AssertionError("Bootstrap draw accounting failed.")
    if compute_vertices and counts["valid_vertices"] + counts["nonconcave"] + counts["nonfinite_vertex"] != len(ds):
        raise AssertionError("Maximum draw accounting failed.")
    result = {"counts": counts, "seed": seed, "compute_vertices": compute_vertices,
              "interval_status": "available" if len(ds) else "no_valid_draws",
              "maximum_interval_status": "available" if len(arr) else ("no_valid_maxima" if compute_vertices else "not_requested"),
              "delta_90_CI": quantile(ds, [.05, .95]).tolist() if len(ds) else None,
              "vertices_95_CI": np.quantile(arr, [.025, .975], axis=0, method="linear").tolist() if len(arr) else None,
              "draw_log": logs}
    return result, ds, arr


def curve_grid(fit, X, centres, lq, use_range):
    cols = list(X); ix = {name: i for i, name in enumerate(cols)}
    v, b, L0, U0 = np.asarray(fit.cov_params()), np.asarray(fit.params), *centres
    crit = st.t.ppf(.975, fit.df_resid); rows = []
    for lev, l in zip(["lower", "median", "upper"], lq):
        for u in np.linspace(*use_range, 151):
            x = X.mean().to_numpy(copy=True)
            x[ix["literacy_c"]], x[ix["use_c"]] = l - L0, u - U0
            x[ix["use_sq"]], x[ix["literacy_use"]] = (u - U0) ** 2, (l - L0) * (u - U0)
            mu, se = float(x @ b), _se(x, v)
            rows.append({"literacy_level": lev, "literacy": l, "use": u, "mean": mu,
                         "CI_low": mu - crit * se, "CI_high": mu + crit * se})
    return pd.DataFrame(rows)


def export_curve_inputs(fit, X, centres, out):
    names = ["adjusted_intercept", "literacy_c", "use_c", "use_sq", "literacy_use"]
    cols = list(X); H = np.zeros((5, len(cols)))
    H[0] = X.mean().to_numpy()
    for c in names[1:]:
        H[0, cols.index(c)] = 0.
    for i, c in enumerate(names[1:], 1):
        H[i, cols.index(c)] = 1.
    b = H @ np.asarray(fit.params)
    V = H @ np.asarray(fit.cov_params()) @ H.T
    pd.DataFrame({"parameter": names, "estimate": b}).to_csv(out / "curve_parameters.csv", index=False)
    pd.DataFrame(V, index=names, columns=names).to_csv(out / "curve_covariance.csv")
    write_json(out / "curve_definition.json", {"parameter_order": names, "centres": list(centres), "df_resid": fit.df_resid,
        "formula": "a + bL*(L-L0) + bU*(U-U0) + bU2*(U-U0)^2 + bLU*(L-L0)*(U-U0)",
        "interval": "pointwise t(n-k) with transformed full HC3 covariance"})


def run_role(d, cfg, role, out, B):
    out.mkdir(parents=True, exist_ok=True)
    lq = quantile(d.literacy, [.25, .5, .75]); uq = quantile(d.use, [.05, .25, .5, .75, .95])
    knots = quantile(d.use, [.05, .275, .5, .725, .95]); X, centres = design(d, cfg, role)
    if not np.all(np.diff(lq) > 0):
        raise ValueError("Quartile cutpoints are not distinct. Review support stratification.")
    write_json(out / "shape_inputs.json", {"L_mean": centres[0], "U_mean": centres[1], "L_quantiles": lq,
        "U_quantiles": uq, "U_min": d.use.min(), "U_max": d.use.max(), "RCS_knots": knots,
        "quantile_type": 7, "n": len(d), "bootstrap_B": B, "bootstrap_seed": cfg["seed"],
        "evaluation_points": "fixed original-sample values", "scale_bounds": [1, 7],
        "bootstrap_nonconcave": "excluded only from maximum intervals; counts reported"})
    for outcome in ["capability", "dependence"]:
        o = out / outcome; y = d[outcome].to_numpy(float); f = fit_export(X, y, o)
        blockcols = [[c for c in X if c not in ["use_c", "use_sq", "literacy_use"]],
                     [c for c in X if c not in ["use_sq", "literacy_use"]], [c for c in X if c != "literacy_use"], list(X)]
        blocks, last = [], None
        for label, cols in zip(["baseline", "linear_use", "squared_use", "literacy_interaction"], blockcols):
            m = sm.OLS(y, X[cols]).fit(cov_type="HC3", use_t=True); p = np.nan
            if last is not None:
                added = [c for c in cols if c not in last[0]]; R = np.zeros((len(added), len(cols)))
                for i, c in enumerate(added):
                    R[i, cols.index(c)] = 1.
                p = float(m.wald_test(R, use_f=True, scalar=True).pvalue)
            blocks.append({"block": label, "n": len(y), "k": len(cols), "R2": m.rsquared,
                "adjusted_R2": m.rsquared_adj, "delta_R2": np.nan if last is None else m.rsquared - last[1], "HC3_block_p": p})
            last = cols, m.rsquared
        pd.DataFrame(blocks).to_csv(o / "hierarchical_blocks.csv", index=False)
        result, ds, vs = paired_bootstrap(X, y, lq, centres, B, cfg["seed"], compute_vertices=outcome == "capability")
        log = result.pop("draw_log")
        pd.DataFrame(log).to_csv(o / "bootstrap_draw_log.csv", index=False)
        result["delta_point_estimate"] = blocks[-1]["R2"] - blocks[1]["R2"]
        write_json(o / "bootstrap_summary.json", result)
        pd.DataFrame({"draw": [r["draw"] for r in log if r["delta_status"] == "valid"], "delta_R2": ds}).to_csv(o / "delta_bootstrap.csv", index=False)
        if outcome == "capability":
            vf = pd.DataFrame(vs, columns=["low", "median", "high", "displacement"])
            vf.insert(0, "draw", [r["draw"] for r in log if r["vertex_status"] == "valid"])
            vf.to_csv(o / "vertex_bootstrap.csv", index=False)
            rows, maxima = [], []
            for label, l in zip(["low", "median", "high"], lq):
                lo, hi = slope(f, list(X), centres, uq[0], l), slope(f, list(X), centres, uq[-1], l)
                m = vertex(f.params, X.columns, centres, l); maxima.append(m)
                finite = bool(np.isfinite(m))
                row = {"literacy_level": label, "literacy": l, "maximum": m, "p_U": max(lo["p_positive"], hi["p_negative"]),
                       "interior_observed": finite and bool(d.use.min() < m < d.use.max()),
                       "interior_Q05_Q95": finite and bool(uq[0] < m < uq[-1]),
                       "n_beyond_pooled": int((d.use > m).sum()) if finite else np.nan,
                       "n_near_pooled": int((abs(d.use - m) <= .5).sum()) if finite else np.nan}
                for prefix, data in [("lower", lo), ("upper", hi),
                    ("min", slope(f, list(X), centres, float(d.use.min()), l)),
                    ("max", slope(f, list(X), centres, float(d.use.max()), l)),
                    ("scale1", slope(f, list(X), centres, 1., l)), ("scale7", slope(f, list(X), centres, 7., l))]:
                    row.update({prefix + "_" + k: v for k, v in data.items()})
                rows.append(row)
            pd.DataFrame(rows).to_csv(o / "shape_tests.csv", index=False)
            support_ok = bool(np.isfinite(maxima).all())
            if support_ok:
                support_frame(d.use, d.literacy, maxima, lq).to_csv(o / "joint_support.csv", index=False)
            write_json(o / "shape_status.json", {"finite_concave_maxima": support_ok,
                "joint_support": "written" if support_ok else "not_applicable_nonconcave_or_nonfinite_maximum"})
            export_curve_inputs(f, X, centres, o)
            curve_grid(f, X, centres, lq, (float(d.use.min()), float(d.use.max()))).to_csv(o / "curve_data.csv", index=False)
        else:
            rows = []
            for u in np.linspace(1, 7, 61):
                rows.append({"use": u, "literacy": centres[0], **slope(f, list(X), centres, float(u), centres[0])})
            pd.DataFrame(rows).to_csv(o / "mean_literacy_slopes.csv", index=False)
            b2, b1 = float(f.params["use_sq"]), float(f.params["use_c"])
            write_json(o / "stationary_point.json", {"literacy": centres[0], "curvature": b2,
                "type": "maximum" if b2 < 0 else ("minimum" if b2 > 0 else "linear"),
                "stationary_use": centres[1] - b1 / (2 * b2) if b2 != 0 else None})
        xc, _ = design(d, cfg, role, expanded=False, centres=centres)
        fit_export(xc, y, o / "reliable_core")
    for usecol in ["support_use", "substitution_use"]:
        xd, _ = design(d, cfg, role, use_col=usecol)
        fit_export(xd, d.capability.to_numpy(), out / "decomposition" / usecol)
    for trust in ["belief_trust", None]:
        xd, _ = design(d, cfg, role, centres=centres, trust=trust)
        fit_export(xd, d.dependence.to_numpy(), out / "trust_sensitivity" / str(trust))
    for dim in ["AW", "PU", "CE", "EJ"]:
        xd, _ = design(d, cfg, role, literacy_col="literacy_" + dim)
        fit_export(xd, d.capability.to_numpy(), out / "dimension_models" / dim)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True); p.add_argument("--out", required=True); p.add_argument("--config")
    p.add_argument("--bootstrap", type=int, default=5000)
    a = p.parse_args()
    if a.bootstrap < 1:
        p.error("bootstrap must be positive.")
    cfg, out = config(a.config), Path(a.out)
    for role in ["student", "teacher"]:
        run_role(read_scored(a.data, role, cfg), cfg, role, out / role, a.bootstrap)


if __name__ == "__main__":
    main()
