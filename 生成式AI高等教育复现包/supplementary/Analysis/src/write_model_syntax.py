from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

from common import BASE, config, write_json


def loadings(factors, marker=True):
    return [f"{f} =~ " + " + ".join([("1*" if marker and i == 0 else "") + v for i, v in enumerate(items)])
            for f, items in factors.items()]


def covs(names):
    return [a + " ~~ " + b for a, b in combinations(names, 2)]


def first(factors, marker=True):
    return "\n".join(loadings(factors, marker) + covs(factors)) + "\n"


def second(factors):
    dims = ["AW", "PU", "CE", "EJ"]; other = [a for a in factors if a not in dims]
    zeros = [a + " ~~ 0*" + b for a, b in combinations(dims, 2)] + [a + " ~~ 0*" + b for a in dims for b in other]
    return "\n".join(loadings(factors) + ["G =~ 1*AW + PU + CE + EJ"] + ["G ~~ " + a for a in other] + covs(other) + zeros) + "\n"


def specifications(cfg):
    scales = cfg["scales"]
    L = {x: scales["literacy_" + x] for x in ["AW", "PU", "CE", "EJ"]}
    if sum(L.values(), []) != scales["literacy"]:
        raise ValueError("The four literacy dimensions must partition the 12-item composite in order.")
    parallel = {**L, **{f: scales[c] for f, c in {"USE": "use", "TR": "trust", "HB": "habit", "RK": "risk"}.items()}}
    anc = {f: scales[c] for f, c in {"PE": "performance_expectancy", "EE": "effort_expectancy", "SI": "social_influence",
           "FC": "facilitating_conditions", "EQ": "equity", "LLL": "lifelong_learning"}.items()}
    records, texts = [], {}
    def save(name, text, factors, role=None, stdlv=False, group_equal=None, reliability=False):
        texts[name + ".lav"] = text
        records.append({"model_id": name, "estimator": "ML", "meanstructure": True,
            "std_lv": stdlv, "group_equal": group_equal or [], "group": "cohort" if role is None else None,
            "role": role, "factors": factors, "auto_cov_lv_x": False, "auto_cov_y": False,
            "reliability": reliability, "higher_order": "_second" in name})
    for role in ["student", "teacher"]:
        factors = {**parallel, "CAP": scales["capability_" + role], "DEP": scales["dependence_" + role]}
        save(role + "_focal_first", first(factors), factors, role, reliability=True)
        save(role + "_focal_second", second(factors), factors, role)
        allfactor = {"ALL": sum(factors.values(), [])}
        save(role + "_focal_one", first(allfactor), allfactor, role)
        save(role + "_literacy_one", first({"LIT": sum(L.values(), [])}), {"LIT": sum(L.values(), [])}, role, reliability=True)
        save(role + "_literacy_four", first(L), L, role, reliability=True)
        save(role + "_literacy_second", second(L), L, role, reliability=True)
        for f, items in anc.items():
            save(role + "_ancillary_" + f, first({f: items}), {f: items}, role, reliability=True)
        pair = {"TR": parallel["TR"], "DEP": factors["DEP"]}
        save(role + "_trust_free", first(pair, False), pair, role, stdlv=True)
        save(role + "_trust_085", "\n".join(loadings(pair, False) + ["TR ~~ 0.85*DEP"]) + "\n", pair, role, stdlv=True)
        one = {"RELIANCE": sum(pair.values(), [])}
        save(role + "_trust_one", first(one, False), one, role, stdlv=True)
    for name, eq in [("configural", []), ("metric", ["loadings"]), ("scalar", ["loadings", "intercepts"])]:
        save("invariance_" + name, first(parallel), parallel, group_equal=eq)
    return texts, records


def export_models(cfg, out):
    out = Path(out)
    texts, records = specifications(cfg)
    out.mkdir(parents=True, exist_ok=True)
    for name, text in texts.items():
        (out / name).write_text(text, encoding="utf-8")
    write_json(out / "model_inventory.json", records)
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config")
    p.add_argument("--out", default=str(BASE / "models"))
    a = p.parse_args()
    records = export_models(config(a.config), a.out)
    print(len(records), "CFA specifications written")


if __name__ == "__main__":
    main()
