#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "src"))
from common import config
from write_model_syntax import export_models

STAGES = ("measurement", "inference", "supplementary", "institution", "prediction", "report")
FAMILIES = ("ridge_linear", "ridge_nonlinear", "rf", "xgboost", "lightgbm")


def run_record(data: Path, out: Path, cfgpath: Path, args, R):
    """Write the provenance record: software versions, input fingerprint, seeds and arguments."""
    def sha256(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ("numpy", "pandas", "scipy", "statsmodels", "sklearn", "xgboost", "lightgbm", "shap", "joblib"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            versions[mod] = None
    if R:
        try:
            versions["R"] = subprocess.run([R, "-e", "cat(R.version.string); cat('|'); cat(as.character(packageVersion('lavaan')))"],
                                           capture_output=True, text=True, check=True).stdout.strip()
        except (subprocess.CalledProcessError, OSError):
            versions["R"] = "unavailable"
    cfg = json.loads(cfgpath.read_text(encoding="utf-8"))
    record = {"timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "input_file": str(data), "input_sha256": sha256(data), "input_bytes": data.stat().st_size,
              "config_file": str(cfgpath), "config_sha256": sha256(cfgpath),
              "seed": cfg.get("seed"), "bootstrap_B": args.get("bootstrap"), "stages": args.get("stages"),
              "families": args.get("families"), "jobs": args.get("jobs"), "software": versions}
    (out / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("data", help="Item-level CSV after the study's sample exclusions")
    p.add_argument("out", help="New or empty output directory")
    p.add_argument("--config", default=str(BASE / "config" / "analysis.json"))
    p.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    p.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    p.add_argument("--bootstrap", type=int, help="Bootstrap draws; defaults to config.bootstrap_B")
    p.add_argument("--jobs", type=int, default=1, help="Parallel jobs for inner CV; -1 uses all CPUs")
    a = p.parse_args(argv)
    data, out, cfgpath = Path(a.data).resolve(), Path(a.out).resolve(), Path(a.config).resolve()
    cfg = config(cfgpath)
    B = cfg["bootstrap_B"] if a.bootstrap is None else a.bootstrap
    if B < 1 or (a.jobs < 1 and a.jobs != -1):
        p.error("bootstrap must be positive; jobs must be positive or -1.")
    if len(a.families) != len(set(a.families)):
        p.error("Model families must be distinct.")
    if not data.is_file():
        p.error(f"Input not found: {data}")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        p.error("Use a new or empty output directory to avoid overwriting model results.")
    stages = set(a.stages)
    if "supplementary" in stages:
        stages.add("measurement")
    R = shutil.which("Rscript")
    if "measurement" in stages and R is None:
        p.error("Rscript is required for CFA. Install R, then run Rscript R/install.R.")
    out.mkdir(parents=True, exist_ok=True)
    run_record(data, out, cfgpath, {"bootstrap": B, "stages": sorted(stages), "families": a.families, "jobs": a.jobs}, R)
    scored = out / "scoring" / "scored_private.csv"
    shared = ["--data", str(scored), "--config", str(cfgpath)]
    commands = [("scoring", [sys.executable, "src/score.py", "--data", str(data),
                 "--out", str(out / "scoring"), "--config", str(cfgpath)])]
    if stages - {"report"} == {"prediction"}:
        commands[0][1].append("--allow-missing-controls")
    if "measurement" in stages:
        syntax_dir = out / "model_specifications"
        export_models(cfg, syntax_dir)
        commands.append(("measurement", [R, "R/measurement.R", str(scored),
                         str(out / "measurement"), str(cfgpath), str(syntax_dir)]))
    if "inference" in stages:
        commands.append(("inference", [sys.executable, "src/inference.py", *shared,
                         "--out", str(out / "inference"), "--bootstrap", str(B)]))
    if "supplementary" in stages:
        commands.append(("supplementary", [sys.executable, "src/supplementary.py", *shared,
                         "--out", str(out / "supplementary"), "--bootstrap", str(B),
                         "--measurement", str(out / "measurement")]))
    if "institution" in stages:
        commands.append(("institution", [sys.executable, "src/wild_cluster.py", *shared,
                         "--out", str(out / "institution")]))
    if "prediction" in stages:
        commands.append(("prediction", [sys.executable, "src/predict.py", *shared,
                         "--out", str(out / "prediction"), "--families", *a.families,
                         "--jobs", str(a.jobs)]))
    if "report" in stages:
        commands.append(("report", [sys.executable, "src/plot_outputs.py", str(out)]))
        commands.append(("report", [sys.executable, "src/audit_outputs.py", str(out)]))
    env = os.environ.copy()
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        env.setdefault(key, "1")
    for stage, command in commands:
        print(f"[{stage}]", flush=True)
        try:
            subprocess.run(command, cwd=BASE, env=env, check=True)
        except (subprocess.CalledProcessError, OSError) as e:
            raise SystemExit(f"Stopped during {stage}: {e}") from e
    print(f"Selected model results written to {out}")


if __name__ == "__main__":
    main()
