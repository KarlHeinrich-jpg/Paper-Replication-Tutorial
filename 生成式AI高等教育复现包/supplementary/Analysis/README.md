# Analysis code for "Calibrating Generative AI Use for Sustainable Higher Education"

Manuscript ID sustainability-4495862. This package contains the complete scoring, measurement,
inferential, sensitivity, institutional, and nested-prediction code used in the manuscript, the
33 CFA model-syntax files, the scoring dictionary, an output manifest that links every exported
file to its manuscript location, and the scripts that regenerate the figures and check the exported
outputs against one another.

Participant-level responses, and the model outputs derived from them, are **not** distributed with
the manuscript (see the Data Availability Statement); every reported quantity appears in the
manuscript tables and appendices. The package documents how each of those tables was produced and
regenerates the complete set of outputs when run on the data under the stated access conditions.
`docs/OUTPUT_MANIFEST.csv` lists every file the pipeline writes, its content, the manuscript table
or section that reports it, and whether it is an aggregate record or a participant-level file
(names ending in `_private`, `fit_private.rds`, `models/*.joblib`).

## Layout

```
run.py                       driver: scoring -> measurement (R) -> inference -> supplementary -> institution -> prediction -> report
run.sh                       thin wrapper around run.py
requirements.txt             pinned Python dependencies
config/analysis.json         item sets, controls, category levels, seeds, and every analysis choice
config/scoring_dictionary.csv item id, data column, construct, cohort, English wording, range
config/frequency_anchors.csv the seven task-frequency anchors and how they are scored
models/*.lav                 the 33 lavaan model files (also regenerated into <out>/model_specifications by run.py)
models/model_inventory.json  estimator, identification, grouping and reliability flags per model
R/measurement.R              CFA estimation, fit export, reliability (alpha, omega, CR, AVE), higher-order omega,
                             trust-dependence comparisons, invariance sequence, literacy factor scores
R/install.R                  installs lavaan and jsonlite
src/score.py                 scoring rules, item summaries, composite summaries, correlations, first-component diagnostic
src/inference.py             HC3 OLS models, hierarchical blocks, inverted-U tests, model-implied maxima, paired case bootstrap,
                             joint support strata, curve inputs and plotted-curve data, decomposition/trust/dimension models
src/supplementary.py         influence deletion, RESET, Breusch-Pagan, restricted cubic spline, institution FE and
                             leave-one-institution-out refits, factor-score and indicator-pruning sensitivity, HTMT bootstrap,
                             ICCs, pooled cross-role contrasts
src/wild_cluster.py          null-imposed wild-cluster bootstrap with exhaustive enumeration of the 6^5 Webb weight combinations
src/predict.py               nested 5x4 cross-validation for the five model families, model manifest, held-out TreeSHAP
src/transforms.py            training-fold construction of the squared-use and literacy-by-use terms for nonlinear ridge
src/write_model_syntax.py    generates the .lav files and model inventory from config/analysis.json
src/plot_outputs.py          regenerates Figures 2-5 and the SHAP display data from a completed output directory
src/audit_outputs.py         consistency checks across the exported outputs
src/common.py                configuration validation, type-7 quantiles, RMSE identity check, design construction
docs/OUTPUT_MANIFEST.csv     output file -> content -> manuscript table/section -> shareable
```

## Requirements

* Python 3.10 or later with the packages in `requirements.txt`
  (`pip install -r requirements.txt`). The pipeline was run with the pinned versions; later
  versions of pandas (3.x) are also supported.
* R 4.1 or later with `lavaan` (0.6-17 or later) and `jsonlite` (`Rscript R/install.R`).
  R is needed only for the measurement stage.

## Input

One CSV containing both cohorts after the sample exclusions described in Section 3.2, with:

* `respondent_id`, `cohort` (`student` or `teacher`), `institution` (five institutions);
* every item column listed in `config/scoring_dictionary.csv` as integer codes 1-7
  (cohort-specific items are blank for the other cohort);
* the background controls: `age`, `study_year`, `ai_tenure`, `overall_frequency`, `gender`,
  `discipline` for students, and `teaching_experience`, `academic_rank`, `ai_tenure`,
  `overall_frequency`, `gender`, `discipline` for teachers, coded as in `config/analysis.json`.

No rows are deleted silently: incomplete focal items or controls stop the run with an explicit
message, because the complete-case analytical cohort is fixed (Section 3.2).

## Running

```
python run.py data.csv outputs/              # complete pipeline (writes to a new, empty directory)
python run.py data.csv outputs/ --stages inference institution report --bootstrap 200   # subset
python src/audit_outputs.py outputs/                                                     # consistency checks
```

The full run takes several hours on a laptop; the nested XGBoost, random-forest and LightGBM
searches dominate (`--jobs -1` parallelises the inner search). Every stage writes to its own
folder; `run_record.json` at the root records software versions, the SHA-256 fingerprint of the
input file, the seeds and the arguments.

Seeds: fold assignment and all bootstraps use 251101; the inner-selection and model seeds are
251101 + outer fold index (251102-251106). Quantiles are Hyndman-Fan type 7 throughout.

## What the checks verify

`src/audit_outputs.py` reproduces, from the exported files alone, the identities that connect the
reported quantities: CFA degrees of freedom against sample moments and free parameters; AVE
against the mean squared standardised loading of the displayed items; additivity of the
hierarchical R2 blocks; the model-implied maxima, boundary slopes, HC3 intervals and plotted
bands against the exported curve parameters and covariance blocks; the beyond-maximum counts
against the exported use percentiles; the bootstrap draw accounting; p = tail count / 7776 with
even tail counts for the wild-cluster tests; the RMSE identity for every pooled prediction
record; and the fold/seed structure of the model manifest. The report is written to
`audit_report.csv`.
