# XSmell+ — MLCQ → CK metrics → smell detection → SHAP → constraint-aware refactoring recommendation

Reproducible pipeline behind the revised XSmell+ paper. It rebuilds a feature table from the public MLCQ labels, trains four
binary smell detectors under leakage-safe splits, explains them with SHAP, turns the explanations into refactoring
recommendations, and checks a small set of real refactorings before/after.

| Contribution | What it is | Steps |
|---|---|---|
| 1. Detection + explanation | Four binary detectors (Blob, Data Class, Feature Envy, Long Method) on CK metrics; project-wise / organisation-wise splits; SHAP and its fidelity | 01–07, 09, 10 |
| 2. Recommendation | XSmell+: SHAP evidence + structural preconditions + simulated impact + cost, conflict-free top-k, abstention | 08, 08b |
| 3. Effectiveness | Before/after case study: CK metrics, Maintainability Index, SonarQube debt, unit tests | 11a, apply_case, 11, 13, 14 |

> **Scope statement (read before citing any number).** MLCQ has one snapshot per repository, so it contains no "after refactoring"
> data. Contribution 2's impact numbers are model-based simulations, and Contribution 3 is a 5-case study in which the author applied
> the refactorings (not blinded). Neither is evidence that XSmell+ recommends better than a static rule or SHAP alone. See
> [Limitations](#limitations-to-report).

---

## 1. Prerequisites

- Python 3.10+ and `pip install -r requirements.txt` (includes `javalang` for steps 13, 14 and `xs_quality.py`)
- `git`, Java **17+**, Maven (`mvn`) in PATH
- **CK jar.** The GitHub `ck-master.zip` is source code, not a jar. Build it once:
  `unzip ck-master.zip && cd ck-master && mvn clean compile package -DskipTests`, then copy
  `target/ck-0.7.1-SNAPSHOT-jar-with-dependencies.jar` to `tools/ck.jar`
- **MLCQ**: `MLCQCodeSmellSamples.csv` from https://doi.org/10.5281/zenodo.3666840 (Madeyski & Lewowski, EASE 2020)
- 10–30 GB of free disk and a stable connection; step 02 clones hundreds of repositories (sources are deleted after metrics
  are computed unless you pass `--keep`)
- Only for step 14: a SonarQube server (Section 5.4)

Versions used for the reported results: Python 3.13, pandas 3.0.2, numpy 2.4.3, scipy 1.17.1, scikit-learn 1.8.0, shap 0.51.0,
matplotlib 3.10.9.

## 2. Quick start

```bash
# Dataset (once; step 02 takes hours and is resumable)
python 01_prepare_labels.py --csv MLCQCodeSmellSamples.csv --out data/labels.csv
python 02_fetch_and_run_ck.py --labels data/labels.csv --ck-jar tools/ck.jar --limit 3   # test run first
python 02_fetch_and_run_ck.py --labels data/labels.csv --ck-jar tools/ck.jar --jobs 2    # full run
python 03_build_dataset.py --labels data/labels.csv --ck data/ck_raw --out data
python 04_diagnose_unmatched.py                                                          # optional

# Experiments, SHAP, recommender, figures (steps 05-10, 08b, 12)
python run_all.py             # full run
python run_all.py --skip-05   # reuse results/per_*_<split>.csv (step 05 is the slow part)
python run_all.py --quick     # smoke test: 1 seed, 3 folds, 2 models. Numbers are NOT for the paper
```

`run_all.py` runs: 05 (four splits) → 07 → 06 → 08 `--sensitivity` → 08b → 09 → 10 → 12.
Steps 11a, 11, 13 and 14 are **not** in `run_all.py` (they need refactoring cases, Java/Maven and SonarQube; Section 5).
If a run is interrupted, re-run the same command: steps are independent, and 08/09 reuse `results/cache/`.

## 3. Pipeline reference

| Step | Script | Purpose / output |
|---|---|---|
| 01 | `01_prepare_labels.py` | Aggregate reviewer severities into one label per (sample, smell) → `data/labels.csv` |
| 02 | `02_fetch_and_run_ck.py` | Fetch each repo at the labelled commit, run CK → `data/ck_raw/` (+ `_run_log.csv` for failures) |
| 03 | `03_build_dataset.py` | Join labels with CK → `dataset_class.csv` (blob, data_class), `dataset_method.csv` (feature_envy, long_method), `join_report.csv` |
| 04 | `04_diagnose_unmatched.py` | Why labelled samples could not be joined → `unmatched_class.csv` |
| 05 | `05_run_experiments.py --split project\|org\|random [--dedup]` | Cross-validated detection, 5 seeds, Mean ± SD → `per_seed_*`, `per_fold_*`, `summary_*`, `confusion_*`, `dataset_audit_*` |
| 07 | `07_stats_tests.py [--min-pos 10]` | `stats_tests.csv`, `split_comparison.csv`, `org_breakdown.csv` |
| 06 | `06_make_figures.py` | `figures/` + SHAP (weak detectors, MCC < 0.30, are labelled "reference only") |
| 08 | `08_recommender.py [--sensitivity]` | XSmell+ recommender + ablation → `recommendations.csv`, `recommender_ablation.csv`, `recommender_examples.txt` |
| 08b | `08b_sensitivity_report.py` | Sensitivity to ρ / λ → `recommender_sensitivity.csv` (needs `08 --sensitivity`) |
| 09 | `09_fidelity.py [--seeds 5]` | SHAP deletion/insertion fidelity → `fidelity*.csv` |
| 10 | `10_cost.py [--reps 5]` | Training / inference / SHAP / recommendation time → `cost.csv` (report your hardware) |
| 11a | `11a_select_cases.py` | Candidate pool for the case study → `cases_todo.csv` |
| – | `apply_case.py` | Apply one prepared patch to your fork and register the commit in `cases_todo.csv` |
| 11 | `11_before_after.py` | Before/after CK, MI, debt, detector probability, unit tests → `before_after.csv`, `before_after_summary.csv` |
| 13 | `13_mi_metrics.py` | Maintainability Index (approximate) → `mi_*` columns |
| 14 | `14_sonar_debt.py prepare\|collect` | SonarQube technical debt → `debt_*`, `issues_*` columns |
| 12 | `12_paper_figures.py` | `fig1_pipeline.png`, ablation and fidelity figures (reads CSVs only, trains nothing) |

Shared modules: `xs_common.py` (out-of-fold prediction and SHAP under project-wise splits, imported by 08–11; keep it next to
`05_run_experiments.py`), `xs_quality.py` (MI and PMD-based debt proxy used by 11), `step_utils.py` (folder naming shared by 02/03/11).
The first run of 08/09 caches out-of-fold SHAP in `results/cache/` (RF SHAP takes minutes). Keep the cache unless data or models change.

## 4. Experimental protocol (what the numbers mean)

**Labels.** Each MLCQ review gives a severity in {none, minor, major, critical}. A sample is positive if more than half of its
reviewers gave severity ≠ none, negative if fewer than half did; exact ties are dropped (150 samples). Each (sample, smell) pair
is one instance. Blob and Data Class are class-level, Feature Envy and Long Method are method-level.

**Tasks.** Four binary tasks (smelly vs clean). MLCQ has no multi-class "which smell" label, so the old 15/16-label formulation is not reproduced.

**Leakage guard.** `severity_median` is never a feature (it separates the label perfectly). `load()` in step 05 aborts if any
feature has AUC > 0.995 or < 0.005 against the label.

**Splits.**
- `project`: StratifiedGroupKFold grouped by repository (main result)
- `org`: leave-one-organisation-out for the 6 largest organisations (apache alone is ~58% of the data)
- `random`: stratified K-fold, reference only, used to quantify the gap
- `project_dedup`: project split after removing rows with identical features and label

SMOTE runs inside the training fold only (imblearn `Pipeline`). Test folds keep the natural imbalance.

**Models.** Logistic regression, Random Forest, XGBoost, each with and without SMOTE, hyper-parameters fixed (no tuning).

**Metrics.** Macro-F1, Weighted-F1, Precision / Recall / F1 of the smelly class, MCC, PR-AUC, ROC-AUC, confusion matrices.
Computed per test fold, averaged over folds within a seed, then Mean ± SD over 5 seeds. In `org` the folds are fixed, so SD only
reflects model randomness.

**Statistics.** Paired Wilcoxon signed-rank on matched (seed, fold) results with Holm correction, within each family. Only
SMOTE-vs-none is inferential; "best vs others" rows are descriptive (the best configuration is picked on the same folds). Fold
results from different seeds re-use the same samples, so p-values are exploratory and effect sizes matter more. This is **not**
McNemar's test and must be described as such.

**Best configuration.** Chosen per smell by CV Macro-F1 on the same folds that are reported, so best-model scores are slightly
optimistic. Report the full table next to it.

### Dataset facts (checked on the real MLCQ file)

- 14,739 reviews → 9,517 (sample × smell) instances, 522 repositories, one commit per repository, 29 GitHub organisations.
- 74% of samples have a single reviewer.
- After the CK join: 9,367 labelled samples → **8,088 matched** (blob 1,964; data_class 1,993; feature_envy 2,043; long_method 2,088; match rate ≥ 99.6% of samples with CK output). Lost: 25 repositories that could not be fetched (mostly Eclipse), some CK runs without class output (e.g. SAP/SapMachine).
- Smelly samples used in the experiments (`dataset_audit_project.csv`): blob 214, data_class 268, feature_envy 59, long_method 219.
- 32–115 rows per task have identical features and label; `project_dedup` shows the effect of removing them.

### Main results (project-wise split, best configuration per smell, Mean ± SD over 5 seeds)

| Smell | Config | Macro-F1 | MCC | PR-AUC |
|---|---|---|---|---|
| long_method | XGBoost + SMOTE | 0.862 ± 0.005 | 0.726 | 0.828 |
| data_class | RF + SMOTE | 0.746 ± 0.009 | 0.498 | 0.626 |
| blob | RF + SMOTE | 0.722 ± 0.005 | 0.446 | 0.518 |
| feature_envy | LogReg | 0.579 ± 0.006 | 0.267 | 0.258 |

Random vs project split are nearly identical (e.g. data_class 0.753 vs 0.746), so the ~4.2-point drop reported in the earlier paper is
not reproduced, and no drop should be attributed to SMOTE. Full tables: `results/summary_<split>.csv`, `results/split_comparison.csv`.

## 5. Contribution 2: XSmell+ recommender (step 08)

For every sample the detector flags (out-of-fold P ≥ τ = 0.5):

1. **Evidence.** SHAP values are aggregated into semantic metric groups, keeping only the direction a refactoring can act on.
   CK direction: **LCOM\*: higher = LESS cohesive (bad). TCC/LCC: higher = MORE cohesive (good).**
2. **Feasibility F.** Structural preconditions of each candidate refactoring, checked on CK metrics.
3. **Impact I = p(x) − p(T_r(x)).** A what-if transformation of the metrics (ρ = fraction of code affected). Model-based, not observed.
4. **Score** S(r|x) = F · (λE + (1 − λ)I) / (1 + κ(C_r − 1)), with E = SHAP evidence share and C_r = ordinal cost 1–3 (a convention, not a measurement).
5. **Selection.** Top-k set without conflicting refactorings (exact search, ≤ 5 candidates).
6. **Abstention** when the detector is weak (cross-validated MCC < `mcc_min` = 0.30, currently Feature Envy) or P < τ.

Defaults: ρ = 0.5, λ = 0.5, κ = 0.25; `--sensitivity` repeats with ρ ∈ {0.3, 0.5, 0.7}, λ ∈ {0.25, 0.5, 0.75}.

**Ablation configurations** (`recommender_ablation.csv`): `xsmell_full`, `xsmell_E` (evidence only), `xsmell_noF` (no feasibility),
`xsmell_noDir` (no direction), `rule_only`, `shap_only`.

**How to read step 08 (important)**
- `feasible_top1` is 1.0 for XSmell+ **by construction** (preconditions are part of the score). Compare `coverage` (share of flagged
  samples that get a recommendation) and the baselines instead; rule-only and SHAP-only often recommend something whose
  preconditions fail (e.g. blob: 0.80 and 0.86 vs 1.00).
- `sim_dp_top1` is a what-if estimate. A static rule can match or beat XSmell+ on it (blob, ρ = 0.5: rule_only 0.098 vs full 0.055). It grows with ρ by construction, so only the order of configurations and trends are meaningful.
- Abstained smells must not be presented as working recommendations.
- SHAP fidelity (step 09) shows the explanation follows the model, not that the recommended refactoring is correct. For long_method local SHAP is about as faithful as global importance; do not claim it is superior.
- Explanation text in `recommender_examples.txt` is illustrative and has not been evaluated.

## 6. Contribution 3: before/after refactoring case study

### 6.1 Workflow

`11a_select_cases.py` → `apply_case.py` (per case) → `11_before_after.py` → `13_mi_metrics.py` → `14_sonar_debt.py prepare/collect` → `11_before_after.py` again.

**Selection (11a).** Candidates are samples the detector flagged and reviewers also labelled smelly, where the recommender did not abstain,
p(smelly) ≥ 0.6, the element is small enough to refactor by hand, at most 2 per repository, Feature Envy excluded; ranked by
p(smelly) then size, with a fixed-seed shuffle to break ties. Cases that cannot be built are kept with `status = skipped` and a reason.

**Cases done (5).** `blob_15` (hadoop-common, Extract Class), `long_method_04` (aries-jax-rs-whiteboard), `long_method_06`,
`long_method_19` (lucene-solr), `long_method_09` (cloudstack), all Extract Method. Patches are in `patches/`; commits live on branches
`xsmell-<case_id>` of the fork `thzynu`; `cases_todo.csv` records `repo_url_after`, `commit_after`, `status`.
These four items are the **originals of Contribution 3: `cases_todo.csv`, `patches/`, the fork branches** (plus `results/`, `data/ck_ba/`, `sonar_in/`).

### 6.2 Apply a patch to your fork

```bash
git config --global user.name "NAME" && git config --global user.email "EMAIL"
python apply_case.py --case long_method_06 --user thzynu             # shallow-fetch commit_before, apply patch, commit, push
python apply_case.py --case long_method_06 --user thzynu --no-push   # dry run
```
Fork the upstream repo on GitHub first (lucene-solr only once for cases 06 and 19). The script writes `repo_url_after`, `commit_after` and `status` into `cases_todo.csv`.

### 6.3 Run the measurement

```bash
pip install javalang
python 11_before_after.py --self-test                                       # checks the comparison logic
python 11_before_after.py --cases cases_todo.csv --ck-jar tools/ck.jar --score-detector --run-tests
python 13_mi_metrics.py --cases cases_todo.csv --workdir work_ba            # run after 11 (reuses work_ba)
python 11_before_after.py --cases cases_todo.csv --ck-jar tools/ck.jar --score-detector --run-tests   # adds MI/debt columns
```
Java setup used: `java` = JDK 21, `JAVA_HOME` = JDK 8 for the Maven builds. CK on lucene-solr, cloudstack and hadoop-common takes minutes per case.
To force a full re-download, delete `data/ck_ba/` **and** `work_ba/` together (see 6.6).

**Maintainability Index** (`13_mi_metrics.py`, Visual Studio variant, 0–100, higher is better):
`MI = max(0, (171 − 5.2·ln V − 0.23·G − 16.2·ln LOC) · 100/171)`, with Halstead volume V from the Java token stream, G = 1 + #(if, for, while,
case, catch, &&, ||, ?), LOC = lines with code. It is an approximation, not the value of a commercial tool. `xs_quality.py` uses CK WMC and LOC
instead of G and LOC counted from tokens; state which definition produced the reported numbers. Method-level cases report the MI of the method plus `mi_class` (LOC-weighted MI of the containing class).

### 6.4 Technical debt with SonarQube (step 14)

SonarQube reports debt per file, not per method. The script analyses only the one Java file of each case (before and after) as its own tiny project and sums
the remediation minutes of CODE_SMELL issues whose line lies inside the refactored element (1 day = 8 h).

> **Do not subscribe to a paid plan.** SonarQube Cloud may offer a 14-day Team trial that later bills. Choose **Free** (public projects) or use Option B.

**Option A, SonarQube Cloud (Free):** sign in at sonarcloud.io with GitHub, create an organization (note its key), generate a token
(Account > Security), install SonarScanner CLI (`sonar-scanner --version` in a new window).
```bat
copy cases_todo.csv cases_todo_backup.csv
python 14_sonar_debt.py prepare --cases cases_todo.csv --org YOUR_ORG_KEY
set SONAR_TOKEN=PASTE_TOKEN_HERE
sonar_in\run_all.bat
python 14_sonar_debt.py collect --cases cases_todo.csv --org YOUR_ORG_KEY
python 11_before_after.py --cases cases_todo.csv --ck-jar tools\ck.jar --score-detector --run-tests
```
**Option B, local SonarQube Community Build (free, no account):** unzip to a short path (e.g. `C:\sonarqube`), Java 17+, do **not** run as Administrator,
start `bin\windows-x86-64\StartSonar.bat`, open http://localhost:9000 (admin/admin), generate a token, then use the same commands without `--org` and with
`--host http://localhost:9000` on both `prepare` and `collect`.

Troubleshooting: "project not found" → create the project by hand with exactly the `sonar.projectKey` in that folder's `sonar-project.properties`;
scanner needs Java 17+ (use a JDK 17/21 `JAVA_HOME` in that window, not JDK 8); if `collect` prints `WARNING whole-file sum != sqale_index`, check that line.

### 6.5 Results on the 5 cases

| Level | n | WMC | LOC | MI | Detector P (before → after) |
|---|---|---|---|---|---|
| class (Extract Class) | 1 | 13 → 5 | 91 → 55 | 46.5 → 51.5 | p_blob 0.98 → 0.31 |
| method (Extract Method) | 4 | 10.5 → 5.25 | 33 → 17.75 | 38.8 → 47.8 | p_long_method 0.75 → 0.56 |

Verdicts: 4 improved, 1 neutral (`long_method_09`, 5 → 5 LOC, WMC unchanged). CBO is unchanged for the class case. Technical debt is 0 min before and after in every case.
Full per-case table: `results/before_after.csv`; summary by level: `results/before_after_summary.csv`.

### 6.6 Cleaning disk space

| Folder | Safe to delete? |
|---|---|
| `work_cases/`, `work_cases2/` | Yes. Temporary checkouts of `apply_case.py`; commits are on GitHub and in `cases_todo.csv`. Only re-created if you re-run `apply_case.py`. |
| `work_ba/` | Yes if you accept re-downloading. Before/after source used by 11, 13, 14. Re-running 11 with `--run-tests` while keeping `data/ck_ba/` fails (CK is cached, sources are gone): delete `data/ck_ba/` too, then expect 20–30 min for the large repos. |

Back up first: `results/`, `data/ck_ba/`, `sonar_in/`, `cases_todo.csv`, `patches/`. These are enough to rebuild every table of Contribution 3.

## 7. Outputs

- `data/dataset_class.csv`, `data/dataset_method.csv`, `data/join_report.csv`, `data/labels.csv`, `data/ck_raw/_run_log.csv`
- `results/per_seed_<split>.csv`, `per_fold_<split>.csv`, `summary_<split>.csv`, `confusion_<split>.csv`, `dataset_audit_<split>.csv` (`<split>` = project, org, random, project_dedup)
- `results/stats_tests.csv`, `split_comparison.csv`, `org_breakdown.csv`
- `results/recommendations.csv`, `recommender_ablation.csv`, `recommender_sensitivity.csv`, `recommender_examples.txt`
- `results/fidelity.csv`, `fidelity_summary.csv`, `fidelity_aopc.csv`; `results/cost.csv`
- `results/before_after.csv`, `before_after_summary.csv`, `test_logs/`; `results/sonar_file_metrics.csv`
- `results/cache/` — out-of-fold predictions and SHAP values (do not delete)
- `figures/*.png`, `figures/figure_manifest.csv` (model, seed and sample behind each figure), `shap_importance.csv`, `shap_stability.csv`

Figures are generated from code (replacing the four incorrect figures of the conference paper): `fig1_pipeline`, `fig2_f1_per_smell`,
`fig4_shap_summary_*`, `fig5_shap_waterfall_*`, plus `fig_class_distribution`, `fig_split_comparison`, `fig_confusion_matrices`,
`fig_correlation_heatmap`, `fig_ablation_feasibility`, `fig_ablation_impact`, `fig_fidelity_curves`.

## Things to state in the paper (Section 4.2)

- MLCQ is the dataset; labels come from its reviewers (cite Madeyski & Lewowski, EASE 2020).
- Your own work: source retrieval at the labelled commit, CK metric extraction, the label aggregation rule (majority, ties dropped), joining, project-wise splits.
- Samples lost (repo deleted, CK failure, unmatched) taken from the logs above.
- CK direction: LCOM\* higher = less cohesive; TCC/LCC higher = more cohesive. Keep this direction in the SHAP-to-refactoring rules.

## Limitations to report

1. **Feature Envy** has 59 positives; RF/XGBoost recall ≈ 0, only logistic regression shows weak signal (MCC ≈ 0.27). Do not draw conclusions from its SHAP plots or present its recommendations as working.
2. **No "after refactoring" in MLCQ.** SHAP explains the model's prediction; it is not evidence that a refactoring removes the smell.
3. **Recommendation quality is not shown to beat baselines.** Simulated impact favours the static rule in some settings; feasibility is 1.0 by construction. The 5 cases do not discriminate XSmell+ from rule-only or SHAP-only: the four Extract Method cases match what a static rule would also output, and for `blob_15` the algorithm recommends Move Method while Extract Class was applied.
4. **Case study is thin.** 5 cases, refactored by the author (not blinded); data_class has 0 cases; LCOM\* and TCC exist for one class-level case only; Extract Method moves complexity rather than removing it (file-level SonarQube Cognitive Complexity stays about the same, Complexity rises by 1).
5. **Technical debt is 0 everywhere** (profile "Sonar way", single file, no libraries), so it does not discriminate. Single-file scans also cannot compare the Extract Class case (code moved to a new file); use CK class-level metrics there. Absolute minutes are not those of a full-project scan. Report the date and ruleset.
6. **Tests.** Only `long_method_04` has a test run (before and after, see `results/test_logs/`); it does not cover the edited method, so it shows the build still works, not behaviour preservation. The other four repos build with ant/ivy or are too large. A test that fails *before* the refactoring too is a build/Java problem, not evidence.
7. **Structural validity** of recommendations is checked as conditions on CK metrics, not by AST analysis or a refactoring engine. Change cost is an ordinal 1–3 convention.
8. **Splits and tuning.** Hyper-parameters are fixed; no cross-dataset evaluation; the best configuration is selected on the reported folds. In `org` the SD reflects model randomness only.
9. **Comparators.** Only LogReg, RF and XGBoost are run. A CNN-on-AST baseline or a user study (e.g. 30 developers) has no data here; cite other work as non-comparable or omit.
10. **Patches** were checked to apply cleanly on the original commit with valid Java syntax; apart from `long_method_04`, they were not compiled (missing dependencies).
