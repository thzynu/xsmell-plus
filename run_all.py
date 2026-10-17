"""Run the whole analysis once:  python run_all.py   (add --quick for a ~5-minute smoke test)

Prerequisite: data/dataset_class.csv and data/dataset_method.csv already exist
(steps 01-03: labels -> CK metrics -> dataset; see README).  Steps 05-10, 08b and 12 are what this script runs (11, 13, 14 need the refactoring cases and SonarQube; run them separately, see README).
"""
import argparse, os, subprocess, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true", help="1 seed, 3 folds, 2 models (smoke test)")
ap.add_argument("--skip-05", action="store_true", help="reuse results/per_*_<split>.csv from an earlier step-05 run (the slow part)")
a = ap.parse_args()

for f in ["data/dataset_class.csv", "data/dataset_method.csv"]:
    if not os.path.exists(f):
        sys.exit(f"Missing {f}. Run steps 01-03 first (see README).")

extra = ["--seeds", "1", "--folds", "3", "--models", "logreg,rf"] if a.quick else []
quick08 = ["--seeds", "1"] if a.quick else []
steps = []
if not a.skip_05:
    steps += [
        ("project-wise split (main result)", ["05_run_experiments.py", "--split", "project"] + extra),
        ("leave-one-organisation-out", ["05_run_experiments.py", "--split", "org"] + extra),
        ("random split (for the gap)", ["05_run_experiments.py", "--split", "random"] + extra),
        ("project-wise, duplicates removed", ["05_run_experiments.py", "--split", "project", "--dedup"] + extra),
    ]
steps += [
    ("statistical tests + split comparison + organisation breakdown", ["07_stats_tests.py"]),
    ("figures + SHAP", ["06_make_figures.py"]),
    ("XSmell+ recommender + ablation + sensitivity (rho, lambda)", ["08_recommender.py", "--sensitivity"] + quick08),
    ("sensitivity report", ["08b_sensitivity_report.py"]),
    ("SHAP fidelity", ["09_fidelity.py"] + quick08),
    ("computational cost", ["10_cost.py"]),
    ("paper figures (pipeline, ablation, fidelity)", ["12_paper_figures.py"]),
]
t0 = time.time()
for name, cmd in steps:
    print(f"\n######## {name}", flush=True)
    r = subprocess.run([sys.executable, "-u"] + cmd)
    if r.returncode != 0:
        sys.exit(f"FAILED at step '{name}' (exit {r.returncode}). Fix it and re-run.")
print(f"\nAll done in {(time.time() - t0) / 60:.1f} min. See results/ and figures/.")
