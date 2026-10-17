"""Step 10 - Computational cost (Section 6.4): training, inference, SHAP and recommendation latency.

Usage: python 10_cost.py [--reps 5]      Output: results/cost.csv (+ hardware line)
Timings are wall-clock on the machine that runs the script; report the hardware printed at the end.
peak_python_mem_MB comes from tracemalloc (Python/numpy allocations only; native tree memory is NOT counted),
so it under-estimates real memory: report it as a lower bound or measure RSS with an external tool.
"""
import argparse, os, platform, time, tracemalloc
import numpy as np
import pandas as pd
import xs_common as xc
import importlib.util

spec = importlib.util.spec_from_file_location("rec", "08_recommender.py")
rec = importlib.util.module_from_spec(spec); spec.loader.exec_module(rec)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    cfgs = xc.best_configs()
    rows = []
    for smell, cfg in cfgs.items():
        df, feats, *_ = xc.exp.load(smell, "data")
        X = df[feats].values.astype(float); y = df["label"].astype(int).values
        fit_t, peak = [], []
        for r in range(a.reps):
            m = xc.exp.make_model(cfg["model"], r, cfg["smote"])
            tracemalloc.start(); t = time.perf_counter(); m.fit(X, y)
            fit_t.append(time.perf_counter() - t); peak.append(tracemalloc.get_traced_memory()[1] / 2**20); tracemalloc.stop()
        n_ev = min(500, len(X))
        t = time.perf_counter(); m.predict_proba(X[:n_ev]); pred_ms = (time.perf_counter() - t) / n_ev * 1000
        t = time.perf_counter(); V, _ = xc.explain(m, cfg["model"], X, X[:n_ev]); shap_ms = (time.perf_counter() - t) / n_ev * 1000
        res = xc.oof_explain(smell, 0)
        t = time.perf_counter(); rec.run_smell(res, 0.5, 0.5, 0.25, 0.5, 3, 0.3)
        rec_ms = (time.perf_counter() - t) / len(res["X"]) * 1000
        rows.append(dict(smell=smell, config=f"{cfg['model']}{'+SMOTE' if cfg['smote'] else ''}", n_train=len(X),
                         n_features=len(feats), fit_s_mean=round(np.mean(fit_t), 2), fit_s_sd=round(np.std(fit_t), 2),
                         peak_python_mem_MB=round(np.mean(peak), 1), predict_ms_per_sample=round(pred_ms, 3),
                         shap_ms_per_sample=round(shap_ms, 2), recommend_ms_per_sample=round(rec_ms, 2)))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv("results/cost.csv", index=False)
    print(f"\nHardware: {platform.processor() or platform.machine()}, {os.cpu_count()} logical CPUs, "
          f"Python {platform.python_version()}, {platform.system()}")
    print("Note: recommend_ms includes the what-if re-predictions for every catalogue entry (batch over all samples).")


if __name__ == "__main__":
    main()
