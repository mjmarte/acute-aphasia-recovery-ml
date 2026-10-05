#!/usr/bin/env python3
"""Runs one algorithm (Ridge, SVR, random forest, XGBoost) on one feature set (FS1 to FS4) with the nested cross-validation."""
import sys
import argparse
import time
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    FEATURE_SETS, CANDIDATE_MODELS, CV_SETTINGS,
    ID_COL, LESION_VOLUME_COL,
    WAB_AQ_COL_12M, WAB_AQ_THRESHOLD, WAB_AQ_DIRECTION,
    CATEGORICAL_ENCODINGS,
    EXPERIMENTS_DIR, MASTER_CSV,
)
from feature_eng import compute_density_weights
from cv_engine import (
    CVConfig, RFEConfig, ModelConfig, FeatureSetConfig,
    get_default_models,
    run_nested_cv, print_results_table, save_results,
    compute_metrics, add_bootstrap_cis, add_conformal_intervals,
    add_posthoc_classification,
    compare_models_wilcoxon, compare_models_hotelling_williams,
    run_shap_stability,
)

def build_feature_sets(df):
    configs = {}
    for fs_key, fs_def in FEATURE_SETS.items():
        present = [c for c in fs_def["features"] if c in df.columns]
        missing = [c for c in fs_def["features"] if c not in df.columns]
        if missing:
            import warnings
            warnings.warn(f"{fs_key}: {len(missing)} columns missing from CSV: {missing}")

        resid_set   = set(fs_def["residualize"])
        skip_resid  = [c for c in present if c not in resid_set]

        configs[fs_key] = FeatureSetConfig(
            name=fs_def["name"],
            columns=present,
            skip_residualize=skip_resid,
        )
    return configs

def get_models(random_state=42):
    models = {
        "ridge": ModelConfig(
            name="ridge",
            param_grid={
                "alpha": [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
            }),
        "svr": ModelConfig(
            name="svr",
            param_grid={
                "kernel": ["rbf"],
                "C": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
                "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
                "epsilon": [0.01, 0.1, 0.5, 1.0, 2.0],
            }),
        "rf": ModelConfig(
            name="rf",
            param_grid={
                "n_estimators": [100, 300, 500],
                "max_depth": [3, 5, 8, 12, None],
                "min_samples_leaf": [1, 2, 5, 10],
                "max_features": ["sqrt", "log2", 0.3, 0.5, 0.7, 1.0],
            }),
    }
    try:
        from xgboost import XGBRegressor
        models["xgb"] = ModelConfig(
            name="xgb",
            param_grid={
                "n_estimators": [100, 300],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.05, 0.1],
                "subsample": [0.7, 0.9],
                "colsample_bytree": [0.5, 0.7, 1.0],
                "reg_alpha": [0, 0.1],
                "reg_lambda": [1.0],
            })
    except ImportError:
        pass
    return models

def load_and_prepare():
    df = pd.read_csv(MASTER_CSV)

    for col, mapping in CATEGORICAL_ENCODINGS.items():
        if col not in df.columns:
            continue
        if col == "race":
            df[col] = df[col].map(lambda v: mapping.get(v, 1))
        else:
            df[col] = df[col].map(mapping)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    y_raw = pd.to_numeric(df[WAB_AQ_COL_12M], errors="coerce").values
    vol   = pd.to_numeric(df[LESION_VOLUME_COL], errors="coerce").values

    valid = ~np.isnan(y_raw) & ~np.isnan(vol)
    df    = df[valid].reset_index(drop=True)
    y     = y_raw[valid]
    vol   = vol[valid]

    subject_ids = df[ID_COL].tolist()

    feature_sets = build_feature_sets(df)

    all_feature_names = []
    for fs in feature_sets.values():
        for c in fs.columns:
            if c not in all_feature_names:
                all_feature_names.append(c)

    X_raw   = df[all_feature_names].apply(pd.to_numeric, errors="coerce").values
    weights, _ = compute_density_weights(y)

    outcome_metadata = {
        "outcome_col":           WAB_AQ_COL_12M,
        "threshold":             WAB_AQ_THRESHOLD,
        "threshold_direction":   WAB_AQ_DIRECTION,
        "outcome_description":   "WAB AQ >= 93.8 at 12m (aphasia resolution)",
    }

    return {
        "X_raw": X_raw, "y": y, "vol": vol, "n": int(valid.sum()),
        "weights": weights, "all_feature_names": all_feature_names,
        "feature_sets": feature_sets, "df": df,
        "subject_ids": subject_ids,
        "outcome_metadata": outcome_metadata,
    }

def extract_fs_data(prepared, fs_name):
    fs_config = prepared["feature_sets"][fs_name]
    all_names = prepared["all_feature_names"]

    col_idx = [all_names.index(c) for c in fs_config.columns if c in all_names]
    X_fs = prepared["X_raw"][:, col_idx]
    fs_names = [all_names[i] for i in col_idx]

    nan_cols = np.isnan(X_fs).all(axis=0)
    if nan_cols.any():
        keep = ~nan_cols
        X_fs = X_fs[:, keep]
        fs_names = [f for f, k in zip(fs_names, keep) if k]

    return X_fs, fs_names, fs_config

def enumerate_tasks(feature_sets, model_configs):
    return [(fs, m) for fs in sorted(feature_sets) for m in sorted(model_configs)]

def run_single_task(prepared, fs_name, model_name, cv_config, rfe_config,
                    model_configs, bootstrap=True, verbose=1):
    X_fs, fs_names, fs_config = extract_fs_data(prepared, fs_name)
    if not fs_names:
        return None

    model_config = model_configs[model_name]
    fs_rfe = RFEConfig(
        enabled=rfe_config.enabled and len(fs_names) > 5,
        min_features=rfe_config.min_features,
        step=rfe_config.step,
        scoring=rfe_config.scoring,
        n_inner_cv=rfe_config.n_inner_cv,
        selector_model=rfe_config.selector_model,
    )

    result = run_nested_cv(
        X_fs, prepared["y"], fs_names, model_config, cv_config,
        rfe_config=fs_rfe, vol=prepared["vol"], residualize=True,
        skip_residualize=fs_config.skip_residualize,
        weights=prepared["weights"], verbose=verbose,
        subject_ids=prepared["subject_ids"],
        outcome_metadata=prepared["outcome_metadata"],
    )
    result["feature_set"] = fs_name

    if bootstrap:
        add_bootstrap_cis(result, n_bootstrap=2000, ci=95,
                          random_state=cv_config.random_state)
        add_conformal_intervals(result)

    om = prepared["outcome_metadata"]
    add_posthoc_classification(result, om["threshold"], om["threshold_direction"])

    return result

def main():
    parser = argparse.ArgumentParser(
        description="Nested cross-validation")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true",
                      help="Run all tasks locally")
    mode.add_argument("--task", type=int, default=None,
                      help="Run one task by ID")
    mode.add_argument("--list-tasks", action="store_true",
                      help="Print task list and exit")

    parser.add_argument("--quick", action="store_true",
                        help="3×5 CV, small grids (debugging)")
    parser.add_argument("--no-bootstrap", action="store_true")
    parser.add_argument("--shap", action="store_true",
                        help="SHAP stability on best model")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Subset of models (e.g. --models ridge rf xgb)")
    parser.add_argument("--feature-sets", nargs="+", default=None,
                        help="Subset of feature sets (e.g. --feature-sets FS1_clinical FS6_full_apriori)")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.quick:
        cv_config = CVConfig(n_outer_repeats=3, n_outer_folds=5,
                             n_inner_folds=3, stratify_outcome=True)
        rfe_config = RFEConfig(enabled=True, min_features=1, step=0.3,
                               n_inner_cv=3, selector_model="rf")
        model_configs = get_default_models(42)
        mode_label = "QUICK (3×5 outer, 3 inner)"
    else:
        cv_config = CVConfig(n_outer_repeats=10, n_outer_folds=10,
                             n_inner_folds=10, stratify_outcome=True)
        rfe_config = RFEConfig(enabled=True, min_features=1, step=0.1,
                               scoring="neg_mean_absolute_error",
                               n_inner_cv=10, selector_model="rf")
        model_configs = get_models(42)
        mode_label = "10 x 10 outer folds, 10 inner folds"

    if args.models:
        model_configs = {k: v for k, v in model_configs.items()
                         if k in args.models}

    prepared = load_and_prepare()
    feature_sets = prepared["feature_sets"]
    if args.feature_sets:
        feature_sets = {k: v for k, v in feature_sets.items()
                        if k in args.feature_sets}
        prepared["feature_sets"] = feature_sets

    tasks = enumerate_tasks(feature_sets, model_configs)

    if args.list_tasks:
        print(f"\n  TASK LIST ({len(tasks)} tasks)")
        print(f"  {'ID':>4s}  {'Feature Set':<35s}  {'Model':<12s}  Features")
        print(f"  {'-'*4}  {'-'*35}  {'-'*12}  {'-'*8}")
        for i, (fs, m) in enumerate(tasks):
            n_feat = len(feature_sets[fs].columns)
            print(f"  {i:4d}  {fs:<35s}  {m:<12s}  {n_feat}")
        return

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = EXPERIMENTS_DIR / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    n_models = len(model_configs)
    n_fs = len(feature_sets)
    print("=" * 70)
    print("  NESTED CROSS-VALIDATION")
    print("=" * 70)
    print(f"  Mode:   {mode_label}")
    print(f"  N:      {prepared['n']} subjects")
    print(f"  Models: {list(model_configs.keys())}")
    print(f"  FS:     {list(feature_sets.keys())}")
    print(f"  Tasks:  {len(tasks)} ({n_fs} feature sets × {n_models} models)")
    print(f"  CV:     {cv_config.n_outer_repeats}×{cv_config.n_outer_folds} "
          f"outer, {cv_config.n_inner_folds} inner "
          f"= {cv_config.total_outer_folds} outer folds")
    print(f"  RFE:    step={rfe_config.step}, inner={rfe_config.n_inner_cv}")
    print(f"  Output: {output_dir}")
    print("=" * 70)

    config_record = {
        "timestamp": timestamp,
        "mode": "quick" if args.quick else "full",
        "n_samples": prepared["n"],
        "n_features_total": len(prepared["all_feature_names"]),
        "outcome_col": WAB_AQ_COL_12M,
        "threshold": WAB_AQ_THRESHOLD,
        "threshold_direction": WAB_AQ_DIRECTION,
        "cv_config": {
            "n_outer_repeats": cv_config.n_outer_repeats,
            "n_outer_folds": cv_config.n_outer_folds,
            "n_inner_folds": cv_config.n_inner_folds,
            "total_outer_folds": cv_config.total_outer_folds,
            "stratify": cv_config.stratify_outcome,
        },
        "rfe_config": {
            "enabled": rfe_config.enabled,
            "step": rfe_config.step,
            "min_features": rfe_config.min_features,
            "n_inner_cv": rfe_config.n_inner_cv,
        },
        "models": {n: {"grid_size": len(mc.param_grid)}
                   for n, mc in model_configs.items()},
        "feature_sets": {k: {
            "n_features": len(v.columns),
            "columns": v.columns,
        } for k, v in feature_sets.items()},
        "tasks": tasks,
        "n_tasks": len(tasks),
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_record, f, indent=2, default=str)

    if args.task is not None:
        task_id = args.task
        if task_id < 0 or task_id >= len(tasks):
            print(f"  ERROR: task {task_id} out of range [0, {len(tasks)-1}]")
            sys.exit(1)

        fs_name, model_name = tasks[task_id]
        print(f"\n  Task {task_id}: {fs_name} × {model_name}")

        t0 = time.time()
        result = run_single_task(
            prepared, fs_name, model_name,
            cv_config, rfe_config, model_configs,
            bootstrap=not args.no_bootstrap, verbose=args.verbose)
        elapsed = time.time() - t0

        if result is None:
            sys.exit(0)

        m = result["overall_metrics"]
        ms = result["metrics_summary"]
        print(f"\n  Result: MAE={m['mae']:.2f}±{ms.get('mae',{}).get('std',0):.2f}  "
              f"R²={m['r2']:.3f}  r={m['pearson_r']:.3f}  "
              f"time={elapsed:.0f}s")
        if "bootstrap_ci" in result:
            for k in ["mae", "r2", "pearson_r"]:
                if k in result["bootstrap_ci"]:
                    lo, hi = result["bootstrap_ci"][k]
                    print(f"  {k:>10s} 95% CI: [{lo:.3f}, {hi:.3f}]")
        if "conformal_summary" in result:
            cs = result["conformal_summary"]
            print(f"  Conformal 90% PI: ±{cs.get('interval_width_90', 0):.1f} "
                  f"(empirical coverage: {cs.get('coverage_90', 0):.1%})")

        result_save = {k: v for k, v in result.items() if k != "fold_results"}
        result_save["fold_summary"] = [
            {"repeat": fr.repeat, "fold": fr.fold,
             "n_features_selected": fr.n_features_selected,
             "selected_features": fr.selected_features,
             "best_params": fr.best_params,
             "train_time_s": fr.train_time_s}
            for fr in result["fold_results"]
        ]
        pkl_file = output_dir / f"task_{task_id:03d}_{fs_name}_{model_name}.pkl"
        with open(pkl_file, "wb") as f:
            pickle.dump(result_save, f)

        summary = {
            "task_id": task_id, "feature_set": fs_name, "model": model_name,
            "n_features": result["n_features_input"],
            "n_samples": result["n_samples"],
            "metrics": {k: float(v) for k, v in m.items()
                        if isinstance(v, (int, float, np.floating))},
            "metrics_std": {k: float(v.get("std", 0))
                           for k, v in ms.items() if isinstance(v, dict)},
            "bootstrap_ci": {k: [float(v[0]), float(v[1])]
                             for k, v in result.get("bootstrap_ci", {}).items()
                             if not (np.isnan(v[0]) or np.isnan(v[1]))},
            "classification_metrics": result.get("classification_metrics", {}),
            "calibration": result.get("calibration", {}),
            "classification_bootstrap_ci": {
                k: [float(v[0]), float(v[1])]
                for k, v in result.get("classification_bootstrap_ci", {}).items()
                if not (np.isnan(v[0]) or np.isnan(v[1]))
            },
            "conformal": {k: float(v)
                          for k, v in result.get("conformal_summary", {}).items()
                          if isinstance(v, (int, float, np.floating))
                          and not np.isnan(v)},
            "feature_stability_top10": sorted(
                result.get("feature_stability", {}).items(),
                key=lambda x: x[1], reverse=True)[:10],
            "elapsed_s": elapsed,
        }
        json_file = output_dir / f"task_{task_id:03d}_{fs_name}_{model_name}.json"
        with open(json_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"  Saved: {pkl_file.name}, {json_file.name}")
        return

    if not args.full:
        print("\n  Specify --full or --task N")
        print("  Use --list-tasks to see available tasks")
        sys.exit(1)

    t0_total = time.time()
    all_cv_results = {}

    for i, (fs_name, model_name) in enumerate(tasks):
        print(f"\n  [{i+1}/{len(tasks)}] {fs_name} × {model_name}")
        result = run_single_task(
            prepared, fs_name, model_name,
            cv_config, rfe_config, model_configs,
            bootstrap=not args.no_bootstrap, verbose=args.verbose)
        if result:
            all_cv_results[(fs_name, model_name)] = result

    print_results_table(all_cv_results,
                        title="NESTED CV RESULTS (ALL MODELS × FEATURE SETS)")

    print("\n  Best model per feature set:")
    for fs_name in feature_sets:
        fs_results = [(k, r) for k, r in all_cv_results.items() if k[0] == fs_name]
        if not fs_results:
            continue
        best_key = min(fs_results, key=lambda x: x[1]["overall_metrics"]["mae"])[0]
        m = all_cv_results[best_key]["overall_metrics"]
        print(f"    {fs_name}: {best_key[1]} "
              f"(MAE={m['mae']:.2f}, R²={m['r2']:.3f})")

    comparison_results = []
    for fs_name in feature_sets:
        model_list = sorted(model_configs)
        for i, m1 in enumerate(model_list):
            for m2 in model_list[i+1:]:
                k1, k2 = (fs_name, m1), (fs_name, m2)
                if k1 in all_cv_results and k2 in all_cv_results:
                    w = compare_models_wilcoxon(all_cv_results[k1],
                                                all_cv_results[k2])
                    hw = compare_models_hotelling_williams(
                        all_cv_results[k1], all_cv_results[k2], prepared["y"])
                    comparison_results.append({
                        "feature_set": fs_name, "model_a": m1, "model_b": m2,
                        "mae_a": w.get("mean_a"), "mae_b": w.get("mean_b"),
                        "wilcoxon_p": w["p_value"],
                        "effect_r": w.get("effect_size_r"),
                        "hw_p": hw["p_value"], "hw_t": hw["statistic"],
                    })

    shap_results = None
    if args.shap and all_cv_results:
        best_key = min(all_cv_results,
                       key=lambda k: all_cv_results[k]["overall_metrics"]["mae"])
        fs_n, mod_n = best_key
        print(f"\n  SHAP stability: {mod_n} @ {fs_n}...")
        X_fs, fs_names, fs_cfg = extract_fs_data(prepared, fs_n)
        shap_results = run_shap_stability(
            X_fs, prepared["y"], fs_names, model_configs[mod_n], cv_config,
            vol=prepared["vol"], residualize=True,
            skip_residualize=fs_cfg.skip_residualize,
            n_folds_for_shap=10, random_state=42)

    save_results({"cv_results": all_cv_results,
                  "comparisons": comparison_results, "shap": shap_results},
                 output_dir)

    if comparison_results:
        pd.DataFrame(comparison_results).to_csv(
            output_dir / "run_model_comparisons.csv", index=False)

    if shap_results:
        pd.DataFrame({
            "feature": shap_results["feature_names"],
            "mean_abs_shap": shap_results["mean_abs_shap"],
        }).sort_values("mean_abs_shap", ascending=False).to_csv(
            output_dir / "shap_stability.csv", index=False)

    total = time.time() - t0_total
    print(f"\n  COMPLETE — {total/3600:.1f}h ({total:.0f}s)")
    print(f"  Results: {output_dir}")

if __name__ == "__main__":
    main()
