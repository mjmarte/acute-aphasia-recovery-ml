"""Nested cross-validation: 10-fold outer loop repeated 10 times (stratified on outcome quartiles), 10-fold inner grid search,
recursive feature elimination inside each training fold (FS2 to FS4), preprocessing fit on training folds only, bootstrap
intervals (2,000 resamples) and post-hoc classification at the clinical threshold."""
import warnings
import time
import json
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any

from sklearn.linear_model import Ridge, LinearRegression, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.model_selection import (
    RepeatedStratifiedKFold, RepeatedKFold, StratifiedKFold,
    KFold, GridSearchCV, cross_val_score
)
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.feature_selection import RFECV, RFE
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    median_absolute_error, explained_variance_score
)
from sklearn.base import clone
from scipy import stats

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

@dataclass
class CVConfig:
    n_outer_repeats: int = 10
    n_outer_folds: int = 5
    n_inner_folds: int = 5
    stratify_outcome: bool = True
    stratify_n_bins: int = 4
    random_state: int = 42

    @property
    def total_outer_folds(self):
        return self.n_outer_repeats * self.n_outer_folds

@dataclass
class RFEConfig:
    enabled: bool = True
    min_features: int = 1
    step: float = 0.2
    scoring: str = "neg_mean_absolute_error"
    n_inner_cv: int = 5
    selector_model: str = "rf"

@dataclass
class PermutationConfig:
    n_permutations: int = 1000
    metric: str = "mae"
    random_state: int = 42

@dataclass
class ModelConfig:
    name: str = "ridge"
    param_grid: Dict = field(default_factory=dict)

    def get_estimator(self, random_state=42):
        if self.name == "ridge":
            return Ridge()
        elif self.name == "svr":
            return SVR()
        elif self.name == "elastic_net":
            return ElasticNet(max_iter=10000, random_state=random_state)
        elif self.name == "gpr":
            kernel = ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(1.0)
            return GaussianProcessRegressor(
                kernel=kernel, n_restarts_optimizer=5,
                random_state=random_state, normalize_y=True
            )
        elif self.name == "rf":
            return RandomForestRegressor(random_state=random_state, n_jobs=1)
        elif self.name == "xgb":
            if not HAS_XGB:
                raise ImportError("xgboost not installed")
            return XGBRegressor(random_state=random_state, verbosity=0, n_jobs=1)
        else:
            raise ValueError(f"Unknown model: {self.name}")

@dataclass
class FeatureSetConfig:
    name: str = ""
    columns: List[str] = field(default_factory=list)
    confound_col: Optional[str] = None
    skip_residualize: List[str] = field(default_factory=list)

@dataclass
class FoldResult:
    repeat: int = 0
    fold: int = 0
    train_idx: np.ndarray = field(default_factory=lambda: np.array([]))
    test_idx: np.ndarray = field(default_factory=lambda: np.array([]))
    y_true: np.ndarray = field(default_factory=lambda: np.array([]))
    y_pred: np.ndarray = field(default_factory=lambda: np.array([]))
    best_params: Dict = field(default_factory=dict)
    selected_features: List[str] = field(default_factory=list)
    n_features_selected: int = 0
    train_time_s: float = 0.0

def get_default_models(random_state=42):
    models = {
        "ridge": ModelConfig(
            name="ridge",
            param_grid={
                "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
            }
        ),
        "svr": ModelConfig(
            name="svr",
            param_grid={
                "kernel": ["rbf"],
                "C": [0.1, 1.0, 10.0, 100.0],
                "gamma": ["scale", "auto"],
                "epsilon": [0.1, 0.5, 1.0],
            }
        ),
        "rf": ModelConfig(
            name="rf",
            param_grid={
                "n_estimators": [100, 300],
                "max_depth": [3, 5, 8, None],
                "min_samples_leaf": [2, 5, 10],
                "max_features": ["sqrt", "log2", 0.5],
            }
        ),
    }
    if HAS_XGB:
        models["xgb"] = ModelConfig(
            name="xgb",
            param_grid={
                "n_estimators": [100, 300],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.05, 0.1],
                "subsample": [0.7, 0.9],
                "colsample_bytree": [0.5, 0.8, 1.0],
                "reg_alpha": [0, 0.1, 1.0],
                "reg_lambda": [1.0, 5.0],
            }
        )
    return models

def residualize_train_test(X_train, X_test, feature_names, vol_train, vol_test,
                           skip_cols=None):
    if skip_cols is None:
        skip_cols = set()

    _always_skip = {
        "wab_aq_acute", "age_at_stroke", "sex", "education_yrs",
        "race", "prior_stroke", "lesion_volume_mL",
        "Age", "Sex", "Edu", "LANG_comp_acute", "Hand",
        "Aphasic?", "Any SSRI 0-3m?", "SLP Tx session (0-3m)",
        "Days Post Testing (acute)", "tract_lesion_volume_mL",
        "brain_age_gap", "predicted_age",
    }
    skip_cols = set(skip_cols) | _always_skip

    X_tr = X_train.copy().astype(float)
    X_te = X_test.copy().astype(float)
    resid_r2 = {}

    for i, name in enumerate(feature_names):
        if name in skip_cols:
            resid_r2[name] = np.nan
            continue

        col_tr = X_tr[:, i]
        col_te = X_te[:, i]
        v_tr = vol_train.reshape(-1, 1)
        v_te = vol_test.reshape(-1, 1)

        valid_tr = ~(np.isnan(col_tr) | np.isnan(v_tr.ravel()))
        if valid_tr.sum() < 5:
            resid_r2[name] = np.nan
            continue

        reg = LinearRegression()
        reg.fit(v_tr[valid_tr], col_tr[valid_tr])

        pred_tr = reg.predict(v_tr[valid_tr])
        ss_res = np.sum((col_tr[valid_tr] - pred_tr) ** 2)
        ss_tot = np.sum((col_tr[valid_tr] - col_tr[valid_tr].mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        resid_r2[name] = r2

        X_tr[valid_tr, i] = col_tr[valid_tr] - pred_tr

        valid_te = ~(np.isnan(col_te) | np.isnan(v_te.ravel()))
        if valid_te.sum() > 0:
            pred_te = reg.predict(v_te[valid_te])
            X_te[valid_te, i] = col_te[valid_te] - pred_te

    return X_tr, X_te, resid_r2

def standardize_train_test(X_train, X_test):
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)
    return X_train_std, X_test_std, scaler

def preprocess_fold(X_train, X_test, feature_names, vol_train, vol_test,
                    residualize=True, skip_residualize=None):
    info = {}

    if residualize and vol_train is not None:
        X_train, X_test, resid_r2 = residualize_train_test(
            X_train, X_test, feature_names,
            vol_train, vol_test,
            skip_cols=skip_residualize
        )
        info["resid_r2"] = resid_r2

    train_medians = np.nanmedian(X_train, axis=0)
    for j in range(X_train.shape[1]):
        nan_tr = np.isnan(X_train[:, j])
        nan_te = np.isnan(X_test[:, j])
        if nan_tr.any():
            X_train[nan_tr, j] = train_medians[j]
        if nan_te.any():
            X_test[nan_te, j] = train_medians[j]

    X_train, X_test, scaler = standardize_train_test(X_train, X_test)
    info["scaler"] = scaler

    return X_train, X_test, info

def run_rfe_in_fold(X_train, y_train, feature_names, rfe_config, random_state=42):
    if not rfe_config.enabled or X_train.shape[1] <= rfe_config.min_features:
        return (np.ones(X_train.shape[1], dtype=bool),
                list(feature_names),
                np.ones(X_train.shape[1], dtype=int),
                X_train.shape[1])

    if rfe_config.selector_model == "rf":
        selector_est = RandomForestRegressor(
            n_estimators=200, max_depth=5, min_samples_leaf=5,
            random_state=random_state, n_jobs=1
        )
    elif rfe_config.selector_model == "ridge":
        selector_est = Ridge(alpha=1.0)
    else:
        selector_est = RandomForestRegressor(
            n_estimators=200, max_depth=5, min_samples_leaf=5,
            random_state=random_state, n_jobs=1
        )

    inner_cv = KFold(
        n_splits=min(rfe_config.n_inner_cv, len(y_train)),
        shuffle=True, random_state=random_state
    )

    rfecv = RFECV(
        estimator=selector_est,
        step=rfe_config.step,
        cv=inner_cv,
        scoring=rfe_config.scoring,
        min_features_to_select=rfe_config.min_features,
        n_jobs=1,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rfecv.fit(X_train, y_train)

    selected_mask = rfecv.support_
    selected_names = [f for f, s in zip(feature_names, selected_mask) if s]
    ranking = rfecv.ranking_
    n_optimal = rfecv.n_features_

    return selected_mask, selected_names, ranking, n_optimal

def compute_metrics(y_true, y_pred):
    if len(y_true) < 2:
        return {k: np.nan for k in [
            "mae", "rmse", "r2", "median_ae", "explained_var",
            "pearson_r", "pearson_p", "spearman_rho", "spearman_p",
            "max_error", "n"
        ]}

    rho, rho_p = stats.spearmanr(y_true, y_pred)
    r, r_p = stats.pearsonr(y_true, y_pred)

    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
        "median_ae": median_absolute_error(y_true, y_pred),
        "explained_var": explained_variance_score(y_true, y_pred),
        "pearson_r": r,
        "pearson_p": r_p,
        "spearman_rho": rho,
        "spearman_p": rho_p,
        "max_error": np.max(np.abs(y_true - y_pred)),
        "n": len(y_true),
    }

def _make_outer_cv(cv_config, y):
    if cv_config.stratify_outcome:
        n_bins = min(cv_config.stratify_n_bins, len(np.unique(y)))
        if n_bins < 2:
            n_bins = 2
        binner = KBinsDiscretizer(
            n_bins=n_bins, encode="ordinal", strategy="quantile"
        )
        y_binned = binner.fit_transform(y.reshape(-1, 1)).ravel().astype(int)

        cv = RepeatedStratifiedKFold(
            n_splits=cv_config.n_outer_folds,
            n_repeats=cv_config.n_outer_repeats,
            random_state=cv_config.random_state,
        )
        return cv, y_binned
    else:
        cv = RepeatedKFold(
            n_splits=cv_config.n_outer_folds,
            n_repeats=cv_config.n_outer_repeats,
            random_state=cv_config.random_state,
        )
        return cv, y

def run_nested_cv(X, y, feature_names, model_config, cv_config,
                  rfe_config=None, vol=None, residualize=True,
                  skip_residualize=None, weights=None, verbose=1,
                  subject_ids=None, outcome_metadata=None):
    n, p = X.shape
    if rfe_config is None:
        rfe_config = RFEConfig(enabled=False)

    outer_cv, y_for_split = _make_outer_cv(cv_config, y)

    fold_results = []
    y_preds_by_repeat = np.full((n, cv_config.n_outer_repeats), np.nan)
    feature_selection_counts = {f: 0 for f in feature_names}
    total_folds = 0

    t_start = time.time()
    fold_idx = 0

    for train_idx, test_idx in outer_cv.split(X, y_for_split):
        repeat = fold_idx // cv_config.n_outer_folds
        fold = fold_idx % cv_config.n_outer_folds
        fold_idx += 1
        total_folds += 1

        t_fold = time.time()

        X_train_raw = X[train_idx].copy()
        X_test_raw = X[test_idx].copy()
        y_train = y[train_idx].copy()
        y_test = y[test_idx].copy()
        w_train = weights[train_idx] if weights is not None else None

        vol_train = vol[train_idx] if vol is not None else None
        vol_test = vol[test_idx] if vol is not None else None

        X_train, X_test, preproc_info = preprocess_fold(
            X_train_raw, X_test_raw, feature_names,
            vol_train, vol_test,
            residualize=residualize,
            skip_residualize=skip_residualize,
        )

        current_features = list(feature_names)
        current_X_train = X_train
        current_X_test = X_test

        selected_names = list(feature_names)
        if rfe_config.enabled and p > rfe_config.min_features:
            sel_mask, selected_names, ranking, n_opt = run_rfe_in_fold(
                current_X_train, y_train, current_features,
                rfe_config, random_state=cv_config.random_state + fold_idx
            )
            current_X_train = current_X_train[:, sel_mask]
            current_X_test = current_X_test[:, sel_mask]

            for fname in selected_names:
                if fname in feature_selection_counts:
                    feature_selection_counts[fname] += 1

        estimator = model_config.get_estimator(cv_config.random_state)
        best_params = {}

        if model_config.param_grid:
            inner_cv = KFold(
                n_splits=min(cv_config.n_inner_folds, len(y_train)),
                shuffle=True,
                random_state=cv_config.random_state + fold_idx
            )
            gs = GridSearchCV(
                estimator, model_config.param_grid,
                cv=inner_cv,
                scoring="neg_mean_absolute_error",
                n_jobs=-1, refit=True,
                error_score=np.nan,
            )
            try:
                if w_train is not None:
                    try:
                        gs.fit(current_X_train, y_train, sample_weight=w_train)
                    except TypeError:
                        gs.fit(current_X_train, y_train)
                else:
                    gs.fit(current_X_train, y_train)
                best_params = gs.best_params_
                fitted_model = gs.best_estimator_
            except Exception as e:
                warnings.warn(f"GridSearch failed fold {fold_idx}: {e}")
                estimator.fit(current_X_train, y_train)
                fitted_model = estimator
        else:
            fitted_model = clone(estimator)
            try:
                if w_train is not None:
                    try:
                        fitted_model.fit(current_X_train, y_train,
                                         sample_weight=w_train)
                    except TypeError:
                        fitted_model.fit(current_X_train, y_train)
                else:
                    fitted_model.fit(current_X_train, y_train)
            except Exception as e:
                warnings.warn(f"Fit failed fold {fold_idx}: {e}")
                fitted_model.fit(current_X_train, y_train)

        y_pred_fold = fitted_model.predict(current_X_test)
        y_preds_by_repeat[test_idx, repeat] = y_pred_fold

        fold_time = time.time() - t_fold

        fr = FoldResult(
            repeat=repeat, fold=fold,
            train_idx=train_idx, test_idx=test_idx,
            y_true=y_test, y_pred=y_pred_fold,
            best_params=best_params,
            selected_features=selected_names,
            n_features_selected=len(selected_names),
            train_time_s=fold_time,
        )
        fold_results.append(fr)

        if verbose >= 2:
            mae = mean_absolute_error(y_test, y_pred_fold)
            print(f"  Repeat {repeat+1}, Fold {fold+1}: "
                  f"MAE={mae:.2f}, n_feat={len(selected_names)}, "
                  f"time={fold_time:.1f}s")

    total_time = time.time() - t_start

    metrics_per_repeat = []
    for rep in range(cv_config.n_outer_repeats):
        rep_mask = ~np.isnan(y_preds_by_repeat[:, rep])
        if rep_mask.sum() > 0:
            m = compute_metrics(y[rep_mask], y_preds_by_repeat[rep_mask, rep])
            m["repeat"] = rep
            metrics_per_repeat.append(m)

    metrics_summary = {}
    if metrics_per_repeat:
        metric_keys = [k for k in metrics_per_repeat[0] if k != "repeat"]
        for k in metric_keys:
            vals = [m[k] for m in metrics_per_repeat if not np.isnan(m[k])]
            if vals:
                metrics_summary[k] = {
                    "mean": np.mean(vals),
                    "std": np.std(vals),
                    "min": np.min(vals),
                    "max": np.max(vals),
                    "values": vals,
                }

    y_pred_mean = np.nanmean(y_preds_by_repeat, axis=1)
    never_tested = np.isnan(y_pred_mean)
    if never_tested.any():
        warnings.warn(f"{never_tested.sum()} subjects never in test set")

    overall_metrics = compute_metrics(
        y[~never_tested], y_pred_mean[~never_tested]
    )

    feature_stability = {
        f: count / total_folds
        for f, count in feature_selection_counts.items()
    } if rfe_config.enabled else {f: 1.0 for f in feature_names}

    config_str = json.dumps({
        "cv": asdict(cv_config),
        "rfe": asdict(rfe_config),
        "model": model_config.name,
        "n_features": p,
        "n_samples": n,
    }, sort_keys=True, default=str)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:12]

    if verbose >= 1:
        m = overall_metrics
        print(f"  {model_config.name:>6s} | "
              f"MAE={m['mae']:.2f}±{metrics_summary.get('mae',{}).get('std',0):.2f}  "
              f"R²={m['r2']:.3f}±{metrics_summary.get('r2',{}).get('std',0):.3f}  "
              f"r={m['pearson_r']:.3f}  ρ={m['spearman_rho']:.3f}  "
              f"[{total_time:.1f}s]")

    return {
        "y_true": y,
        "y_pred_all": y_preds_by_repeat,
        "y_pred_mean": y_pred_mean,
        "fold_results": fold_results,
        "metrics_per_repeat": metrics_per_repeat,
        "metrics_summary": metrics_summary,
        "overall_metrics": overall_metrics,
        "feature_stability": feature_stability,
        "config_hash": config_hash,
        "total_time_s": total_time,
        "model_name": model_config.name,
        "n_features_input": p,
        "n_samples": n,
        "feature_names": list(feature_names),
        "subject_ids": list(subject_ids) if subject_ids is not None else None,
        "outcome_metadata": outcome_metadata or {},
    }

def run_single_permutation(X, y, feature_names, model_config, cv_config,
                           rfe_config, vol, residualize, skip_residualize,
                           weights, perm_idx, random_state_base):
    rng = np.random.RandomState(random_state_base + perm_idx)
    y_perm = rng.permutation(y)

    result = run_nested_cv(
        X, y_perm, feature_names, model_config, cv_config,
        rfe_config=rfe_config, vol=vol, residualize=residualize,
        skip_residualize=skip_residualize, weights=weights, verbose=0
    )

    return {
        "perm_idx": perm_idx,
        "mae": result["overall_metrics"]["mae"],
        "r2": result["overall_metrics"]["r2"],
        "pearson_r": result["overall_metrics"]["pearson_r"],
        "spearman_rho": result["overall_metrics"]["spearman_rho"],
    }

def run_permutation_test(X, y, feature_names, model_config, cv_config,
                         rfe_config=None, vol=None, residualize=True,
                         skip_residualize=None, weights=None,
                         observed_metrics=None, perm_config=None,
                         verbose=1):
    if perm_config is None:
        perm_config = PermutationConfig()

    if rfe_config is None:
        rfe_config = RFEConfig(enabled=False)

    null_maes = []
    null_r2s = []
    null_rs = []
    null_rhos = []

    if verbose >= 1:
        print(f"\n  Running {perm_config.n_permutations} permutations "
              f"for {model_config.name}...")

    for i in range(perm_config.n_permutations):
        perm_result = run_single_permutation(
            X, y, feature_names, model_config, cv_config,
            rfe_config, vol, residualize, skip_residualize,
            weights, i, perm_config.random_state
        )
        null_maes.append(perm_result["mae"])
        null_r2s.append(perm_result["r2"])
        null_rs.append(perm_result["pearson_r"])
        null_rhos.append(perm_result["spearman_rho"])

        if verbose >= 1 and (i + 1) % 50 == 0:
            print(f"    Permutation {i+1}/{perm_config.n_permutations}")

    null_maes = np.array(null_maes)
    null_r2s = np.array(null_r2s)
    null_rs = np.array(null_rs)
    null_rhos = np.array(null_rhos)

    obs_mae = observed_metrics["mae"]
    obs_r2 = observed_metrics["r2"]
    obs_r = observed_metrics["pearson_r"]
    obs_rho = observed_metrics["spearman_rho"]

    p_mae = (np.sum(null_maes <= obs_mae) + 1) / (perm_config.n_permutations + 1)
    p_r2 = (np.sum(null_r2s >= obs_r2) + 1) / (perm_config.n_permutations + 1)
    p_r = (np.sum(null_rs >= obs_r) + 1) / (perm_config.n_permutations + 1)
    p_rho = (np.sum(null_rhos >= obs_rho) + 1) / (perm_config.n_permutations + 1)

    return {
        "p_values": {
            "mae": p_mae, "r2": p_r2,
            "pearson_r": p_r, "spearman_rho": p_rho,
        },
        "null_distribution": {
            "mae": null_maes, "r2": null_r2s,
            "pearson_r": null_rs, "spearman_rho": null_rhos,
        },
        "observed": {
            "mae": obs_mae, "r2": obs_r2,
            "pearson_r": obs_r, "spearman_rho": obs_rho,
        },
        "n_permutations": perm_config.n_permutations,
    }

def compare_models_wilcoxon(results_a, results_b, metric="mae"):
    vals_a = [m[metric] for m in results_a["metrics_per_repeat"]]
    vals_b = [m[metric] for m in results_b["metrics_per_repeat"]]

    n = min(len(vals_a), len(vals_b))
    vals_a = vals_a[:n]
    vals_b = vals_b[:n]

    if n < 5:
        return {"statistic": np.nan, "p_value": np.nan, "n": n}

    stat, p = stats.wilcoxon(vals_a, vals_b, alternative="two-sided")

    diffs = np.array(vals_a) - np.array(vals_b)
    r_effect = 1 - (2 * stat) / (n * (n + 1) / 2) if n > 0 else np.nan

    return {
        "statistic": stat,
        "p_value": p,
        "effect_size_r": r_effect,
        "mean_a": np.mean(vals_a),
        "mean_b": np.mean(vals_b),
        "mean_diff": np.mean(diffs),
        "n": n,
    }

def compare_models_hotelling_williams(results_a, results_b, y_true):
    pred_a = results_a["y_pred_mean"]
    pred_b = results_b["y_pred_mean"]

    valid = (~np.isnan(pred_a)) & (~np.isnan(pred_b)) & (~np.isnan(y_true))
    if valid.sum() < 10:
        return {"statistic": np.nan, "p_value": np.nan}

    r_ay = stats.pearsonr(pred_a[valid], y_true[valid])[0]
    r_by = stats.pearsonr(pred_b[valid], y_true[valid])[0]
    r_ab = stats.pearsonr(pred_a[valid], pred_b[valid])[0]
    n = valid.sum()

    r_det = 1 - r_ay**2 - r_by**2 - r_ab**2 + 2*r_ay*r_by*r_ab
    r_bar = (r_ay + r_by) / 2
    f = (1 - r_ab) / (2 * r_det / (n - 3) + (1 - r_ab)**3 / (4 * (n - 1)))

    denom = np.sqrt(2 * (1 - r_ab) * f) if (1 - r_ab) * f > 0 else 1e-10
    t_stat = (r_ay - r_by) * np.sqrt((n - 1) * (1 + r_ab)) / denom

    p_value = 2 * stats.t.sf(abs(t_stat), df=n-3)

    return {
        "statistic": t_stat,
        "p_value": p_value,
        "r_a": r_ay,
        "r_b": r_by,
        "r_ab": r_ab,
        "n": n,
    }

def run_shap_analysis(X, y, feature_names, model_config, vol=None,
                      residualize=True, skip_residualize=None,
                      random_state=42):
    if not HAS_SHAP:
        warnings.warn("shap not installed; skipping SHAP analysis")
        return None

    X_proc = X.copy()
    if residualize and vol is not None:
        skip = set(skip_residualize or [])
        skip.update({"Age", "Sex", "Edu", "LANG_comp_acute", "Hand",
                     "Aphasic?", "Any SSRI 0-3m?", "SLP Tx session (0-3m)",
                     "Days Post Testing (acute)", "tract_lesion_volume_mL",
                     "brain_age_gap", "predicted_age"})
        for i, name in enumerate(feature_names):
            if name in skip:
                continue
            col = X_proc[:, i]
            v = vol.reshape(-1, 1)
            valid = ~(np.isnan(col) | np.isnan(v.ravel()))
            if valid.sum() >= 5:
                reg = LinearRegression().fit(v[valid], col[valid])
                X_proc[valid, i] = col[valid] - reg.predict(v[valid])

    medians = np.nanmedian(X_proc, axis=0)
    for j in range(X_proc.shape[1]):
        nans = np.isnan(X_proc[:, j])
        if nans.any():
            X_proc[nans, j] = medians[j]
    scaler = StandardScaler()
    X_proc = scaler.fit_transform(X_proc)

    estimator = model_config.get_estimator(random_state)
    if model_config.param_grid:
        cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
        gs = GridSearchCV(estimator, model_config.param_grid, cv=cv,
                          scoring="neg_mean_absolute_error", n_jobs=-1, refit=True)
        gs.fit(X_proc, y)
        model = gs.best_estimator_
    else:
        model = clone(estimator)
        model.fit(X_proc, y)

    if model_config.name in ("rf", "xgb"):
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.KernelExplainer(model.predict, X_proc[:50])

    shap_values = explainer.shap_values(X_proc)

    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    importance_order = np.argsort(mean_abs_shap)[::-1]

    return {
        "shap_values": shap_values,
        "feature_names": list(feature_names),
        "mean_abs_shap": mean_abs_shap,
        "importance_order": importance_order,
        "fitted_model": model,
        "X_processed": X_proc,
    }

def run_full_analysis(X, y, feature_names, model_configs, feature_set_configs,
                      cv_config=None, rfe_config=None, perm_config=None,
                      vol=None, residualize=True, skip_residualize=None,
                      weights=None, run_permutations=True, run_shap=True,
                      verbose=1):
    if cv_config is None:
        cv_config = CVConfig()
    if rfe_config is None:
        rfe_config = RFEConfig(enabled=False)
    if perm_config is None:
        perm_config = PermutationConfig()

    all_results = {}
    all_feature_names = list(feature_names)

    for fs_name, fs_config in feature_set_configs.items():
        col_idx = [i for i, f in enumerate(all_feature_names)
                   if f in fs_config.columns]
        if not col_idx:
            warnings.warn(f"Feature set '{fs_name}' has no matching columns")
            continue

        X_fs = X[:, col_idx]
        fs_names = [all_feature_names[i] for i in col_idx]
        skip_resid = fs_config.skip_residualize or skip_residualize

        if verbose >= 1:
            print(f"\n{'='*70}")
            print(f"  Feature set: {fs_name} ({len(fs_names)} features)")
            print(f"{'='*70}")

        for model_name, model_config in model_configs.items():
            key = (fs_name, model_name)

            if verbose >= 1:
                print(f"\n  Running {model_name} on {fs_name}...")

            fs_rfe = deepcopy(rfe_config)
            if len(fs_names) <= 5:
                fs_rfe.enabled = False

            result = run_nested_cv(
                X_fs, y, fs_names, model_config, cv_config,
                rfe_config=fs_rfe, vol=vol, residualize=residualize,
                skip_residualize=skip_resid, weights=weights,
                verbose=verbose,
            )
            result["feature_set"] = fs_name
            all_results[key] = result

    if run_permutations:
        for fs_name, fs_config in feature_set_configs.items():
            col_idx = [i for i, f in enumerate(all_feature_names)
                       if f in fs_config.columns]
            if not col_idx:
                continue
            X_fs = X[:, col_idx]
            fs_names = [all_feature_names[i] for i in col_idx]
            skip_resid = fs_config.skip_residualize or skip_residualize

            best_key = None
            best_mae = np.inf
            for (fn, mn), res in all_results.items():
                if fn == fs_name and res["overall_metrics"]["mae"] < best_mae:
                    best_mae = res["overall_metrics"]["mae"]
                    best_key = (fn, mn)

            if best_key is None:
                continue

            model_config = model_configs[best_key[1]]
            fs_rfe = deepcopy(rfe_config)
            if len(fs_names) <= 5:
                fs_rfe.enabled = False

            perm_result = run_permutation_test(
                X_fs, y, fs_names, model_config, cv_config,
                rfe_config=fs_rfe, vol=vol, residualize=residualize,
                skip_residualize=skip_resid, weights=weights,
                observed_metrics=all_results[best_key]["overall_metrics"],
                perm_config=perm_config, verbose=verbose,
            )
            all_results[best_key]["permutation_test"] = perm_result

    comparison_results = []
    feature_sets_tested = list(feature_set_configs.keys())
    model_names_tested = list(model_configs.keys())

    for fs_name in feature_sets_tested:
        for i, m1 in enumerate(model_names_tested):
            for m2 in model_names_tested[i+1:]:
                key1 = (fs_name, m1)
                key2 = (fs_name, m2)
                if key1 in all_results and key2 in all_results:
                    wilcox = compare_models_wilcoxon(
                        all_results[key1], all_results[key2], metric="mae"
                    )
                    hw = compare_models_hotelling_williams(
                        all_results[key1], all_results[key2], y
                    )
                    comparison_results.append({
                        "feature_set": fs_name,
                        "model_a": m1, "model_b": m2,
                        "wilcoxon_p": wilcox["p_value"],
                        "wilcoxon_stat": wilcox["statistic"],
                        "effect_size_r": wilcox.get("effect_size_r", np.nan),
                        "hw_p": hw["p_value"],
                        "hw_stat": hw["statistic"],
                        "mae_a": wilcox.get("mean_a", np.nan),
                        "mae_b": wilcox.get("mean_b", np.nan),
                    })

    shap_results = None
    if run_shap:
        best_overall_key = min(
            all_results.keys(),
            key=lambda k: all_results[k]["overall_metrics"]["mae"]
        )
        best_fs = best_overall_key[0]
        best_model = best_overall_key[1]
        fs_config = feature_set_configs[best_fs]
        col_idx = [i for i, f in enumerate(all_feature_names)
                   if f in fs_config.columns]
        X_best = X[:, col_idx]
        fs_names = [all_feature_names[i] for i in col_idx]

        if verbose >= 1:
            print(f"\n  Running SHAP on best model: {best_model} @ {best_fs}")

        shap_results = run_shap_analysis(
            X_best, y, fs_names, model_configs[best_model],
            vol=vol, residualize=residualize,
            skip_residualize=fs_config.skip_residualize or skip_residualize,
            random_state=cv_config.random_state,
        )

    return {
        "cv_results": all_results,
        "comparisons": comparison_results,
        "shap": shap_results,
        "cv_config": asdict(cv_config),
        "rfe_config": asdict(rfe_config),
        "perm_config": asdict(perm_config) if perm_config else None,
    }

def print_results_table(all_results, title="NESTED CV RESULTS"):
    print(f"\n{'='*100}")
    print(f"  {title}")
    print(f"{'='*100}")
    print(f"\n  {'Feature Set':<30s} {'Model':<8s} {'p':>3s} {'N':>3s} "
          f"{'MAE':>8s} {'RMSE':>8s} {'R²':>10s} "
          f"{'r':>8s} {'ρ':>8s} {'Time':>6s}")
    print(f"  {'-'*30} {'-'*8} {'-'*3} {'-'*3} "
          f"{'-'*8} {'-'*8} {'-'*10} "
          f"{'-'*8} {'-'*8} {'-'*6}")

    for (fs, model), res in sorted(all_results.items()):
        m = res["overall_metrics"]
        ms = res["metrics_summary"]
        mae_str = f"{m['mae']:.2f}±{ms.get('mae',{}).get('std',0):.2f}"
        rmse_str = f"{m['rmse']:.2f}"
        r2_str = f"{m['r2']:.3f}±{ms.get('r2',{}).get('std',0):.3f}"
        r_str = f"{m['pearson_r']:.3f}"
        rho_str = f"{m['spearman_rho']:.3f}"

        perm_str = ""
        if "permutation_test" in res:
            p_val = res["permutation_test"]["p_values"]["mae"]
            perm_str = f"  p={p_val:.4f}"

        print(f"  {fs:<30s} {model:<8s} "
              f"{res['n_features_input']:3d} {res['n_samples']:3d} "
              f"{mae_str:>8s} {rmse_str:>8s} {r2_str:>10s} "
              f"{r_str:>8s} {rho_str:>8s} "
              f"{res['total_time_s']:5.0f}s{perm_str}")

def save_results(results, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for (fs, model), res in results["cv_results"].items():
        m = res["overall_metrics"]
        ms = res["metrics_summary"]
        row = {
            "feature_set": fs, "model": model,
            "n_features": res["n_features_input"],
            "n_samples": res["n_samples"],
            "mae": m["mae"],
            "mae_std": ms.get("mae", {}).get("std", np.nan),
            "rmse": m["rmse"],
            "r2": m["r2"],
            "r2_std": ms.get("r2", {}).get("std", np.nan),
            "pearson_r": m["pearson_r"],
            "spearman_rho": m["spearman_rho"],
            "total_time_s": res["total_time_s"],
        }
        if "bootstrap_ci" in res:
            for metric, ci in res["bootstrap_ci"].items():
                row[f"{metric}_ci_lo"] = ci[0]
                row[f"{metric}_ci_hi"] = ci[1]
        if "permutation_test" in res:
            for pk, pv in res["permutation_test"]["p_values"].items():
                row[f"perm_p_{pk}"] = pv
        if "conformal_summary" in res:
            for ck, cv in res["conformal_summary"].items():
                row[f"conformal_{ck}"] = cv
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "run_results_summary.csv", index=False)

    if results["comparisons"]:
        comp_df = pd.DataFrame(results["comparisons"])
        comp_df.to_csv(output_dir / "run_model_comparisons.csv", index=False)

    for (fs, model), res in results["cv_results"].items():
        repeat_df = pd.DataFrame(res["metrics_per_repeat"])
        fname = f"repeat_metrics_{fs}_{model}.csv"
        repeat_df.to_csv(output_dir / fname, index=False)

    for (fs, model), res in results["cv_results"].items():
        if res["feature_stability"]:
            stab = sorted(res["feature_stability"].items(),
                          key=lambda x: x[1], reverse=True)
            stab_df = pd.DataFrame(stab, columns=["feature", "selection_freq"])
            fname = f"feature_stability_{fs}_{model}.csv"
            stab_df.to_csv(output_dir / fname, index=False)

    for (fs, model), res in results["cv_results"].items():
        pred_data = {
            "subject_id": res["subject_ids"] if res.get("subject_ids") is not None
                          else list(range(res["n_samples"])),
            "y_true": res["y_true"],
            "y_pred_mean": res["y_pred_mean"],
        }
        if "conformal" in res:
            for conf_key, conf_res in res["conformal"].items():
                if isinstance(conf_res, dict) and "lower" in conf_res:
                    pred_data[f"{conf_key}_lower"] = conf_res["lower"]
                    pred_data[f"{conf_key}_upper"] = conf_res["upper"]
        pred_df = pd.DataFrame(pred_data)
        fname = f"predictions_{fs}_{model}.csv"
        pred_df.to_csv(output_dir / fname, index=False)

    print(f"\n  Results saved to {output_dir}/")
    return output_dir

def compute_bootstrap_ci(y_true, y_pred, n_bootstrap=2000, ci=95,
                         random_state=42):
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    alpha = (100 - ci) / 2

    boot_metrics = {k: [] for k in ["mae", "rmse", "r2", "pearson_r", "spearman_rho"]}

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            m = compute_metrics(y_true[idx], y_pred[idx])
            for k in boot_metrics:
                if k in m and not np.isnan(m[k]):
                    boot_metrics[k].append(m[k])
        except Exception:
            continue

    cis = {}
    for k, vals in boot_metrics.items():
        if len(vals) >= 100:
            arr = np.array(vals)
            cis[k] = (np.percentile(arr, alpha), np.percentile(arr, 100 - alpha))
        else:
            cis[k] = (np.nan, np.nan)

    return cis

def add_bootstrap_cis(cv_result, n_bootstrap=2000, ci=95, random_state=42):
    y_true = cv_result["y_true"]
    y_pred = cv_result["y_pred_mean"]
    valid = ~np.isnan(y_pred) & ~np.isnan(y_true)

    if valid.sum() < 10:
        cv_result["bootstrap_ci"] = {}
        return

    cis = compute_bootstrap_ci(
        y_true[valid], y_pred[valid],
        n_bootstrap=n_bootstrap, ci=ci, random_state=random_state
    )
    cv_result["bootstrap_ci"] = cis

def compute_classification_metrics(y_true_cont, y_pred_cont, threshold, direction=">="):
    from sklearn.metrics import (
        f1_score, matthews_corrcoef, precision_score, recall_score,
        balanced_accuracy_score, roc_auc_score, confusion_matrix,
    )

    if direction == ">=":
        y_true_bin = (y_true_cont >= threshold).astype(int)
        y_pred_bin = (y_pred_cont >= threshold).astype(int)
    else:
        y_true_bin = (y_true_cont <= threshold).astype(int)
        y_pred_bin = (y_pred_cont <= threshold).astype(int)

    n = len(y_true_bin)
    n_pos = y_true_bin.sum()
    n_neg = n - n_pos

    if n_pos == 0 or n_neg == 0:
        return {k: np.nan for k in [
            "f1", "mcc", "precision", "recall", "specificity",
            "balanced_accuracy", "auc_roc", "ppv", "npv", "youdens_j",
            "n_true_pos", "n_true_neg", "prevalence",
        ]}

    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    youdens_j = recall + specificity - 1 if not np.isnan(recall) else np.nan

    try:
        auc = roc_auc_score(y_true_bin, y_pred_cont)
    except ValueError:
        auc = np.nan

    return {
        "f1": f1_score(y_true_bin, y_pred_bin, zero_division=0),
        "mcc": matthews_corrcoef(y_true_bin, y_pred_bin),
        "precision": precision_score(y_true_bin, y_pred_bin, zero_division=0),
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy_score(y_true_bin, y_pred_bin),
        "auc_roc": auc,
        "ppv": ppv,
        "npv": npv,
        "youdens_j": youdens_j,
        "n_true_pos": int(n_pos),
        "n_true_neg": int(n_neg),
        "prevalence": float(n_pos / n),
    }

def compute_calibration_stats(y_true, y_pred):
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    if valid.sum() < 5:
        return {"calibration_slope": np.nan, "calibration_intercept": np.nan}

    yt, yp = y_true[valid], y_pred[valid]
    reg = LinearRegression()
    reg.fit(yp.reshape(-1, 1), yt)
    return {
        "calibration_slope": float(reg.coef_[0]),
        "calibration_intercept": float(reg.intercept_),
    }

def add_posthoc_classification(cv_result, threshold, direction=">=",
                                n_bootstrap=2000, ci=95, random_state=42):
    y_true = cv_result["y_true"]
    y_pred = cv_result["y_pred_mean"]
    valid = ~np.isnan(y_pred) & ~np.isnan(y_true)

    if valid.sum() < 10:
        cv_result["classification_metrics"] = {}
        cv_result["calibration"] = {}
        cv_result["classification_bootstrap_ci"] = {}
        return

    yt, yp = y_true[valid], y_pred[valid]

    cls_metrics = compute_classification_metrics(yt, yp, threshold, direction)
    cv_result["classification_metrics"] = cls_metrics

    cal_stats = compute_calibration_stats(yt, yp)
    cv_result["calibration"] = cal_stats

    rng = np.random.RandomState(random_state)
    n = len(yt)
    alpha_pct = (100 - ci) / 2

    cls_keys = [k for k in cls_metrics if k not in
                ("n_true_pos", "n_true_neg", "prevalence")]
    cal_keys = list(cal_stats.keys())
    all_keys = cls_keys + cal_keys

    boot_vals = {k: [] for k in all_keys}

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(yt[idx])) < 2:
            continue
        if direction == ">=":
            bin_true = (yt[idx] >= threshold).astype(int)
        else:
            bin_true = (yt[idx] <= threshold).astype(int)
        if bin_true.sum() == 0 or bin_true.sum() == n:
            continue

        try:
            cm = compute_classification_metrics(yt[idx], yp[idx], threshold, direction)
            for k in cls_keys:
                if k in cm and not np.isnan(cm[k]):
                    boot_vals[k].append(cm[k])

            cs = compute_calibration_stats(yt[idx], yp[idx])
            for k in cal_keys:
                if k in cs and not np.isnan(cs[k]):
                    boot_vals[k].append(cs[k])
        except Exception:
            continue

    boot_cis = {}
    for k, vals in boot_vals.items():
        if len(vals) >= 100:
            arr = np.array(vals)
            boot_cis[k] = (float(np.percentile(arr, alpha_pct)),
                           float(np.percentile(arr, 100 - alpha_pct)))
        else:
            boot_cis[k] = (np.nan, np.nan)

    cv_result["classification_bootstrap_ci"] = boot_cis

def compute_conformal_intervals(y_true, y_pred, confidence=0.90,
                                method="split"):
    valid = ~np.isnan(y_pred) & ~np.isnan(y_true)
    yt, yp = y_true[valid], y_pred[valid]
    n = len(yt)

    residuals = np.abs(yt - yp)

    if method == "split":
        alpha = 1 - confidence
        q_level = min((1 - alpha) * (1 + 1/n), 1.0)
        interval_width = np.quantile(residuals, q_level)

        lower = np.full_like(y_pred, np.nan)
        upper = np.full_like(y_pred, np.nan)
        lower[valid] = yp - interval_width
        upper[valid] = yp + interval_width

    elif method == "adaptive":
        n_bins = min(5, n // 5)
        if n_bins < 2:
            return compute_conformal_intervals(
                y_true, y_pred, confidence, method="split")

        bin_edges = np.percentile(yp, np.linspace(0, 100, n_bins + 1))
        bin_edges[0] -= 1
        bin_edges[-1] += 1
        bin_idx = np.digitize(yp, bin_edges) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)

        scale = np.ones(n)
        for b in range(n_bins):
            mask = bin_idx == b
            if mask.sum() > 2:
                local_mad = np.median(residuals[mask])
                scale[mask] = max(local_mad, 1e-6)
            else:
                scale[mask] = np.median(residuals)

        norm_residuals = residuals / scale

        alpha = 1 - confidence
        q_level = min((1 - alpha) * (1 + 1/n), 1.0)
        q_norm = np.quantile(norm_residuals, q_level)

        widths = q_norm * scale
        interval_width = np.median(widths)

        lower = np.full_like(y_pred, np.nan)
        upper = np.full_like(y_pred, np.nan)
        lower[valid] = yp - widths
        upper[valid] = yp + widths

    else:
        raise ValueError(f"Unknown conformal method: {method}")

    coverage = np.mean((yt >= lower[valid]) & (yt <= upper[valid]))

    return {
        "interval_width": float(interval_width),
        "lower": lower,
        "upper": upper,
        "coverage": float(coverage),
        "confidence": confidence,
        "residuals": residuals,
        "method": method,
        "n_calibration": n,
    }

def add_conformal_intervals(cv_result, confidence_levels=None):
    if confidence_levels is None:
        confidence_levels = [0.80, 0.90, 0.95]

    y_true = cv_result["y_true"]
    y_pred = cv_result["y_pred_mean"]

    valid = ~np.isnan(y_pred) & ~np.isnan(y_true)
    if valid.sum() < 10:
        cv_result["conformal"] = {}
        return

    conformal_results = {}
    for conf in confidence_levels:
        for method in ["split", "adaptive"]:
            key = f"{method}_{int(conf*100)}"
            conformal_results[key] = compute_conformal_intervals(
                y_true, y_pred, confidence=conf, method=method
            )

    cv_result["conformal"] = conformal_results

    primary = conformal_results.get("split_90", {})
    cv_result["conformal_summary"] = {
        "interval_width_90": primary.get("interval_width", np.nan),
        "coverage_90": primary.get("coverage", np.nan),
        "interval_width_80": conformal_results.get(
            "split_80", {}).get("interval_width", np.nan),
        "interval_width_95": conformal_results.get(
            "split_95", {}).get("interval_width", np.nan),
    }

def run_shap_stability(X, y, feature_names, model_config, cv_config,
                       vol=None, residualize=True, skip_residualize=None,
                       n_folds_for_shap=10, random_state=42,
                       subject_ids=None):
    if not HAS_SHAP:
        warnings.warn("shap not installed; skipping SHAP stability")
        return None

    n, p = X.shape
    cv = KFold(n_splits=n_folds_for_shap, shuffle=True, random_state=random_state)

    all_shap = np.zeros((n, p))
    shap_counts = np.zeros(n)
    fold_rankings = []
    fold_mean_abs = []
    fold_test_indices = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        fold_test_indices.append(test_idx.tolist())
        X_train_raw = X[train_idx].copy()
        X_test_raw = X[test_idx].copy()
        y_train = y[train_idx]

        vol_train = vol[train_idx] if vol is not None else None
        vol_test = vol[test_idx] if vol is not None else None

        X_train, X_test, _ = preprocess_fold(
            X_train_raw, X_test_raw, feature_names,
            vol_train, vol_test,
            residualize=residualize,
            skip_residualize=skip_residualize,
        )

        estimator = model_config.get_estimator(random_state)
        if model_config.param_grid:
            inner_cv = KFold(n_splits=5, shuffle=True,
                             random_state=random_state + fold_idx)
            gs = GridSearchCV(
                estimator, model_config.param_grid,
                cv=inner_cv, scoring="neg_mean_absolute_error",
                n_jobs=-1, refit=True, error_score=np.nan,
            )
            gs.fit(X_train, y_train)
            model = gs.best_estimator_
        else:
            model = clone(estimator)
            model.fit(X_train, y_train)

        try:
            if model_config.name in ("rf", "xgb"):
                explainer = shap.TreeExplainer(model)
            else:
                bg = X_train[:min(50, len(X_train))]
                explainer = shap.KernelExplainer(model.predict, bg)

            sv = explainer.shap_values(X_test)
            all_shap[test_idx] += sv
            shap_counts[test_idx] += 1

            fold_abs = np.mean(np.abs(sv), axis=0)
            fold_mean_abs.append(fold_abs)
            ranking = np.argsort(fold_abs)[::-1]
            fold_rankings.append(ranking)

        except Exception as e:
            warnings.warn(f"SHAP failed on fold {fold_idx}: {e}")
            continue

    valid_shap = shap_counts > 0
    if valid_shap.any():
        all_shap[valid_shap] /= shap_counts[valid_shap, np.newaxis]

    mean_abs_shap = np.mean(np.abs(all_shap[valid_shap]), axis=0)

    rank_stability = {}
    for k in [3, 5, 10]:
        for fi, fname in enumerate(feature_names):
            key = f"top{k}_{fname}"
            appearances = sum(1 for ranks in fold_rankings if fi in ranks[:k])
            rank_stability[key] = appearances / len(fold_rankings) if fold_rankings else 0

    return {
        "mean_abs_shap": mean_abs_shap,
        "all_shap_values": all_shap,
        "shap_counts": shap_counts,
        "fold_mean_abs_shap": fold_mean_abs,
        "fold_rankings": fold_rankings,
        "feature_names": list(feature_names),
        "rank_stability": rank_stability,
        "n_folds": len(fold_rankings),
        "fold_test_indices": fold_test_indices,
        "subject_ids": list(subject_ids) if subject_ids is not None else None,
    }
