"""Residualization of imaging features against lesion volume, inverse-density sample weights, standardization."""
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    LESION_VOLUME_COL,
    LOW_OUTCOME_THRESHOLD,
)
try:
    from data_loader import load_modeling_data, get_feature_matrix, get_layer_columns
except ImportError:
    pass

def residualize_features(X, feature_names, lesion_vol, features_to_skip=None):
    if features_to_skip is None:
        features_to_skip = set()
    else:
        features_to_skip = set(features_to_skip)

    _always_skip = {
        "wab_aq_acute", "age_at_stroke", "sex", "education_yrs",
        "race", "prior_stroke", LESION_VOLUME_COL,
        "Age", "Sex", "Edu", "LANG_comp_acute", "Hand",
        "Aphasic?", "Any SSRI 0-3m?", "SLP Tx session (0-3m)",
        "Days Post Testing (acute)",
    }
    features_to_skip |= _always_skip

    X_resid = X.copy().astype(float)
    vol = lesion_vol.reshape(-1, 1).astype(float)
    resid_info = {}

    for i, name in enumerate(feature_names):
        if name in features_to_skip:
            resid_info[name] = {"r2": np.nan, "residualized": False}
            continue

        col = X_resid[:, i]

        valid = ~(np.isnan(col) | np.isnan(vol.ravel()))
        if valid.sum() < 5:
            warnings.warn(f"Skipping residualization of '{name}': "
                          f"only {valid.sum()} valid cases")
            resid_info[name] = {"r2": np.nan, "residualized": False}
            continue

        reg = LinearRegression()
        reg.fit(vol[valid], col[valid])
        predicted = reg.predict(vol[valid])
        residuals = col[valid] - predicted

        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((col[valid] - col[valid].mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        X_resid[valid, i] = residuals
        resid_info[name] = {"r2": r2, "residualized": True, "beta": reg.coef_[0]}

    return X_resid, resid_info

def print_residualization_summary(resid_info, top_n=15):
    items = [(k, v) for k, v in resid_info.items()
             if v.get("residualized", False) and not np.isnan(v["r2"])]
    items.sort(key=lambda x: x[1]["r2"], reverse=True)

    print(f"\nResidualization summary ({len(items)} features residualized):\n")
    print(f"  {'Feature':<50s}  R²(vol)   beta")
    print(f"  {'-'*50}  -------  ------")

    for name, info in items[:top_n]:
        print(f"  {name:<50s}  {info['r2']:.3f}    {info['beta']:+.4f}")

    if len(items) > top_n:
        remaining = items[top_n:]
        avg_r2 = np.mean([v["r2"] for _, v in remaining])
        print(f"  ... +{len(remaining)} more (mean R²={avg_r2:.3f})")

def compute_density_weights(y, bandwidth="auto", cap=5.0):
    y_clean = y[~np.isnan(y)]

    if bandwidth == "auto":
        silverman = 1.06 * np.std(y_clean) * len(y_clean) ** (-1/5)
        bandwidth = min(silverman, 5.0)
        bandwidth = max(bandwidth, 1.0)

    kde = stats.gaussian_kde(y_clean, bw_method=bandwidth / np.std(y_clean))
    density = kde(y)

    raw_weights = 1.0 / (density + 1e-10)

    weights = raw_weights / raw_weights.mean()
    weights = np.clip(weights, a_min=None, a_max=cap)

    weights = weights / weights.mean()

    return weights, density

def compute_threshold_weights(y, threshold=None, rare_weight=3.0):
    if threshold is None:
        threshold = LOW_OUTCOME_THRESHOLD

    weights = np.where(y < threshold, rare_weight, 1.0)
    weights = weights / weights.mean()
    return weights

def handle_missing(X, y, feature_names, strategy="drop"):
    if strategy == "drop":
        valid = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        return X[valid], y[valid], valid

    elif strategy == "impute":
        X_imp = X.copy()
        for j in range(X.shape[1]):
            col = X_imp[:, j]
            nans = np.isnan(col)
            if nans.any():
                median_val = np.nanmedian(col)
                X_imp[nans, j] = median_val
        valid = ~np.isnan(y)
        return X_imp[valid], y[valid], valid

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def prepare_modeling_set(data, layers, mode="apriori",
                         residualize=True, standardize=True,
                         weighting="density", nan_strategy="drop"):
    if isinstance(layers, int):
        layers = [layers]

    X, y, names = get_feature_matrix(data, layers, mode=mode)
    ids = data["ids"]
    qc = data["qc_mask"]

    resid_info = None
    if residualize and LESION_VOLUME_COL in data["df"].columns:
        vol = pd.to_numeric(
            data["df"][LESION_VOLUME_COL], errors="coerce"
        ).values
        X, resid_info = residualize_features(X, names, vol)

    n_before = len(y)
    X, y, valid_mask = handle_missing(X, y, names, strategy=nan_strategy)
    ids = ids[valid_mask]
    qc = qc[valid_mask]
    n_dropped = n_before - len(y)

    scaler = None
    if standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    weights = None
    if weighting == "density":
        weights, _ = compute_density_weights(y)
    elif weighting == "threshold":
        weights = compute_threshold_weights(y)

    return {
        "X": X,
        "y": y,
        "names": names,
        "weights": weights,
        "ids": ids,
        "qc_mask": qc,
        "scaler": scaler,
        "resid_info": resid_info,
        "n_dropped": n_dropped,
    }

if __name__ == "__main__":
    print("=" * 70)
    print("  FEATURE ENGINEERING DIAGNOSTIC")
    print("=" * 70)

    data = load_modeling_data()
    print(f"\nLoaded N={data['n']} subjects, outcome range "
          f"[{data['y'].min():.1f}, {data['y'].max():.1f}]")

    for max_layer in range(5):
        layers = list(range(max_layer + 1))
        result = prepare_modeling_set(
            data, layers, mode="apriori",
            residualize=(max_layer >= 2),
            standardize=False,
            weighting=None,
            nan_strategy="drop",
        )
        X, y, names = result["X"], result["y"], result["names"]
        print(f"\n  Layers 0..{max_layer} (apriori): "
              f"{X.shape[1]} features, N={X.shape[0]}, "
              f"dropped={result['n_dropped']}")

    print("\n" + "=" * 70)
    print("  FULL PIPELINE: Layers 0-4, apriori, residualized, weighted")
    print("=" * 70)

    result = prepare_modeling_set(
        data, layers=[0, 1, 2, 3, 4], mode="apriori",
        residualize=True, standardize=True,
        weighting="density", nan_strategy="drop",
    )

    X = result["X"]
    y = result["y"]
    names = result["names"]
    weights = result["weights"]

    print(f"\n  X shape: {X.shape}")
    print(f"  y shape: {y.shape}, range [{y.min():.1f}, {y.max():.1f}]")
    print(f"  Dropped for NaN: {result['n_dropped']}")
    print(f"  Weight range: [{weights.min():.2f}, {weights.max():.2f}], "
          f"mean={weights.mean():.2f}")

    if result["resid_info"]:
        print_residualization_summary(result["resid_info"])

    print("\n  Weight distribution by outcome quartile:")
    for q_low, q_high in [(0, 25), (25, 50), (50, 75), (75, 100)]:
        mask = (y >= q_low) & (y < q_high + 0.01)
        if mask.sum() > 0:
            print(f"    AQ [{q_low:3d}, {q_high:3d}): "
                  f"N={mask.sum():3d}, mean_weight={weights[mask].mean():.2f}")

    assert not np.isnan(X).any(), "NaN detected in final X!"
    assert not np.isnan(y).any(), "NaN detected in final y!"
    print("\n  ✓ No NaN in final matrices")
    print("\n  ✓ Pipeline complete — ready for modeling")
