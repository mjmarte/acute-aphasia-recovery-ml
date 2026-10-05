#!/usr/bin/env python3
"""Prepares the data for one analysis (outcome, acute predictors, sample) and runs one algorithm on one feature set."""
import os, sys, json, pickle, time, argparse, importlib.util, warnings
from dataclasses import asdict, is_dataclass
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

SRC = Path(os.environ.get('ENGINE_SRC', Path(__file__).resolve().parents[1] / 'engine')); sys.path.insert(0, str(SRC))
REV = Path(os.environ.get('REV_ROOT', os.environ.get('REV_ROOT', '.')))
DATA = REV / 'data' / 'dataset.csv'
REDUCED = REV / 'data' / 'reduced_inventory.csv'
OUTROOT = REV / 'experiments' / 'models'

from config import FEATURE_SETS, CATEGORICAL_ENCODINGS, LESION_VOLUME_COL, ID_COL, EXCLUDED_SUBJECTS
from feature_eng import compute_density_weights
from cv_engine import CVConfig, RFEConfig, FeatureSetConfig

def import_runner():
    spec = importlib.util.spec_from_file_location('run_models', SRC / 'run_models.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

VARIANTS = {
    'wab_base':   dict(outcome='wab_aq_12mo',       thr=93.8, acute=['wab_aq_acute'],                        subset=None),
    'nct_base':   dict(outcome='nct_total_cu_12mo', thr=22.1, acute=['wab_aq_acute'],                        subset=None),
    'c_wab_own':  dict(outcome='wab_aq_12mo',       thr=93.8, acute=['wab_aq_acute'],                        subset='common'),
    'c_wab_cu':   dict(outcome='wab_aq_12mo',       thr=93.8, acute=['nct_total_cu_acute'],                  subset='common'),
    'c_wab_both': dict(outcome='wab_aq_12mo',       thr=93.8, acute=['wab_aq_acute', 'nct_total_cu_acute'],  subset='common'),
    'c_nct_own':  dict(outcome='nct_total_cu_12mo', thr=22.1, acute=['nct_total_cu_acute'],                  subset='common'),
    'c_nct_aq':   dict(outcome='nct_total_cu_12mo', thr=22.1, acute=['wab_aq_acute'],                        subset='common'),
    'c_nct_both': dict(outcome='nct_total_cu_12mo', thr=22.1, acute=['wab_aq_acute', 'nct_total_cu_acute'],  subset='common'),
    'red_aq':     dict(outcome='cu_reduced_12mo',   thr=17.7, acute=['wab_aq_acute'],                        subset='reduced12'),
    'red_own':    dict(outcome='cu_reduced_12mo',   thr=17.7, acute=['cu_reduced_acute'],                    subset='common_reduced'),
    'res_wab':    dict(outcome='resid_wab',         thr=0.0,  acute=[],                                      subset=None),
    'res_nct':    dict(outcome='resid_nct',         thr=0.0,  acute=[],                                      subset='common'),
    'sub_wab':    dict(outcome='wab_aq_12mo',       thr=93.8, acute=['wab_aq_acute'],                        subset='aphasic'),
}

def prepare(variant):
    v = VARIANTS[variant]
    df = pd.read_csv(DATA)
    red = pd.read_csv(REDUCED)[['patient_id', 'cu_full_12mo', 'cu_reduced_12mo', 'cu_full_acute', 'cu_reduced_acute']]
    df = df.merge(red, on='patient_id', how='left')
    for col, mapping in CATEGORICAL_ENCODINGS.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].map(mapping), errors='coerce')
    for s in EXCLUDED_SUBJECTS:
        df = df[df[ID_COL] != s]
    if v['outcome'] == 'resid_wab':
        m = df.dropna(subset=['wab_aq_12mo', 'wab_aq_acute'])
        b = np.polyfit(m.wab_aq_acute, m.wab_aq_12mo, 1)
        df['resid_wab'] = df.wab_aq_12mo - np.polyval(b, df.wab_aq_acute)
    if v['outcome'] == 'resid_nct':
        m = df.dropna(subset=['nct_total_cu_12mo', 'nct_total_cu_acute'])
        b = np.polyfit(m.nct_total_cu_acute, m.nct_total_cu_12mo, 1)
        df['resid_nct'] = df.nct_total_cu_12mo - np.polyval(b, df.nct_total_cu_acute)
    sub = v['subset']
    if sub == 'common':
        df = df[df.nct_total_cu_acute.notna() & df.nct_total_cu_12mo.notna() & df.wab_aq_12mo.notna()]
    elif sub == 'common_reduced':
        df = df[df.cu_reduced_acute.notna() & df.cu_reduced_12mo.notna()]
    elif sub == 'reduced12':
        df = df[df.cu_reduced_12mo.notna()]
    elif sub == 'aphasic':
        df = df[df.wab_aq_acute < 93.8]
    df = df.reset_index(drop=True)
    y_raw = pd.to_numeric(df[v['outcome']], errors='coerce').values
    vol = pd.to_numeric(df[LESION_VOLUME_COL], errors='coerce').values
    acute_ok = np.ones(len(df), bool)
    for a in v['acute']:
        acute_ok &= pd.to_numeric(df[a], errors='coerce').notna().values
    valid = ~np.isnan(y_raw) & ~np.isnan(vol) & acute_ok
    df = df[valid].reset_index(drop=True); y = y_raw[valid]; vol = vol[valid]
    fsets = {}
    for k, d in FEATURE_SETS.items():
        cols = []
        for c in d['features']:
            if c == 'wab_aq_acute':
                cols.extend(v['acute'])
            else:
                cols.append(c)
        cols = [c for c in cols if c in df.columns]
        resid = set(d['residualize'])
        fsets[k] = FeatureSetConfig(name=d['name'], columns=cols, skip_residualize=[c for c in cols if c not in resid])
    all_names = []
    for fs in fsets.values():
        for c in fs.columns:
            if c not in all_names: all_names.append(c)
    X_raw = df[all_names].apply(pd.to_numeric, errors='coerce').values
    weights, _ = compute_density_weights(y)
    return dict(X_raw=X_raw, y=y, vol=vol, n=int(valid.sum()), weights=weights, all_feature_names=all_names,
                feature_sets=fsets, df=df, subject_ids=df[ID_COL].tolist(),
                outcome_metadata=dict(outcome_col=v['outcome'], threshold=v['thr'], threshold_direction='>=',
                                      outcome_description=f"{variant}: {v['outcome']} >= {v['thr']}"))

def to_jsonable(o):
    if isinstance(o, dict): return {str(k): to_jsonable(x) for k, x in o.items()}
    if isinstance(o, (list, tuple)): return [to_jsonable(x) for x in o]
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, (np.floating, np.integer)): return o.item()
    if is_dataclass(o): return to_jsonable(asdict(o))
    return o

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', required=True, choices=list(VARIANTS))
    ap.add_argument('--fs', required=True); ap.add_argument('--model', required=True)
    ap.add_argument('--quick', action='store_true'); ap.add_argument('--outroot', default=str(OUTROOT)); ap.add_argument('--no-rfe', action='store_true', help='fit the full feature set in every fold (no within-fold RFE); outputs go to <variant>_norfe')
    a = ap.parse_args()
    runner = import_runner()
    if a.quick:
        cv = CVConfig(n_outer_repeats=2, n_outer_folds=5, n_inner_folds=3, stratify_outcome=True)
        rfe = RFEConfig(enabled=True, min_features=1, step=0.3, n_inner_cv=3, selector_model='rf')
        models = runner.get_default_models(42)
    else:
        cv = CVConfig(n_outer_repeats=10, n_outer_folds=10, n_inner_folds=10, stratify_outcome=True)
        rfe = RFEConfig(enabled=True, min_features=1, step=0.1, scoring='neg_mean_absolute_error', n_inner_cv=10, selector_model='rf')
        models = runner.get_models(42)
    if a.no_rfe: rfe = RFEConfig(enabled=False)
    out = Path(a.outroot) / (a.variant + ('_norfe' if a.no_rfe else '')); out.mkdir(parents=True, exist_ok=True)
    prepared = prepare(a.variant)
    t0 = time.time()
    res = runner.run_single_task(prepared, a.fs, a.model, cv, rfe, models, bootstrap=not a.quick, verbose=1)
    if res is None:
        print('no features'); return
    el = time.time() - t0
    stem = f"task_{a.fs}_{a.model}"
    with open(out / f"{stem}.pkl", 'wb') as f: pickle.dump(res, f)
    yp = np.asarray(res['y_pred_all'])
    pred = pd.DataFrame({'patient_id': res['subject_ids'], 'y_true': res['y_true'], 'y_pred_mean': res['y_pred_mean']})
    for r in range(yp.shape[1]): pred[f'rep{r+1}'] = yp[:, r]
    pred.to_csv(out / f"predictions_{a.fs}_{a.model}.csv", index=False)
    fr = [to_jsonable(x) for x in res['fold_results']]
    with open(out / f"folds_{a.fs}_{a.model}.json", 'w') as f: json.dump(fr, f)
    summ = dict(variant=a.variant + ('_norfe' if a.no_rfe else ''), feature_set=a.fs, model=a.model, n=res['n_samples'], quick=a.quick, elapsed_s=el,
                outcome_metadata=res.get('outcome_metadata'), overall_metrics=res['overall_metrics'],
                metrics_summary={k: {kk: vv for kk, vv in d.items() if kk != 'values'} for k, d in res['metrics_summary'].items()},
                bootstrap_ci=res.get('bootstrap_ci'), classification_metrics=res.get('classification_metrics'),
                classification_bootstrap_ci=res.get('classification_bootstrap_ci'), calibration=res.get('calibration'),
                conformal=res.get('conformal_intervals'), feature_stability=res['feature_stability'], feature_names=res['feature_names'],
                n_features_input=res['n_features_input'])
    with open(out / f"{stem}.json", 'w') as f: json.dump(to_jsonable(summ), f, indent=1, default=str)
    m = res['overall_metrics']; c = res.get('classification_metrics') or {}
    print(f"DONE {a.variant} {a.fs} {a.model} n={res['n_samples']} R2={m.get('r2'):.3f} MAE={m.get('mae'):.2f} F1={c.get('f1', float('nan')):.3f} [{el/60:.1f} min]")

if __name__ == '__main__':
    main()
