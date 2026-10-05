#!/usr/bin/env python3
"""SHAP attributions computed on the held-out patients of each of the 100 outer folds."""
import sys, json, pickle, argparse, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
SRC = Path(str(Path(__file__).resolve().parents[1] / 'engine')); sys.path.insert(0, str(SRC)); sys.path.insert(0, str(Path(__file__).parent))
import run_model as rt
from cv_engine import preprocess_fold
import shap

ap = argparse.ArgumentParser(); ap.add_argument('--variant', required=True); ap.add_argument('--fs', required=True); ap.add_argument('--model', required=True)
a = ap.parse_args()
base_variant = a.variant.replace('_norfe', ''); runner = rt.import_runner(); prepared = rt.prepare(base_variant)
X, names, fs_cfg = runner.extract_fs_data(prepared, a.fs); y = prepared['y']; vol = prepared['vol']; w = prepared['weights']; ids = prepared['subject_ids']
tdir = rt.OUTROOT / a.variant; res = pickle.load(open(tdir / f'task_{a.fs}_{a.model}.pkl', 'rb'))
models = runner.get_models(42); mc = models[a.model]
per_fold = []; per_subject = []
for fr in res['fold_results']:
    tr, te = np.asarray(fr.train_idx), np.asarray(fr.test_idx)
    Xtr, Xte, _ = preprocess_fold(X[tr].copy(), X[te].copy(), names, vol[tr], vol[te], residualize=True, skip_residualize=fs_cfg.skip_residualize)
    sel = [names.index(f) for f in fr.selected_features]; Xtr, Xte = Xtr[:, sel], Xte[:, sel]; fnames = list(fr.selected_features)
    est = mc.get_estimator(42); est.set_params(**fr.best_params)
    try: est.fit(Xtr, y[tr], sample_weight=w[tr])
    except TypeError: est.fit(Xtr, y[tr])
    if a.model in ('rf', 'xgb'):
        ex = shap.TreeExplainer(est); sv = ex.shap_values(Xte)
    elif a.model == 'ridge':
        ex = shap.LinearExplainer(est, Xtr); sv = ex.shap_values(Xte)
    else:
        bg = shap.kmeans(Xtr, min(20, len(Xtr))); ex = shap.KernelExplainer(est.predict, bg); sv = ex.shap_values(Xte, nsamples=200, silent=True)
    sv = np.asarray(sv); mabs = dict(zip(fnames, np.abs(sv).mean(0)))
    per_fold.append(dict(repeat=fr.repeat, fold=fr.fold, **{f: mabs.get(f, 0.0) for f in names}))
    for k, i in enumerate(te):
        row = dict(patient_id=ids[i], repeat=fr.repeat, fold=fr.fold); row.update({f: 0.0 for f in names}); row.update(dict(zip(fnames, sv[k]))); per_subject.append(row)
PF = pd.DataFrame(per_fold); PS = pd.DataFrame(per_subject)
summ = PF[names].mean().sort_values(ascending=False).rename('mean_abs_shap').reset_index().rename(columns={'index': 'feature'})
summ['rank'] = np.arange(1, len(summ) + 1); summ['retention'] = summ.feature.map(res['feature_stability'])
summ.to_csv(tdir / f'shap_rank_{a.fs}_{a.model}.csv', index=False); PF.to_csv(tdir / f'shap_folds_{a.fs}_{a.model}.csv', index=False); PS.to_csv(tdir / f'shap_subjects_{a.fs}_{a.model}.csv', index=False)
print(summ.head(15).to_string(index=False))
