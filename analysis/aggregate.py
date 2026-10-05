#!/usr/bin/env python3
"""Collects results across analyses; corrected resampled paired t-test (Nadeau and Bengio 2003) on fold-level error; feature retention frequencies."""
import os
import json, glob, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
ROOT = Path(os.environ.get('REV_ROOT', '.')) / 'experiments' / 'models'
OUT = Path(os.environ.get('REV_ROOT', '.')) / 'results' / 'aggregate'; OUT.mkdir(parents=True, exist_ok=True)

def load_all():
    rows = []
    for f in sorted(ROOT.glob('*/task_*.json')):
        j = json.load(open(f)); m = j['overall_metrics']; c = j.get('classification_metrics') or {}; bc = j.get('bootstrap_ci') or {}; cc = j.get('classification_bootstrap_ci') or {}
        def ci(d, k):
            v = d.get(k); return (v[0], v[1]) if v and not any(pd.isna(x) for x in v) else (np.nan, np.nan)
        r = dict(variant=j['variant'], fs=j['feature_set'], model=j['model'], n=j['n'], threshold=j['outcome_metadata'].get('threshold'),
                 r2=m.get('r2'), mae=m.get('mae'), pearson_r=m.get('pearson_r'), spearman=m.get('spearman_rho'),
                 r2_sd=j['metrics_summary'].get('r2', {}).get('std'), mae_sd=j['metrics_summary'].get('mae', {}).get('std'),
                 f1=c.get('f1'), sens=c.get('recall'), spec=c.get('specificity'), ppv=c.get('ppv'), npv=c.get('npv'), bal_acc=c.get('balanced_accuracy'), mcc=c.get('mcc'), auc=c.get('auc_roc'),
                 cal_slope=(j.get('calibration') or {}).get('calibration_slope'), cal_intercept=(j.get('calibration') or {}).get('calibration_intercept'), elapsed_min=j.get('elapsed_s', 0) / 60)
        thr = r['threshold']; pf = f.parent / f"predictions_{j['feature_set']}_{j['model']}.csv"
        if pf.exists() and thr is not None:
            P = pd.read_csv(pf); yt, yp = P.y_true >= thr, P.y_pred_mean >= thr
            r.update(tp=int((yt & yp).sum()), tn=int((~yt & ~yp).sum()), fp=int((~yt & yp).sum()), fn=int((yt & ~yp).sum()))
        for k in ['r2', 'mae', 'pearson_r']: r[f'{k}_lo'], r[f'{k}_hi'] = ci(bc, k)
        for k, kk in [('f1', 'f1'), ('sensitivity', 'recall'), ('specificity', 'specificity'), ('ppv', 'ppv'), ('npv', 'npv'), ('balanced_accuracy', 'balanced_accuracy'), ('mcc', 'mcc'), ('auc', 'auc_roc')]: r[f'{k}_lo'], r[f'{k}_hi'] = ci(cc, kk)
        fs_stab = j.get('feature_stability') or {}; r['top_selected'] = '; '.join(f"{k}:{v:.2f}" for k, v in sorted(fs_stab.items(), key=lambda x: -x[1])[:6])
        r['n_feat_input'] = j.get('n_features_input'); rows.append(r)
    return pd.DataFrame(rows)

def fold_errors(variant, fs, model):
    f = ROOT / variant / f'folds_{fs}_{model}.json'
    if not f.exists(): return None
    fr = json.load(open(f)); out = []
    for x in fr:
        yt = np.asarray(x['y_true'], float); yp = np.asarray(x['y_pred'], float)
        out.append(dict(repeat=x['repeat'], fold=x['fold'], mae=np.mean(np.abs(yt - yp)), n_test=len(yt), n_train=len(x['train_idx']), selected=x['selected_features']))
    return pd.DataFrame(out)

def corrected_t(d, n_train, n_test):
    d = np.asarray(d, float); J = len(d); m = d.mean(); v = d.var(ddof=1)
    se = np.sqrt(v * (1.0 / J + n_test / n_train)) if v > 0 else np.nan
    t = m / se if se and se > 0 else np.nan; p = 2 * stats.t.sf(abs(t), J - 1) if not np.isnan(t) else np.nan
    return m, se, t, p

def main():
    df = load_all(); df.to_csv(OUT / 'all_tasks.csv', index=False)
    print(df[['variant', 'fs', 'model', 'n', 'r2', 'mae', 'f1', 'mcc', 'elapsed_min']].round(3).to_string(index=False))
    comps = []
    for variant in df.variant.unique():
        sub = df[df.variant == variant]
        for model in sub.model.unique():
            fss = sorted(sub[sub.model == model].fs.unique())
            fe = {fs: fold_errors(variant, fs, model) for fs in fss}
            for a, b in zip(fss[:-1], fss[1:]):
                if fe[a] is None or fe[b] is None: continue
                mrg = fe[a].merge(fe[b], on=['repeat', 'fold'], suffixes=('_a', '_b'))
                d = mrg.mae_a - mrg.mae_b
                m, se, t, p = corrected_t(d, mrg.n_train_a.mean(), mrg.n_test_a.mean())
                comps.append(dict(variant=variant, model=model, comparison=f'{a} -> {b}', metric='MAE', mean_diff=m, se_corrected=se, t=t, df=len(d) - 1, p=p, ci_lo=m - stats.t.ppf(0.975, len(d) - 1) * se, ci_hi=m + stats.t.ppf(0.975, len(d) - 1) * se))
            if 'FS1' in fe and 'FS4' in fe and fe['FS1'] is not None and fe['FS4'] is not None:
                mrg = fe['FS1'].merge(fe['FS4'], on=['repeat', 'fold'], suffixes=('_a', '_b')); d = mrg.mae_a - mrg.mae_b
                m, se, t, p = corrected_t(d, mrg.n_train_a.mean(), mrg.n_test_a.mean())
                comps.append(dict(variant=variant, model=model, comparison='FS1 -> FS4', metric='MAE', mean_diff=m, se_corrected=se, t=t, df=len(d) - 1, p=p, ci_lo=m - stats.t.ppf(0.975, len(d) - 1) * se, ci_hi=m + stats.t.ppf(0.975, len(d) - 1) * se))
        for fs in sub.fs.unique():
            fr_r = fold_errors(variant, fs, 'ridge')
            if fr_r is None: continue
            for model in [m for m in sub[sub.fs == fs].model.unique() if m != 'ridge']:
                fm = fold_errors(variant, fs, model)
                if fm is None: continue
                mrg = fr_r.merge(fm, on=['repeat', 'fold'], suffixes=('_a', '_b')); d = mrg.mae_a - mrg.mae_b
                m, se, t, p = corrected_t(d, mrg.n_train_a.mean(), mrg.n_test_a.mean())
                comps.append(dict(variant=variant, model=model, comparison=f'{fs}: ridge -> {model}', metric='MAE', mean_diff=m, se_corrected=se, t=t, df=len(d) - 1, p=p, ci_lo=m - stats.t.ppf(0.975, len(d) - 1) * se, ci_hi=m + stats.t.ppf(0.975, len(d) - 1) * se))
    C = pd.DataFrame(comps); C.to_csv(OUT / 'corrected_t_comparisons.csv', index=False)
    if len(C): print(C.round(4).to_string(index=False))
    sel = []
    for f in sorted(ROOT.glob('*/folds_*.json')):
        variant = f.parent.name; fs, model = f.stem.replace('folds_', '').split('_', 1); fr = json.load(open(f))
        cnt = {}
        for x in fr:
            for s in x['selected_features']: cnt[s] = cnt.get(s, 0) + 1
        nsel = np.mean([len(x['selected_features']) for x in fr])
        for k, v in cnt.items(): sel.append(dict(variant=variant, fs=fs, model=model, feature=k, selected_frac=v / len(fr), mean_n_selected=nsel))
    pd.DataFrame(sel).to_csv(OUT / 'rfe_selection_frequency.csv', index=False)
    json.dump(dict(n_tasks=len(df), variants=sorted(df.variant.unique())), open(OUT / 'status.json', 'w'))

if __name__ == '__main__':
    main()
