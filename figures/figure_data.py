#!/usr/bin/env python3
"""Exports the values plotted in the figures."""
import os
import json
from pathlib import Path
import numpy as np, pandas as pd
REV = Path(os.environ.get('REV_ROOT', '.')); EXP = REV / 'experiments' / 'models'; OUT = REV / 'results' / 'figdata'; OUT.mkdir(exist_ok=True)
A = pd.read_csv(REV / 'results' / 'aggregate' / 'all_tasks.csv'); N = json.load(open(REV / 'manuscript' / 'numbers.json'))
def best(variant, fs=None):
    d = A[A.variant == variant]; d = d[d.fs == fs] if fs else d
    return d.assign(_f1r=d.f1.round(2)).sort_values(['_f1r', 'mcc'], ascending=False).iloc[0] if len(d) else None
rows = []
for key, variant, thr in [('aphasia_resolution', 'wab_base', 93.8), ('discourse_content', 'nct_base', 22.1)]:
    b = best(variant)
    if b is None: continue
    P = pd.read_csv(EXP / variant / f'predictions_{b.fs}_{b.model}.csv'); reps = [c for c in P.columns if c.startswith('rep')]
    P['outcome'] = key; P['fs'] = b.fs; P['model'] = b.model; P['thr'] = thr; P['pred_sd'] = P[reps].std(axis=1); P['pred_min'] = P[reps].min(axis=1); P['pred_max'] = P[reps].max(axis=1)
    P['cls_true'] = (P.y_true >= thr).astype(int); P['cls_pred'] = (P.y_pred_mean >= thr).astype(int); P['correct'] = P.cls_true == P.cls_pred; rows.append(P)
pd.concat(rows).to_csv(OUT / 'best_predictions.csv', index=False)
cells = {'c_wab_own': ('12-month WAB-AQ', 'Own'), 'c_wab_cu': ('12-month WAB-AQ', 'Other'), 'c_wab_both': ('12-month WAB-AQ', 'Both'), 'c_nct_own': ('12-month content units', 'Own'), 'c_nct_aq': ('12-month content units', 'Other'), 'c_nct_both': ('12-month content units', 'Both')}
cr = []
for v, (oc, pr) in cells.items():
    for fs in ['FS1', 'FS4']:
        b = best(v, fs)
        if b is not None: cr.append(dict(outcome=oc, predictor=pr, fs=fs, model=b.model, r2=b.r2, r2_lo=b.r2_lo, r2_hi=b.r2_hi, f1=b.f1, f1_lo=b.f1_lo, f1_hi=b.f1_hi, mae=b.mae))
pd.DataFrame(cr).to_csv(OUT / 'cells.csv', index=False)
sr = []
for key, variant in [('aphasia_resolution', 'wab_base'), ('discourse_content', 'nct_base')]:
    b = best(variant)
    if b is None: continue
    f = EXP / f'{variant}_norfe' / f'shap_rank_FS4_{b.model}.csv'; fs_ = EXP / variant / f'shap_rank_FS4_{b.model}.csv'
    if f.exists():
        s = pd.read_csv(f); s['outcome'] = key; s['model'] = b.model
        if fs_.exists(): s = s.drop(columns=['retention']).merge(pd.read_csv(fs_)[['feature', 'retention']], on='feature', how='left')
        sr.append(s)
if sr: pd.concat(sr).to_csv(OUT / 'shap_ranks.csv', index=False)
df = pd.read_csv(REV / 'data' / 'dataset.csv')[['patient_id', 'wab_aq_acute', 'wab_aq_12mo', 'nct_total_cu_acute', 'nct_total_cu_12mo']]; df.to_csv(OUT / 'outcomes.csv', index=False)
print('exported', [p.name for p in OUT.glob('*.csv')])
