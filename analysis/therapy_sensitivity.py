#!/usr/bin/env python3
"""Prediction residuals by speech-language therapy receipt; SHAP rank correlation in therapy recipients."""
import os
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
REV = Path(os.environ.get('REV_ROOT', '.')); EXP = REV / 'experiments' / 'models'
A = pd.read_csv(REV / 'results' / 'aggregate' / 'all_tasks.csv'); df = pd.read_csv(REV / 'data' / 'dataset.csv')
df['slp'] = (pd.to_numeric(df.slp_tx_sessions_0to12mo.astype(str).str.extract(r'^\s*(\d+)')[0], errors='coerce').fillna(0) > 0).astype(int)
out = {}
for key, variant in [('wab', 'wab_base'), ('nct', 'nct_base')]:
    b = A[A.variant == variant].assign(_f1r=lambda x: x.f1.round(2)).sort_values(['_f1r', 'mcc'], ascending=False).iloc[0]
    P = pd.read_csv(EXP / variant / f'predictions_{b.fs}_{b.model}.csv').merge(df[['patient_id', 'slp']], on='patient_id'); P['resid'] = P.y_true - P.y_pred_mean
    u = stats.mannwhitneyu(P.resid[P.slp == 1], P.resid[P.slp == 0], alternative='two-sided')
    out[f'slp:{key}:med_yes'] = f"{P.resid[P.slp == 1].median():.1f}"; out[f'slp:{key}:med_no'] = f"{P.resid[P.slp == 0].median():.1f}"; out[f'slp:{key}:p'] = f"{u.pvalue:.3f}".lstrip('0'); out[f'slp:{key}:n_yes'] = int(P.slp.sum()); out[f'slp:{key}:n_no'] = int((P.slp == 0).sum())
    f = EXP / variant / f'shap_subjects_FS4_{b.model}.csv'
    if f.exists():
        S = pd.read_csv(f).merge(df[['patient_id', 'slp']], on='patient_id'); feats = [c for c in S.columns if c not in ('patient_id', 'repeat', 'fold', 'slp')]
        full = S[feats].abs().mean().sort_values(ascending=False); sub = S[S.slp == 1][feats].abs().mean()
        top = full.head(15).index; rho = stats.spearmanr(full[top], sub[top]).correlation; out[f'slp:{key}:rho'] = f"{rho:.2f}"
    else: out[f'slp:{key}:rho'] = 'PENDING'
json.dump(out, open(REV / 'results' / 'local' / 'slp_sensitivity.json', 'w'), indent=1); print(out)
