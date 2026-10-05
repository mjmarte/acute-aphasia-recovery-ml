#!/usr/bin/env python3
"""Participant characteristics, outcome distributions and ordinary least squares baselines under the same outer folds."""
import os
import json, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, mean_absolute_error, f1_score, roc_auc_score, balanced_accuracy_score
import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'engine'))
from config import FEATURE_SETS, CATEGORICAL_ENCODINGS
from feature_eng import compute_density_weights

REV = Path(os.environ.get('REV_ROOT', '.')); OUT = REV / 'results' / 'local'; OUT.mkdir(exist_ok=True)
df = pd.read_csv(REV / 'data' / 'dataset.csv')
red = pd.read_csv(REV / 'data' / 'reduced_inventory.csv')[['patient_id', 'cu_reduced_12mo', 'cu_reduced_acute', 'cu_full_12mo']]
df = df.merge(red, on='patient_id', how='left')
S = {}
df['resolved'] = (df.wab_aq_12mo >= 93.8).astype(int); df['resolved967'] = (df.wab_aq_12mo >= 96.7).astype(int)
df['aphasic_ac'] = (df.wab_aq_acute < 93.8).astype(int); df['cu_norm'] = np.where(df.nct_total_cu_12mo.notna(), (df.nct_total_cu_12mo >= 22.1).astype(float), np.nan)
df['has_cu12'] = df.nct_total_cu_12mo.notna(); df['has_cuac'] = df.nct_total_cu_acute.notna(); df['common'] = df.has_cu12 & df.has_cuac
code = {'0.0': 'No aphasia', '0': 'No aphasia', '1': 'Global', '2': "Broca's", '3': 'Isolation', '4': 'Transcortical motor', '5': "Wernicke's", '6': 'Transcortical sensory', '7': 'Conduction', '8': 'Anomic'}
for tp in ['acute', '12mo']:
    c = f'wab_aphasia_type_{tp}'; t = df[c].astype(str).map(lambda s: code.get(s, s if s != 'nan' else np.nan))
    aq = df[f'wab_aq_{tp}']; t = t.where(~(t.isna() & (aq >= 93.8)), 'No aphasia'); df[f'type_{tp}'] = t

def welch(a, b):
    a, b = pd.Series(a).dropna(), pd.Series(b).dropna(); t, p = stats.ttest_ind(a, b, equal_var=False); return t, p
def fisher(x, g):
    ct = pd.crosstab(x, g); 
    if ct.shape == (2, 2): return stats.fisher_exact(ct.values)[1]
    return stats.chi2_contingency(ct.values)[1] if ct.size else np.nan
rows = []
def t1_cont(name, col, grp, gcol):
    o = df[col]; a = df.loc[df[gcol] == 1, col]; b = df.loc[df[gcol] == 0, col]; t, p = welch(a, b)
    rows.append(dict(table=grp, characteristic=name, overall=f"{o.mean():.1f} ({o.std():.1f})", g1=f"{a.mean():.1f} ({a.std():.1f})", g0=f"{b.mean():.1f} ({b.std():.1f})", p=p, n_overall=int(o.notna().sum()), n_g1=int(a.notna().sum()), n_g0=int(b.notna().sum())))
def t1_pct(name, mask, grp, gcol, testcol=None):
    o = mask.mean() * 100; a = mask[df[gcol] == 1].mean() * 100; b = mask[df[gcol] == 0].mean() * 100
    p = fisher(df[testcol] if testcol else mask.astype(int), df[gcol]) if testcol or True else np.nan
    rows.append(dict(table=grp, characteristic=name, overall=f"{o:.1f}", g1=f"{a:.1f}", g0=f"{b:.1f}", p=p))
for grp, gcol, sub in [('by_resolution', 'resolved', df.index), ('by_cu_norm', 'cu_norm', df.index[df.has_cu12])]:
    d0 = df; df = df.loc[sub].copy(); df[gcol] = df[gcol].astype(int)
    t1_cont('Age, y', 'age_at_stroke', grp, gcol); t1_pct('Sex, % female', (df.sex == 'F'), grp, gcol)
    for r, code_ in [('Asian', 'A'), ('Black', 'B'), ('Other', 'O'), ('White', 'W')]: t1_pct(f'Race {r}, %', (df.race == code_), grp, gcol)
    t1_cont('Education, y', 'education_yrs', grp, gcol); t1_pct('Prior stroke, %', (df.prior_stroke == 'Y'), grp, gcol)
    t1_cont('Acute WAB-AQ', 'wab_aq_acute', grp, gcol); t1_cont('12-month WAB-AQ', 'wab_aq_12mo', grp, gcol)
    t1_cont('Acute content units', 'nct_total_cu_acute', grp, gcol); t1_cont('12-month content units', 'nct_total_cu_12mo', grp, gcol)
    t1_cont('Lesion volume, mL', 'lesion_volume_mL', grp, gcol); t1_cont('Days post onset', 'dpo_acute', grp, gcol)
    t1_pct('Aphasic at acute visit (WAB-AQ < 93.8), %', (df.wab_aq_acute < 93.8), grp, gcol)
    for ty in ['No aphasia', 'Anomic', "Broca's", 'Conduction', 'Global', 'Isolation', 'Transcortical motor', 'Transcortical sensory', "Wernicke's"]:
        t1_pct(f'Acute type {ty}, %', (df.type_acute == ty), grp, gcol)
    S[f'{grp}_n'] = dict(overall=len(df), g1=int((df[gcol] == 1).sum()), g0=int((df[gcol] == 0).sum()))
    df = d0
pd.DataFrame(rows).to_csv(OUT / 'table1.csv', index=False)
S['race_pct'] = (df.race.value_counts(normalize=True) * 100).round(1).to_dict()
S['types_acute'] = df.type_acute.value_counts(dropna=False).to_dict(); S['types_12mo'] = df.type_12mo.value_counts(dropna=False).to_dict()
S['type_missing_acute_ids'] = df.loc[df.type_acute.isna(), 'patient_id'].tolist(); S['type_missing_12mo_ids'] = df.loc[df.type_12mo.isna(), 'patient_id'].tolist()

S['n'] = len(df); S['resolved_938'] = int(df.resolved.sum()); S['resolved_967'] = int(df.resolved967.sum()); S['reclassified_938_to_967'] = int(((df.resolved == 1) & (df.resolved967 == 0)).sum())
S['aphasic_acute'] = int(df.aphasic_ac.sum()); S['resolved_among_aphasic'] = int(df.loc[df.aphasic_ac == 1, 'resolved'].sum()); S['nonaphasic_acute'] = int((df.aphasic_ac == 0).sum())
S['relapsed_nonaphasic'] = int(((df.aphasic_ac == 0) & (df.resolved == 0)).sum())
S['n_cu12'] = int(df.has_cu12.sum()); S['cu_norm_n'] = int(np.nansum(df.cu_norm)); S['n_cu_acute'] = int(df.has_cuac.sum()); S['n_common'] = int(df.common.sum())
S['resolved_prop_with_cu12'] = round(df.loc[df.has_cu12, 'resolved'].mean(), 3); S['resolved_prop_without_cu12'] = round(df.loc[~df.has_cu12, 'resolved'].mean(), 3)
S['resolved_with_without_p'] = fisher(df.has_cu12.astype(int), df.resolved)
na = df[(df.aphasic_ac == 0) & df.has_cu12]; S['nonaphasic_with_cu12'] = len(na); S['nonaphasic_not_normalized'] = int((na.cu_norm == 0).sum())
d938 = df[df.has_cu12]; S['resolved_not_normalized_938'] = int(((d938.resolved == 1) & (d938.cu_norm == 0)).sum()); S['resolved_not_normalized_967'] = int(((d938.resolved967 == 1) & (d938.cu_norm == 0)).sum())
S['normalized_not_resolved_938'] = int(((d938.resolved == 0) & (d938.cu_norm == 1)).sum())
for col, lab in [('wab_aq_12mo', 'aq12'), ('nct_total_cu_12mo', 'cu12'), ('wab_aq_acute', 'aqac'), ('nct_total_cu_acute', 'cuac')]:
    x = df[col].dropna(); S[f'dist_{lab}'] = dict(n=len(x), mean=round(x.mean(), 1), sd=round(x.std(), 1), median=float(x.median()), min=float(x.min()), max=float(x.max()), skew=round(stats.skew(x), 2), n_ge98=int((x >= 98).sum()), n_eq100=int((x == 100).sum()), n_top10pct=int((x >= 0.9 * x.max()).sum()))
ps = df.prior_stroke == 'Y'; S['prior_stroke'] = dict(n=int(ps.sum()), resolved_pct=round(df.loc[ps, 'resolved'].mean() * 100, 1), noprior_resolved_pct=round(df.loc[~ps, 'resolved'].mean() * 100, 1), p_res=fisher(ps.astype(int), df.resolved),
    cu_norm_pct=round(df.loc[ps & df.has_cu12, 'cu_norm'].mean() * 100, 1), noprior_cu_norm_pct=round(df.loc[~ps & df.has_cu12, 'cu_norm'].mean() * 100, 1), p_cu=fisher(ps[df.has_cu12].astype(int), df.loc[df.has_cu12, 'cu_norm']),
    aq_ac_mean=round(df.loc[ps, 'wab_aq_acute'].mean(), 1), aq_ac_noprior=round(df.loc[~ps, 'wab_aq_acute'].mean(), 1), aq12_mean=round(df.loc[ps, 'wab_aq_12mo'].mean(), 1), aq12_noprior=round(df.loc[~ps, 'wab_aq_12mo'].mean(), 1))

def smd(a, b):
    a, b = pd.Series(a).dropna(), pd.Series(b).dropna(); sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2); return (a.mean() - b.mean()) / sp if sp > 0 else np.nan
sm = []
for name, col in [('Age', 'age_at_stroke'), ('Education', 'education_yrs'), ('Acute WAB-AQ', 'wab_aq_acute'), ('12-month WAB-AQ', 'wab_aq_12mo'), ('Lesion volume', 'lesion_volume_mL'), ('log lesion volume', None), ('12-month content units', 'nct_total_cu_12mo'), ('Days post onset', 'dpo_acute')]:
    x = np.log1p(df.lesion_volume_mL) if col is None else df[col]
    sm.append(dict(variable=name, common_mean=round(x[df.common].mean(), 2), other_mean=round(x[~df.common].mean(), 2), smd=round(smd(x[df.common], x[~df.common]), 2), n_common=int(x[df.common].notna().sum()), n_other=int(x[~df.common].notna().sum())))
for name, mask in [('Female', df.sex == 'F'), ('Prior stroke', df.prior_stroke == 'Y'), ('Black', df.race == 'B'), ('Aphasic acutely', df.aphasic_ac == 1), ('Resolved at 12 mo', df.resolved == 1)]:
    p1, p0 = mask[df.common].mean(), mask[~df.common].mean(); h = 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p0)))
    sm.append(dict(variable=name + ' (proportion; Cohen h)', common_mean=round(p1, 2), other_mean=round(p0, 2), smd=round(h, 2), n_common=int(df.common.sum()), n_other=int((~df.common).sum())))
pd.DataFrame(sm).to_csv(OUT / 'smd_common_vs_rest.csv', index=False)

def cv_reg(X, y, thr, weights=None):
    X = np.asarray(X, float); y = np.asarray(y, float)
    bins = KBinsDiscretizer(4, encode='ordinal', strategy='quantile').fit_transform(y.reshape(-1, 1)).ravel().astype(int)
    rk = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42); splits = list(rk.split(X, bins))
    R2, MAE, F1, AUC, BA = [], [], [], [], []
    for rep in range(10):
        pred = np.zeros_like(y)
        for tr, te in splits[rep * 10:(rep + 1) * 10]:
            Xtr, Xte = X[tr].copy(), X[te].copy(); med = np.nanmedian(Xtr, 0); Xtr = np.where(np.isnan(Xtr), med, Xtr); Xte = np.where(np.isnan(Xte), med, Xte)
            m = LinearRegression().fit(Xtr, y[tr], sample_weight=None if weights is None else weights[tr]); pred[te] = m.predict(Xte)
        R2.append(r2_score(y, pred)); MAE.append(mean_absolute_error(y, pred)); F1.append(f1_score(y >= thr, pred >= thr)); BA.append(balanced_accuracy_score(y >= thr, pred >= thr))
        try: AUC.append(roc_auc_score(y >= thr, pred))
        except Exception: AUC.append(np.nan)
    return dict(R2=np.mean(R2), R2_sd=np.std(R2), MAE=np.mean(MAE), F1=np.mean(F1), F1_sd=np.std(F1), AUC=np.nanmean(AUC), BA=np.mean(BA))
def cv_logit(X, y01, weights=None):
    X = np.asarray(X, float); y01 = np.asarray(y01, int)
    rk = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42); splits = list(rk.split(X, y01)); F1, AUC, BA = [], [], []
    for rep in range(10):
        prob = np.zeros(len(y01))
        for tr, te in splits[rep * 10:(rep + 1) * 10]:
            Xtr, Xte = X[tr].copy(), X[te].copy(); med = np.nanmedian(Xtr, 0); Xtr = np.where(np.isnan(Xtr), med, Xtr); Xte = np.where(np.isnan(Xte), med, Xte)
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9; m = LogisticRegression(C=1.0, max_iter=1000).fit((Xtr - mu) / sd, y01[tr]); prob[te] = m.predict_proba((Xte - mu) / sd)[:, 1]
        F1.append(f1_score(y01, prob >= 0.5)); AUC.append(roc_auc_score(y01, prob)); BA.append(balanced_accuracy_score(y01, prob >= 0.5))
    return dict(F1=np.mean(F1), AUC=np.mean(AUC), BA=np.mean(BA))
enc = df.copy(); enc['sex'] = enc.sex.map(CATEGORICAL_ENCODINGS['sex']); enc['prior_stroke'] = enc.prior_stroke.map(CATEGORICAL_ENCODINGS['prior_stroke'])
FS1 = ['wab_aq_acute', 'age_at_stroke', 'sex', 'education_yrs', 'prior_stroke']; FS2 = FS1 + ['lesion_volume_mL']
ols = []
def run_block(label, d, ycol, thr, own):
    y = d[ycol].values; w, _ = compute_density_weights(y)
    specs = [(f'{own} only', [own]), (f'{own} + lesion volume', [own, 'lesion_volume_mL']), ('FS1 (with ' + own + ')', [own] + FS1[1:]), ('FS2 (with ' + own + ')', [own] + FS2[1:])]
    for name, cols in specs:
        for wl, ww in [('unweighted', None), ('KDE-weighted', w)]:
            r = cv_reg(d[cols].values, y, thr, ww); ols.append(dict(block=label, outcome=ycol, n=len(d), spec=name, weighting=wl, **{k: round(float(v), 4) for k, v in r.items()}))
run_block('WAB n=73', enc, 'wab_aq_12mo', 93.8, 'wab_aq_acute')
run_block('CU n=61 (acute AQ)', enc[enc.has_cu12], 'nct_total_cu_12mo', 22.1, 'wab_aq_acute')
c = enc[enc.common]
run_block('common WAB own', c, 'wab_aq_12mo', 93.8, 'wab_aq_acute'); run_block('common WAB from acute CU', c, 'wab_aq_12mo', 93.8, 'nct_total_cu_acute')
run_block('common CU own', c, 'nct_total_cu_12mo', 22.1, 'nct_total_cu_acute'); run_block('common CU from acute AQ', c, 'nct_total_cu_12mo', 22.1, 'wab_aq_acute')
for lab, cols, ycol, thr in [('common WAB both', ['wab_aq_acute', 'nct_total_cu_acute'], 'wab_aq_12mo', 93.8), ('common CU both', ['wab_aq_acute', 'nct_total_cu_acute'], 'nct_total_cu_12mo', 22.1)]:
    y = c[ycol].values; w, _ = compute_density_weights(y)
    for wl, ww in [('unweighted', None), ('KDE-weighted', w)]:
        r = cv_reg(c[cols].values, y, thr, ww); ols.append(dict(block=lab, outcome=ycol, n=len(c), spec='both acute measures', weighting=wl, **{k: round(float(v), 4) for k, v in r.items()}))
rr = enc[enc.cu_reduced_12mo.notna()]; run_block('reduced CU n=59 (acute AQ)', rr, 'cu_reduced_12mo', 17.7, 'wab_aq_acute')
rc = enc[enc.cu_reduced_12mo.notna() & enc.cu_reduced_acute.notna()]; run_block('reduced CU common own', rc, 'cu_reduced_12mo', 17.7, 'cu_reduced_acute')
for name, cols in [('wab_aq_acute only', ['wab_aq_acute']), ('FS2', FS2)]:
    r = cv_reg(enc[cols].values, enc.wab_aq_12mo.values, 96.7, None); ols.append(dict(block='WAB n=73 at 96.7', outcome='wab_aq_12mo', n=73, spec=name, weighting='unweighted', **{k: round(float(v), 4) for k, v in r.items()}))
pd.DataFrame(ols).to_csv(OUT / 'ols_baselines.csv', index=False)
lg = []
for thr in [93.8, 96.7]:
    for name, cols in [('FS1', FS1), ('FS2', FS2), ('acute AQ only', ['wab_aq_acute'])]:
        r = cv_logit(enc[cols].values, (enc.wab_aq_12mo >= thr).astype(int)); lg.append(dict(threshold=thr, spec=name, n_pos=int((enc.wab_aq_12mo >= thr).sum()), **{k: round(float(v), 4) for k, v in r.items()}))
pd.DataFrame(lg).to_csv(OUT / 'logistic_classifiers.csv', index=False)

feats = FEATURE_SETS['FS4']['features'] + ['nct_total_cu_acute']
X = enc[[f for f in feats if f in enc.columns]].copy(); X['log_lesion_volume'] = np.log1p(enc.lesion_volume_mL)
C = X.corr(method='spearman'); C.to_csv(OUT / 'predictor_correlations_spearman.csv')
S['r_aq_logvol_spearman'] = round(C.loc['wab_aq_acute', 'log_lesion_volume'], 3); S['r_aq_vol_pearson'] = round(enc[['wab_aq_acute', 'lesion_volume_mL']].corr().iloc[0, 1], 3)
S['r_aq_logvol_pearson'] = round(np.corrcoef(enc.wab_aq_acute, np.log1p(enc.lesion_volume_mL))[0, 1], 3)
S['r_aqac_cuac_pearson'] = round(c[['wab_aq_acute', 'nct_total_cu_acute']].corr().iloc[0, 1], 3); S['r_aq12_cu12_pearson'] = round(enc[enc.has_cu12][['wab_aq_12mo', 'nct_total_cu_12mo']].corr().iloc[0, 1], 3)
img = [f for f in FEATURE_SETS['FS4']['residualize'] if f in C.columns]
S['imaging_vs_aq_abs_r_median'] = round(C.loc[img, 'wab_aq_acute'].abs().median(), 2); S['imaging_vs_aq_abs_r_max'] = round(C.loc[img, 'wab_aq_acute'].abs().max(), 2)
S['imaging_vs_logvol_abs_r_median'] = round(C.loc[img, 'log_lesion_volume'].abs().median(), 2)
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(13, 11)); im = ax.imshow(C.values, cmap='RdBu_r', vmin=-1, vmax=1); ax.set_xticks(range(len(C))); ax.set_yticks(range(len(C))); ax.set_xticklabels(C.columns, rotation=90, fontsize=7); ax.set_yticklabels(C.columns, fontsize=7); plt.colorbar(im, ax=ax, fraction=0.03, label='Spearman rho'); plt.tight_layout(); plt.savefig(OUT / 'predictor_correlations_spearman.png', dpi=300); plt.close()
json.dump(S, open(OUT / 'summary.json', 'w'), indent=1, default=str)
print(json.dumps(S, indent=1, default=str)[:6000])
print(pd.DataFrame(ols)[['block', 'spec', 'weighting', 'R2', 'MAE', 'F1', 'AUC']].to_string(index=False))
print(pd.DataFrame(lg).to_string(index=False)); print(pd.DataFrame(sm).to_string(index=False))
