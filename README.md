# Acute-phase machine learning prediction of 12-month aphasia and discourse recovery

Analysis code for Marte et al., *Brain Communications* (BRAINCOM-2026-600).

- `engine/`: nested cross-validation (10-fold outer loop repeated 10 times, 10-fold inner grid search), recursive feature elimination within each training fold, inverse-density sample weighting, residualization of imaging features against lesion volume, and four algorithms (Ridge, SVR, random forest, XGBoost) predicting continuous 12-month outcomes, thresholded post hoc.
- `analysis/`: each model and sensitivity analysis, the comparison of feature sets and algorithms (corrected resampled paired t-test), SHAP attributions, participant characteristics and ordinary least squares baselines, and the therapy sensitivity check.
- `figures/`: figures and graphical abstract.

Patient data are not included. They are available as described in the article's Data availability statement. Set `REV_ROOT` to the directory that holds `data/` and `experiments/`. Subject identifiers used for exclusions are read from local, git-ignored files (see `engine/config.py`).

Python 3.11 (scikit-learn 1.4, XGBoost 2.0, SHAP 0.44, NumPy 1.26, SciPy 1.11): `pip install -r requirements.txt`. R ≥ 4.3 with ggplot2, patchwork, ragg and ggseg.
