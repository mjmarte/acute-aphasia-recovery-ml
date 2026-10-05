"""Feature sets, outcome definitions and model settings."""
from pathlib import Path

PROJECT_ROOT    = Path(__file__).parent.parent
DATA_RAW        = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED  = PROJECT_ROOT / "data" / "processed"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
FIGURES_DIR     = PROJECT_ROOT / "figures"

MASTER_CSV = DATA_RAW / "dataset.csv"

ID_COL = "patient_id"

WAB_AQ_COL_12M   = "wab_aq_12mo"
WAB_AQ_THRESHOLD = 93.8
WAB_AQ_DIRECTION = ">="

NCT_CU_COL_12M   = "nct_total_cu_12mo"
NCT_CU_THRESHOLD = 22.1
NCT_CU_DIRECTION = ">="

N_WAB = 73
N_NCT = 61

import os as _os
def _ids(name):
    f = _os.path.join(_os.path.dirname(__file__), name)
    return [l.strip() for l in open(f) if l.strip()] if _os.path.exists(f) else []
EXCLUDED_SUBJECTS = _ids("excluded_subjects.txt")
QC_FLAG_SUBJECTS = {k: "QC flag" for k in _ids("qc_flag_subjects.txt")}

CATEGORICAL_ENCODINGS = {
    "sex":          {"M": 0, "F": 1},
    "prior_stroke": {"N": 0, "Y": 1},
}

CLINICAL_BASELINE = [
    "wab_aq_acute",
    "age_at_stroke",
    "sex",
    "education_yrs",
    "prior_stroke",
]

LESION_VOLUME_COL = "lesion_volume_mL"

APRIORI_BPM_LANGUAGE_CORTICAL = [
    "BPM_IFG_opercularis_L",
    "BPM_IFG_triangularis_L",
    "BPM_STG_L",
    "BPM_STG_L_pole",
    "BPM_MTG_L",
    "BPM_SMG_L",
    "BPM_AG_L",
    "BPM_Ins_L",
    "BPM_FuG_L",
    "BPM_MFG_L",
]

APRIORI_TRACT_LANGUAGE = [
    "SLF_L_prob",
    "SLFt_L_prob",
    "IFOF_L_prob",
    "ILF_L_prob",
    "UF_L_prob",
]

APRIORI_ANATOMY = APRIORI_BPM_LANGUAGE_CORTICAL + APRIORI_TRACT_LANGUAGE

APRIORI_NEMO_COMPOSITES = [
    "lang_right_mean_chaco",
    "lang_interhemispheric_disconnection",
]

APRIORI_NEMO_PAIRS = [
    "pair_POP-STG",
    "pair_POP-SMG",
    "pair_PTR-STG",
    "pair_POP-MTG",
    "pair_PTR-MTG",
    "pair_SMG-STG",
    "pair_IPG-STG",
    "pair_ITG-PTR",
    "pair_FG-PTR",
    "pair_ITG-STG",
    "pair_IN-POP",
    "pair_IN-STG",
    "pair_PrCG-POP",
    "pair_PrCG-STG",
    "pair_SFG-POP",
    "pair_POP-PTR",
    "pair_L.POP-R.POP",
    "pair_L.STG-R.STG",
    "pair_L.SMG-R.SMG",
    "pair_L.TH-POP",
    "pair_L.TH-STG",
    "pair_L.PU-POP",
]

APRIORI_NETWORK = APRIORI_NEMO_PAIRS

FEATURE_SETS = {
    "FS1": {
        "name":        "Clinical baseline",
        "description": "Admission-available clinical and demographic features",
        "features":    CLINICAL_BASELINE,
        "residualize": [],
        "rfe":         False,
    },
    "FS2": {
        "name":        "Clinical + volume",
        "description": "Does stroke size add beyond clinical?",
        "features":    CLINICAL_BASELINE + [LESION_VOLUME_COL],
        "residualize": [],
        "rfe":         False,
    },
    "FS3": {
        "name":        "Clinical + anatomy",
        "description": "Does lesion location add beyond volume? (BPM + Tractotron)",
        "features":    CLINICAL_BASELINE + [LESION_VOLUME_COL] + APRIORI_ANATOMY,
        "residualize": APRIORI_ANATOMY,
        "rfe":         True,
    },
    "FS4": {
        "name":        "Clinical + anatomy + network",
        "description": "Does lesion-network disruption add beyond anatomy? (NeMo ChaCo, residualized)",
        "features":    CLINICAL_BASELINE + [LESION_VOLUME_COL] + APRIORI_ANATOMY + APRIORI_NEMO_PAIRS,
        "residualize": APRIORI_ANATOMY + APRIORI_NEMO_PAIRS,
        "rfe":         True,
    },
}

RANDOM_STATE = 42

CANDIDATE_MODELS = {
    "ridge": {
        "class": "sklearn.linear_model.Ridge",
        "param_grid": {
            "alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        },
    },
    "svr": {
        "class": "sklearn.svm.SVR",
        "param_grid": {
            "kernel":  ["rbf"],
            "C":       [0.1, 1.0, 10.0, 100.0],
            "gamma":   ["scale", "auto"],
            "epsilon": [0.1, 0.5, 1.0],
        },
    },
    "rf": {
        "class": "sklearn.ensemble.RandomForestRegressor",
        "param_grid": {
            "n_estimators":     [100, 300],
            "max_depth":        [3, 5, 8, None],
            "min_samples_leaf": [2, 5, 10],
            "max_features":     ["sqrt", "log2", 0.5],
        },
    },
    "xgb": {
        "class": "xgboost.XGBRegressor",
        "param_grid": {
            "n_estimators":     [100],
            "max_depth":        [3, 5],
            "learning_rate":    [0.05, 0.1],
            "subsample":        [0.8],
            "colsample_bytree": [0.7],
            "reg_alpha":        [0, 0.1],
            "reg_lambda":       [1.0],
        },
    },
}

CV_SETTINGS = {
    "outer_k":       10,
    "outer_repeats": 10,
    "inner_k":       10,
    "bootstrap_n":   2000,
    "random_state":  RANDOM_STATE,
}

LOW_OUTCOME_THRESHOLD = 50
