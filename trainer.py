import numpy as np, shap
import os, pickle, yaml
import xgboost as xgb
from munch import munchify
from scipy.stats import iqr
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score
from sklearn.impute import KNNImputer
from xgboost import XGBClassifier

feature_names = [
    # ECG
    "HR",
    "RSA",
    # EDA
    "SCL mean",
    "SCL slope",
    "SCR count",
    "SCR amp",
    "SCR rise",
    # RSP
    "RSP period",
    "RSP depth",
    "RVT",
    # SKT
    "T mean",
    "T slope",
]


def task_exists(sub, tasks):
    for task in tasks:
        task_file = f"{sub}_{task}.csv"
        if os.path.exists(cfg.root + task_file):
            return True
    print(f"{sub} does not have any of the tasks {tasks}.")
    return False


def load_session(sub, task, every=1, scale=False, recovery=False):
    X, y = [], []

    task_file = f"{sub}_{task}_free.npy"
    if os.path.exists(cfg.token_root + task_file):
        baseline = np.load(cfg.token_root + task_file)
        if len(baseline) > 60:
            baseline = baseline[:60]
        mean, std = np.median(baseline, axis=0), iqr(baseline) + 1e-6

    task_file = f"{sub}_{task}_free.npy"
    if os.path.exists(cfg.token_root + task_file):
        this_X = np.load(cfg.token_root + task_file)[::every]
        this_X = (this_X - mean) / std if scale else this_X - mean
        X.append(this_X)
        y.append([0] * len(this_X))

    task_file = f"{sub}_{task}_recov.npy" if recovery else f"{sub}_{task}_event.npy"
    if os.path.exists(cfg.token_root + task_file):
        this_X = np.load(cfg.token_root + task_file)[::every]
        this_X = (this_X - mean) / std if scale else this_X - mean
        X.append(this_X)
        y.append([1] * len(this_X))

    return X, y


# Load config file
with open("config.yaml", "r") as f:
    cfg = munchify(yaml.safe_load(f))

loso = LeaveOneOut()
subs = [f"P{i}" for i in range(1, 34) if i not in [3, 8]]
scores = {t: [] for t in cfg.test_tasks}
shap_v = {sub: {s: [] for s in cfg.test_tasks} for sub in subs}  # SHAP values
shap_d = {sub: {s: [] for s in cfg.test_tasks} for sub in subs}  # Test data
shap_e = {sub: {s: [] for s in cfg.test_tasks} for sub in subs}  # Explainers

feature_ids = []
if "ECG" in cfg.modalities:
    feature_ids.extend([0, 1])
if "EDA" in cfg.modalities:
    feature_ids.extend([2, 3, 4, 5, 6])
if "RSP" in cfg.modalities:
    feature_ids.extend([7, 8, 9])
if "SKT" in cfg.modalities:
    feature_ids.extend([10, 11])
feature_names = [feature_names[i] for i in feature_ids]

# Outer loop: Leave-One-Subject-Out
for i, (train_index, test_index) in enumerate(loso.split(subs)):
    train_subs = np.array(subs)[train_index]
    test_sub = np.array(subs)[test_index][0]
    print(f"\nTest subject: {test_sub}")

    if not task_exists(test_sub, cfg.test_tasks):
        continue

    # Inner loop: subject subsampling (N=10)
    for fold in range(10):
        # Bootstrap the training subjects
        train_subs_cv = np.random.choice(train_subs, size=len(train_subs), replace=True)

        # Load the training data
        X_train, y_train = [], []
        for sub in train_subs_cv:
            for task in cfg.train_tasks:
                X, y = load_session(sub, task, every=5, scale=cfg.scale)
                X_train.extend(X)
                y_train.extend(y)

        X_train = np.concatenate(X_train, axis=0)[:, feature_ids]
        y_train = np.concatenate(y_train, axis=0)
        if cfg.permute:
            y_train = np.random.permutation(y_train)

        # Impute missing values
        imp = KNNImputer(missing_values=2048.0)
        X_train = imp.fit_transform(X_train)

        scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)
        model = XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=cfg.seed,
            reg_lambda=10,
            device="cuda",
        )
        dtrain = xgb.DMatrix(X_train, label=y_train)
        model.fit(X_train, y_train)

        # Evaluate on the test subject
        for task in cfg.test_tasks:
            print("Testing on", test_sub, task)
            if not task_exists(test_sub, cfg.test_tasks):
                continue

            # Load the test data
            X_test, y_test = load_session(
                test_sub, task, every=1, scale=cfg.scale, recovery=False
            )
            if len(X_test) == 0:
                continue

            y_test = np.concatenate(y_test, axis=0)
            X_test = np.concatenate(X_test, axis=0)[:, feature_ids]
            X_test = imp.transform(X_test)

            # Explain model predictions
            if cfg.use_shap:
                explainer = shap.Explainer(model, X_train, feature_names=feature_names)
                explanation = explainer(X_test)
                shap_values = explainer.shap_values(X_test)
                shap_v[test_sub][task].append(shap_values)
                shap_d[test_sub][task].append(X_test)
                shap_e[test_sub][task].append(explanation)

            # Evaluate performance
            dtest = xgb.DMatrix(X_test, label=y_test)
            y_proba = model.get_booster().predict(dtest)
            auroc = roc_auc_score(y_test, y_proba)
            scores[task].append(auroc)
            print(f"ROC-AUC: {auroc:.4f}")

            # Save the predictions
            probas_0 = y_proba[np.where(y_test == 0)[0]]
            probas_1 = y_proba[np.where(y_test == 1)[0]]

            tr_tasks = "".join(cfg.train_tasks)
            is_rand = "_rand" if cfg.permute else ""
            is_mod = f"_{cfg.modalities[0]}" if len(cfg.modalities) == 1 else ""

            folder = f"results{is_rand}{is_mod}_recov"
            os.makedirs(folder, exist_ok=True)
            np.save(f"{folder}/{test_sub}_{tr_tasks}_{task}0_s{fold}.npy", probas_0)
            np.save(f"{folder}/{test_sub}_{tr_tasks}_{task}1_s{fold}.npy", probas_1)

    # Save the SHAP values
    if cfg.use_shap:
        for task in cfg.test_tasks:
            if len(shap_v[test_sub][task]) == 0:
                continue
            os.makedirs("shaps", exist_ok=True)
            shap_values = np.array(shap_values)
            np.save(f"shaps/{test_sub}_{task}_shap.npy", shap_v[test_sub][task])
            np.save(f"shaps/{test_sub}_{task}_data.npy", shap_d[test_sub][task])
            with open(f"shaps/{test_sub}_{task}_explainer.pkl", "wb") as f:
                pickle.dump(shap_e[test_sub][task], f)

# Save SHAP values
if cfg.use_shap:
    os.makedirs("shap_values", exist_ok=True)
    for sub in subs:
        for task in cfg.test_tasks:
            if len(shap_v[sub][task]) > 0:
                shap_values = np.concatenate(shap_v[sub][task], axis=0)
                shap_data = np.concatenate(shap_d[sub][task], axis=0)
                np.save(f"shap_values/{sub}_{task}_shap.npy", shap_values)
                np.save(f"shap_values/{sub}_{task}_data.npy", shap_data)

print("\nAverage scores:")
for task in cfg.test_tasks:
    print(f"{task}: {np.median(scores[task]):.4f} ± {np.std(scores[task]):.4f}")
