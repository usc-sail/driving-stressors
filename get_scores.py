import numpy as np, os
from tqdm import tqdm
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

np.random.seed(79911092)
mod = ""


def get_scores(path0, path1):
    pred_0 = np.load(path0)
    pred_1 = np.load(path1)

    y_pred = np.concatenate([pred_0, pred_1])
    y_true = np.concatenate([np.zeros(len(pred_0)), np.ones(len(pred_1))])

    return {
        "ROC-AUC": roc_auc_score(y_true, y_pred),
        "Proba 0": np.mean(pred_0),
        "Proba 1": np.mean(pred_1),
    }


train_tasks = "IMS"
test_tasks = ["I", "M", "S"]

subs = [f"P{i}" for i in range(1, 34) if i not in [3, 8]]
folds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
scores = {
    "ROC-AUC": [],
    "Proba 0": [],
    "Proba 1": [],
}

for sub in tqdm(subs):
    for task in test_tasks:
        for fold in folds:
            path0 = f"results{mod}/{sub}_{train_tasks}_{task}0_s{fold}.npy"
            path1 = f"results{mod}/{sub}_{train_tasks}_{task}1_s{fold}.npy"
            if not os.path.exists(path0):
                continue

            score = get_scores(path0, path1)
            scores["ROC-AUC"].append(score["ROC-AUC"])
            scores["Proba 0"].append(score["Proba 0"])
            scores["Proba 1"].append(score["Proba 1"])


def compute_bootstrap_ci(proba_0_list, proba_1_list, n_iter=1000):
    aurocs, effects = [], []
    probas_0, probas_1 = [], []
    for _ in range(n_iter):
        bootstrap_indices_0 = np.random.choice(
            range(len(proba_0_list)), size=len(proba_0_list), replace=True
        )
        bootstrap_indices_1 = np.random.choice(
            range(len(proba_1_list)), size=len(proba_1_list), replace=True
        )
        proba_0_bootstrap = np.array(proba_0_list)[bootstrap_indices_0]
        proba_1_bootstrap = np.array(proba_1_list)[bootstrap_indices_1]
        auroc = roc_auc_score(
            np.concatenate(
                [np.zeros(len(proba_0_bootstrap)), np.ones(len(proba_1_bootstrap))]
            ),
            np.concatenate([proba_0_bootstrap, proba_1_bootstrap]),
        )

        stat, _ = wilcoxon(proba_0_bootstrap, proba_1_bootstrap)
        n = len(proba_0_bootstrap)
        r = stat / (n * (n + 1) / 2)

        diffs = proba_1_bootstrap - proba_0_bootstrap
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs, ddof=1)
        r = mean_diff / std_diff

        aurocs.append(auroc)
        effects.append(r)
        probas_0.append(np.median(proba_0_bootstrap))
        probas_1.append(np.median(proba_1_bootstrap))

    lower_auroc = np.percentile(aurocs, 2.5)
    upper_auroc = np.percentile(aurocs, 97.5)
    central_auroc = np.percentile(aurocs, 50)

    lower_effect = np.percentile(effects, 2.5)
    upper_effect = np.percentile(effects, 97.5)
    central_effect = np.percentile(effects, 50)

    lower_proba_0 = np.percentile(probas_0, 2.5)
    upper_proba_0 = np.percentile(probas_0, 97.5)
    central_proba_0 = np.percentile(probas_0, 50)

    lower_proba_1 = np.percentile(probas_1, 2.5)
    upper_proba_1 = np.percentile(probas_1, 97.5)
    central_proba_1 = np.percentile(probas_1, 50)

    print(f"ROC-AUC: {central_auroc:.3f} [{lower_auroc:.3f}, {upper_auroc:.3f}]")
    print(f"Effect: {central_effect:.3f} [{lower_effect:.3f}, {upper_effect:.3f}]")
    print(f"Proba 0: {central_proba_0:.3f} [{lower_proba_0:.3f}, {upper_proba_0:.3f}]")
    print(f"Proba 1: {central_proba_1:.3f} [{lower_proba_1:.3f}, {upper_proba_1:.3f}]")

    return aurocs, effects, central_auroc


free_proba = scores["Proba 0"]
event_proba = scores["Proba 1"]
aurocs, effects, avg = compute_bootstrap_ci(free_proba, event_proba)

scores = {"ROC-AUC": [], "Proba 0": [], "Proba 1": []}
for sub in tqdm(subs):
    for task in ["I", "M", "S"]:
        for fold in folds:
            path0 = f"results_rand{mod}/{sub}_{train_tasks}_{task}0_s{fold}.npy"
            path1 = f"results_rand{mod}/{sub}_{train_tasks}_{task}1_s{fold}.npy"
            if not os.path.exists(path0):
                continue

            score = get_scores(path0, path1)
            scores["ROC-AUC"].append(score["ROC-AUC"])
            scores["Proba 0"].append(score["Proba 0"])
            scores["Proba 1"].append(score["Proba 1"])

proba_0_list = scores["Proba 0"]
proba_1_list = scores["Proba 1"]

counter = 0
for _ in range(1000):
    bootstrap_indices_0 = np.random.choice(
        range(len(proba_0_list)), size=len(proba_0_list), replace=True
    )
    bootstrap_indices_1 = np.random.choice(
        range(len(proba_1_list)), size=len(proba_1_list), replace=True
    )
    proba_0_bootstrap = np.array(proba_0_list)[bootstrap_indices_0]
    proba_1_bootstrap = np.array(proba_1_list)[bootstrap_indices_1]
    auroc = roc_auc_score(
        np.concatenate(
            [np.zeros(len(proba_0_bootstrap)), np.ones(len(proba_1_bootstrap))]
        ),
        np.concatenate([proba_0_bootstrap, proba_1_bootstrap]),
    )
    if auroc >= avg:
        counter += 1

print("Permutation test (p-value):", counter / 1000)
