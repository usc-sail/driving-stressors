import numpy as np, os, yaml
from munch import munchify
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def task_exists(sub, tasks):
    for task in tasks:
        task_file = f"{sub}_{task}.csv"
        if os.path.exists(cfg.root + task_file):
            return True
    print(f"{sub} does not have any of the tasks {tasks}.")
    return False


# Load config file
os.makedirs("ckpt", exist_ok=True)
with open("config.yaml", "r") as f:
    cfg = munchify(yaml.safe_load(f))

loso = LeaveOneOut()
subs = [f"P{i}" for i in range(1, 34) if i != 3]
INCLUDE_VIDEO = False

scores = {t: [] for t in cfg.tasks}
for i, (train_index, test_index) in enumerate(loso.split(subs)):
    train_subs = np.array(subs)[train_index]
    test_sub = np.array(subs)[test_index][0]
    print(f"\nTest subject: {test_sub}")

    if not task_exists(test_sub, cfg.tasks):
        continue

    X_train, y_train = [], []
    for sub in train_subs:
        for task in cfg.tasks:
            task_file = f"{sub}_{task}_video.npy"
            if os.path.exists(cfg.token_root + task_file):
                baseline = np.load(cfg.token_root + task_file)
                mean, std = baseline.mean(axis=0), baseline.std(axis=0)

                if INCLUDE_VIDEO:
                    this_X = np.load(cfg.token_root + task_file)[::30]
                    this_X = this_X - mean
                    X_train.append(this_X)
                    y_train.append([0] * len(this_X))

            task_file = f"{sub}_{task}_free.npy"
            if os.path.exists(cfg.token_root + task_file):
                this_X = np.load(cfg.token_root + task_file)[::5]
                this_X = this_X - mean
                X_train.append(this_X)
                y_train.append([0] * len(this_X))

            task_file = f"{sub}_{task}_event.npy"
            if os.path.exists(cfg.token_root + task_file):
                this_X = np.load(cfg.token_root + task_file)[::5]
                this_X = this_X - mean
                X_train.append(this_X)
                y_train.append([1] * len(this_X))

    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    # permute y_train
    # y_train = np.random.permutation(y_train)

    print(X_train.shape, y_train.shape)

    imp = KNNImputer(n_neighbors=3)
    X_train = np.where(X_train == 2048.0, np.nan, X_train)
    X_train = imp.fit_transform(X_train)

    # Load the model
    # model = LogisticRegression(
    #     class_weight="balanced", max_iter=5000, solver="newton-cholesky", C=100
    # )
    model = XGBClassifier(
        eval_metric="logloss", reg_lambda=5, random_state=cfg.seed, subsample=0.8
    )
    model.fit(X_train, y_train)

    for task in cfg.tasks:
        print("Testing on", test_sub, task)
        if not task_exists(test_sub, cfg.tasks):
            continue

        # Load the data
        X_test, y_test = [], []

        task_file = f"{test_sub}_{task}_video.npy"
        if os.path.exists(cfg.token_root + task_file):
            baseline = np.load(cfg.token_root + task_file)
            mean, std = baseline.mean(axis=0), baseline.std(axis=0)

        task_file = f"{test_sub}_{task}_free.npy"
        if os.path.exists(cfg.token_root + task_file):
            this_X = np.load(cfg.token_root + task_file)
            this_X = this_X - mean
            X_test.append(this_X)
            y_test.append([0] * len(this_X))

        task_file = f"{test_sub}_{task}_event.npy"
        if os.path.exists(cfg.token_root + task_file):
            this_X = np.load(cfg.token_root + task_file)
            this_X = this_X - mean
            X_test.append(this_X)
            y_test.append([1] * len(this_X))

        if len(X_test) == 0:
            print(f"No test data for {task}")
            continue

        X_test = np.concatenate(X_test, axis=0)
        y_test = np.concatenate(y_test, axis=0)

        X_test = np.where(X_test == 2048.0, np.nan, X_test)
        X_test = imp.transform(X_test)

        # Evaluate performance
        y_proba = model.predict_proba(X_test)
        y_pred = np.argmax(y_proba, axis=1)
        y_proba = y_proba[:, 1]

        auroc = roc_auc_score(y_test, y_proba)
        print(f"ROC-AUC: {auroc:.4f}")
        scores[task].append(auroc)

        # Save the predictions
        probas_0 = y_proba[np.where(y_test == 0)[0]]
        probas_1 = y_proba[np.where(y_test == 1)[0]]

        tasks_text = "".join(cfg.tasks)
        folder = f"results_svm{cfg.seed}"
        os.makedirs(folder, exist_ok=True)
        np.save(f"{folder}/{test_sub}_{tasks_text}_{task}0.npy", probas_0)
        np.save(f"{folder}/{test_sub}_{tasks_text}_{task}1.npy", probas_1)

print("\nAverage scores:")
for task in cfg.tasks:
    print(f"{task}: {np.median(scores[task]):.4f} ± {np.std(scores[task]):.4f}")

print("\nOverall mean score:", np.mean([np.median(scores[task]) for task in cfg.tasks]))
print("Overall std score:", np.std([np.mean(scores[task]) for task in cfg.tasks]))
