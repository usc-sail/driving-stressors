import pandas as pd, os
import numpy as np, json
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
from plotting import plot_proba_ts, plot_events_boxplot


def get_scores(path0, path1):
    pred_0 = np.load(path0)
    pred_1 = np.load(path1)

    y_pred = np.concatenate([pred_0, pred_1])
    y_true = np.concatenate([np.zeros(len(pred_0)), np.ones(len(pred_1))])

    return (
        {
            "ROC-AUC": roc_auc_score(y_true, y_pred),
            "Proba 0": np.mean(pred_0),
            "Proba 1": np.mean(pred_1),
        },
        y_pred,
        y_true,
    )


def get_event_scores(event, y_pred, bounds):
    event_proba = y_pred[int(event) : int(event) + int(30)]
    free_proba = y_pred[: bounds[0]]

    y_pred_event = np.concatenate([free_proba, event_proba])
    y_true_event = np.concatenate(
        [np.zeros(len(free_proba)), np.ones(len(event_proba))]
    )

    return {
        "ROC-AUC": roc_auc_score(y_true_event, y_pred_event),
        "Proba 0": np.mean(free_proba),
        "Proba 1": np.mean(event_proba),
    }


def create_offsets(sub, this_mode):
    event_f = csv_path + f"{sub}_{this_mode}.csv"
    df = pd.read_csv(event_f, low_memory=False)
    df = df[["time", "event"]].dropna().reset_index(drop=True)
    df["time"] = df["time"] - df

    offset_list, event_list, flag = [], [], False
    for i, row in df.iterrows():
        if row["event"] == start_event[this_mode]:
            flag = True
        if flag:
            offset_list.append(int(row["time"]))
            event_list.append(row["event"])
        if row["event"] in ["Clear Text", "Experiment End"]:
            break

    # save the offset + event lists
    os.makedirs("offsets", exist_ok=True)
    offsets = {"offsets": offset_list, "events": event_list}
    with open(f"offsets/{sub}_{this_mode}.json", "w") as f:
        json.dump(offsets, f, indent=4)

    return offset_list, event_list


csv_path = f"/media/data/toyota/processed_data/trina_33_final/"
this_mode, train_mode = "S", "IMS"
with open(f"events_mapping.json", "rb") as f:
    events_mapping = json.load(f)[this_mode]

subs = [f"P{i}" for i in range(1, 34) if i != 3]
scores, event_scores = {}, {}

folds = [0, 1, 2, 3, 4]
folders = [f"results_svm{fold}" for fold in folds]


for sub in tqdm(subs):
    scores[sub] = {}

    for folder in folders:
        path0 = f"{folder}/{sub}_{train_mode}_{this_mode}0.npy"
        path1 = f"{folder}/{sub}_{train_mode}_{this_mode}1.npy"
        if not os.path.exists(path0):
            continue

        scores[sub][folder], y_pred, y_true = get_scores(path0, path1)

        if sub == "P14" and this_mode == "I":
            continue

        if os.path.exists(f"offsets/{sub}_{this_mode}.json"):
            with open(f"offsets/{sub}_{this_mode}.json", "r") as f:
                offsets = json.load(f)
                offset_list = offsets["offsets"]
                event_list = offsets["events"]
        else:
            start_event = {
                "I": "Fog Start",
                "M": "Timer Start",
                "S": "Crash",
            }
            event_f = csv_path + f"{sub}_{this_mode}.csv"
            offset_list, event_list = create_offsets(
                event_f, sub, this_mode, start_event
            )

        bounds = [sum(y_true == 0), sum(y_true == 1)]
        bounds = np.cumsum(bounds)
        offset_list += bounds[0]

        plot_proba_ts(sub, this_mode, y_pred, bounds, offset_list, event_list)

        # get probabilities close to the events
        for e, event in enumerate(offset_list):
            try:
                mapped_event = events_mapping[event_list[e]]
            except KeyError:
                continue

            this_event_score = get_event_scores(event, y_pred, bounds)
            if mapped_event not in event_scores:
                event_scores[mapped_event] = {
                    k: [v] for k, v in this_event_score.items()
                }
            else:
                for k, v in this_event_score.items():
                    event_scores[mapped_event][k].append(v)

        # add the session as an event
        if "Session" not in event_scores:
            event_scores["Session"] = {k: [v] for k, v in scores[sub][folder].items()}
        else:
            for k, v in scores[sub][folder].items():
                event_scores["Session"][k].append(v)


def compute_bootstrap_ci(proba_0_list, proba_1_list, n_iter=1000):
    aurocs, effects = [], []
    for _ in range(n_iter):
        bootstrap_indices = np.random.choice(
            range(len(proba_0_list)), size=len(proba_0_list), replace=True
        )
        proba_0_bootstrap = np.array(proba_0_list)[bootstrap_indices]
        proba_1_bootstrap = np.array(proba_1_list)[bootstrap_indices]
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

    lower_auroc = np.percentile(aurocs, 2.5)
    upper_auroc = np.percentile(aurocs, 97.5)
    central_auroc = np.percentile(aurocs, 50)

    lower_effect = np.percentile(effects, 2.5)
    upper_effect = np.percentile(effects, 97.5)
    central_effect = np.percentile(effects, 50)

    print(f"ROC-AUC: {central_auroc:.3f} [{lower_auroc:.3f}, {upper_auroc:.3f}]")
    print(f"Effect: {central_effect:.3f} [{lower_effect:.3f}, {upper_effect:.3f}]")

    return aurocs, effects


bootstrap_scores = {}
for e in event_scores.keys():
    print(f"\nEvent: {e}")
    proba_0_list = event_scores[e]["Proba 0"]
    proba_1_list = event_scores[e]["Proba 1"]

    aurocs, effects = compute_bootstrap_ci(proba_0_list, proba_1_list)
    bootstrap_scores[e] = {"ROC-AUC": aurocs, "Effect": effects}

    stat, p = wilcoxon(proba_0_list, proba_1_list)
    print(f"Wilcoxon p-value = {p:.8f}")

if this_mode == "I":
    events = [
        "Session",
        "Fog Start",
        "Slow Cars",
        "Brakes",
    ]
elif this_mode == "M":
    events = [
        "Session",
        "Timer Start",
        "Amazon Truck",
        "Pace Car",
        "Construction 1",
        "Construction 2",
    ]
else:
    events = [
        "Session",
        "Crash",
        "Barrel",
    ]

plot_events_boxplot(bootstrap_scores, events, this_mode, train_mode)

# line plot across events, with 2 lines, one green (proba 0) and one red (proba 1)
data = [event_scores[ev]["Proba 0"] for ev in events]
fig, ax = plt.subplots(figsize=(len(events), 4), dpi=200)


# Bootstrap function
def bootstrap_ci(data, num_samples=1000, ci=95):
    boot_means = [
        np.mean(np.random.choice(data, size=len(data), replace=True))
        for _ in range(num_samples)
    ]
    lower = np.percentile(boot_means, (100 - ci) / 2)
    upper = np.percentile(boot_means, 100 - (100 - ci) / 2)
    central = np.percentile(boot_means, 50)
    return central, lower, upper


central, lower, upper = bootstrap_ci(data[0])
for i in range(len(events) - 1):
    central_next, lower_next, upper_next = bootstrap_ci(data[i + 1])
    plt.plot(
        [i + 1, i + 2],
        [central, central_next],
        color="green",
        linewidth=1.5,
        marker="o",
        label="Proba 0",
    )
    # put confidence intervals
    plt.fill_between(
        [i + 1, i + 2],
        [lower, lower_next],
        [upper, upper_next],
        color="green",
        alpha=0.2,
        linewidth=0,
    )
    central, lower, upper = central_next, lower_next, upper_next

data = [event_scores[ev]["Proba 1"] for ev in events]
central, lower, upper = bootstrap_ci(data[0])
for i in range(len(events) - 1):
    central_next, lower_next, upper_next = bootstrap_ci(data[i + 1])
    plt.plot(
        [i + 1, i + 2],
        [central, central_next],
        color="red",
        linewidth=1.5,
        marker="o",
        label="Proba 1",
    )
    # put confidence intervals
    plt.fill_between(
        [i + 1, i + 2],
        [lower, lower_next],
        [upper, upper_next],
        color="red",
        alpha=0.2,
        linewidth=0,
    )
    central, lower, upper = central_next, lower_next, upper_next

# Formatting
plt.ylabel("Anomaly probability")
plt.yticks(np.arange(0, 1.1, 0.1))
plt.xticks(range(1, len(events) + 1), events, rotation=45, ha="right")
plt.ylim(0, 1)
plt.grid(linestyle="--", alpha=0.75, linewidth=0.75)

# remove up and right spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Save the plot
plt.savefig(f"figures/lineplot_{this_mode}_{train_mode}.png", bbox_inches="tight")
