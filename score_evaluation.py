import pandas as pd, json, os
import pickle, numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score


def open_pkl(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data, len(data["ecg"])


this_mode = "I"
fig_name = f"{this_mode}_only.png"
with open("results/nn_scores_I.json", "r") as f:
    data = json.load(f)
    data = {k: v for k, v in data.items() if v}

os.makedirs("results/figures", exist_ok=True)
pkl_path = "/media/data/toyota/processed_data/trina_33_samples_fupd/"
csv_path = f"/media/data/toyota/processed_data/trina_33/"

scores = {}
subject_list = [f.split("_")[0] for f in os.listdir(csv_path)]
for d in list(set(subject_list)):
    print(f"\nSubject: {d}")

    modes = []
    if os.path.exists(csv_path + f"{d}_I.csv"):
        modes.append("I")
    if os.path.exists(csv_path + f"{d}_M.csv"):
        modes.append("M")
    if os.path.exists(csv_path + f"{d}_S.csv"):
        modes.append("S")

    if this_mode not in modes or d in ["P9", "P14", "P21"]:
        print(f"{d} does not have {this_mode}")
        continue

    proba = data[d]
    # smooth the probability by 3 points
    proba = np.convolve(proba, np.ones(3) / 3, mode="same")
    # recover the first 3 and last 3 points
    proba[:3] = data[d][:3]
    proba[-3:] = data[d][-3:]

    if "I" in modes:
        free_I, len_free_I = open_pkl(pkl_path + f"{d}_I_free.pkl")
        event_I, len_event_I = open_pkl(pkl_path + f"{d}_I_event.pkl")
    if "M" in modes:
        free_M, len_free_M = open_pkl(pkl_path + f"{d}_M_free.pkl")
        event_M, len_event_M = open_pkl(pkl_path + f"{d}_M_event.pkl")
    if "S" in modes:
        free_S, len_free_S = open_pkl(pkl_path + f"{d}_S_free.pkl")
        event_S, len_event_S = open_pkl(pkl_path + f"{d}_S_event.pkl")

    bounds = []
    if "I" in modes:
        bounds.extend([len_free_I, len_event_I])
    if "M" in modes:
        bounds.extend([len_free_M, len_event_M])
    if "S" in modes:
        bounds.extend([len_free_S, len_event_S])
    bounds = np.cumsum(bounds)

    if this_mode == "I":
        bounds = [0, bounds[0], bounds[1]]
    elif this_mode == "M" and "I" in modes:
        bounds = [bounds[1], bounds[2], bounds[3]]
    elif this_mode == "M" and "I" not in modes:
        bounds = [0, bounds[0], bounds[1]]
    elif this_mode == "S" and "M" in modes and "I" in modes:
        bounds = [bounds[3], bounds[4], bounds[5]]
    elif this_mode == "S" and "M" in modes and "I" not in modes:
        bounds = [bounds[1], bounds[2], bounds[3]]
    elif this_mode == "S" and "M" not in modes and "I" in modes:
        bounds = [bounds[1], bounds[2], bounds[3]]
    else:
        bounds = [0, bounds[0], bounds[1]]

    free_proba = proba[bounds[0] : bounds[1]]
    event_proba = proba[bounds[1] : bounds[2]]
    y_pred = np.concatenate([free_proba, event_proba])
    y_true = np.concatenate([np.zeros(len(free_proba)), np.ones(len(event_proba))])

    auroc = roc_auc_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred > 0.5, average="macro")
    bac = balanced_accuracy_score(y_true, y_pred > 0.5)
    scores[d] = {"ROC-AUC": auroc, "F1 Macro": f1_macro, "BAC": bac}
    print(scores[d])

    start_event = {
        "I": "Fog Start",
        "M": "Timer Start",
        "S": "Crash",
    }

    event_f = csv_path + f"{d}_{this_mode}.csv"
    df = pd.read_csv(event_f, low_memory=False)
    df = df[["time", "event"]].dropna().reset_index(drop=True)
    idx = df[df["event"] == start_event[this_mode]].index[0]
    df["time"] = df["time"] - df["time"].iloc[idx]

    offset_list, event_list, flag = [], [], False
    for i, row in df.iterrows():
        if row["event"] == start_event[this_mode]:
            flag = True
        if flag:
            offset_list.append(row["time"])
            event_list.append(row["event"])
        if row["event"] in ["Clear Text", "Experiment End"]:
            break

    new_change = offset_list + bounds[1]
    new_change_i = [i for i, c in enumerate(new_change) if c >= bounds[1]]
    new_change = [new_change[i] for i in new_change_i]
    label_list = [event_list[i] for i in new_change_i]

    plt.figure(figsize=(15, 3))
    plt.plot(np.array(proba), color="blue", linewidth=0.99)

    for xc in new_change:
        if xc >= bounds[1]:
            plt.axvline(x=xc, color="red")

    plt.title(f"{d} - {this_mode} {label_list}")
    plt.ylabel("Anomaly probability")
    plt.xlabel("Time (s)")
    plt.xticks(np.arange(0, len(proba), 30), np.arange(0, len(proba), 30))
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.xlim(bounds[0] - 5, bounds[2] + 5)
    plt.savefig(f"results/figures/{d}_{fig_name}", bbox_inches="tight", dpi=300)

# print average scores
auroc_list, f1_list, bac_list = [], [], []
for d in scores:
    auroc_list.append(scores[d]["ROC-AUC"])
    f1_list.append(scores[d]["F1 Macro"])
    bac_list.append(scores[d]["BAC"])

print("\nAverage ROC-AUC:", np.mean(auroc_list).round(3))
print("Average F1 Macro:", np.mean(f1_list).round(3))
print("Average B. Acc:", np.mean(bac_list).round(3))
