import pandas as pd, json, os
import pickle, numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score


def open_pkl(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data, len(data["ecg"])


def update_bounds(modes):
    if "I" in modes:
        _, len_video_I = open_pkl(pkl_path + f"{d}_I_video.pkl")
        _, len_free_I = open_pkl(pkl_path + f"{d}_I_free.pkl")
        _, len_event_I = open_pkl(pkl_path + f"{d}_I_event.pkl")
    if "M" in modes:
        _, len_video_M = open_pkl(pkl_path + f"{d}_M_video.pkl")
        _, len_free_M = open_pkl(pkl_path + f"{d}_M_free.pkl")
        _, len_event_M = open_pkl(pkl_path + f"{d}_M_event.pkl")
    if "S" in modes:
        _, len_video_S = open_pkl(pkl_path + f"{d}_S_video.pkl")
        _, len_free_S = open_pkl(pkl_path + f"{d}_S_free.pkl")
        _, len_event_S = open_pkl(pkl_path + f"{d}_S_event.pkl")

    bounds = []
    if "I" in modes:
        bounds.extend([len_video_I, len_free_I, len_event_I])
    if "M" in modes:
        bounds.extend([len_video_M, len_free_M, len_event_M])
    if "S" in modes:
        bounds.extend([len_video_S, len_free_S, len_event_S])

    return np.cumsum(bounds)


def configure_eval1(bounds, eval_task):
    assert eval_task in ["VvF", "FvE", "VvE"]

    if eval_task == "VvF":
        if this_mode == "I":
            bounds = [0, bounds[0], bounds[1]]
        elif this_mode == "M" and "I" in modes:
            bounds = [bounds[2], bounds[3], bounds[4]]
        elif this_mode == "M" and "I" not in modes:
            bounds = [0, bounds[0], bounds[1]]
        elif this_mode == "S" and "M" in modes and "I" in modes:
            bounds = [bounds[5], bounds[6], bounds[7]]
        elif this_mode == "S" and "M" in modes and "I" not in modes:
            bounds = [bounds[2], bounds[3], bounds[4]]
        elif this_mode == "S" and "M" not in modes and "I" in modes:
            bounds = [bounds[2], bounds[3], bounds[4]]
        else:
            bounds = [0, bounds[0], bounds[1]]

    elif eval_task == "FvE":
        if this_mode == "I":
            bounds = [bounds[0], bounds[1], bounds[2]]
        elif this_mode == "M" and "I" in modes:
            bounds = [bounds[3], bounds[4], bounds[5]]
        elif this_mode == "M" and "I" not in modes:
            bounds = [bounds[0], bounds[1], bounds[2]]
        elif this_mode == "S" and "M" in modes and "I" in modes:
            bounds = [bounds[6], bounds[7], bounds[8]]
        elif this_mode == "S" and "M" in modes and "I" not in modes:
            bounds = [bounds[3], bounds[4], bounds[5]]
        elif this_mode == "S" and "M" not in modes and "I" in modes:
            bounds = [bounds[3], bounds[4], bounds[5]]
        else:
            bounds = [bounds[0], bounds[1], bounds[2]]

    elif eval_task == "VvE":
        if this_mode == "I":
            bounds = [0, bounds[0], bounds[1], bounds[2]]
        elif this_mode == "M" and "I" in modes:
            bounds = [bounds[2], bounds[3], bounds[4], bounds[5]]
        elif this_mode == "M" and "I" not in modes:
            bounds = [0, bounds[0], bounds[1], bounds[2]]
        elif this_mode == "S" and "M" in modes and "I" in modes:
            bounds = [bounds[5], bounds[6], bounds[7], bounds[8]]
        elif this_mode == "S" and "M" in modes and "I" not in modes:
            bounds = [bounds[2], bounds[3], bounds[4], bounds[5]]
        elif this_mode == "S" and "M" not in modes and "I" in modes:
            bounds = [bounds[2], bounds[3], bounds[4], bounds[5]]
        else:
            bounds = [0, bounds[0], bounds[1], bounds[2]]

    return bounds


def configure_eval2(bounds, eval_task):
    if eval_task == "VvF":
        return [0, bounds[0], bounds[1]]
    elif eval_task == "FvE":
        return [bounds[0], bounds[1], bounds[2]]
    elif eval_task == "VvE":
        return [0, bounds[0], bounds[1], bounds[2]]
    else:
        raise ValueError("Invalid evaluation task")


this_mode = "M"
eval_mode = "FvE"
fig_name = f"{this_mode}.png"
with open("results/nn_scores_final.json", "r") as f:
    data = json.load(f)
    data = {k: v for k, v in data.items() if v}

os.makedirs("results/figures", exist_ok=True)
pkl_path = "/media/data/toyota/processed_data/trina_33_samples_final/"
csv_path = f"/media/data/toyota/processed_data/trina_33_final/"

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

    if this_mode not in modes or d not in data:
        print(f"{d} does not have {this_mode}")
        continue

    proba = data[d]
    # smooth the probability by 3 points
    proba = np.convolve(proba, np.ones(3) / 3, mode="same")
    proba[:3] = data[d][:3]
    proba[-3:] = data[d][-3:]

    video, len_video = open_pkl(pkl_path + f"{d}_{this_mode}_video.pkl")
    free, len_free = open_pkl(pkl_path + f"{d}_{this_mode}_free.pkl")
    event, len_event = open_pkl(pkl_path + f"{d}_{this_mode}_event.pkl")
    bounds = [len_video, len_free, len_event]

    bounds = np.cumsum(bounds)
    if len(proba) > bounds[-1]:
        bounds = update_bounds(modes)
        bounds = configure_eval1(bounds, eval_mode)
    else:
        bounds = configure_eval2(bounds, eval_mode)

    free_proba = proba[bounds[0] : bounds[1]]
    event_proba = (
        proba[bounds[1] : bounds[2]]
        if len(bounds) == 3
        else proba[bounds[2] : bounds[3]]
    )
    y_pred = np.concatenate([free_proba, event_proba])
    y_true = np.concatenate([np.zeros(len(free_proba)), np.ones(len(event_proba))])

    auroc = roc_auc_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred >= 0.5, average="macro")
    bac = balanced_accuracy_score(y_true, y_pred >= 0.5)
    scores[d] = {"ROC-AUC": auroc, "F1 Macro": f1_macro, "BAC": bac}
    print(scores[d])

    if d == "P14" and this_mode == "I":
        continue

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
    if "E" in eval_mode:
        for i, row in df.iterrows():
            if row["event"] == start_event[this_mode]:
                flag = True
            if flag:
                offset_list.append(row["time"])
                event_list.append(row["event"])
            if row["event"] in ["Clear Text", "Experiment End"]:
                break

        bound_of_interest = bounds[1] if len(bounds) == 3 else bounds[2]
        new_change = offset_list + bound_of_interest
        new_change_i = [i for i, c in enumerate(new_change) if c >= bound_of_interest]
        new_change = [new_change[i] for i in new_change_i]
        label_list = [event_list[i] for i in new_change_i]

    if eval_mode == "FvE":
        plt.figure(figsize=(15, 3))
        plt.plot(np.array(proba), color="navy", linewidth=0.99, alpha=0.75)

        for xc in new_change:
            if xc >= bounds[1]:
                plt.axvline(x=xc, color="crimson")

        plt.title(f"{d} - {this_mode} {label_list}")
        plt.ylabel("Anomaly probability")
        plt.xlabel("Time (s)")
        plt.xticks(np.arange(0, len(proba), 30), np.arange(0, len(proba), 30))
        plt.grid(axis="x", linestyle="--", alpha=0.5, linewidth=0.75)
        plt.xlim(bounds[0] - 5, bounds[2] + 5)
        plt.ylim(-0.05, 1.05)
        plt.savefig(f"results/figures/{d}_{fig_name}", bbox_inches="tight", dpi=300)

    # get probabilities close to the events
    if "E" in eval_mode:
        all_event_auroc = []
        for e, event in enumerate(new_change[:-1]):
            print(f"Event: {label_list[e]}")
            event_proba = proba[int(event) - 0 : int(event) + 30]
            y_pred_event = np.concatenate([free_proba, event_proba])
            y_true_event = np.concatenate(
                [np.zeros(len(free_proba)), np.ones(len(event_proba))]
            )
            auroc = roc_auc_score(y_true_event, y_pred_event)
            f1_macro = f1_score(y_true_event, y_pred_event >= 0.5, average="macro")
            bac = balanced_accuracy_score(y_true_event, y_pred_event >= 0.5)
            print(f"ROC-AUC: {auroc:.3f}, F1 Macro: {f1_macro:.3f}, BAC: {bac:.3f}")
            all_event_auroc.append(auroc)

        print(f"Average event ROC-AUC: {np.mean(all_event_auroc):.3f}")

# print average scores
auroc_list, f1_list, bac_list = [], [], []
for d in scores:
    auroc_list.append(scores[d]["ROC-AUC"])
    f1_list.append(scores[d]["F1 Macro"])
    bac_list.append(scores[d]["BAC"])

print("\nAverage ROC-AUC:", np.mean(auroc_list).round(3))
print("Average F1 Macro:", np.mean(f1_list).round(3))
print("Average B. Acc:", np.mean(bac_list).round(3))
