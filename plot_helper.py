import numpy as np, os, json
import matplotlib.pyplot as plt


def plot_proba_ts(sub, this_mode, y_pred, bounds, offset_list, event_list):
    y_pred = np.array(y_pred)
    y_pred_t = np.mean(y_pred, axis=0)

    plt.figure(figsize=(15, 3))
    plt.plot(y_pred_t, color="navy", linewidth=0.99, alpha=0.75)

    for xc in offset_list:
        plt.axvline(x=xc, color="crimson")

    plt.title(f"{sub} - {this_mode} {event_list}")
    plt.ylabel("Anomaly probability")
    plt.xlabel("Time (s)")
    plt.xticks(np.arange(0, len(y_pred_t), 30), np.arange(0, len(y_pred_t), 30))
    plt.grid(linestyle="--", alpha=0.5, linewidth=0.75)
    plt.xlim(-5, bounds[1] + 5)
    plt.ylim(-0.05, 1.05)

    os.makedirs("figures", exist_ok=True)
    plt.savefig(f"figures/{sub}_{this_mode}.png", bbox_inches="tight", dpi=300)
    plt.close()

    # Save the data
    to_json = {
        "y_pred": y_pred.tolist(),
        "offset_list": offset_list.tolist(),
        "event_names": event_list,
    }
    os.makedirs("ts_data", exist_ok=True)
    with open(f"ts_data/{sub}_{this_mode}.json", "w") as f:
        json.dump(to_json, f, indent=4)


def plot_events_boxplot(scores, events, this_mode, train_mode, score_type="ROC-AUC"):
    data = [scores[ev][score_type] for ev in events]
    _, ax = plt.subplots(figsize=(len(events), 4), dpi=200)
    box = ax.boxplot(
        data,
        patch_artist=True,
        boxprops=dict(facecolor="lightcoral", alpha=0.75, linewidth=0.75),
        medianprops=dict(color="crimson", linewidth=2),
        whiskerprops=dict(color="black", linewidth=0.75),
    )
    # Set different color for the first box
    colors = ["steelblue"] + (len(data) - 1) * ["lightblue"]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)

    # Formatting
    if score_type == "ROC-AUC":
        plt.ylabel("ROC-AUC")
        plt.ylim(0.401, 1)
    else:
        plt.ylabel("Effect Size (Cohen's d)")
        plt.ylim(-0.35, 1.19)

    plt.xticks(range(1, len(events) + 1), events, rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.75, linewidth=0.75)

    # Remove up and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Save the plot
    plt.savefig(
        f"figures/boxplot_{this_mode}_{train_mode}_{score_type}.png",
        bbox_inches="tight",
    )
    plt.close()
