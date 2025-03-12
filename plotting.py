import numpy as np, os
import matplotlib.pyplot as plt


def plot_proba_ts(sub, this_mode, y_pred, bounds, offset_list, event_list):
    plt.figure(figsize=(15, 3))
    plt.plot(np.array(y_pred), color="navy", linewidth=0.99, alpha=0.75)

    for xc in offset_list:
        plt.axvline(x=xc, color="crimson")

    plt.title(f"{sub} - {this_mode} {event_list}")
    plt.ylabel("Anomaly probability")
    plt.xlabel("Time (s)")
    plt.xticks(np.arange(0, len(y_pred), 30), np.arange(0, len(y_pred), 30))
    plt.grid(linestyle="--", alpha=0.5, linewidth=0.75)
    plt.xlim(-5, bounds[1] + 5)
    plt.ylim(-0.05, 1.05)

    os.makedirs("figures", exist_ok=True)
    plt.savefig(f"figures/{sub}_{this_mode}.png", bbox_inches="tight", dpi=300)
    plt.close()


def plot_events_boxplot(scores, events, this_mode, train_mode):
    data = [scores[ev]["Effect"] for ev in events]
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
    plt.ylabel("Effect Size")
    # plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xticks(range(1, len(events) + 1), events, rotation=45, ha="right")
    # plt.ylim(0, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.75, linewidth=0.75)

    # Remove up and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Save the plot
    plt.savefig(f"figures/boxplot_{this_mode}_{train_mode}_r.png", bbox_inches="tight")
    plt.close()
