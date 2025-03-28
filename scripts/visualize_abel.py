import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Update this path as needed
METRICS_FILE = "logs/self_play_20250327_172850/abel_self_play/metrics.json"
PLOT_DIR = "logs/self_play_20250327_172850/plots"
os.makedirs(PLOT_DIR, exist_ok=True)


def load_metrics(metrics_file):
    with open(metrics_file, "r") as f:
        return json.load(f)


def smooth(values, window_size=10):
    return pd.Series(values).rolling(window=window_size, min_periods=1).mean().tolist()


def plot_metric(iterations, values, name, ylabel, color="blue", smooth_window=10):
    plt.figure(figsize=(10, 5))
    plt.plot(iterations, values, alpha=0.3, label=f"{name} (raw)", color=color)
    if len(values) >= smooth_window:
        plt.plot(iterations, smooth(values, smooth_window), label=f"{name} (smoothed)", color=color)
    plt.xlabel("Iterations")
    plt.ylabel(ylabel)
    plt.title(f"{name} Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{name.replace(' ', '_').lower()}.png"))
    plt.close()


def main():
    metrics = load_metrics(METRICS_FILE)
    iterations = metrics["iterations"]

    # Standard metrics to plot
    tracked = [
        ("win_rates", "Win Rate (%)", "blue"),
        ("loss_values", "Loss", "red"),
        ("exploration_rates", "Exploration Rate (Epsilon)", "green"),
        ("avg_rewards", "Average Reward", "purple"),
        ("bluff_success_rates", "Bluff Success Rate (%)", "orange"),
        ("fold_rates", "Fold Rate (%)", "gray"),
        ("aggression_factors", "Aggression Factor", "brown"),
        ("raise_sizes", "Raise Size", "teal"),
        ("hand_strength_correlations", "Hand Strength Correlation", "darkcyan")
    ]

    for key, label, color in tracked:
        values = metrics.get(key, [])
        if values and any(v is not None for v in values):
            filtered = [v if v is not None else np.nan for v in values]
            plot_metric(iterations, filtered, label, label, color=color)

    # Action frequencies breakdown (aggregated)
    action_freqs = metrics.get("action_frequencies", [])
    aggregated = {"fold": 0, "call": 0, "raise": 0}
    count = 0
    for entry in action_freqs:
        if entry:
            for key in aggregated:
                aggregated[key] += entry.get(key, 0)
            count += 1

    if count > 0:
        plt.figure(figsize=(6, 6))
        actions = list(aggregated.keys())
        values = [aggregated[a] for a in actions]
        plt.bar(actions, values, color=["lightblue", "lightgreen", "salmon"])
        plt.title("Total Action Frequencies")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, "action_frequencies.png"))
        plt.close()

    print(f"All visualizations saved to {PLOT_DIR}")


if __name__ == "__main__":
    main()