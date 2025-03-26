import os
import re
import csv
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(
    description="This script parses and visualizes the results of hyperparameter grid search experiments for the Abel poker agent. It generates bar plots of self-play win rates, displays top N experiments, and exports a CSV summary for further analysis."
)
parser.add_argument("--base_dir", type=str, default="logs/grid_search", help="Base directory containing grid search experiment logs")
parser.add_argument("--top_n", type=int, default=5, help="Number of top experiments to display")
args = parser.parse_args()

base_dir = args.base_dir
experiments = [d for d in os.listdir(base_dir) if d.startswith("exp_")]
experiments.sort()

exp_labels = []
win_rates = []
summary_table = []

# Extract win rates and hyperparameters from experiment logs
for exp in experiments:
    log_file = os.path.join(base_dir, exp, "experiment.log")
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            lines = f.readlines()
            eps_decay, lr, gamma, batch_size, buffer_size = None, None, None, None, None
            for line in lines:
                if "eps_decay" in line:
                    match = re.search(r"eps_decay=(.*?), lr=(.*?), gamma=(.*?), batch_size=(.*?), buffer=(.*?)\n", line)
                    if match:
                        eps_decay, lr, gamma, batch_size, buffer_size = match.groups()
                        break
            for line in lines[::-1]:  # reverse search for latest entry
                match = re.search(r"Win Rate = ([0-9\.]+)%", line)
                if match:
                    label = f"{exp}\nED:{eps_decay}, LR:{lr}, G:{gamma}, BS:{batch_size}"
                    win_rate = float(match.group(1))
                    exp_labels.append(label)
                    win_rates.append(win_rate)
                    summary_table.append({
                        "experiment": exp,
                        "win_rate": win_rate,
                        "eps_decay": eps_decay,
                        "lr": lr,
                        "gamma": gamma,
                        "batch_size": batch_size,
                        "buffer_size": buffer_size
                    })
                    break

# Plot win rates
plt.figure(figsize=(14, 6))
plt.bar(exp_labels, win_rates, color="skyblue")
plt.xlabel("Experiment + Hyperparameters")
plt.ylabel("Self-Play Win Rate (%)")
plt.title("Grid Search Results: Self-Play Win Rate by Experiment")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.grid(axis="y")
plt.savefig(f"{base_dir}/grid_search_results.png")
plt.show()

# Print top N experiments
summary_table = sorted(summary_table, key=lambda x: x["win_rate"], reverse=True)
print(f"\n=== Top {args.top_n} Experiments ===")
for i, entry in enumerate(summary_table[:args.top_n]):
    print(f"{i+1}. {entry['experiment']} | Win Rate: {entry['win_rate']}% | ED:{entry['eps_decay']}, LR:{entry['lr']}, G:{entry['gamma']}, BS:{entry['batch_size']}, BUF:{entry['buffer_size']}")

# Export to CSV
csv_path = f"{base_dir}/grid_search_summary.csv"
with open(csv_path, mode='w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["experiment", "win_rate", "eps_decay", "lr", "gamma", "batch_size", "buffer_size"])
    writer.writeheader()
    writer.writerows(summary_table)

print(f"\nCSV exported to {csv_path}")
print(f"Grid search results plot saved as {base_dir}/grid_search_results.png")
