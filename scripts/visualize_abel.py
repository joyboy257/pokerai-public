import matplotlib.pyplot as plt
import numpy as np
import re

LOG_FILE = "logs/local_test.log"

def extract_win_rates(log_file=LOG_FILE):
    """ Extracts win rates from the log file. """
    win_rates = []
    with open(log_file, "r") as f:
        for line in f:
            if "Win Rate" in line:
                rate = float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[-1])
                win_rates.append(rate)
    return win_rates

def extract_q_values(log_file=LOG_FILE):
    """ Extracts Q-values from logs (if logged). """
    q_values = []
    with open(log_file, "r") as f:
        for line in f:
            if "Q-Value" in line:
                value = float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[-1])
                q_values.append(value)
    return q_values

def extract_bluffing_stats(log_file=LOG_FILE):
    """ Extracts bluffing attempts and success rate. """
    total_bluffs = 0
    successful_bluffs = 0

    with open(log_file, "r") as f:
        logs = f.readlines()

    for i, line in enumerate(logs):
        if "BLUFFED" in line:
            total_bluffs += 1
            # Check if bluff was successful (if Abel won the round after bluffing)
            if i + 2 < len(logs) and "won the round" in logs[i + 2]:
                successful_bluffs += 1

    bluff_success_rate = (successful_bluffs / total_bluffs) * 100 if total_bluffs > 0 else 0
    return total_bluffs, successful_bluffs, bluff_success_rate

def plot_win_rate(win_rates):
    """ Plots win rate over time. """
    plt.figure(figsize=(10, 5))
    plt.plot(win_rates, marker='o', linestyle='-', color='blue', label="Win Rate")
    plt.xlabel('Episodes')
    plt.ylabel('Win Rate (%)')
    plt.title('Abel Win Rate Over Time')
    plt.legend()
    plt.grid()
    plt.show()

def plot_q_values(q_values):
    """ Plots Q-values over time. """
    plt.figure(figsize=(10, 5))
    plt.plot(q_values, marker='x', linestyle='-', color='red', label="Q-Values")
    plt.xlabel('Episodes')
    plt.ylabel('Q-Values')
    plt.title('Q-Values Over Time')
    plt.legend()
    plt.grid()
    plt.show()

def plot_bluffing_stats(total_bluffs, successful_bluffs):
    """ Plots bluffing attempts and success rate. """
    labels = ["Total Bluffs", "Successful Bluffs"]
    values = [total_bluffs, successful_bluffs]
    colors = ["red", "green"]

    plt.figure(figsize=(6, 6))
    plt.bar(labels, values, color=colors)
    plt.xlabel("Bluffing Metrics")
    plt.ylabel("Count")
    plt.title("Abel Bluffing Analysis")
    plt.show()

if __name__ == "__main__":
    print("\n=== Analyzing Abel's Performance ===\n")

    # Extract performance metrics
    win_rates = extract_win_rates()
    q_values = extract_q_values()
    total_bluffs, successful_bluffs, bluff_success_rate = extract_bluffing_stats()

    # Display bluffing statistics
    print(f"Total Bluffs: {total_bluffs}")
    print(f"Successful Bluffs: {successful_bluffs}")
    print(f"Bluff Success Rate: {bluff_success_rate:.2f}%\n")

    # Plot results
    if win_rates:
        plot_win_rate(win_rates)
    if q_values:
        plot_q_values(q_values)
    if total_bluffs > 0:
        plot_bluffing_stats(total_bluffs, successful_bluffs)
