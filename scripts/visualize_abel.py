# scripts/visualize_abel.py
import matplotlib.pyplot as plt

def plot_win_rate(win_rates):
    plt.plot(win_rates)
    plt.xlabel('Episodes')
    plt.ylabel('Win Rate (%)')
    plt.title('Abel Win Rate Over Time')
    plt.show()

def plot_q_values(q_values):
    plt.plot(q_values)
    plt.xlabel('Episodes')
    plt.ylabel('Q-Values')
    plt.title('Q-Values Over Time')
    plt.show()
