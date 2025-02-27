import matplotlib.pyplot as plt

def visualize_results():
    games = []
    wins = []
    win_rates = []
    bluff_freqs = []
    strategy_switches = []

    with open("logs/kane_results.txt", "r") as result_file:
        lines = result_file.readlines()
        total_games = int(lines[0].split(":")[1].strip())
        total_wins = int(lines[1].split(":")[1].strip())
        win_rate = float(lines[2].split(":")[1].strip().replace('%', ''))
        bluff_freq = float(lines[3].split(":")[1].strip().replace('%', ''))
        strategy_switch = int(lines[4].split(":")[1].strip())

    games.append(total_games)
    wins.append(total_wins)
    win_rates.append(win_rate)
    bluff_freqs.append(bluff_freq)
    strategy_switches.append(strategy_switch)

    # Plot Win Rate
    plt.figure(figsize=(10, 5))
    plt.plot(games, win_rates, label='Win Rate', color='blue')
    plt.xlabel('Games Played')
    plt.ylabel('Win Rate (%)')
    plt.title('Win Rate Over Games')
    plt.legend()
    plt.grid(True)
    plt.savefig('logs/kane_win_rate.png')
    plt.show()

    # Plot Bluffing Frequency
    plt.figure(figsize=(10, 5))
    plt.plot(games, bluff_freqs, label='Bluffing Frequency', color='red')
    plt.xlabel('Games Played')
    plt.ylabel('Bluffing Frequency (%)')
    plt.title('Bluffing Frequency Over Games')
    plt.legend()
    plt.grid(True)
    plt.savefig('logs/kane_bluff_freq.png')
    plt.show()

    # Plot Strategy Switches
    plt.figure(figsize=(10, 5))
    plt.plot(games, strategy_switches, label='Strategy Switches', color='green')
    plt.xlabel('Games Played')
    plt.ylabel('Strategy Switches')
    plt.title('Strategy Switches Over Games')
    plt.legend()
    plt.grid(True)
    plt.savefig('logs/kane_strategy_switches.png')
    plt.show()

if __name__ == "__main__":
    visualize_results()
