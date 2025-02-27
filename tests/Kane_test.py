import logging
from pypokerengine.api.game import setup_config, start_poker
from ai.Kane.Kane import RuleBasedPlayer
from ai.Kane.strategies.AggressiveStrategy import AggressiveStrategy
from ai.Kane.strategies.DefensiveStrategy import DefensiveStrategy

def run_game(num_games=10000):
    win_count = 0
    bluff_count = 0
    strategy_switches = 0
    total_games = 0

    for i in range(num_games):
        config = setup_config(max_round=5, initial_stack=1000, small_blind_amount=10)
        config.register_player(name="Kane", algorithm=RuleBasedPlayer())
        config.register_player(name="Opponent", algorithm=RuleBasedPlayer(strategy=DefensiveStrategy()))

        game_result = start_poker(config, verbose=0)
        total_games += 1

        # Check for Kane's win
        if game_result["players"][0]["name"] == "Kane" and game_result["players"][0]["stack"] > game_result["players"][1]["stack"]:
            win_count += 1

        # Collect bluffing and strategy data
        with open('logs/kane_log.txt', 'r') as log_file:
            logs = log_file.read()
            bluff_count += logs.count('Bluffing activated!')
            strategy_switches += logs.count('Strategy switched to')

    # Calculate Win Rate
    win_rate = (win_count / total_games) * 100

    # Display Results
    print("\n=== Kane Training Summary ===")
    print(f"Total Games Played: {total_games}")
    print(f"Total Wins: {win_count}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Bluffing Frequency: {bluff_count / total_games:.2%}")
    print(f"Strategy Switches: {strategy_switches}")

    # Save Results for Visualization
    with open("logs/kane_results.txt", "w") as result_file:
        result_file.write(f"Total Games Played: {total_games}\n")
        result_file.write(f"Total Wins: {win_count}\n")
        result_file.write(f"Win Rate: {win_rate:.2f}%\n")
        result_file.write(f"Bluffing Frequency: {bluff_count / total_games:.2%}\n")
        result_file.write(f"Strategy Switches: {strategy_switches}\n")

if __name__ == "__main__":
    run_game()
