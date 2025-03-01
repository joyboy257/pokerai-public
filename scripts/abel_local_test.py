import os
import logging
from ai.Abel.agents.RLBasedPlayer import RLBasedPlayer  # Use RLBasedPlayer for both players
from pypokerengine.api.game import setup_config, start_poker

# Local Test Configuration
NUM_TEST_GAMES = 50
LOG_DIR = "logs"

# Setup Logging
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(filename=os.path.join(LOG_DIR, "local_test.log"), level=logging.INFO, format="%(asctime)s - %(message)s")

def local_test_abel(num_games=NUM_TEST_GAMES):
    """
    Runs a quick test of Abel for local verification.
    """
    state_size = 6
    action_size = 3
    abel = RLBasedPlayer(state_size, action_size)
    opponent = RLBasedPlayer(state_size, action_size)  # Using RL-based opponent
    win_count = 0

    print("\n=== Running Local Test ===")
    for game_number in range(num_games):
        config = setup_config(max_round=5, initial_stack=1000, small_blind_amount=10)
        config.register_player(name="Abel", algorithm=abel)
        config.register_player(name="Opponent", algorithm=opponent)
        game_result = start_poker(config, verbose=1)

        # Check if Abel won
        abel_stack = next(player["stack"] for player in game_result["players"] if player["name"] == "Abel")
        opponent_stack = next(player["stack"] for player in game_result["players"] if player["name"] == "Opponent")

        if abel_stack > opponent_stack:
            win_count += 1

        # Save model checkpoint every 10 games
        if (game_number + 1) % 10 == 0:
            abel.save_model()

    # Summary
    total_games = num_games
    win_rate = (win_count / total_games) * 100
    print(f"\n=== Local Test Summary ===")
    print(f"Total Games Played: {total_games}")
    print(f"Total Wins: {win_count}")
    print(f"Win Rate: {win_rate:.2f}%")

    # Logging Summary
    logging.info(f"Local Test Summary: Total Games={total_games}, Wins={win_count}, Win Rate={win_rate:.2f}%")

if __name__ == "__main__":
    local_test_abel()
