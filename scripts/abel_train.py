# scripts/abel_train.py
#1 milly games set rn
import os
import logging
from ai.Abel.agents.RLBasedPlayer import RLBasedPlayer
from pypokerengine.api.game import setup_config, start_poker

# Configure Logging
log_dir = "models/logs"
logging.basicConfig(filename="abel_training.log", level=logging.INFO, format="%(asctime)s - %(message)s")

def train_abel(num_games=1000):
    state_size = 6
    action_size = 3
    abel = RLBasedPlayer(state_size, action_size)
    opponent = RLBasedPlayer(state_size, action_size)  # <-- Fix: Pass required arguments

    win_count = 0

    for game_number in range(num_games):
        config = setup_config(max_round=5, initial_stack=1000, small_blind_amount=10)
        config.register_player(name="Abel", algorithm=abel)
        config.register_player(name="Opponent", algorithm=opponent)  # Fix: Properly initialized
        game_result = start_poker(config, verbose=0)

        if game_result["players"][0]["name"] == "Abel" and game_result["players"][0]["stack"] > game_result["players"][1]["stack"]:
            win_count += 1

        if game_number % 100 == 0:
            abel.save_model()

    print(f"\n=== Training Summary ===")
    print(f"Total Games Played: {num_games}")
    print(f"Total Wins: {win_count}")
    print(f"Win Rate: {(win_count / num_games) * 100:.2f}%")

if __name__ == "__main__":
    train_abel()
