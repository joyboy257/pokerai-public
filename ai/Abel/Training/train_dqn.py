from ai.Abel.agents.RLBasedPlayer import RLBasedPlayer
from pypokerengine.api.game import setup_config, start_poker
import logging

# Configure logging
logging.basicConfig(filename="training.log", level=logging.INFO, format="%(asctime)s - %(message)s")

def train_abel_dqn(num_games=10000):
    state_size = 6
    action_size = 3
    abel = RLBasedPlayer(state_size, action_size)
    win_count = 0
    bluffing_wins = 0

    for game_number in range(num_games):
        config = setup_config(max_round=5, initial_stack=1000, small_blind_amount=10)
        config.register_player(name="Abel", algorithm=abel)
        config.register_player(name="Opponent", algorithm=RLBasedPlayer(state_size, action_size))  # Self-play
        game_result = start_poker(config, verbose=0)

        abel_stack = next(player["stack"] for player in game_result["players"] if player["name"] == "Abel")
        opponent_stack = next(player["stack"] for player in game_result["players"] if player["name"] == "Opponent")

        if abel_stack > opponent_stack:
            win_count += 1
            # Check if Abel bluffed
            if len(abel.opponent_history) > 0 and "raise" in abel.opponent_history[-1]:
                bluffing_wins += 1

        if game_number % 100 == 0:
            abel.cfr_player.update_regrets(str(game_result), action_idx=0, utility=0)  # Reward computed after round
            abel.model.save_model(abel.model_path)

        if game_number % 1000 == 0:
            logging.info(f"Game {game_number}: Win Rate={(win_count / (game_number + 1)) * 100:.2f}%, Bluff Wins={(bluffing_wins / max(1, win_count)) * 100:.2f}%")

    print(f"\n=== Training Summary ===")
    print(f"Total Games Played: {num_games}")
    print(f"Total Wins: {win_count}")
    print(f"Bluff Wins: {bluffing_wins}")
    print(f"Win Rate: {(win_count / num_games) * 100:.2f}%")
