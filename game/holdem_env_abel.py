import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU


import sys
import os

# Add the project root directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pypokerengine.api.game import setup_config, start_poker
from ai.Abel_rl_based import RLBasedPlayer
from pypokerengine.players import BasePokerPlayer
import json


class RandomPlayer(BasePokerPlayer):
    def declare_action(self, valid_actions, hole_card, round_state):
        action = valid_actions[1]['action']  # Always calls
        amount = valid_actions[1]['amount']
        return action, amount

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass


def train_abel(num_games=1000):
    state_size = 11  # Number of features in encode_state method
    action_size = 3  # Fold, Call, Raise

    abel = RLBasedPlayer(state_size, action_size)
    abel_wins = 0
    random_player_wins = 0

    # Track results in batches of 100 games
    batch_results = []

    for game_number in range(1, num_games + 1):
        # Setup and play a game
        config = setup_config(max_round=10, initial_stack=1000, small_blind_amount=10)
        config.register_player(name="Abel", algorithm=abel)
        config.register_player(name="RandomPlayer", algorithm=RandomPlayer())
        game_result = start_poker(config, verbose=0)

        # Evaluate results
        if game_result['players'][0]['name'] == "Abel" and game_result['players'][0]['stack'] > game_result['players'][1]['stack']:
            abel_wins += 1
        elif game_result['players'][1]['name'] == "RandomPlayer" and game_result['players'][1]['stack'] > game_result['players'][0]['stack']:
            random_player_wins += 1

        # Decay epsilon
        abel.adjust_epsilon()

        # Track results in batches of 100
        if game_number % 100 == 0:
            win_rate = abel_wins / 100 * 100
            batch_results.append(f"Batch {game_number // 100}: Abel Win Rate = {win_rate:.2f}%")
            print(f"Progress: {game_number}/{num_games} games completed.")
            print(batch_results[-1])  # Print the latest batch result
            
            # Reset win counters for the next batch
            abel_wins = 0
            random_player_wins = 0

    # Save results to a file
    with open("training_results_log.txt", "w") as f:
        f.write("\n".join(batch_results))

    # Final Results
    print(f"Training completed. See 'training_results_log.txt' for batch results.")


if __name__ == "__main__":
    train_abel(1000)