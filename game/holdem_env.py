import sys
import os

# Add the project root directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import logging
from ai.Kane_rule_based import RuleBasedPlayer
from pypokerengine.api.game import setup_config, start_poker
from pypokerengine.players import BasePokerPlayer

class RandomPlayer(BasePokerPlayer):
    def declare_action(self, valid_actions, hole_card, round_state):
        action = valid_actions[1]['action']  # 'call'
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


def simulate_games(num_games=100):
    kane_wins = 0
    random_player_wins = 0

    for i in range(num_games):
        # Setup the game
        config = setup_config(max_round=10, initial_stack=1000, small_blind_amount=10)
        config.register_player(name="Kane", algorithm=RuleBasedPlayer())
        config.register_player(name="RandomPlayer", algorithm=RandomPlayer())
        
        # Run the game
        game_result = start_poker(config, verbose=0)
        
        # Evaluate results
        if game_result['players'][0]['name'] == "Kane" and game_result['players'][0]['stack'] > game_result['players'][1]['stack']:
            kane_wins += 1
        elif game_result['players'][1]['name'] == "RandomPlayer" and game_result['players'][1]['stack'] > game_result['players'][0]['stack']:
            random_player_wins += 1

    print(f"Kane Wins: {kane_wins}")
    print(f"RandomPlayer Wins: {random_player_wins}")


if __name__ == "__main__":
    simulate_games(100)
