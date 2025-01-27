from pypokerengine.api.game import setup_config, start_poker
from ai.Kane_rule_based import RuleBasedPlayer
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


def setup_poker_game():
    config = setup_config(max_round=10, initial_stack=1000, small_blind_amount=10)
    config.register_player(name="Kane", algorithm=RuleBasedPlayer())
    config.register_player(name="RandomPlayer", algorithm=RandomPlayer())
    return config


def play_game():
    config = setup_poker_game()
    game_result = start_poker(config, verbose=1)
    print(game_result)


if __name__ == "__main__":
    play_game()
