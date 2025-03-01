# ai/Abel/training/train_dqn.py
from ai.Abel.agents.RLBasedPlayer import RLBasedPlayer
from pypokerengine.api.game import setup_config, start_poker
from pypokerengine.players import BasePokerPlayer

def train_abel_dqn(num_games=10000):
    state_size = 6
    action_size = 3
    abel = RLBasedPlayer(state_size, action_size)
    win_count = 0

    for game_number in range(num_games):
        config = setup_config(max_round=5, initial_stack=1000, small_blind_amount=10)
        config.register_player(name="Abel", algorithm=abel)
        config.register_player(name="Opponent", algorithm=BasePokerPlayer())
        game_result = start_poker(config, verbose=0)

        if game_result["players"][0]["name"] == "Abel" and game_result["players"][0]["stack"] > game_result["players"][1]["stack"]:
            win_count += 1

        if game_number % 100 == 0:
            abel.model.save_model(abel.model_path)

    print(f"\n=== Training Summary ===")
    print(f"Total Games Played: {num_games}")
    print(f"Total Wins: {win_count}")
    print(f"Win Rate: {(win_count / num_games) * 100:.2f}%")
