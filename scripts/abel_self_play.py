# scripts/abel_self_play.py
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pypokerengine.api.game import setup_config, start_poker
from pypokerengine.players import BasePokerPlayer

from ai.Abel.agents.RLBasedPlayer import RLBasedPlayer
from ai.Abel.utils.hand_history_logger import HandHistoryLogger
from ai.Abel.utils.training_metrics import TrainingMetricsTracker
from ai.Abel.utils.hand_strength import HandStrengthCalculator
from ai.Abel.utils.decision_evaluator import DecisionEvaluator

# Setup directories
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f"logs/self_play_{timestamp}"
model_dir = "models/checkpoints"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Logging setup
logging.basicConfig(
    filename=f"{log_dir}/self_play.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

def train_abel(num_games=10000, eval_interval=500, save_interval=250):
    logging.info(f"Starting self-play training for {num_games} games")

    state_size, action_size = 6, 3
    hand_logger = HandHistoryLogger(log_dir=f"{log_dir}/hand_history")
    metrics_tracker = TrainingMetricsTracker(log_dir=log_dir, experiment_name="abel_self_play")
    decision_eval = DecisionEvaluator(log_dir=f"{log_dir}/decisions")

    abel_1 = RLBasedPlayer(state_size, action_size, model_path="models/abel_main.h5", decision_evaluator=decision_eval)
    abel_2 = RLBasedPlayer(state_size, action_size, model_path="models/abel_opponent.h5", decision_evaluator=decision_eval)
    has_bluff_tracking = hasattr(abel_1, "get_bluff_success_rate")

    wins, total_profit = 0, 0

    for game in range(num_games):
        config = setup_config(max_round=5, initial_stack=1000, small_blind_amount=10)
        if game % 2 == 0:
            config.register_player(name="Abel_1", algorithm=abel_1)
            config.register_player(name="Abel_2", algorithm=abel_2)
        else:
            config.register_player(name="Abel_2", algorithm=abel_2)
            config.register_player(name="Abel_1", algorithm=abel_1)

        abel_position = "SB" if game % 2 == 0 else "BB"
        table_info = {
            "table_id": f"self_play_game_{game}",
            "small_blind": 10,
            "big_blind": 20
        }
        player_info = [
            {"name": "Abel_1", "position": abel_position, "stack": 1000},
            {"name": "Abel_2", "position": "BB" if abel_position == "SB" else "SB", "stack": 1000}
        ]

        hand_logger.start_hand()
        hand_logger.start_new_hand(table_info, player_info)

        game_result = start_poker(config, verbose=0)

        abel_player = next(p for p in game_result["players"] if p["name"] == "Abel_1")
        opponent = next(p for p in game_result["players"] if p["name"] == "Abel_2")

        profit = abel_player["stack"] - 1000
        total_profit += profit
        if abel_player["stack"] > opponent["stack"]:
            wins += 1

        # Log hand result every 10 games
        if game % 10 == 0:
            try:
                hand_logger.log_hand_result(
                    winners=[abel_player["name"] if abel_player["stack"] > opponent["stack"] else opponent["name"]],
                    pot_distribution={},
                    player_final_states=[
                        {"name": abel_player["name"], "stack": abel_player["stack"]},
                        {"name": opponent["name"], "stack": opponent["stack"]}
                    ]
                )
            except Exception as e:
                logging.warning(f"Hand logging failed: {e}")

        # Evaluation + Metrics
        if game % eval_interval == 0 or game == num_games - 1:
            try:
                decisions = decision_eval.get_decisions()
                total_hand_strength = 0
                raise_sizes = []
                action_counts = {"fold": 0, "call": 0, "raise": 0}
                pot_odds_calls = {"justified": 0, "unjustified": 0}

                for d in decisions:
                    chosen = d["chosen_action"]
                    strength = d["hand_strength"]
                    pot = d["pot_size"]
                    is_bluff = d.get("is_bluff", False)

                    total_hand_strength += strength
                    if chosen == "raise":
                        raise_sizes.append(pot)
                    if chosen in action_counts:
                        action_counts[chosen] += 1
                    if pot > 0:
                        if strength > 0.5:
                            pot_odds_calls["justified"] += 1
                        else:
                            pot_odds_calls["unjustified"] += 1

                avg_strength = total_hand_strength / max(1, len(decisions))
                avg_raise = np.mean(raise_sizes) if raise_sizes else 0

                metrics = {
                    "win_rate": (wins / (game + 1)) * 100,
                    "avg_reward": total_profit / (game + 1),
                    "exploration_rate": abel_1.epsilon,
                    "bluff_success_rate": abel_1.get_bluff_success_rate() if has_bluff_tracking else 0,
                    "hand_strength_correlation": avg_strength,
                    "raise_sizes": avg_raise,
                    "action_frequency": action_counts,
                    "pot_odds_calls": pot_odds_calls,
                    "aggression_factor": abel_1.get_aggression_factor(),
                    "fold_rate": abel_1.get_fold_rate(),
                    "average_raise_size": abel_1.get_average_raise_size(),
                    "opponent": "self-play"
                }

                metrics_tracker.log_iteration(game, metrics)
                decision_eval.reset()
            except Exception as e:
                logging.warning(f"Failed to log metrics: {e}")

        if game % save_interval == 0 or game == num_games - 1:
            try:
                abel_1.save_model()
                ckpt_path = f"{model_dir}/abel_self_play_{game}.h5"
                abel_1.model.save_model(ckpt_path)
                logging.info(f"Saved checkpoint: {ckpt_path}")
            except Exception as e:
                logging.error(f"Error saving checkpoint: {e}")

        abel_1.adjust_epsilon()
        abel_2.adjust_epsilon()

    # Final plots + summary
    try:
        metrics_tracker.plot_win_rate()
        metrics_tracker.plot_learning_metrics()
        if has_bluff_tracking:
            metrics_tracker.plot_playing_style_metrics()
        metrics_tracker.generate_training_summary()
    except Exception as e:
        logging.error(f"Final plot/gen error: {e}")

    print("\n=== Self-Play Training Summary ===")
    print(f"Games Played: {num_games}")
    print(f"Abel Wins: {wins} ({(wins / num_games) * 100:.2f}%)")
    print(f"Avg Profit: {total_profit / num_games:.2f}")
    print(f"Final Epsilon: {abel_1.epsilon:.4f}")
    print(f"Logs saved to: {log_dir}")
    
    if decision_eval:
        decision_eval.save()
        
    return abel_1

if __name__ == "__main__":
    trained = train_abel(num_games=1000)
