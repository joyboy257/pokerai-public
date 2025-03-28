# scripts/abel_local_test.py
import sys
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pypokerengine.api.game import setup_config, start_poker
from pypokerengine.players import BasePokerPlayer

# Import Abel
from ai.Abel.agents.RLBasedPlayer import RLBasedPlayer

# Import analysis tools if available (with error handling if they're not installed yet)
try:
    from ai.Abel.utils.hand_history_logger import HandHistoryLogger
    from ai.Abel.utils.training_metrics import TrainingMetricsTracker
    from ai.Abel.utils.hand_strength import HandStrengthCalculator
    from ai.Abel.utils.decision_evaluator import DecisionEvaluator
    analysis_tools_available = True
except ImportError:
    print("Note: Advanced analysis tools not available. Running with basic logging only.")
    analysis_tools_available = False

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Configure directories
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f"logs/local_test_{timestamp}"
os.makedirs(log_dir, exist_ok=True)

# Configure logging
logging.basicConfig(
    filename=f"{log_dir}/local_test.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

# Simple baseline player
class SimplePlayer(BasePokerPlayer):
    def declare_action(self, valid_actions, hole_card, round_state):
        call_action_info = valid_actions[1]
        return call_action_info['action'], call_action_info['amount']
    def receive_game_start_message(self, game_info): pass
    def receive_round_start_message(self, round_count, hole_card, seats): pass
    def receive_street_start_message(self, street, round_state): pass
    def receive_game_update_message(self, action, round_state): pass
    def receive_round_result_message(self, winners, hand_info, round_state): pass

def run_local_test(num_games=20, use_self_play=True):
    print(f"Starting local test with {num_games} games...")
    logging.info(f"Starting local test with {num_games} games")

    # Initialize analysis tools if available
    if analysis_tools_available:
        print("Using advanced analysis tools...")
        hand_logger = HandHistoryLogger(log_dir=f"{log_dir}/hand_history")
        metrics_tracker = TrainingMetricsTracker(log_dir=log_dir, experiment_name="local_test")
        decision_eval = DecisionEvaluator(log_dir=f"{log_dir}/decisions")
    else:
        decision_eval = None

    state_size = 6
    action_size = 3
    abel = RLBasedPlayer(state_size, action_size, decision_evaluator=decision_eval)

    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    main_model_path = f"{model_dir}/abel_main.h5"
    if not os.path.exists(main_model_path):
        try:
            abel.save_model()
            print(f"Initial model saved to {main_model_path}")
        except Exception as e:
            print(f"Warning: Could not save initial model: {e}")

    has_adjust_epsilon = hasattr(abel, 'adjust_epsilon')
    has_save_model = hasattr(abel, 'save_model')
    has_bluff_tracking = hasattr(abel, 'get_bluff_success_rate')

    games_played, abel_wins, total_profit = 0, 0, 0

    if use_self_play:
        abel_2 = RLBasedPlayer(state_size, action_size, decision_evaluator=decision_eval)
        print("Running self-play test...")

        for game in range(num_games):
            config = setup_config(max_round=5, initial_stack=1000, small_blind_amount=10)
            if game % 2 == 0:
                config.register_player(name="Abel", algorithm=abel)
                config.register_player(name="Abel_2", algorithm=abel_2)
            else:
                config.register_player(name="Abel_2", algorithm=abel_2)
                config.register_player(name="Abel", algorithm=abel)

            game_result = start_poker(config, verbose=1)

            abel_player = next((p for p in game_result["players"] if p["name"] == "Abel"), None)
            other_player = next((p for p in game_result["players"] if p["name"] == "Abel_2"), None)

            if abel_player and other_player:
                games_played += 1
                if abel_player["stack"] > other_player["stack"]:
                    abel_wins += 1
                profit = abel_player["stack"] - 1000
                total_profit += profit

                print(f"Game {game+1}: {'Won' if profit > 0 else 'Lost'}, Profit: {profit}, Stack: {abel_player['stack']}")

                if analysis_tools_available:
                    try:
                        hand_logger.log_hand_result(
                            winners=["Abel" if abel_player["stack"] > other_player["stack"] else "Abel_2"],
                            pot_distribution={},
                            player_final_states=[
                                {"name": "Abel", "stack": abel_player["stack"]},
                                {"name": "Abel_2", "stack": other_player["stack"]}
                            ]
                        )

                        metrics = {
                            "win_rate": (abel_wins / games_played) * 100,
                            "avg_reward": total_profit / games_played,
                            "exploration_rate": abel.epsilon,
                            "opponent": "self-play",
                            "fold_rate": abel.get_fold_rate(),
                            "aggression_factor": abel.get_aggression_factor(),
                            "raise_size": abel.get_average_raise_size(),
                            "pot_odds_calls": abel.get_pot_odds_call_justification(),
                        }

                        if has_bluff_tracking:
                            metrics["bluff_success_rate"] = abel.get_bluff_success_rate()

                        # Collect extended decision-level metrics
                        if decision_eval:
                            total_hand_strength = 0
                            raise_sizes = []
                            action_counts = {"fold": 0, "call": 0, "raise": 0}
                            pot_odds_calls = {"justified": 0, "unjustified": 0}

                            decisions = decision_eval.get_decisions()
                            for decision in decisions:
                                chosen = decision["chosen_action"]
                                strength = decision["hand_strength"]
                                is_bluff = decision["is_bluff"]
                                pot_size = decision["pot_size"]

                                total_hand_strength += strength
                                if chosen == "raise":
                                    raise_sizes.append(pot_size)
                                if chosen in action_counts:
                                    action_counts[chosen] += 1
                                if pot_size > 0:
                                    if strength > 0.5:
                                        pot_odds_calls["justified"] += 1
                                    else:
                                        pot_odds_calls["unjustified"] += 1

                            if decisions:
                                metrics["hand_strength_avg"] = total_hand_strength / len(decisions)
                                metrics["action_frequency"] = action_counts
                                metrics["raise_size"] = np.mean(raise_sizes) if raise_sizes else 0
                                metrics["pot_odds_calls"] = pot_odds_calls

                            decision_eval.reset()

                        metrics_tracker.log_iteration(game, metrics)
                    except Exception as e:
                        print(f"Warning: Error logging analysis data: {e}")

            if has_adjust_epsilon:
                abel.adjust_epsilon()
                abel_2.adjust_epsilon()

    # Final metric visuals
    if analysis_tools_available:
        try:
            metrics_tracker.plot_win_rate(save_path=f"{log_dir}/win_rate.png")
            metrics_tracker.plot_learning_metrics(save_path=f"{log_dir}/learning_metrics.png")
            if has_bluff_tracking:
                metrics_tracker.plot_playing_style_metrics(save_path=f"{log_dir}/playing_style.png")
            metrics_tracker.generate_training_summary(save_path=f"{log_dir}/training_summary.json")
            print(f"Metrics saved to: {log_dir}")
        except Exception as e:
            print(f"Warning: Error generating analysis plots: {e}")

    win_rate = (abel_wins / games_played) * 100 if games_played else 0
    avg_profit = total_profit / games_played if games_played else 0

    print("\n=== TEST SUMMARY ===")
    print(f"Games Played: {games_played}")
    print(f"Abel Wins: {abel_wins} ({win_rate:.2f}%)")
    print(f"Average Profit: {avg_profit:.2f} chips")
    print(f"Final Epsilon: {abel.epsilon:.4f}")
    print(f"Logs saved to: {log_dir}")

    if has_save_model:
        try:
            abel.save_model()
            print(f"Model saved to {main_model_path}")
        except Exception as e:
            print(f"Warning: Could not save model: {e}")

    return win_rate, avg_profit

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Local test for Abel poker AI')
    parser.add_argument('--games', type=int, default=20, help='Number of games to play')
    parser.add_argument('--opponent', choices=['self', 'simple'], default='self', help='Type of opponent (self-play or simple player)')
    args = parser.parse_args()

    use_self_play = (args.opponent == 'self')
    win_rate, avg_profit = run_local_test(num_games=args.games, use_self_play=use_self_play)
    print(f"\nTest completed successfully with win rate {win_rate:.2f}% and average profit {avg_profit:.2f} chips.")

class BasePokerPlayer:
    def declare_action(self, valid_actions, hole_card, round_state):
        raise NotImplementedError()
    def receive_game_start_message(self, game_info): pass
    def receive_round_start_message(self, round_count, hole_card, seats): pass
    def receive_street_start_message(self, street, round_state): pass
    def receive_game_update_message(self, action, round_state): pass
    def receive_round_result_message(self, winners, hand_info, round_state): pass
