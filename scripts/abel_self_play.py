# scripts/self_play.py
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pypokerengine.api.game import setup_config, start_poker
from pypokerengine.players import BasePokerPlayer

# Imports
from ai.Abel.agents.RLBasedPlayer import RLBasedPlayer
from ai.utils.hand_history_logger import HandHistoryLogger
from ai.utils.training_metrics import TrainingMetricsTracker
from ai.utils.hand_strength import HandStrengthCalculator
from ai.utils.decision_evaluator import DecisionEvaluator

# Configure Timestamp and Directories
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_log_dir = "logs/training"
log_dir = f"{base_log_dir}/{timestamp}"
models_dir = "models/checkpoints"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# Configure Logging
logging.basicConfig(
    filename=f"{log_dir}/abel_training.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

# Create a simple baseline player for evaluation
class BaselinePlayer(BasePokerPlayer):
    def declare_action(self, valid_actions, hole_card, round_state):
        action = valid_actions[1]['action']  # Always call
        return action, valid_actions[1]['amount']
        
    def receive_game_start_message(self, game_info): pass
    def receive_round_start_message(self, round_count, hole_card, seats): pass
    def receive_street_start_message(self, street, round_state): pass
    def receive_game_update_message(self, action, round_state): pass
    def receive_round_result_message(self, winners, hand_info, round_state): pass

def evaluate_performance(agent, num_eval_games=100):
    """Evaluate agent performance against the baseline player."""
    win_count = 0
    total_profit = 0
    
    for _ in range(num_eval_games):
        # Alternate positions for fairness
        if _ % 2 == 0:
            config = setup_config(max_round=5, initial_stack=1000, small_blind_amount=10)
            config.register_player(name="Abel", algorithm=agent)
            config.register_player(name="Baseline", algorithm=BaselinePlayer())
        else:
            config = setup_config(max_round=5, initial_stack=1000, small_blind_amount=10)
            config.register_player(name="Baseline", algorithm=BaselinePlayer())
            config.register_player(name="Abel", algorithm=agent)
        
        game_result = start_poker(config, verbose=0)
        
        # Find Abel's result
        abel_player = next((p for p in game_result["players"] if p["name"] == "Abel"), None)
        baseline_player = next((p for p in game_result["players"] if p["name"] == "Baseline"), None)
        
        if abel_player and baseline_player:
            if abel_player["stack"] > baseline_player["stack"]:
                win_count += 1
            total_profit += abel_player["stack"] - 1000  # Initial stack was 1000
    
    win_rate = (win_count / num_eval_games) * 100
    avg_profit = total_profit / num_eval_games
    
    return win_rate, avg_profit

def train_abel(num_games=10000, save_interval=100, eval_interval=500):
    """
    Train Abel using self-play with improved tracking and analysis.
    
    Args:
        num_games (int): Total number of games to play
        save_interval (int): Interval to save model checkpoints
        eval_interval (int): Interval to evaluate and log performance
    """
    print(f"Starting Abel training for {num_games} games...")
    logging.info(f"Starting Abel training for {num_games} games")
    
    # Initialize our tracking/analysis tools
    hand_logger = HandHistoryLogger(log_dir=f"{log_dir}/hand_history")
    metrics_tracker = TrainingMetricsTracker(log_dir=log_dir, experiment_name="abel_self_play")
    hand_strength_calc = HandStrengthCalculator()
    decision_eval = DecisionEvaluator(log_dir=f"{log_dir}/decisions")
    
    # Initialize Abel agents
    state_size = 6
    action_size = 3
    abel_1 = RLBasedPlayer(state_size, action_size, model_path="models/abel_main.h5")
    abel_2 = RLBasedPlayer(state_size, action_size, model_path="models/abel_opponent.h5")
    
    # Check if Abel has bluff tracking capabilities
    has_bluff_tracking = hasattr(abel_1, 'get_bluff_success_rate')
    
    # Training metrics
    win_counts = 0
    total_profits = 0
    
    # Training loop
    for game_number in range(num_games):
        # Alternate positions for fair training
        if game_number % 2 == 0:
            config = setup_config(max_round=5, initial_stack=1000, small_blind_amount=10)
            config.register_player(name="Abel_1", algorithm=abel_1)
            config.register_player(name="Abel_2", algorithm=abel_2)
            first_player = "Abel_1"
        else:
            config = setup_config(max_round=5, initial_stack=1000, small_blind_amount=10)
            config.register_player(name="Abel_2", algorithm=abel_2)
            config.register_player(name="Abel_1", algorithm=abel_1)
            first_player = "Abel_2"
        
        # Play the game
        game_result = start_poker(config, verbose=0)
        
        # Record game outcome
        abel_1_player = next((p for p in game_result["players"] if p["name"] == "Abel_1"), None)
        abel_2_player = next((p for p in game_result["players"] if p["name"] == "Abel_2"), None)
        
        if abel_1_player and abel_2_player:
            # Track wins and profit for primary Abel
            if abel_1_player["stack"] > abel_2_player["stack"]:
                win_counts += 1
            profit = abel_1_player["stack"] - 1000  # Initial stack was 1000
            total_profits += profit
            
            # Record hand history (simplified for self-play)
            if game_number % 10 == 0:  # Only record every 10th game to avoid excessive logging
                try:
                    # Create simplified hand data structure
                    hand_data = {
                        "hand_id": game_number,
                        "player_info": [
                            {"name": "Abel_1", "position": "SB" if first_player == "Abel_1" else "BB", "stack": 1000},
                            {"name": "Abel_2", "position": "BB" if first_player == "Abel_1" else "SB", "stack": 1000}
                        ],
                        "streets": {
                            "preflop": {"actions": [], "community_cards": []},
                            "flop": {"actions": [], "community_cards": []},
                            "turn": {"actions": [], "community_cards": []},
                            "river": {"actions": [], "community_cards": []}
                        },
                        "results": {
                            "winners": ["Abel_1" if abel_1_player["stack"] > abel_2_player["stack"] else "Abel_2"],
                            "player_final_states": [
                                {"name": "Abel_1", "stack": abel_1_player["stack"]},
                                {"name": "Abel_2", "stack": abel_2_player["stack"]}
                            ]
                        }
                    }
                    hand_logger.log_hand_result(
                        winners=["Abel_1" if abel_1_player["stack"] > abel_2_player["stack"] else "Abel_2"],
                        pot_distribution={},
                        player_final_states=[
                            {"name": "Abel_1", "stack": abel_1_player["stack"]},
                            {"name": "Abel_2", "stack": abel_2_player["stack"]}
                        ]
                    )
                except Exception as e:
                    logging.error(f"Error logging hand history: {e}")
        
        # Run evaluation at intervals
        if game_number % eval_interval == 0 or game_number == num_games - 1:
            # Evaluate against baseline
            win_rate, avg_profit = evaluate_performance(abel_1, num_eval_games=100)
            
            # Get bluff success rate if available
            bluff_rate = 0
            if has_bluff_tracking:
                try:
                    bluff_rate = abel_1.get_bluff_success_rate()
                except:
                    bluff_rate = 0
            
            # Current training stats
            current_win_rate = (win_counts / max(1, game_number + 1)) * 100
            current_avg_profit = total_profits / max(1, game_number + 1)
            
            # Log metrics
            metrics = {
                "win_rate": win_rate,  # Against baseline
                "avg_reward": avg_profit,  # Against baseline
                "self_play_win_rate": current_win_rate,  # Self-play
                "self_play_profit": current_avg_profit,  # Self-play
                "exploration_rate": abel_1.epsilon,
                "bluff_success_rate": bluff_rate,
                "opponent": "baseline"  # For this evaluation
            }
            metrics_tracker.log_iteration(game_number, metrics)
            
            # Log progress
            logging.info(f"Game {game_number}: Win Rate vs Baseline={win_rate:.2f}%, "
                        f"Avg Profit={avg_profit:.2f}, Epsilon={abel_1.epsilon:.3f}")
            
            print(f"Game {game_number}/{num_games}: Win Rate vs Baseline={win_rate:.2f}%, "
                 f"Self-play Win Rate={current_win_rate:.2f}%, Epsilon={abel_1.epsilon:.3f}")
            
            # Generate plots every few evaluations
            if game_number % (eval_interval * 5) == 0 or game_number == num_games - 1:
                try:
                    metrics_tracker.plot_win_rate()
                    metrics_tracker.plot_learning_metrics()
                    if has_bluff_tracking:
                        metrics_tracker.plot_playing_style_metrics()
                except Exception as e:
                    logging.error(f"Error generating plots: {e}")
        
        # Save model checkpoints
        if game_number % save_interval == 0 or game_number == num_games - 1:
            checkpoint_path = f"{models_dir}/abel_self_play_{game_number}.h5"
            try:
                abel_1.save_model()
                # Also save a numbered checkpoint
                abel_1.model.save_model(checkpoint_path)
                logging.info(f"Checkpoint saved: {checkpoint_path}")
            except Exception as e:
                logging.error(f"Error saving model: {e}")
        
        # Decay exploration rates
        abel_1.adjust_epsilon()
        abel_2.adjust_epsilon()
    
    # Save final model
    abel_1.save_model()
    
    # Generate final evaluation report
    metrics_tracker.generate_training_summary()
    
    # Print training summary
    print(f"\n=== Training Summary ===")
    print(f"Total Games Played: {num_games}")
    print(f"Final Win Rate vs Baseline: {win_rate:.2f}%")
    print(f"Final Self-play Win Rate: {current_win_rate:.2f}%")
    print(f"Final Exploration Rate: {abel_1.epsilon:.4f}")
    print(f"Logs and checkpoints saved to {log_dir} and {models_dir}")
    
    return abel_1

# Run training if this script is executed directly
if __name__ == "__main__":
    trained_abel = train_abel(num_games=10000)