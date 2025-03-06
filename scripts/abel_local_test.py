# scripts/abel_local_test.py
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
    from ai.utils.hand_history_logger import HandHistoryLogger
    from ai.utils.training_metrics import TrainingMetricsTracker
    from ai.utils.hand_strength import HandStrengthCalculator
    from ai.utils.decision_evaluator import DecisionEvaluator
    analysis_tools_available = True
except ImportError:
    print("Note: Advanced analysis tools not available. Running with basic logging only.")
    analysis_tools_available = False

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
        # Always call
        call_action_info = valid_actions[1]
        return call_action_info['action'], call_action_info['amount']
        
    def receive_game_start_message(self, game_info): pass
    def receive_round_start_message(self, round_count, hole_card, seats): pass
    def receive_street_start_message(self, street, round_state): pass
    def receive_game_update_message(self, action, round_state): pass
    def receive_round_result_message(self, winners, hand_info, round_state): pass

def run_local_test(num_games=20, use_self_play=True):
    """
    Run a small local test to verify that Abel works correctly
    
    Args:
        num_games (int): Number of games to play
        use_self_play (bool): Whether to use self-play or play against a simple opponent
    
    Returns:
        tuple: (win_rate, avg_profit)
    """
    print(f"Starting local test with {num_games} games...")
    logging.info(f"Starting local test with {num_games} games")
    
    # Initialize analysis tools if available
    if analysis_tools_available:
        print("Using advanced analysis tools...")
        hand_logger = HandHistoryLogger(log_dir=f"{log_dir}/hand_history")
        metrics_tracker = TrainingMetricsTracker(log_dir=log_dir, experiment_name="local_test")
    
    # Initialize Abel
    state_size = 6
    action_size = 3
    abel = RLBasedPlayer(state_size, action_size)
    
    # Set up model paths
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    main_model_path = f"{model_dir}/abel_main.h5"
    
    # Save initial model if it doesn't exist
    if not os.path.exists(main_model_path):
        try:
            abel.save_model()
            print(f"Initial model saved to {main_model_path}")
        except Exception as e:
            print(f"Warning: Could not save initial model: {e}")
    
    # Check if Abel has methods needed for training
    has_adjust_epsilon = hasattr(abel, 'adjust_epsilon')
    has_save_model = hasattr(abel, 'save_model')
    has_bluff_tracking = hasattr(abel, 'get_bluff_success_rate')
    
    if not has_adjust_epsilon:
        print("Warning: Abel does not have 'adjust_epsilon' method, epsilon decay won't work")
    if not has_save_model:
        print("Warning: Abel does not have 'save_model' method, model saving won't work")
    
    # Statistics
    games_played = 0
    abel_wins = 0
    total_profit = 0
    
    if use_self_play:
        # Self-play
        abel_2 = RLBasedPlayer(state_size, action_size)
        print("Running self-play test...")
        
        for game in range(num_games):
            # Alternate positions for fair testing
            if game % 2 == 0:
                config = setup_config(max_round=5, initial_stack=1000, small_blind_amount=10)
                config.register_player(name="Abel", algorithm=abel)
                config.register_player(name="Abel_2", algorithm=abel_2)
                first_player = "Abel"
            else:
                config = setup_config(max_round=5, initial_stack=1000, small_blind_amount=10)
                config.register_player(name="Abel_2", algorithm=abel_2)
                config.register_player(name="Abel", algorithm=abel)
                first_player = "Abel_2"
            
            # Play game
            game_result = start_poker(config, verbose=1)  # Verbose to see what's happening
            
            # Record game outcome
            abel_player = next((p for p in game_result["players"] if p["name"] == "Abel"), None)
            other_player = next((p for p in game_result["players"] if p["name"] == "Abel_2"), None)
            
            if abel_player and other_player:
                games_played += 1
                
                # Track stats
                if abel_player["stack"] > other_player["stack"]:
                    abel_wins += 1
                profit = abel_player["stack"] - 1000  # Initial stack was 1000
                total_profit += profit
                
                # Log
                print(f"Game {game+1}: {'Won' if abel_player['stack'] > other_player['stack'] else 'Lost'}, "
                      f"Profit: {profit}, Stack: {abel_player['stack']}")
                
                # Record in analysis tools if available
                if analysis_tools_available:
                    try:
                        # Log hand result
                        hand_logger.log_hand_result(
                            winners=["Abel" if abel_player["stack"] > other_player["stack"] else "Abel_2"],
                            pot_distribution={},
                            player_final_states=[
                                {"name": "Abel", "stack": abel_player["stack"]},
                                {"name": "Abel_2", "stack": other_player["stack"]}
                            ]
                        )
                        
                        # Log metrics
                        metrics = {
                            "win_rate": (abel_wins / games_played) * 100,
                            "avg_reward": total_profit / games_played,
                            "exploration_rate": abel.epsilon if has_adjust_epsilon else 0.0,
                            "opponent": "self-play"
                        }
                        
                        # Add bluff success rate if available
                        if has_bluff_tracking:
                            try:
                                metrics["bluff_success_rate"] = abel.get_bluff_success_rate()
                            except:
                                pass
                                
                        metrics_tracker.log_iteration(game, metrics)
                    except Exception as e:
                        print(f"Warning: Error logging analysis data: {e}")
            
            # Decay epsilon if method exists
            if has_adjust_epsilon:
                abel.adjust_epsilon()
                abel_2.adjust_epsilon()
    else:
        # Play against simple opponent
        print("Running test against simple opponent...")
        
        for game in range(num_games):
            # Alternate positions
            if game % 2 == 0:
                config = setup_config(max_round=5, initial_stack=1000, small_blind_amount=10)
                config.register_player(name="Abel", algorithm=abel)
                config.register_player(name="Simple", algorithm=SimplePlayer())
                abel_position = "SB"
            else:
                config = setup_config(max_round=5, initial_stack=1000, small_blind_amount=10)
                config.register_player(name="Simple", algorithm=SimplePlayer())
                config.register_player(name="Abel", algorithm=abel)
                abel_position = "BB"
            
            # Play game
            game_result = start_poker(config, verbose=1)  # Verbose to see what's happening
            
            # Record game outcome
            abel_player = next((p for p in game_result["players"] if p["name"] == "Abel"), None)
            simple_player = next((p for p in game_result["players"] if p["name"] == "Simple"), None)
            
            if abel_player and simple_player:
                games_played += 1
                
                # Track stats
                if abel_player["stack"] > simple_player["stack"]:
                    abel_wins += 1
                profit = abel_player["stack"] - 1000  # Initial stack was 1000
                total_profit += profit
                
                # Log
                print(f"Game {game+1}: {'Won' if abel_player['stack'] > simple_player['stack'] else 'Lost'}, "
                      f"Profit: {profit}, Stack: {abel_player['stack']}")
                
                # Record in analysis tools if available
                if analysis_tools_available:
                    try:
                        # Log hand result
                        hand_logger.log_hand_result(
                            winners=["Abel" if abel_player["stack"] > simple_player["stack"] else "Simple"],
                            pot_distribution={},
                            player_final_states=[
                                {"name": "Abel", "stack": abel_player["stack"]},
                                {"name": "Simple", "stack": simple_player["stack"]}
                            ]
                        )
                        
                        # Log metrics
                        metrics = {
                            "win_rate": (abel_wins / games_played) * 100,
                            "avg_reward": total_profit / games_played,
                            "exploration_rate": abel.epsilon if has_adjust_epsilon else 0.0,
                            "opponent": "simple"
                        }
                        
                        # Add bluff success rate if available
                        if has_bluff_tracking:
                            try:
                                metrics["bluff_success_rate"] = abel.get_bluff_success_rate()
                            except:
                                pass
                                
                        metrics_tracker.log_iteration(game, metrics)
                    except Exception as e:
                        print(f"Warning: Error logging analysis data: {e}")
            
            # Decay epsilon if method exists
            if has_adjust_epsilon:
                abel.adjust_epsilon()
    
    # Calculate results
    win_rate = (abel_wins / games_played) * 100 if games_played > 0 else 0
    avg_profit = total_profit / games_played if games_played > 0 else 0
    
    # Generate plots if analysis tools available
    if analysis_tools_available:
        try:
            metrics_tracker.plot_win_rate(save_path=f"{log_dir}/win_rate.png")
            metrics_tracker.plot_learning_metrics(save_path=f"{log_dir}/learning_metrics.png")
            
            if has_bluff_tracking:
                metrics_tracker.plot_playing_style_metrics(save_path=f"{log_dir}/playing_style.png")
                
            metrics_tracker.generate_training_summary(save_path=f"{log_dir}/test_summary.json")
        except Exception as e:
            print(f"Warning: Error generating analysis plots: {e}")
    
    # Print summary
    print("\n=== TEST SUMMARY ===")
    print(f"Games Played: {games_played}")
    print(f"Abel Wins: {abel_wins} ({win_rate:.2f}%)")
    print(f"Average Profit: {avg_profit:.2f} chips")
    if has_adjust_epsilon:
        print(f"Final Epsilon: {abel.epsilon:.4f}")
    print(f"Logs saved to: {log_dir}")
    
    # Try to save the model
    if has_save_model:
        try:
            abel.save_model()
            print(f"Model saved to {main_model_path}")
        except Exception as e:
            print(f"Warning: Could not save model: {e}")
    
    return win_rate, avg_profit

# Command-line arguments if running directly
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Local test for Abel poker AI')
    parser.add_argument('--games', type=int, default=20, help='Number of games to play')
    parser.add_argument('--opponent', choices=['self', 'simple'], default='self', 
                        help='Type of opponent (self-play or simple player)')
    
    args = parser.parse_args()
    
    use_self_play = (args.opponent == 'self')
    win_rate, avg_profit = run_local_test(num_games=args.games, use_self_play=use_self_play)
    
    print(f"\nTest completed successfully with win rate {win_rate:.2f}% and average profit {avg_profit:.2f} chips.")