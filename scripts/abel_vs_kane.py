# scripts/abel_vs_kane.py
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pypokerengine.api.game import setup_config, start_poker

# Imports
from ai.Abel.agents.RLBasedPlayer import RLBasedPlayer
from ai.Kane.Kane import RuleBasedPlayer
from ai.utils.hand_history_logger import HandHistoryLogger
from ai.utils.training_metrics import TrainingMetricsTracker
from ai.utils.opponent_analyzer import OpponentAnalyzer
from ai.utils.decision_evaluator import DecisionEvaluator

# Configure Timestamp and Directories
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_log_dir = "logs/abel_vs_kane"
log_dir = f"{base_log_dir}/{timestamp}"
models_dir = "models/checkpoints"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# Configure Logging
logging.basicConfig(
    filename=f"{log_dir}/abel_vs_kane.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

def train_abel_vs_kane(num_games=5000, save_interval=100, eval_interval=250, 
                     preliminary_analysis_games=200, analyze_every=10):
    """
    Train Abel against Kane with detailed tracking and analysis.
    
    Args:
        num_games (int): Total number of games to play
        save_interval (int): Interval to save model checkpoints
        eval_interval (int): Interval to evaluate and log performance
        preliminary_analysis_games (int): Games to play for initial Kane analysis
        analyze_every (int): Frequency of games to record for analysis
    """
    print(f"Starting Abel vs Kane training for {num_games} games...")
    logging.info(f"Starting Abel vs Kane training for {num_games} games")
    
    # Initialize our tracking/analysis tools
    hand_logger = HandHistoryLogger(log_dir=f"{log_dir}/hand_history")
    metrics_tracker = TrainingMetricsTracker(log_dir=log_dir, experiment_name="abel_vs_kane")
    opponent_analyzer = OpponentAnalyzer(log_dir=f"{log_dir}/opponent_analysis")
    decision_eval = DecisionEvaluator(log_dir=f"{log_dir}/decisions")
    
    # Initialize Abel and Kane
    state_size = 6
    action_size = 3
    abel = RLBasedPlayer(state_size, action_size, model_path="models/abel_main.h5")
    kane = RuleBasedPlayer()  # Kane is rule-based with strategy switching
    
    # Check if Abel has bluff tracking capabilities
    has_bluff_tracking = hasattr(abel, 'get_bluff_success_rate')
    
    print("Running preliminary analysis of Kane...")
    logging.info("Running preliminary analysis of Kane")
    
    # Run preliminary games to analyze Kane's behavior
    for game_number in range(preliminary_analysis_games):
        # Alternate positions
        if game_number % 2 == 0:
            config = setup_config(max_round=5, initial_stack=1000, small_blind_amount=10)
            config.register_player(name="Abel", algorithm=abel)
            config.register_player(name="Kane", algorithm=kane)
            abel_position = "SB"
        else:
            config = setup_config(max_round=5, initial_stack=1000, small_blind_amount=10)
            config.register_player(name="Kane", algorithm=kane)
            config.register_player(name="Abel", algorithm=abel)
            abel_position = "BB"
        
        # Play with more verbosity for analysis
        game_result = start_poker(config, verbose=1)
        
        # Only record some games for analysis (to avoid excessive logging)
        if game_number % analyze_every == 0:
            try:
                # Create simplified hand data
                hand_data = {
                    "hand_id": game_number,
                    "player_info": [
                        {"name": "Abel", "position": abel_position, "stack": 1000},
                        {"name": "Kane", "position": "BB" if abel_position == "SB" else "SB", "stack": 1000}
                    ],
                    "streets": {
                        "preflop": {"actions": [], "community_cards": []},
                        "flop": {"actions": [], "community_cards": []},
                        "turn": {"actions": [], "community_cards": []},
                        "river": {"actions": [], "community_cards": []}
                    },
                    "results": {
                        "winners": [],  # Would need actual winner data
                        "player_final_states": [
                            {"name": p["name"], "stack": p["stack"]} for p in game_result["players"]
                        ]
                    }
                }
                
                # Record hand for opponent analysis
                opponent_analyzer.record_hand(hand_data)
            except Exception as e:
                logging.error(f"Error recording hand data: {e}")
    
    # Generate Kane analysis report
    kane_profile = opponent_analyzer.get_opponent_profile("Kane")
    counter_strategy = opponent_analyzer.suggest_counter_strategy("Kane")
    
    # Log Kane's profile
    logging.info(f"Kane Profile - Strategy: {kane_profile.get('detected_strategy', 'Unknown')}")
    logging.info(f"VPIP: {kane_profile.get('vpip', 0):.1f}%, PFR: {kane_profile.get('pfr', 0):.1f}%, "
                f"Aggression Factor: {kane_profile.get('aggression_factor', 0):.2f}")
    
    # Log counter strategy
    logging.info(f"Suggested Counter Strategy: {counter_strategy.get('strategy', 'Unknown')}")
    for suggestion in counter_strategy.get("suggestions", []):
        logging.info(f"- {suggestion}")
    
    print(f"\nKane Analysis Complete:")
    print(f"Detected Strategy: {kane_profile.get('detected_strategy', 'Unknown')}")
    print(f"VPIP: {kane_profile.get('vpip', 0):.1f}%, PFR: {kane_profile.get('pfr', 0):.1f}%, " 
          f"Aggression: {kane_profile.get('aggression_factor', 0):.2f}")
    print("\nRecommended Counter-Strategy:")
    for suggestion in counter_strategy.get("suggestions", []):
        print(f"- {suggestion}")
    print("\nBeginning main training phase...\n")
    
    # Generate visualizations of Kane's play style
    try:
        opponent_analyzer.plot_opponent_tendencies("Kane")
        opponent_analyzer.generate_opponent_report("Kane")
    except Exception as e:
        logging.error(f"Error generating Kane analysis: {e}")
    
    # Metrics for tracking
    total_games = 0
    abel_wins = 0
    kane_wins = 0
    
    # Create a new Abel instance - keep opponent analyzer's Abel unchanged
    abel = RLBasedPlayer(state_size, action_size, model_path="models/abel_main.h5")
    
    # Main training loop
    for game_number in range(num_games):
        # Alternate positions
        if game_number % 2 == 0:
            config = setup_config(max_round=5, initial_stack=1000, small_blind_amount=10)
            config.register_player(name="Abel", algorithm=abel)
            config.register_player(name="Kane", algorithm=kane)
            abel_position = "SB"
        else:
            config = setup_config(max_round=5, initial_stack=1000, small_blind_amount=10)
            config.register_player(name="Kane", algorithm=kane)
            config.register_player(name="Abel", algorithm=abel)
            abel_position = "BB"
        
        # Play the game
        game_result = start_poker(config, verbose=0)
        
        # Record game outcome
        abel_player = next((p for p in game_result["players"] if p["name"] == "Abel"), None)
        kane_player = next((p for p in game_result["players"] if p["name"] == "Kane"), None)
        
        if abel_player and kane_player:
            total_games += 1
            
            # Determine winner
            if abel_player["stack"] > kane_player["stack"]:
                abel_wins += 1
                winner = "Abel"
            else:
                kane_wins += 1
                winner = "Kane"
            
            # Record hand history (only every few games)
            if game_number % analyze_every == 0:
                try:
                    # Simplified hand tracking
                    hand_logger.log_hand_result(
                        winners=[winner],
                        pot_distribution={},  # We don't have detailed pot information
                        player_final_states=[
                            {"name": "Abel", "stack": abel_player["stack"]},
                            {"name": "Kane", "stack": kane_player["stack"]}
                        ]
                    )
                    
                    # Update opponent analysis
                    hand_data = {
                        "hand_id": preliminary_analysis_games + game_number,
                        "player_info": [
                            {"name": "Abel", "position": abel_position, "stack": 1000},
                            {"name": "Kane", "position": "BB" if abel_position == "SB" else "SB", "stack": 1000}
                        ],
                        "streets": {
                            "preflop": {"actions": [], "community_cards": []},
                            "flop": {"actions": [], "community_cards": []},
                            "turn": {"actions": [], "community_cards": []},
                            "river": {"actions": [], "community_cards": []}
                        },
                        "results": {
                            "winners": [winner],
                            "player_final_states": [
                                {"name": "Abel", "stack": abel_player["stack"]},
                                {"name": "Kane", "stack": kane_player["stack"]}
                            ]
                        }
                    }
                    opponent_analyzer.record_hand(hand_data)
                except Exception as e:
                    logging.error(f"Error updating hand analysis: {e}")
        
        # Run evaluation at intervals
        if game_number % eval_interval == 0 or game_number == num_games - 1:
            # Calculate current win rate against Kane
            current_win_rate = (abel_wins / total_games) * 100 if total_games > 0 else 0
            
            # Get bluff success rate if available
            bluff_rate = 0
            if has_bluff_tracking:
                try:
                    bluff_rate = abel.get_bluff_success_rate()
                except:
                    bluff_rate = 0
            
            # Update Kane's profile analysis
            kane_profile = opponent_analyzer.get_opponent_profile("Kane")
            kane_strategy = kane_profile.get('detected_strategy', 'Unknown')
            
            # Log metrics
            metrics = {
                "win_rate": current_win_rate,
                "exploration_rate": abel.epsilon,
                "bluff_success_rate": bluff_rate,
                "opponent": "Kane",
                "kane_strategy": kane_strategy
            }
            metrics_tracker.log_iteration(game_number, metrics)
            
            # Log progress
            logging.info(f"Game {game_number}: Win Rate={current_win_rate:.2f}%, "
                        f"Epsilon={abel.epsilon:.3f}, Kane Strategy={kane_strategy}")
            
            print(f"Game {game_number}/{num_games}: Win Rate vs Kane={current_win_rate:.2f}%, "
                 f"Abel Wins={abel_wins}, Kane Wins={kane_wins}, Epsilon={abel.epsilon:.3f}")
            
            # Generate plots every few evaluations
            if game_number % (eval_interval * 4) == 0 or game_number == num_games - 1:
                try:
                    metrics_tracker.plot_win_rate()
                    metrics_tracker.plot_learning_metrics()
                    metrics_tracker.plot_opponent_comparison()
                    if has_bluff_tracking:
                        metrics_tracker.plot_playing_style_metrics()
                    
                    # Re-analyze Kane if enough games have been played
                    if game_number % (eval_interval * 8) == 0 or game_number == num_games - 1:
                        opponent_analyzer.plot_opponent_tendencies("Kane")
                        counter_strategy = opponent_analyzer.suggest_counter_strategy("Kane")
                        logging.info(f"Updated Counter Strategy: {counter_strategy.get('strategy', 'Unknown')}")
                except Exception as e:
                    logging.error(f"Error generating plots: {e}")
        
        # Save model checkpoints
        if game_number % save_interval == 0 or game_number == num_games - 1:
            checkpoint_path = f"{models_dir}/abel_vs_kane_{game_number}.h5"
            try:
                abel.save_model()
                # Also save a numbered checkpoint
                abel.model.save_model(checkpoint_path)
                logging.info(f"Checkpoint saved: {checkpoint_path}")
            except Exception as e:
                logging.error(f"Error saving model: {e}")
        
        # Decay exploration rate
        abel.adjust_epsilon()
    
    # Generate final analysis
    try:
        # Final Kane analysis
        final_kane_profile = opponent_analyzer.get_opponent_profile("Kane")
        final_counter_strategy = opponent_analyzer.suggest_counter_strategy("Kane")
        opponent_analyzer.generate_opponent_report("Kane", f"{log_dir}/final_kane_analysis.txt")
        
        # Generate summary metrics
        metrics_tracker.generate_training_summary(f"{log_dir}/final_training_summary.json")
    except Exception as e:
        logging.error(f"Error generating final analysis: {e}")
    
    # Print final summary
    print(f"\n=== Final Training Summary ===")
    print(f"Total Games Played: {total_games}")
    print(f"Abel Wins: {abel_wins} ({(abel_wins / total_games) * 100:.2f}%)")
    print(f"Kane Wins: {kane_wins} ({(kane_wins / total_games) * 100:.2f}%)")
    print(f"Final Exploration Rate: {abel.epsilon:.4f}")
    print(f"Logs and checkpoints saved to {log_dir} and {models_dir}")
    
    return abel

# Run training if this script is executed directly
if __name__ == "__main__":
    print("\nStarting Abel vs Kane training sequence...")
    trained_abel = train_abel_vs_kane(num_games=5000)
    print("Training complete!")