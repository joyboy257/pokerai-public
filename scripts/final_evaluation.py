# scripts/final_evaluation.py
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pypokerengine.api.game import setup_config, start_poker

# Imports
from ai.Abel.agents.RLBasedPlayer import RLBasedPlayer
from ai.Kane.Kane import RuleBasedPlayer
from ai.utils.hand_history_logger import HandHistoryLogger
from ai.utils.opponent_analyzer import OpponentAnalyzer
from ai.utils.decision_evaluator import DecisionEvaluator

# Configure Timestamp and Directories
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_log_dir = "logs/final_evaluation"
log_dir = f"{base_log_dir}/{timestamp}"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(f"{log_dir}/plots", exist_ok=True)

# Configure Logging
logging.basicConfig(
    filename=f"{log_dir}/evaluation.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

def run_final_evaluation(num_games=1000, abel_model_path="models/abel_vs_kane_final.h5"):
    """
    Run a comprehensive evaluation of Abel vs Kane with detailed analysis.
    
    Args:
        num_games (int): Number of games to play for evaluation
        abel_model_path (str): Path to Abel's trained model
    
    Returns:
        pd.DataFrame: DataFrame with game results
    """
    print(f"Starting final evaluation: Abel vs Kane ({num_games} games)...")
    logging.info(f"Starting final evaluation: Abel vs Kane ({num_games} games)")
    
    # Initialize our analysis tools
    hand_logger = HandHistoryLogger(log_dir=f"{log_dir}/hand_history")
    decision_eval = DecisionEvaluator(log_dir=f"{log_dir}/decisions")
    opponent_analyzer = OpponentAnalyzer(log_dir=f"{log_dir}/opponent_analysis")
    
    # Initialize Abel and Kane
    state_size = 6
    action_size = 3
    abel = RLBasedPlayer(state_size, action_size, model_path=abel_model_path)
    kane = RuleBasedPlayer()  # Kane is rule-based with strategy switching
    
    # Set Abel to evaluation mode (no exploration)
    abel.epsilon = 0
    
    # Statistics
    games_data = []
    
    # Progress tracking
    progress_interval = max(1, num_games // 20)  # Show progress 20 times
    last_progress = 0
    
    for game_number in range(num_games):
        # Alternate positions
        if game_number % 2 == 0:
            config = setup_config(max_round=5, initial_stack=1000, small_blind_amount=10)
            config.register_player(name="Abel", algorithm=abel)
            config.register_player(name="Kane", algorithm=kane)
            abel_position = "SB"
            kane_position = "BB"
        else: