# ai/Abel/utils/hand_history_logger.py
import os
import json
import logging
from datetime import datetime

class HandHistoryLogger:
    """
    Records detailed information about each poker hand for post-game analysis.
    
    This class captures the full history of a poker hand, including cards, actions,
    pot sizes, and outcomes. The data is stored in a structured format for
    later analysis and can be saved to disk as JSON.
    """
    
    def __init__(self, log_dir="logs/hand_history", enable_console_output=False):
        """
        Initialize the hand history logger.
        
        Args:
            log_dir (str): Directory where hand history logs will be saved
            enable_console_output (bool): Whether to also output logs to console
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Set up logging
        self.logger = logging.getLogger("hand_history")
        self.logger.setLevel(logging.INFO)
        
        # File handler
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"{log_dir}/hand_history_{timestamp}.log"
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler (optional)
        if enable_console_output:
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter('%(asctime)s - %(message)s')
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # Current hand data
        self.current_hand = None
        self.hand_counter = 0
        
        # History storage
        self.hand_history = []
    
    def start_new_hand(self, table_info, player_info):
        """
        Start recording a new hand.
        
        Args:
            table_info (dict): Information about the table (blinds, positions, etc.)
            player_info (dict): Information about the players (stack sizes, positions)
        """
        self.hand_counter += 1
        
        self.current_hand = {
            "hand_id": self.hand_counter,
            "timestamp": datetime.now().isoformat(),
            "table_info": table_info,
            "player_info": player_info,
            "streets": {
                "preflop": {"actions": [], "community_cards": []},
                "flop": {"actions": [], "community_cards": []},
                "turn": {"actions": [], "community_cards": []},
                "river": {"actions": [], "community_cards": []}
            },
            "results": None
        }
        
        self.logger.info(f"Hand #{self.hand_counter} started")
        self.logger.info(f"Table info: {json.dumps(table_info)}")
        self.logger.info(f"Player info: {json.dumps(player_info)}")
    
    def log_hole_cards(self, player_name, hole_cards):
        """
        Log the hole cards for a player.
        
        Args:
            player_name (str): Name of the player
            hole_cards (list): List of hole cards
        """
        if self.current_hand is None:
            self.logger.warning("Attempted to log hole cards without starting a hand")
            return
        
        if "hole_cards" not in self.current_hand:
            self.current_hand["hole_cards"] = {}
        
        self.current_hand["hole_cards"][player_name] = hole_cards
        self.logger.info(f"{player_name}'s hole cards: {hole_cards}")
    
    def log_action(self, street, player_name, action, amount, pot_size=None):
        """
        Log a player action during the hand.
        
        Args:
            street (str): Current street (preflop, flop, turn, river)
            player_name (str): Name of the player
            action (str): Action taken (fold, check, call, raise)
            amount (float): Amount of chips committed
            pot_size (float, optional): Current pot size after the action
        """
        if self.current_hand is None:
            self.logger.warning("Attempted to log action without starting a hand")
            return
        
        action_data = {
            "player": player_name,
            "action": action,
            "amount": amount
        }
        
        if pot_size is not None:
            action_data["pot_size"] = pot_size
        
        self.current_hand["streets"][street]["actions"].append(action_data)
        self.logger.info(f"{street.capitalize()}: {player_name} {action}s {amount}")
        
        if pot_size is not None:
            self.logger.info(f"Pot size: {pot_size}")
    
    def log_community_cards(self, street, cards):
        """
        Log community cards revealed on a street.
        
        Args:
            street (str): Current street (flop, turn, river)
            cards (list): List of community cards
        """
        if self.current_hand is None:
            self.logger.warning("Attempted to log community cards without starting a hand")
            return
        
        self.current_hand["streets"][street]["community_cards"] = cards
        self.logger.info(f"{street.capitalize()} cards: {cards}")
    
    def log_hand_result(self, winners, pot_distribution, player_final_states):
        """
        Log the result of the hand.
        
        Args:
            winners (list): List of winning players
            pot_distribution (dict): How the pot was distributed
            player_final_states (dict): Final states of all players
        """
        if self.current_hand is None:
            self.logger.warning("Attempted to log hand result without starting a hand")
            return
        
        result = {
            "winners": winners,
            "pot_distribution": pot_distribution,
            "player_final_states": player_final_states
        }
        
        self.current_hand["results"] = result
        
        winner_str = ", ".join(winners)
        self.logger.info(f"Hand #{self.hand_counter} complete")
        self.logger.info(f"Winners: {winner_str}")
        self.logger.info(f"Pot distribution: {json.dumps(pot_distribution)}")
        
        # Store hand in history and clear current hand
        self.hand_history.append(self.current_hand)
        self._save_hand_to_disk(self.current_hand)
        self.current_hand = None
    
    def _save_hand_to_disk(self, hand_data):
        """
        Save a completed hand to disk as JSON.
        
        Args:
            hand_data (dict): Complete hand data
        """
        hand_id = hand_data["hand_id"]
        filename = f"{self.log_dir}/hand_{hand_id}.json"
        
        with open(filename, 'w') as f:
            json.dump(hand_data, f, indent=2)
    
    def get_hand_history(self):
        """
        Get the full history of recorded hands.
        
        Returns:
            list: List of all recorded hands
        """
        return self.hand_history
    
    def get_hand(self, hand_id):
        """
        Get a specific hand by its ID.
        
        Args:
            hand_id (int): ID of the hand to retrieve
            
        Returns:
            dict: Hand data, or None if not found
        """
        for hand in self.hand_history:
            if hand["hand_id"] == hand_id:
                return hand
        
        # If not found in memory, try to load from disk
        filename = f"{self.log_dir}/hand_{hand_id}.json"
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                return json.load(f)
        
        return None
    
    def get_player_hands(self, player_name):
        """
        Get all hands involving a specific player.
        
        Args:
            player_name (str): Name of the player
            
        Returns:
            list: List of hands involving the player
        """
        player_hands = []
        
        for hand in self.hand_history:
            for player in hand["player_info"]:
                if player["name"] == player_name:
                    player_hands.append(hand)
                    break
        
        return player_hands
    
    def get_player_stats(self, player_name):
        """
        Calculate basic stats for a player.
        
        Args:
            player_name (str): Name of the player
            
        Returns:
            dict: Dictionary of player stats
        """
        player_hands = self.get_player_hands(player_name)
        
        if not player_hands:
            return {
                "total_hands": 0,
                "hands_won": 0,
                "win_rate": 0,
                "total_profit": 0,
                "avg_profit_per_hand": 0
            }
        
        hands_won = 0
        total_profit = 0
        
        for hand in player_hands:
            if hand["results"]:
                # Check if player won
                if player_name in hand["results"]["winners"]:
                    hands_won += 1
                
                # Calculate profit
                for player_state in hand["results"]["player_final_states"]:
                    if player_state["name"] == player_name:
                        initial_stack = None
                        for initial_player in hand["player_info"]:
                            if initial_player["name"] == player_name:
                                initial_stack = initial_player["stack"]
                                break
                        
                        if initial_stack is not None:
                            profit = player_state["stack"] - initial_stack
                            total_profit += profit
        
        total_hands = len(player_hands)
        win_rate = (hands_won / total_hands) * 100 if total_hands > 0 else 0
        avg_profit = total_profit / total_hands if total_hands > 0 else 0
        
        return {
            "total_hands": total_hands,
            "hands_won": hands_won,
            "win_rate": win_rate,
            "total_profit": total_profit,
            "avg_profit_per_hand": avg_profit
        }

# Example usage
if __name__ == "__main__":
    # Initialize logger
    hand_logger = HandHistoryLogger(enable_console_output=True)
    
    # Log a sample hand
    table_info = {
        "table_id": "Table1",
        "small_blind": 10,
        "big_blind": 20
    }
    
    player_info = [
        {"name": "Abel", "position": "SB", "stack": 1000},
        {"name": "Kane", "position": "BB", "stack": 1000}
    ]
    
    # Start new hand
    hand_logger.start_new_hand(table_info, player_info)
    
    # Log hole cards
    hand_logger.log_hole_cards("Abel", ["AS", "KS"])
    hand_logger.log_hole_cards("Kane", ["JH", "QH"])
    
    # Log preflop actions
    hand_logger.log_action("preflop", "Abel", "raise", 60, 80)
    hand_logger.log_action("preflop", "Kane", "call", 40, 120)
    
    # Log flop
    hand_logger.log_community_cards("flop", ["2S", "5C", "KH"])
    hand_logger.log_action("flop", "Kane", "check", 0, 120)
    hand_logger.log_action("flop", "Abel", "bet", 80, 200)
    hand_logger.log_action("flop", "Kane", "call", 80, 280)
    
    # Log turn
    hand_logger.log_community_cards("turn", ["2S", "5C", "KH", "AS"])
    hand_logger.log_action("turn", "Kane", "check", 0, 280)
    hand_logger.log_action("turn", "Abel", "bet", 200, 480)
    hand_logger.log_action("turn", "Kane", "fold", 0, 480)
    
    # Log result
    hand_logger.log_hand_result(
        winners=["Abel"],
        pot_distribution={"Abel": 480},
        player_final_states=[
            {"name": "Abel", "stack": 1240},
            {"name": "Kane", "stack": 760}
        ]
    )
    
    # Print stats
    print("Abel stats:", hand_logger.get_player_stats("Abel"))
    print("Kane stats:", hand_logger.get_player_stats("Kane"))