import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import logging

class OpponentAnalyzer:
    """
    Analyzes opponent tendencies and patterns from poker hand histories.
    
    This class tracks opponent behaviors and calculates metrics like 
    fold frequency, aggression factor, and other patterns to enable 
    adaptive play against rule-based opponents like Kane.
    """
    
    def __init__(self, log_dir="logs/opponent_analysis"):
        """
        Initialize the opponent analyzer.
        
        Args:
            log_dir (str): Directory where analysis logs will be saved
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Set up logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logging.basicConfig(
            filename=f"{log_dir}/opponent_analysis_{timestamp}.log",
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )
        
        # Dictionary to store opponent profiles
        self.opponents = {}
        
        # Store all actions for detailed analysis
        self.action_history = []
    
    def _initialize_opponent(self, opponent_name):
        """
        Initialize tracking for a new opponent.
        
        Args:
            opponent_name (str): Name of the opponent
        """
        if opponent_name not in self.opponents:
            self.opponents[opponent_name] = {
                # Basic stats
                "total_hands": 0,
                "hands_won": 0,
                "total_actions": 0,
                
                # Action counts
                "fold_count": 0,
                "check_count": 0,
                "call_count": 0,
                "raise_count": 0,
                "bet_count": 0,
                
                # Preflop tendencies
                "preflop_fold": 0,
                "preflop_call": 0,
                "preflop_raise": 0,
                "preflop_total": 0,
                
                # Postflop tendencies
                "postflop_fold": 0,
                "postflop_check": 0,
                "postflop_call": 0,
                "postflop_bet": 0,
                "postflop_raise": 0,
                "postflop_total": 0,
                
                # Positional play
                "sb_actions": [],
                "bb_actions": [],
                
                # Betting tendencies
                "bet_sizes": [],
                "bet_sizing_by_pot": [],
                
                # Showdown data
                "showdown_hands": [],
                
                # Advanced patterns
                "continuation_bet_count": 0,
                "continuation_bet_opportunities": 0,
                "check_raise_count": 0,
                "check_raise_opportunities": 0,
                "bluff_caught_count": 0,
                "bluff_attempt_count": 0,
                
                # Street-specific tendencies
                "flop_aggression": [],
                "turn_aggression": [],
                "river_aggression": [],
                
                # Strategy detection
                "recent_actions": [],
                "detected_strategy": None,
                "strategy_confidence": 0.0,
                "adaptation_history": []
            }
            
            logging.info(f"Initialized tracking for opponent: {opponent_name}")
    
    def record_hand(self, hand_data):
        """
        Record a complete hand history for analysis.
        
        Args:
            hand_data (dict): Hand history data
        """
        hand_id = hand_data["hand_id"]
        logging.info(f"Recording hand #{hand_id} for analysis")
        
        # Get player information
        for player_info in hand_data["player_info"]:
            player_name = player_info["name"]
            self._initialize_opponent(player_name)
            self.opponents[player_name]["total_hands"] += 1
        
        # Record winners
        if hand_data["results"] and "winners" in hand_data["results"]:
            for winner in hand_data["results"]["winners"]:
                if winner in self.opponents:
                    self.opponents[winner]["hands_won"] += 1
        
        # Record hole cards for showdown analysis
        if "hole_cards" in hand_data:
            for player_name, cards in hand_data["hole_cards"].items():
                if player_name in self.opponents and hand_data["results"]:
                    # Check if hand went to showdown
                    showdown = any(
                        len(street_data["actions"]) > 0 for street in ["river"] 
                        for street_data in [hand_data["streets"][street]]
                    )
                    
                    if showdown:
                        hand_strength = 0.0  # Calculate or estimate hand strength
                        self.opponents[player_name]["showdown_hands"].append({
                            "hole_cards": cards,
                            "community_cards": hand_data["streets"]["river"]["community_cards"],
                            "hand_strength": hand_strength,
                            "won": player_name in hand_data["results"]["winners"]
                        })
        
        # Process each street
        for street in ["preflop", "flop", "turn", "river"]:
            street_data = hand_data["streets"][street]
            
            if not street_data["actions"]:
                continue
            
            # Track continuation betting opportunities
            if street == "flop":
                preflop_aggressor = None
                for action in hand_data["streets"]["preflop"]["actions"]:
                    if action["action"] == "raise":
                        preflop_aggressor = action["player"]
                
                if preflop_aggressor:
                    self.opponents[preflop_aggressor]["continuation_bet_opportunities"] += 1
            
            # Process actions for each player
            last_action = None
            for action_index, action_data in enumerate(street_data["actions"]):
                player_name = action_data["player"]
                action = action_data["action"]
                amount = action_data["amount"]
                
                if player_name not in self.opponents:
                    self._initialize_opponent(player_name)
                
                # Record the action for later analysis
                action_record = {
                    "hand_id": hand_id,
                    "player": player_name,
                    "street": street,
                    "action": action,
                    "amount": amount,
                    "position": next((p["position"] for p in hand_data["player_info"] if p["name"] == player_name), None),
                    "community_cards": street_data["community_cards"],
                    "pot_size": action_data.get("pot_size", 0)
                }
                self.action_history.append(action_record)
                
                # Update basic action counts
                self.opponents[player_name]["total_actions"] += 1
                
                if action == "fold":
                    self.opponents[player_name]["fold_count"] += 1
                elif action == "check":
                    self.opponents[player_name]["check_count"] += 1
                elif action == "call":
                    self.opponents[player_name]["call_count"] += 1
                elif action == "raise":
                    self.opponents[player_name]["raise_count"] += 1
                    self.opponents[player_name]["bet_sizes"].append(amount)
                    
                    # Record bet sizing relative to pot
                    pot_size = action_data.get("pot_size", 0)
                    if pot_size > 0:
                        bet_ratio = amount / pot_size
                        self.opponents[player_name]["bet_sizing_by_pot"].append(bet_ratio)
                
                # Track street-specific actions
                if street == "preflop":
                    self.opponents[player_name]["preflop_total"] += 1
                    
                    if action == "fold":
                        self.opponents[player_name]["preflop_fold"] += 1
                    elif action == "call":
                        self.opponents[player_name]["preflop_call"] += 1
                    elif action == "raise":
                        self.opponents[player_name]["preflop_raise"] += 1
                else:
                    self.opponents[player_name]["postflop_total"] += 1
                    
                    if action == "fold":
                        self.opponents[player_name]["postflop_fold"] += 1
                    elif action == "check":
                        self.opponents[player_name]["postflop_check"] += 1
                    elif action == "call":
                        self.opponents[player_name]["postflop_call"] += 1
                    elif action == "raise":
                        self.opponents[player_name]["postflop_raise"] += 1
                
                # Track positional play
                position = next((p["position"] for p in hand_data["player_info"] if p["name"] == player_name), None)
                if position == "SB":
                    self.opponents[player_name]["sb_actions"].append(action)
                elif position == "BB":
                    self.opponents[player_name]["bb_actions"].append(action)
                
                # Track continuation betting
                if street == "flop" and action in ["bet", "raise"] and player_name == preflop_aggressor:
                    self.opponents[player_name]["continuation_bet_count"] += 1
                
                # Track check-raising
                if last_action and last_action["player"] == player_name and last_action["action"] == "check" \
                   and action == "raise":
                    self.opponents[player_name]["check_raise_count"] += 1
                
                # Track aggression by street
                if action in ["bet", "raise"]:
                    if street == "flop":
                        self.opponents[player_name]["flop_aggression"].append(1)
                    elif street == "turn":
                        self.opponents[player_name]["turn_aggression"].append(1)
                    elif street == "river":
                        self.opponents[player_name]["river_aggression"].append(1)
                elif action in ["check", "call"]:
                    if street == "flop":
                        self.opponents[player_name]["flop_aggression"].append(0)
                    elif street == "turn":
                        self.opponents[player_name]["turn_aggression"].append(0)
                    elif street == "river":
                        self.opponents[player_name]["river_aggression"].append(0)
                
                # Record for pattern detection
                self.opponents[player_name]["recent_actions"].append({
                    "street": street,
                    "action": action,
                    "amount": amount
                })
                
                # Keep recent actions list manageable
                if len(self.opponents[player_name]["recent_actions"]) > 20:
                    self.opponents[player_name]["recent_actions"].pop(0)
                
                # Save the action for next iteration
                last_action = action_data
        
        # Detect opponent strategy after processing the hand
        for player_name in self.opponents:
            if player_name != "Abel":  # Only analyze opponents
                self._detect_strategy(player_name)
    
    def record_action(self, player_name, street, action, amount, position=None, 
                     community_cards=None, pot_size=None, hand_id=None):
        """
        Record a single action for analysis.
        
        Args:
            player_name (str): Name of the player
            street (str): Current street (preflop, flop, turn, river)
            action (str): Action taken (fold, check, call, raise)
            amount (float): Amount of chips
            position (str, optional): Player position (SB, BB)
            community_cards (list, optional): Current community cards
            pot_size (float, optional): Current pot size
            hand_id (int, optional): Hand identifier
        """
        if player_name not in self.opponents:
            self._initialize_opponent(player_name)
        
        # Record the action
        action_record = {
            "hand_id": hand_id,
            "player": player_name,
            "street": street,
            "action": action,
            "amount": amount,
            "position": position,
            "community_cards": community_cards,
            "pot_size": pot_size
        }
        self.action_history.append(action_record)
        
        # Update action counts
        self.opponents[player_name]["total_actions"] += 1
        
        if action == "fold":
            self.opponents[player_name]["fold_count"] += 1
        elif action == "check":
            self.opponents[player_name]["check_count"] += 1
        elif action == "call":
            self.opponents[player_name]["call_count"] += 1
        elif action == "raise":
            self.opponents[player_name]["raise_count"] += 1
            self.opponents[player_name]["bet_sizes"].append(amount)
            
            if pot_size:
                bet_ratio = amount / pot_size
                self.opponents[player_name]["bet_sizing_by_pot"].append(bet_ratio)
        
        # Track street-specific actions
        if street == "preflop":
            self.opponents[player_name]["preflop_total"] += 1
            
            if action == "fold":
                self.opponents[player_name]["preflop_fold"] += 1
            elif action == "call":
                self.opponents[player_name]["preflop_call"] += 1
            elif action == "raise":
                self.opponents[player_name]["preflop_raise"] += 1
        else:
            self.opponents[player_name]["postflop_total"] += 1
            
            if action == "fold":
                self.opponents[player_name]["postflop_fold"] += 1
            elif action == "check":
                self.opponents[player_name]["postflop_check"] += 1
            elif action == "call":
                self.opponents[player_name]["postflop_call"] += 1
            elif action == "raise":
                self.opponents[player_name]["postflop_raise"] += 1
        
        # Track positional play
        if position == "SB":
            self.opponents[player_name]["sb_actions"].append(action)
        elif position == "BB":
            self.opponents[player_name]["bb_actions"].append(action)
        
        # Record for pattern detection
        self.opponents[player_name]["recent_actions"].append({
            "street": street,
            "action": action,
            "amount": amount
        })
        
        # Keep recent actions list manageable
        if len(self.opponents[player_name]["recent_actions"]) > 20:
            self.opponents[player_name]["recent_actions"].pop(0)
        
        # Update strategy detection
        self._detect_strategy(player_name)
    
    def get_opponent_profile(self, player_name):
        """
        Get a complete profile of an opponent's tendencies.
        
        Args:
            player_name (str): Name of the opponent
            
        Returns:
            dict: Opponent profile with calculated metrics
        """
        if player_name not in self.opponents:
            logging.warning(f"No data for opponent: {player_name}")
            return None
        
        profile = self.opponents[player_name].copy()
        
        # Calculate derived metrics
        
        # Win rate
        if profile["total_hands"] > 0:
            profile["win_rate"] = profile["hands_won"] / profile["total_hands"] * 100
        else:
            profile["win_rate"] = 0
        
        # VPIP (Voluntarily Put $ In Pot)
        if profile["preflop_total"] > 0:
            profile["vpip"] = (profile["preflop_call"] + profile["preflop_raise"]) / profile["preflop_total"] * 100
        else:
            profile["vpip"] = 0
        
        # PFR (Preflop Raise)
        if profile["preflop_total"] > 0:
            profile["pfr"] = profile["preflop_raise"] / profile["preflop_total"] * 100
        else:
            profile["pfr"] = 0
        
        # Aggression Factor: (Raises + Bets) / Calls
        if profile["call_count"] > 0:
            profile["aggression_factor"] = profile["raise_count"] / profile["call_count"]
        else:
            profile["aggression_factor"] = profile["raise_count"] if profile["raise_count"] > 0 else 0
        
        # Fold to Continuation Bet
        if profile["continuation_bet_opportunities"] > 0:
            profile["continuation_bet_frequency"] = profile["continuation_bet_count"] / profile["continuation_bet_opportunities"] * 100
        else:
            profile["continuation_bet_frequency"] = 0
        
        # Check-Raise Frequency
        if profile["check_raise_opportunities"] > 0:
            profile["check_raise_frequency"] = profile["check_raise_count"] / profile["check_raise_opportunities"] * 100
        else:
            profile["check_raise_frequency"] = 0
        
        # Positional Play
        if profile["sb_actions"]:
            profile["sb_fold_rate"] = profile["sb_actions"].count("fold") / len(profile["sb_actions"]) * 100
            profile["sb_aggression"] = (profile["sb_actions"].count("raise") + profile["sb_actions"].count("bet")) / max(1, len(profile["sb_actions"])) * 100
        else:
            profile["sb_fold_rate"] = 0
            profile["sb_aggression"] = 0
        
        if profile["bb_actions"]:
            profile["bb_fold_rate"] = profile["bb_actions"].count("fold") / len(profile["bb_actions"]) * 100
            profile["bb_aggression"] = (profile["bb_actions"].count("raise") + profile["bb_actions"].count("bet")) / max(1, len(profile["bb_actions"])) * 100
        else:
            profile["bb_fold_rate"] = 0
            profile["bb_aggression"] = 0
        
        # Average Bet Sizing
        if profile["bet_sizes"]:
            profile["avg_bet_size"] = sum(profile["bet_sizes"]) / len(profile["bet_sizes"])
        else:
            profile["avg_bet_size"] = 0
        
        if profile["bet_sizing_by_pot"]:
            profile["avg_bet_pot_ratio"] = sum(profile["bet_sizing_by_pot"]) / len(profile["bet_sizing_by_pot"])
        else:
            profile["avg_bet_pot_ratio"] = 0
        
        # Street-specific aggression
        if profile["flop_aggression"]:
            profile["flop_aggression_rate"] = sum(profile["flop_aggression"]) / len(profile["flop_aggression"]) * 100
        else:
            profile["flop_aggression_rate"] = 0
        
        if profile["turn_aggression"]:
            profile["turn_aggression_rate"] = sum(profile["turn_aggression"]) / len(profile["turn_aggression"]) * 100
        else:
            profile["turn_aggression_rate"] = 0
        
        if profile["river_aggression"]:
            profile["river_aggression_rate"] = sum(profile["river_aggression"]) / len(profile["river_aggression"]) * 100
        else:
            profile["river_aggression_rate"] = 0
        
        # Calculating fold percentages for different streets
        profile["preflop_fold_rate"] = (profile["preflop_fold"] / max(1, profile["preflop_total"])) * 100
        profile["postflop_fold_rate"] = (profile["postflop_fold"] / max(1, profile["postflop_total"])) * 100
        
        return profile
    
    def _detect_strategy(self, player_name):
        """
        Detect the opponent's strategy based on their action pattern.
        
        Args:
            player_name (str): Name of the opponent
            
        Returns:
            tuple: (strategy_name, confidence)
        """
        if player_name not in self.opponents:
            return None, 0.0
        
        profile = self.get_opponent_profile(player_name)
        
        # Not enough data
        if profile["total_actions"] < 10:
            strategy = "Unknown"
            confidence = 0.0
        else:
            # Define strategy detection logic
            strategies = {
                "Aggressive": 0.0,
                "Defensive": 0.0,
                "Tight": 0.0,
                "Loose": 0.0,
                "Balanced": 0.0
            }
            
            # Aggressive indicators
            if profile["aggression_factor"] > 2.0:
                strategies["Aggressive"] += 0.3
            if profile["pfr"] > 25:
                strategies["Aggressive"] += 0.2
            if profile["flop_aggression_rate"] > 40:
                strategies["Aggressive"] += 0.1
            if profile["turn_aggression_rate"] > 40:
                strategies["Aggressive"] += 0.1
            if profile["river_aggression_rate"] > 40:
                strategies["Aggressive"] += 0.1
            if profile["avg_bet_pot_ratio"] > 0.7:
                strategies["Aggressive"] += 0.2
            
            # Defensive indicators
            if profile["aggression_factor"] < 0.8:
                strategies["Defensive"] += 0.3
            if profile["preflop_fold_rate"] > 40:
                strategies["Defensive"] += 0.2
            if profile["postflop_fold_rate"] > 30:
                strategies["Defensive"] += 0.2
            if profile["vpip"] < 20:
                strategies["Defensive"] += 0.1
            if profile["raise_count"] / max(1, profile["total_actions"]) < 0.15:
                strategies["Defensive"] += 0.2
            
            # Tight indicators
            if profile["vpip"] < 22:
                strategies["Tight"] += 0.4
            if profile["pfr"] < 15:
                strategies["Tight"] += 0.3
            if profile["preflop_fold_rate"] > 50:
                strategies["Tight"] += 0.3
            
            # Loose indicators
            if profile["vpip"] > 35:
                strategies["Loose"] += 0.4
            if profile["call_count"] / max(1, profile["total_actions"]) > 0.4:
                strategies["Loose"] += 0.3
            if profile["preflop_fold_rate"] < 30:
                strategies["Loose"] += 0.3
            
            # Balanced indicators
            if 20 <= profile["vpip"] <= 30:
                strategies["Balanced"] += 0.3
            if 15 <= profile["pfr"] <= 25:
                strategies["Balanced"] += 0.3
            if 0.8 <= profile["aggression_factor"] <= 2.0:
                strategies["Balanced"] += 0.4
            
            # Determine top strategy
            top_strategy = max(strategies.items(), key=lambda x: x[1])
            strategy = top_strategy[0]
            confidence = top_strategy[1]
            
            # If confidence is too low, might be balanced or unknown
            if confidence < 0.4:
                strategy = "Balanced" if strategies["Balanced"] >= 0.3 else "Unknown"
                confidence = strategies["Balanced"] if strategy == "Balanced" else 0.2
        
        # Store the detected strategy
        self.opponents[player_name]["detected_strategy"] = strategy
        self.opponents[player_name]["strategy_confidence"] = confidence
        
        # Log strategy changes
        if len(self.opponents[player_name]["adaptation_history"]) == 0 or \
           self.opponents[player_name]["adaptation_history"][-1]["strategy"] != strategy:
            self.opponents[player_name]["adaptation_history"].append({
                "time": datetime.now().isoformat(),
                "strategy": strategy,
                "confidence": confidence,
                "total_actions": profile["total_actions"]
            })
        
        return strategy, confidence
    
    def suggest_counter_strategy(self, player_name):
        """
        Suggest a counter-strategy against an opponent.
        
        Args:
            player_name (str): Name of the opponent
            
        Returns:
            dict: Counter-strategy suggestions
        """
        if player_name not in self.opponents:
            return {"error": f"No data for opponent: {player_name}"}
        
        profile = self.get_opponent_profile(player_name)
        strategy, confidence = self._detect_strategy(player_name)
        
        if strategy == "Unknown" or confidence < 0.3:
            return {
                "strategy": "Balanced",
                "confidence": 0.0,
                "suggestions": [
                    "Not enough data to form reliable counter-strategy",
                    "Play a balanced strategy until more data is collected",
                    "Focus on solid fundamental poker play"
                ]
            }
        
        counter = {
            "strategy": f"Counter-{strategy}",
            "confidence": confidence,
            "suggestions": []
        }
        
        if strategy == "Aggressive":
            counter["suggestions"] = [
                "Tighten up your starting hand requirements",
                "Look for opportunities to trap with strong hands",
                f"Call or re-raise their bets when you have strong hands (opponent bets {profile['avg_bet_pot_ratio']:.2f}x pot on average)",
                "Be prepared to fold when facing large bets with marginal hands",
                "Avoid bluffing against this opponent"
            ]
        
        elif strategy == "Defensive":
            counter["suggestions"] = [
                "Increase your aggression level",
                "Bet and raise more frequently",
                f"Bluff more often, especially on the flop (opponent fold rate: {profile['postflop_fold_rate']:.1f}%)",
                "Steal blinds more aggressively",
                "Bet larger sizes when bluffing"
            ]
        
        elif strategy == "Tight":
            counter["suggestions"] = [
                "Steal blinds more frequently",
                f"Continuation bet with high frequency (opponent folds to C-bets: {profile['continuation_bet_frequency']:.1f}%)",
                "Bluff more on later streets",
                "Apply pressure when they check to you",
                "Avoid paying off their big bets when they show strength"
            ]
        
        elif strategy == "Loose":
            counter["suggestions"] = [
                "Tighten up your starting hand requirements",
                "Value bet more frequently with strong hands",
                "Reduce bluffing frequency",
                "Make larger value bets with strong hands",
                "Be patient and wait for strong hands to capitalize on their looseness"
            ]
        
        elif strategy == "Balanced":
            counter["suggestions"] = [
                "Play a fundamentally sound strategy",
                "Look for small exploits in their play",
                "Avoid making big adjustments",
                "Use your position to your advantage",
                "Focus on making correct decisions rather than trying to counter their style"
            ]
        
        return counter
    
    def generate_counters_for_play_style(self, aggression, tightness):
        """
        Generate counter strategy suggestions based on play style metrics.
        
        Args:
            aggression (float): Opponent's aggression factor (0-5)
            tightness (float): Opponent's tightness (0-100, higher means tighter)
            
        Returns:
            dict: Counter strategies
        """
        counters = {}
        
        # Counter high aggression
        if aggression > 2.0:
            counters["vs_aggression"] = [
                "Tighten up starting hand requirements",
                "Look for trap opportunities with strong hands",
                "Be prepared to fold marginal hands to big bets",
                "Re-raise with premium hands to build pots",
                "Avoid bluffing against aggressive opponents"
            ]
        # Counter low aggression
        elif aggression < 0.8:
            counters["vs_aggression"] = [
                "Increase your own aggression",
                "Bluff more frequently",
                "Continuation bet more often",
                "Make smaller value bets since they call often",
                "Raise for value more often with strong hands"
            ]
        # Balanced aggression
        else:
            counters["vs_aggression"] = [
                "Maintain balanced approach to aggression",
                "Mix up your play to avoid being predictable",
                "Focus on position and hand strength for aggression decisions"
            ]
        
        # Counter tight play
        if tightness > 70:
            counters["vs_tightness"] = [
                "Steal blinds more aggressively",
                "Bluff more often, especially on later streets",
                "Fold to their raises when you have marginal hands",
                "Bet smaller for value to induce calls",
                "3-bet their opens less frequently"
            ]
        # Counter loose play
        elif tightness < 30:
            counters["vs_tightness"] = [
                "Value bet more frequently with strong hands",
                "Reduce bluffing frequency",
                "Bet larger with strong hands",
                "Call down lighter when they're being aggressive",
                "Tighten up your own starting hand requirements"
            ]
        # Balanced tightness
        else:
            counters["vs_tightness"] = [
                "Use a fundamentally sound strategy",
                "Pay attention to their positional tendencies",
                "Adapt based on specific scenarios rather than overall style"
            ]
        
        return counters
    
    def plot_opponent_tendencies(self, player_name, output_dir=None):
        """
        Generate visualizations of opponent tendencies.
        
        Args:
            player_name (str): Name of the opponent
            output_dir (str, optional): Directory to save the plots
            
        Returns:
            list: Paths to generated plot files
        """
        if player_name not in self.opponents:
            logging.warning(f"No data for opponent: {player_name}")
            return []
        
        if output_dir is None:
            output_dir = os.path.join(self.log_dir, player_name)
        
        os.makedirs(output_dir, exist_ok=True)
        
        profile = self.get_opponent_profile(player_name)
        plot_paths = []
        
        # 1. Action Distribution
        plt.figure(figsize=(10, 6))
        actions = ['fold_count', 'check_count', 'call_count', 'raise_count']
        action_counts = [profile[action] for action in actions]
        action_labels = ['Fold', 'Check', 'Call', 'Raise']
        
        plt.bar(action_labels, action_counts)
        plt.title(f"{player_name}'s Action Distribution")
        plt.ylabel('Count')
        plt.grid(axis='y', alpha=0.3)
        
        action_path = os.path.join(output_dir, f"{player_name}_actions.png")
        plt.savefig(action_path)
        plt.close()
        plot_paths.append(action_path)
        
        # 2. Preflop vs Postflop Tendencies
        plt.figure(figsize=(12, 6))
        
        # Preflop
        preflop_actions = ['preflop_fold', 'preflop_call', 'preflop_raise']
        preflop_counts = [profile[action] for action in preflop_actions]
        preflop_labels = ['Fold', 'Call', 'Raise']
        
        plt.subplot(1, 2, 1)
        plt.pie(preflop_counts, labels=preflop_labels, autopct='%1.1f%%')
        plt.title(f"{player_name}'s Preflop Actions")
        
        # Postflop
        postflop_actions = ['postflop_fold', 'postflop_check', 'postflop_call', 'postflop_raise']
        postflop_counts = [profile[action] for action in postflop_actions]
        postflop_labels = ['Fold', 'Check', 'Call', 'Raise']
        
        plt.subplot(1, 2, 2)
        plt.pie(postflop_counts, labels=postflop_labels, autopct='%1.1f%%')
        plt.title(f"{player_name}'s Postflop Actions")
        
        plt.tight_layout()
        stages_path = os.path.join(output_dir, f"{player_name}_stages.png")
        plt.savefig(stages_path)
        plt.close()
        plot_paths.append(stages_path)
        
        # 3. Key Metrics Radar Chart
        plt.figure(figsize=(8, 8))
        
        # Define metrics
        metrics = ['vpip', 'pfr', 'aggression_factor', 'preflop_fold_rate', 
                 'postflop_fold_rate', 'flop_aggression_rate']
        metric_values = [profile[metric] for metric in metrics]
        
        # Normalize values for radar chart
        max_values = [100, 100, 3, 100, 100, 100]  # Maximum expected values for each metric
        normalized_values = [min(v / m, 1.0) for v, m in zip(metric_values, max_values)]
        
        # Close the loop for the radar chart
        metrics = [m.replace('_', ' ').title() for m in metrics]
        metrics.append(metrics[0])
        normalized_values.append(normalized_values[0])
        
        # Compute angles for radar chart
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        # Draw radar chart
        ax = plt.subplot(111, polar=True)
        plt.xticks(angles[:-1], metrics[:-1], color='grey', size=8)
        ax.set_rlabel_position(0)
        plt.yticks([0.25, 0.5, 0.75, 1], ["0.25", "0.5", "0.75", "1"], color="grey", size=7)
        plt.ylim(0, 1)
        
        ax.plot(angles, normalized_values, linewidth=1, linestyle='solid')
        ax.fill(angles, normalized_values, 'b', alpha=0.1)
        
        plt.title(f"{player_name}'s Playing Style", size=11, y=1.1)
        
        radar_path = os.path.join(output_dir, f"{player_name}_radar.png")
        plt.savefig(radar_path)
        plt.close()
        plot_paths.append(radar_path)
        
        # 4. Aggression by Street
        plt.figure(figsize=(10, 6))
        streets = ['Flop', 'Turn', 'River']
        aggression_rates = [
            profile['flop_aggression_rate'],
            profile['turn_aggression_rate'],
            profile['river_aggression_rate']
        ]
        
        plt.bar(streets, aggression_rates)
        plt.title(f"{player_name}'s Aggression by Street")
        plt.ylabel('Aggression Rate (%)')
        plt.grid(axis='y', alpha=0.3)
        
        street_path = os.path.join(output_dir, f"{player_name}_street_aggression.png")
        plt.savefig(street_path)
        plt.close()
        plot_paths.append(street_path)
        
        # 5. Positional Play
        if profile["sb_actions"] or profile["bb_actions"]:
            plt.figure(figsize=(10, 6))
            positions = ['Small Blind', 'Big Blind']
            fold_rates = [profile['sb_fold_rate'], profile['bb_fold_rate']]
            aggression_rates = [profile['sb_aggression'], profile['bb_aggression']]
            
            x = np.arange(len(positions))
            width = 0.35
            
            plt.bar(x - width/2, fold_rates, width, label='Fold Rate (%)')
            plt.bar(x + width/2, aggression_rates, width, label='Aggression (%)')
            
            plt.title(f"{player_name}'s Positional Play")
            plt.xticks(x, positions)
            plt.ylabel('Rate (%)')
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
            
            position_path = os.path.join(output_dir, f"{player_name}_position.png")
            plt.savefig(position_path)
            plt.close()
            plot_paths.append(position_path)
        
        return plot_paths
    
    def generate_opponent_report(self, player_name, output_path=None):
        """
        Generate a comprehensive report on an opponent.
        
        Args:
            player_name (str): Name of the opponent
            output_path (str, optional): Path to save the report
            
        Returns:
            str: Path to the saved report
        """
        if player_name not in self.opponents:
            logging.warning(f"No data for opponent: {player_name}")
            return None
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"{self.log_dir}/{player_name}_analysis_{timestamp}.txt"
        
        profile = self.get_opponent_profile(player_name)
        strategy, confidence = self._detect_strategy(player_name)
        counter_strategy = self.suggest_counter_strategy(player_name)
        
        with open(output_path, 'w') as f:
            f.write(f"OPPONENT ANALYSIS: {player_name}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Strategy: {strategy} (Confidence: {confidence:.2f})\n\n")
            
            f.write("BASIC STATS\n")
            f.write("-----------\n")
            f.write(f"Total Hands: {profile['total_hands']}\n")
            f.write(f"Hands Won: {profile['hands_won']} ({profile['win_rate']:.1f}%)\n")
            f.write(f"Total Actions: {profile['total_actions']}\n\n")
            
            f.write("KEY METRICS\n")
            f.write("-----------\n")
            f.write(f"VPIP (Voluntarily Put $ In Pot): {profile['vpip']:.1f}%\n")
            f.write(f"PFR (Preflop Raise): {profile['pfr']:.1f}%\n")
            f.write(f"Aggression Factor: {profile['aggression_factor']:.2f}\n")
            f.write(f"Preflop Fold Rate: {profile['preflop_fold_rate']:.1f}%\n")
            f.write(f"Postflop Fold Rate: {profile['postflop_fold_rate']:.1f}%\n\n")
            
            f.write("STREET-SPECIFIC TENDENCIES\n")
            f.write("-------------------------\n")
            f.write(f"Flop Aggression: {profile['flop_aggression_rate']:.1f}%\n")
            f.write(f"Turn Aggression: {profile['turn_aggression_rate']:.1f}%\n")
            f.write(f"River Aggression: {profile['river_aggression_rate']:.1f}%\n\n")
            
            f.write("BETTING PATTERNS\n")
            f.write("---------------\n")
            f.write(f"Average Bet Size: {profile['avg_bet_size']:.2f}\n")
            f.write(f"Average Bet to Pot Ratio: {profile['avg_bet_pot_ratio']:.2f}\n")
            f.write(f"Continuation Bet Frequency: {profile['continuation_bet_frequency']:.1f}%\n")
            f.write(f"Check-Raise Frequency: {profile['check_raise_frequency']:.1f}%\n\n")
            
            f.write("POSITIONAL PLAY\n")
            f.write("--------------\n")
            f.write(f"Small Blind Fold Rate: {profile['sb_fold_rate']:.1f}%\n")
            f.write(f"Small Blind Aggression: {profile['sb_aggression']:.1f}%\n")
            f.write(f"Big Blind Fold Rate: {profile['bb_fold_rate']:.1f}%\n")
            f.write(f"Big Blind Aggression: {profile['bb_aggression']:.1f}%\n\n")
            
            f.write("COUNTER STRATEGY\n")
            f.write("----------------\n")
            for suggestion in counter_strategy["suggestions"]:
                f.write(f"- {suggestion}\n")
            
            f.write("\n")
            
            f.write("STRATEGY ADAPTATIONS\n")
            f.write("-------------------\n")
            for adaptation in profile["adaptation_history"]:
                f.write(f"- {adaptation['time']} - {adaptation['strategy']} " + 
                       f"(Confidence: {adaptation['confidence']:.2f}, Actions: {adaptation['total_actions']})\n")
        
        logging.info(f"Opponent analysis saved to {output_path}")
        return output_path
    
    def save_opponent_data(self, output_path=None):
        """
        Save opponent data to a JSON file.
        
        Args:
            output_path (str, optional): Path to save the data
            
        Returns:
            str: Path to the saved file
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"{self.log_dir}/opponent_data_{timestamp}.json"
        
        # Convert data to serializable format
        data = {
            "opponents": {},
            "action_history": self.action_history
        }
        
        for player_name, profile in self.opponents.items():
            data["opponents"][player_name] = {}
            
            # Copy most items
            for key, value in profile.items():
                # Skip complex objects or convert them to serializable format
                if key in ["recent_actions", "adaptation_history", "bet_sizes", 
                         "bet_sizing_by_pot", "sb_actions", "bb_actions",
                         "flop_aggression", "turn_aggression", "river_aggression",
                         "showdown_hands"]:
                    data["opponents"][player_name][key] = value
                else:
                    # Ensure serializable values
                    try:
                        json.dumps({key: value})
                        data["opponents"][player_name][key] = value
                    except (TypeError, OverflowError):
                        data["opponents"][player_name][key] = str(value)
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logging.info(f"Opponent data saved to {output_path}")
        return output_path
    
    def load_opponent_data(self, input_path):
        """
        Load opponent data from a JSON file.
        
        Args:
            input_path (str): Path to the saved data
            
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        if not os.path.exists(input_path):
            logging.error(f"Input file not found: {input_path}")
            return False
        
        try:
            with open(input_path, 'r') as f:
                data = json.load(f)
            
            self.opponents = data.get("opponents", {})
            self.action_history = data.get("action_history", [])
            
            logging.info(f"Loaded opponent data from {input_path}")
            return True
        except Exception as e:
            logging.error(f"Error loading opponent data: {e}")
            return False

# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = OpponentAnalyzer()
    
    # Sample hand data
    hand_data = {
        "hand_id": 1,
        "player_info": [
            {"name": "Abel", "position": "SB", "stack": 1000},
            {"name": "Kane", "position": "BB", "stack": 1000}
        ],
        "hole_cards": {
            "Abel": ["AS", "KS"],
            "Kane": ["JH", "QH"]
        },
        "streets": {
            "preflop": {
                "actions": [
                    {"player": "Abel", "action": "raise", "amount": 60, "pot_size": 80},
                    {"player": "Kane", "action": "call", "amount": 40, "pot_size": 120}
                ],
                "community_cards": []
            },
            "flop": {
                "actions": [
                    {"player": "Kane", "action": "check", "amount": 0, "pot_size": 120},
                    {"player": "Abel", "action": "bet", "amount": 80, "pot_size": 200},
                    {"player": "Kane", "action": "call", "amount": 80, "pot_size": 280}
                ],
                "community_cards": ["2S", "5C", "KH"]
            },
            "turn": {
                "actions": [
                    {"player": "Kane", "action": "check", "amount": 0, "pot_size": 280},
                    {"player": "Abel", "action": "bet", "amount": 200, "pot_size": 480},
                    {"player": "Kane", "action": "fold", "amount": 0, "pot_size": 480}
                ],
                "community_cards": ["2S", "5C", "KH", "AS"]
            },
            "river": {
                "actions": [],
                "community_cards": []
            }
        },
        "results": {
            "winners": ["Abel"],
            "pot_distribution": {"Abel": 480},
            "player_final_states": [
                {"name": "Abel", "stack": 1240},
                {"name": "Kane", "stack": 760}
            ]
        }
    }
    
    # Record sample hand
    analyzer.record_hand(hand_data)
    
    # Analyze Kane
    kane_profile = analyzer.get_opponent_profile("Kane")
    print(f"Kane's Profile - VPIP: {kane_profile['vpip']:.1f}%, PFR: {kane_profile['pfr']:.1f}%, Aggression: {kane_profile['aggression_factor']:.2f}")
    
    # Get counter strategy
    counter = analyzer.suggest_counter_strategy("Kane")
    print(f"Counter Strategy: {counter['strategy']} (Confidence: {counter['confidence']:.2f})")
    for suggestion in counter["suggestions"]:
        print(f"- {suggestion}")
    
    # Generate report
    report_path = analyzer.generate_opponent_report("Kane")
    print(f"Report generated: {report_path}")
    
    # Generate visualizations
    plot_paths = analyzer.plot_opponent_tendencies("Kane")
    print(f"Plots generated: {len(plot_paths)} visualizations")