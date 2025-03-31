import logging
import os
import sys
from pypokerengine.api.game import setup_config, start_poker
from ai.Kane.Kane import RuleBasedPlayer
from pypokerengine.players import BasePokerPlayer

# Set up logging to display Kane's decision-making process
logging.basicConfig(level=logging.INFO, 
                   format='%(message)s',
                   handlers=[logging.StreamHandler()])

class SimpleOpponent(BasePokerPlayer):
    """Simple opponent for demonstration purposes"""
    def declare_action(self, valid_actions, hole_card, round_state):
        # Always calls or checks when possible
        action = valid_actions[1]['action']  # 'call' or 'check'
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

def kane_demonstration():
    """Demonstrate Kane's rule-based decision making"""
    print("\n" + "="*50)
    print("KANE AGENT DEMONSTRATION - RULE BASED APPROACH")
    print("="*50)
    
    # Create Kane with both strategy types for demonstration
    kane = RuleBasedPlayer()
    opponent = SimpleOpponent()
    
    # Setup game config
    config = setup_config(max_round=1, initial_stack=1000, small_blind_amount=10)
    config.register_player(name="Kane", algorithm=kane)
    config.register_player(name="Opponent", algorithm=opponent)
    
    # Display Kane's structure
    print("\nKane implements a Finite State Machine with these states:")
    print("- PreflopState: Initial betting")
    print("- FlopState: After first 3 community cards")
    print("- TurnState: After 4th community card")
    print("- RiverState: After 5th community card")
    print("- EndState: Hand complete")
    
    print("\nKane's strategies:")
    print("- AggressiveStrategy: Focus on raising with strong hands")
    print("- DefensiveStrategy: Conservative play, prefers calling/checking")
    
    print("\nDemonstrating Kane's decision process during gameplay:")
    print("-" * 50)
    
    # Play a demonstration hand
    game_result = start_poker(config, verbose=1)
    
    # Show example decision making
    print("\nExample decision process:")
    print("1. Current state (e.g., Preflop) determines context")
    print("2. Strategy selected based on hand strength and game state")
    print("3. Decision made following the selected strategy rules")
    
    # Display results
    print("\nResults summary:")
    for player in game_result['players']:
        print(f"Player: {player['name']}, Stack: {player['stack']}")
    
    return game_result

if __name__ == "__main__":
    kane_demonstration()