# demo.py
import sys
import os
import time

# Add the current directory to the path
sys.path.insert(0, os.getcwd())

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Import your actual models
try:
    # Import here to avoid logging error
    from pypokerengine.api.game import setup_config, start_poker
    from pypokerengine.players import BasePokerPlayer
    
    # Create a "patch" for the logging before importing Kane
    import logging
    logging.basicConfig(
        filename="logs/kane_log.txt",
        level=logging.INFO,
        format="%(asctime)s - %(message)s"
    )
    
    # Now import Kane
    from ai.Kane.Kane import RuleBasedPlayer
    from ai.Kane.strategies.AggressiveStrategy import AggressiveStrategy
    from ai.Kane.strategies.DefensiveStrategy import DefensiveStrategy
    
    # Try to import Abel
    try:
        from ai.Abel.Abel import RLBasedPlayer
        ABEL_LOADED = True
    except ImportError:
        ABEL_LOADED = False
        print("Abel model could not be loaded, but Kane is available")
    
    MODELS_LOADED = True
    print("Successfully loaded Kane model!")
except ImportError as e:
    MODELS_LOADED = False
    ABEL_LOADED = False
    print(f"Error importing models: {e}")
    print("Will run in limited demonstration mode")

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

class SimpleOpponent(BasePokerPlayer):
    """Simple opponent for demonstration"""
    def declare_action(self, valid_actions, hole_card, round_state):
        # Simple strategy - always call
        action = valid_actions[1]['action']
        amount = valid_actions[1]['amount']
        return action, amount
    
    def receive_game_start_message(self, game_info): pass
    def receive_round_start_message(self, round_count, hole_card, seats): pass
    def receive_street_start_message(self, street, round_state): pass
    def receive_game_update_message(self, action, round_state): pass
    def receive_round_result_message(self, winners, hand_info, round_state): pass

def run_kane_demo():
    """Run a demonstration using the actual Kane model"""
    print("\n" + "="*70)
    print(" KANE: RULE-BASED POKER AI DEMONSTRATION ".center(70, "="))
    print("="*70)
    
    # Create Kane with different strategies
    kane_aggressive = RuleBasedPlayer(strategy=AggressiveStrategy())
    opponent = SimpleOpponent()
    
    print("\nRunning Kane with Aggressive Strategy...")
    # Setup game with aggressive Kane
    config = setup_config(max_round=1, initial_stack=1000, small_blind_amount=10)
    config.register_player(name="Kane (Aggressive)", algorithm=kane_aggressive)
    config.register_player(name="Opponent", algorithm=opponent)
    
    # Play one hand
    game_result = start_poker(config, verbose=1)
    
    print("\nKane demonstrates rule-based decision making with:")
    print("- Finite State Machine for game state tracking")
    print("- Strategy pattern for behavioral flexibility")
    print("- Hand strength evaluation using Treys library")
    print("- Dynamic bluffing based on game context")

def run_abel_demo():
    """Run a demonstration using the actual Abel model"""
    if not ABEL_LOADED:
        print("\nAbel model is not available. Skipping demonstration.")
        return
        
    print("\n" + "="*70)
    print(" ABEL: REINFORCEMENT LEARNING POKER AI DEMONSTRATION ".center(70, "="))
    print("="*70)
    
    # Initialize Abel with demo parameters
    state_size = 6
    action_size = 3
    abel = RLBasedPlayer(state_size, action_size)
    opponent = SimpleOpponent()
    
    print("\nRunning Abel against simple opponent...")
    # Setup game
    config = setup_config(max_round=1, initial_stack=1000, small_blind_amount=10)
    config.register_player(name="Abel", algorithm=abel)
    config.register_player(name="Opponent", algorithm=opponent)
    
    # Play one hand
    game_result = start_poker(config, verbose=1)
    
    print("\nAbel demonstrates reinforcement learning with:")
    print("- Deep Q-Network for action value estimation")
    print("- Experience replay for training stability")
    print("- Epsilon-greedy exploration policy")
    print("- State encoding for neural network processing")

def run_competition():
    """Run a competition between Kane and Abel"""
    if not ABEL_LOADED:
        print("\nAbel model is not available. Skipping competition.")
        return
        
    print("\n" + "="*70)
    print(" KANE VS. ABEL COMPETITION ".center(70, "="))
    print("="*70)
    
    # Create agents
    kane = RuleBasedPlayer()
    abel = RLBasedPlayer(state_size=6, action_size=3)
    
    print("\nRunning head-to-head competition...")
    # Setup game
    config = setup_config(max_round=3, initial_stack=1000, small_blind_amount=10)
    config.register_player(name="Kane", algorithm=kane)
    config.register_player(name="Abel", algorithm=abel)
    
    # Play multiple hands
    game_result = start_poker(config, verbose=1)
    
    # Report results
    kane_stack = next(player["stack"] for player in game_result["players"] if player["name"] == "Kane")
    abel_stack = next(player["stack"] for player in game_result["players"] if player["name"] == "Abel")
    
    print("\nFinal Results:")
    print(f"Kane's final stack: {kane_stack}")
    print(f"Abel's final stack: {abel_stack}")
    
    if kane_stack > abel_stack:
        print("Kane wins the demonstration match!")
    else:
        print("Abel wins the demonstration match!")

def main_demo():
    """Run the complete demonstration"""
    clear_screen()
    
    print("\n" + "="*70)
    print(" KANE & ABEL: RULE-BASED VS. REINFORCEMENT LEARNING POKER AI ".center(70, "="))
    print("="*70)
    print("\nAuthor: Deon Allax Quek Wei Xuan".center(70))
    print("\nThis demonstration showcases two contrasting approaches to poker AI:")
    print("1. Kane: A rule-based agent using finite state machines and strategy patterns")
    print("2. Abel: A learning-based agent using deep reinforcement learning")
    print("\nRepository: https://github.com/joyboy257/pokerai-public")
    
    if not MODELS_LOADED:
        print("\nWARNING: Could not load actual models. Running in limited mode.")
    
    input("\nPress Enter to begin Kane demonstration...")
    clear_screen()
    if MODELS_LOADED:
        try:
            run_kane_demo()
        except Exception as e:
            print(f"Error in Kane demonstration: {e}")
            print(f"Details: {type(e).__name__}")
            import traceback
            traceback.print_exc()
    else:
        print("Kane demonstration unavailable in limited mode")
    
    input("\nPress Enter to continue to Abel demonstration...")
    clear_screen()
    if MODELS_LOADED and ABEL_LOADED:
        try:
            run_abel_demo()
        except Exception as e:
            print(f"Error in Abel demonstration: {e}")
            print(f"Details: {type(e).__name__}")
            import traceback
            traceback.print_exc()
    else:
        print("Abel demonstration unavailable in limited mode")
    
    input("\nPress Enter to see Kane vs. Abel competition...")
    clear_screen()
    if MODELS_LOADED and ABEL_LOADED:
        try:
            run_competition()
        except Exception as e:
            print(f"Error in competition: {e}")
            print(f"Details: {type(e).__name__}")
            import traceback
            traceback.print_exc()
    else:
        print("Competition unavailable in limited mode")
    
    print("\n" + "="*70)
    print(" KEY FINDINGS ".center(70, "="))
    print("="*70)
    print("\n1. Well-designed rule-based systems like Kane maintain an edge in structured environments")
    print("2. Reinforcement learning shows promise but requires extensive resources")
    print("3. Self-play training has limitations when facing different opponent types")
    print("4. Future work should explore hybrid approaches combining rules and learning")
    
    print("\n" + "="*70)
    print(" END OF DEMONSTRATION ".center(70, "="))
    print("="*70)
    print("\nThank you for watching this demonstration of the Kane & Abel Poker AI project.")
    print("For more details and the complete code, please visit:")
    print("https://github.com/joyboy257/pokerai-public")

if __name__ == "__main__":
    main_demo()