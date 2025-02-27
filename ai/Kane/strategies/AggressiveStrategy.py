import logging
from ai.Kane.strategies.Strategy import Strategy

class AggressiveStrategy(Strategy):
    """
    Aggressive Strategy for Kane
    - High frequency of raises and bets.
    - Aims to pressure opponents into folding.
    - Occasional bluffing to maintain unpredictability.
    """

    def decide_action(self, player, valid_actions, hole_card, round_state):
        # Aggressive behavior: prioritize raise or bet
        action = "raise"
        max_raise = valid_actions[2]['amount']['max']  # Max raise

        # If max raise is not possible, try to bet
        if max_raise == -1:  # -1 indicates no raise is allowed
            action = "call"
            amount = valid_actions[1]['amount']
        else:
            amount = max_raise

        # Logging the aggressive action
        logging.info(f"[AggressiveStrategy] Decided Action: {action} Amount: {amount}")
        
        return action, amount
