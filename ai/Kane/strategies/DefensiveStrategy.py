import logging
from ai.Kane.strategies.Strategy import Strategy

class DefensiveStrategy(Strategy):
    """
    Defensive Strategy for Kane
    - Cautious and conservative.
    - Avoids large pots and aggressive plays.
    - Focuses on pot control and minimizing losses.
    """

    def decide_action(self, player, valid_actions, hole_card, round_state):
        # Defensive behavior: prioritize call or fold
        action = "fold"
        amount = 0
        
        # If check is available, prefer to check
        for act in valid_actions:
            if act['action'] == 'check':
                action = 'check'
                amount = act['amount']
                break
        
        # If call is available and safe, do a minimal call
        if action == "fold":
            action = "call"
            amount = valid_actions[1]['amount']

        # Logging the defensive action
        logging.info(f"[DefensiveStrategy] Decided Action: {action} Amount: {amount}")
        
        return action, amount
