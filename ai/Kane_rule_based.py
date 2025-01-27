import logging
from pypokerengine.players import BasePokerPlayer

# Configure logging
logging.basicConfig(
    filename="kane_log.txt",  # Log file name
    level=logging.INFO,       # Set logging level to INFO
    format="%(asctime)s - %(message)s"  # Log format
)

class RuleBasedPlayer(BasePokerPlayer):
    def __init__(self):
        super().__init__()

    def evaluate_hand_strength(self, hole_cards, community_cards):
        # Simple hand strength evaluation: number of high cards (10 or above)
        high_cards = ['10', 'J', 'Q', 'K', 'A']
        all_cards = hole_cards + community_cards
        strength = sum(1 for card in all_cards if card[1] in high_cards)
        return strength

    def declare_action(self, valid_actions, hole_card, round_state):
        # Evaluate hand strength
        community_cards = round_state.get("community_card", [])
        hand_strength = self.evaluate_hand_strength(hole_card, community_cards)

        # Define actions based on hand strength
        if hand_strength >= 2:  # Strong hand
            action = "raise"
            amount = valid_actions[2]['amount']['max']  # Max raise
        elif hand_strength == 1:  # Medium hand
            action = "call"
            amount = valid_actions[1]['amount']  # Match the bet
        else:  # Weak hand
            action = "fold"
            amount = 0

        # Log the decision details
        logging.info(
            f"Decision made by Kane: Hand={hole_card}, Community={community_cards}, "
            f"Hand Strength={hand_strength}, Action={action}, Amount={amount}"
        )

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