import logging
from pypokerengine.players import BasePokerPlayer
from treys import Card, Evaluator  # Using Treys for hand evaluation

# Configure logging
logging.basicConfig(
    filename="kane_log.txt",  # Log file name
    level=logging.INFO,       # Set logging level to INFO
    format="%(asctime)s - %(message)s"  # Log format
)

class RuleBasedPlayer(BasePokerPlayer):
    def __init__(self):
        super().__init__()
        self.evaluator = Evaluator()  # Treys Evaluator for hand strength

    def evaluate_hand_strength(self, hole_cards, community_cards):
        """
        Evaluates the hand strength using Treys' Evaluator.
        Returns a hand rank (lower is better).
        """
        # Convert hole cards and community cards to Treys format
        treys_hole = [Card.new(card) for card in hole_cards]
        treys_community = [Card.new(card) for card in community_cards]

        # If no community cards, evaluate hole cards only
        if not community_cards:
            hand_rank = self.evaluator.evaluate([], treys_hole)
        else:
            hand_rank = self.evaluator.evaluate(treys_community, treys_hole)

        # Get hand class (e.g., Pair, Two Pair, etc.)
        hand_class = self.evaluator.get_rank_class(hand_rank)
        hand_strength = 1 - (hand_rank / 7462)  # Normalize hand strength (1 is best, 0 is worst)

        # Log the hand evaluation
        logging.info(
            f"Evaluated Hand: Hole={hole_cards}, Community={community_cards}, "
            f"Hand Rank={hand_rank}, Class={self.evaluator.class_to_string(hand_class)}, "
            f"Strength={hand_strength:.2f}"
        )

        return hand_strength, self.evaluator.class_to_string(hand_class)

    def declare_action(self, valid_actions, hole_card, round_state):
        """
        Makes a decision based on hand strength.
        """
        # Get community cards from round state
        community_cards = round_state.get("community_card", [])

        # Evaluate hand strength
        hand_strength, hand_class = self.evaluate_hand_strength(hole_card, community_cards)

        # Decision making logic
        if hand_strength > 0.8:  # Strong hand
            action = "raise"
            amount = valid_actions[2]['amount']['max']  # Max raise
        elif hand_strength > 0.5:  # Medium hand
            action = "call"
            amount = valid_actions[1]['amount']  # Match the bet
        else:  # Weak hand
            action = "fold"
            amount = 0

        # Log decision details
        logging.info(
            f"Decision made by Kane: Hole={hole_card}, Community={community_cards}, "
            f"Hand Class={hand_class}, Hand Strength={hand_strength:.2f}, "
            f"Action={action}, Amount={amount}"
        )

        return action, amount

    def receive_game_start_message(self, game_info):
        logging.info(f"Game started with info: {game_info}")

    def receive_round_start_message(self, round_count, hole_card, seats):
        logging.info(f"Round {round_count} started with hole cards: {hole_card}")

    def receive_street_start_message(self, street, round_state):
        logging.info(f"New street: {street}")

    def receive_game_update_message(self, action, round_state):
        logging.info(f"Game updated with action: {action}")

    def receive_round_result_message(self, winners, hand_info, round_state):
        logging.info(f"Round ended. Winners: {winners}")
