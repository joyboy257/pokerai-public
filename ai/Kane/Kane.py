import logging
import random
from pypokerengine.players import BasePokerPlayer
from treys import Card, Evaluator  # Using Treys for hand evaluation

# Importing the FSM states
from ai.Kane.states.PreflopState import PreflopState
from ai.Kane.states.FlopState import FlopState
from ai.Kane.states.TurnState import TurnState
from ai.Kane.states.RiverState import RiverState
from ai.Kane.states.EndState import EndState

# Importing Strategies
from ai.Kane.strategies.AggressiveStrategy import AggressiveStrategy
from ai.Kane.strategies.DefensiveStrategy import DefensiveStrategy

# Configure logging
logging.basicConfig(
    filename="logs/kane_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

def convert_to_treys_format(cards):
    """
    Convert cards from PyPokerEngine format to Treys format.
    PyPokerEngine: ['H8', 'C2']
    Treys Format: ['8h', '2c']
    """
    suit_map = {
        'H': 'h',  # Hearts
        'D': 'd',  # Diamonds
        'C': 'c',  # Clubs
        'S': 's'   # Spades
    }
    treys_cards = []

    for card in cards:
        rank = card[1]  # Second character is rank
        suit = card[0]  # First character is suit
        treys_card = rank + suit_map[suit]
        treys_cards.append(treys_card)

    return treys_cards


class RuleBasedPlayer(BasePokerPlayer):
    def __init__(self, strategy=None):
        super().__init__()
        self.evaluator = Evaluator()  # Treys Evaluator for hand strength
        self.current_state = PreflopState()  # Start with Preflop state
        self.strategy = strategy if strategy else AggressiveStrategy()  # Default to Aggressive
        self.aggressive_strategy = AggressiveStrategy()
        self.defensive_strategy = DefensiveStrategy()
        
        # Dynamic Variables
        self.win_streak = 0
        self.lose_streak = 0
        self.strategy_switches = 0
        self.bluffing_chance = 0.1  # Base chance to bluff
        self.bluffing = False
        
        # Tracking Variables
        self.total_rounds = 0
        self.bluffing_attempts = 0
        self.previous_strategy = self.strategy

    def evaluate_hand_strength(self, hole_cards, community_cards):
        """
        Evaluates the hand strength using Treys' Evaluator.
        Returns a hand rank (lower is better).
        """
        # Convert to Treys format
        treys_hole = [Card.new(card) for card in convert_to_treys_format(hole_cards)]
        treys_community = [Card.new(card) for card in convert_to_treys_format(community_cards)]

        # Check if community cards are available
        if len(treys_community) < 3:
            high_cards = ['A', 'K', 'Q', 'J', 'T']
            high_card_count = sum(1 for card in hole_cards if card[1] in high_cards)
            hand_strength = high_card_count / 2
            hand_class = "High Card"
        else:
            hand_rank = self.evaluator.evaluate(treys_community, treys_hole)
            hand_class = self.evaluator.get_rank_class(hand_rank)
            hand_strength = 1 - (hand_rank / 7462)

            logging.info(
                f"Evaluated Hand: Hole={hole_cards}, Community={community_cards}, "
                f"Hand Rank={hand_rank}, Class={self.evaluator.class_to_string(hand_class)}, "
                f"Strength={hand_strength:.2f}"
            )
            hand_class = self.evaluator.class_to_string(hand_class)

        return hand_strength, hand_class

    def adjust_bluffing_chance(self):
        """
        Dynamically adjust bluffing chance based on streaks.
        """
        if self.lose_streak >= 3:
            self.bluffing_chance = min(0.3, self.bluffing_chance + 0.05)
        elif self.win_streak >= 3:
            self.bluffing_chance = max(0.05, self.bluffing_chance - 0.05)
        else:
            self.bluffing_chance = 0.1

        logging.info(f"Adjusted Bluffing Chance: {self.bluffing_chance:.2f}")

    def decide_strategy(self, hand_strength, round_state):
        """
        Dynamically decide which strategy to use.
        """
        # Strategy Decision Logic
        if self.bluffing:
            logging.info("Kane is bluffing this round.")
            return self.aggressive_strategy

        # If strong hand, play aggressively
        if hand_strength > 0.7:
            if not isinstance(self.strategy, AggressiveStrategy):
                self.strategy_switches += 1
            return self.aggressive_strategy
        
        # If weak hand or unfavorable position, play defensively
        if hand_strength < 0.4:
            if not isinstance(self.strategy, DefensiveStrategy):
                self.strategy_switches += 1
            return self.defensive_strategy

        # Default to aggressive strategy
        return self.aggressive_strategy

    def declare_action(self, valid_actions, hole_card, round_state):
        """
        Makes a decision using the FSM pattern and Strategy Pattern.
        """
        self.total_rounds += 1  # Track total rounds
        
        # Evaluate Hand Strength
        community_cards = round_state.get("community_card", [])
        hand_strength, hand_class = self.evaluate_hand_strength(hole_card, community_cards)

        # Adjust Bluffing Chance
        self.adjust_bluffing_chance()

        # Bluffing Mechanism
        if not self.bluffing and random.random() < self.bluffing_chance:
            logging.info("Bluffing activated!")
            self.bluffing = True
            self.bluffing_attempts += 1
        else:
            self.bluffing = False

        # Dynamic Strategy Switching
        self.strategy = self.decide_strategy(hand_strength, round_state)
        
        # Track Strategy Switches
        if self.strategy.__class__.__name__ != self.previous_strategy.__class__.__name__:
            self.strategy_switches += 1
        self.previous_strategy = self.strategy
        
        # FSM: Enter Current State
        self.current_state.enter_state(self, round_state)

        # FSM: Decide Action Based on Current State and Strategy
        action, amount = self.strategy.decide_action(self, valid_actions, hole_card, round_state)

        # FSM: Transition to Next State
        self.current_state = self.current_state.next_state(round_state)

        # Log decision details
        logging.info(
            f"Decision made by Kane: Hole={hole_card}, Community={round_state.get('community_card', [])}, "
            f"Action={action}, Amount={amount}, State={self.current_state.__class__.__name__}, "
            f"Strategy={self.strategy.__class__.__name__}"
        )

        return action, amount

    def receive_game_start_message(self, game_info):
        logging.info(f"Game started with info: {game_info}")
        self.current_state = PreflopState()  # Ensure FSM starts at Preflop

    def receive_round_start_message(self, round_count, hole_card, seats):
        logging.info(f"Round {round_count} started with hole cards: {hole_card}")
        self.current_state = PreflopState()  # Reset to Preflop state

    def receive_street_start_message(self, street, round_state):
        logging.info(f"New street: {street}")

    def receive_game_update_message(self, action, round_state):
        logging.info(f"Game updated with action: {action}")

    def receive_round_result_message(self, winners, hand_info, round_state):
        logging.info(f"Round ended. Winners: {winners}")
        self.current_state = EndState()  # Transition to EndState
