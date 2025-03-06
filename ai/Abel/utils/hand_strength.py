# Save this file to: ai/utils/hand_strength.py

import logging
import numpy as np
import itertools
from treys import Card, Evaluator

class HandStrengthCalculator:
    """
    Evaluates poker hand strength on a 0-1 scale, including incomplete information.
    
    This class provides methods to calculate the strength of a poker hand at any
    stage of a Texas Hold'em game (preflop, flop, turn, river), allowing for 
    decision-making with incomplete information.
    """
    
    def __init__(self):
        """
        Initialize the hand strength calculator.
        """
        # Initialize treys evaluator
        self.evaluator = Evaluator()
        
        # Cache for preflop hand rankings
        self.preflop_rankings = {}
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("HandStrengthCalculator")
    
    def _convert_to_treys_format(self, cards):
        """
        Convert cards from PyPokerEngine format to Treys format.
        
        Args:
            cards (list): List of cards in format like ['H8', 'C2']
            
        Returns:
            list: List of Card objects in Treys format
        """
        suit_map = {
            'H': 'h',  # Hearts
            'D': 'd',  # Diamonds
            'C': 'c',  # Clubs
            'S': 's'   # Spades
        }
        
        treys_cards = []
        
        for card in cards:
            if len(card) != 2:
                self.logger.warning(f"Invalid card format: {card}")
                continue
                
            rank = card[1]
            suit = card[0]
            
            if suit not in suit_map:
                self.logger.warning(f"Invalid suit in card: {card}")
                continue
                
            treys_card = rank + suit_map[suit]
            treys_cards.append(Card.new(treys_card))
        
        return treys_cards
    
    def get_hole_cards_rank(self, hole_cards):
        """
        Get a ranking for hole cards (preflop).
        
        Args:
            hole_cards (list): List of hole cards in poker engine format
            
        Returns:
            float: Ranking value between 0 and 1 (1 is best)
        """
        # Convert to string key for caching
        hole_key = ''.join(sorted(hole_cards))
        
        # Return cached value if available
        if hole_key in self.preflop_rankings:
            return self.preflop_rankings[hole_key]
        
        # If we have exactly two hole cards, use direct evaluation
        if len(hole_cards) == 2:
            # Check for pairs
            if hole_cards[0][1] == hole_cards[1][1]:
                # Rank pairs (AA is best, 22 is worst)
                rank_map = {'A': 14, 'K': 13, 'Q': 12, 'J': 11, 'T': 10, 
                           '9': 9, '8': 8, '7': 7, '6': 6, '5': 5, '4': 4, '3': 3, '2': 2}
                
                rank = rank_map.get(hole_cards[0][1], 2)  # Default to 2 if unknown
                # Scale pairs (AA=1.0, 22=0.5)
                ranking = 0.5 + (rank - 2) / 24
            else:
                # Not a pair - consider other factors
                rank_map = {'A': 14, 'K': 13, 'Q': 12, 'J': 11, 'T': 10, 
                           '9': 9, '8': 8, '7': 7, '6': 6, '5': 5, '4': 4, '3': 3, '2': 2}
                
                # Get numerical ranks
                rank1 = rank_map.get(hole_cards[0][1], 2)
                rank2 = rank_map.get(hole_cards[1][1], 2)
                
                # Suited bonus
                suited_bonus = 0.1 if hole_cards[0][0] == hole_cards[1][0] else 0
                
                # Connected bonus (closer ranks get higher bonus)
                rank_diff = abs(rank1 - rank2)
                connected_bonus = max(0, 0.08 - (rank_diff - 1) * 0.02)
                
                # High card value
                high_card_value = (max(rank1, rank2) - 2) / 12 * 0.3  # Scale high card (0-0.3)
                
                # Combined ranking
                ranking = 0.25 + high_card_value + suited_bonus + connected_bonus
                
                # Cap at just under pair of 2s
                ranking = min(ranking, 0.49)
        else:
            # If we don't have exactly two cards, use a default ranking
            ranking = 0.25  # Below average hand
            self.logger.warning(f"Expected 2 hole cards, got {len(hole_cards)}")
        
        # Cache the result
        self.preflop_rankings[hole_key] = ranking
        
        return ranking
    
    def calc_hand_strength(self, hole_cards, community_cards=[]):
        """
        Calculate hand strength based on current cards.
        
        Args:
            hole_cards (list): List of hole cards in poker engine format
            community_cards (list): List of community cards in poker engine format
            
        Returns:
            tuple: (hand_strength, hand_class)
                - hand_strength: Value between 0-1 representing hand strength
                - hand_class: String describing the hand type
        """
        # Convert cards to treys format
        treys_hole = self._convert_to_treys_format(hole_cards)
        treys_community = self._convert_to_treys_format(community_cards)
        
        # Handle preflop case (no community cards)
        if not community_cards:
            hand_strength = self.get_hole_cards_rank(hole_cards)
            
            # Determine hand class for preflop
            if len(hole_cards) == 2 and hole_cards[0][1] == hole_cards[1][1]:
                hand_class = f"Pair of {hole_cards[0][1]}s"
            elif len(hole_cards) == 2 and hole_cards[0][0] == hole_cards[1][0]:
                hand_class = f"Suited Cards ({hole_cards[0][1]}-{hole_cards[1][1]})"
            elif len(hole_cards) == 2:
                hand_class = f"Offsuit Cards ({hole_cards[0][1]}-{hole_cards[1][1]})"
            else:
                hand_class = "Unknown Preflop Hand"
            
            return hand_strength, hand_class
        
        # Handle postflop with treys evaluator
        hand_rank = self.evaluator.evaluate(treys_community, treys_hole)
        hand_class_id = self.evaluator.get_rank_class(hand_rank)
        hand_class = self.evaluator.class_to_string(hand_class_id)
        
        # Convert rank to strength (lower rank is better in treys)
        # Perfect hand (Royal Flush) has rank 1, worst has rank 7462
        hand_strength = 1 - (hand_rank - 1) / 7461
        
        return hand_strength, hand_class
    
    def calc_hand_potential(self, hole_cards, community_cards):
        """
        Calculate hand potential by simulating future cards.
        
        This method estimates how likely a hand is to improve or deteriorate
        with future community cards.
        
        Args:
            hole_cards (list): List of hole cards in poker engine format
            community_cards (list): List of community cards in poker engine format
            
        Returns:
            tuple: (positive_potential, negative_potential, equity)
                - positive_potential: Probability of improving (0-1)
                - negative_potential: Probability of deteriorating (0-1)
                - equity: Overall equity in the hand (0-1)
        """
        if len(community_cards) >= 5:
            # River - no more potential
            hand_strength, _ = self.calc_hand_strength(hole_cards, community_cards)
            return 0.0, 0.0, hand_strength
        
        # Convert cards to treys format
        treys_hole = self._convert_to_treys_format(hole_cards)
        treys_community = self._convert_to_treys_format(community_cards)
        
        # Get current hand strength
        current_strength, _ = self.calc_hand_strength(hole_cards, community_cards)
        
        # Define relevant variables
        ahead_count = 0
        behind_count = 0
        ahead_improved = 0
        behind_improved = 0
        
        # Create deck without known cards
        all_cards = [
            '2S', '3S', '4S', '5S', '6S', '7S', '8S', '9S', 'TS', 'JS', 'QS', 'KS', 'AS',
            '2H', '3H', '4H', '5H', '6H', '7H', '8H', '9H', 'TH', 'JH', 'QH', 'KH', 'AH',
            '2D', '3D', '4D', '5D', '6D', '7D', '8D', '9D', 'TD', 'JD', 'QD', 'KD', 'AD',
            '2C', '3C', '4C', '5C', '6C', '7C', '8C', '9C', 'TC', 'JC', 'QC', 'KC', 'AC'
        ]
        
        # Remove known cards
        for card in hole_cards + community_cards:
            if card in all_cards:
                all_cards.remove(card)
        
        # For computational efficiency, limit to a random sample if too many combinations
        remaining_cards_needed = 5 - len(community_cards)
        opponent_hole_combinations = min(100, len(list(itertools.combinations(all_cards, 2))))
        future_board_combinations = min(100, len(list(itertools.combinations(all_cards, remaining_cards_needed))))
        
        # Sample opponents' hole cards
        opponent_hands = []
        for _ in range(opponent_hole_combinations):
            opponent_hole = np.random.choice(all_cards, 2, replace=False).tolist()
            opponent_hands.append(opponent_hole)
        
        # For each opponent hand
        for opp_hole in opponent_hands:
            # Convert opponent cards
            treys_opp = self._convert_to_treys_format(opp_hole)
            
            # Remove opponent cards from deck for future community cards
            remaining_cards = [card for card in all_cards if card not in opp_hole]
            
            # Evaluate current hand vs opponent
            opp_rank = self.evaluator.evaluate(treys_community, treys_opp)
            my_rank = self.evaluator.evaluate(treys_community, treys_hole)
            
            # Determine ahead/behind
            if my_rank < opp_rank:  # Lower rank is better in treys
                ahead_count += 1
                ahead_status = True
            else:
                behind_count += 1
                ahead_status = False
            
            # Simulate future boards
            for _ in range(future_board_combinations):
                future_cards = np.random.choice(remaining_cards, remaining_cards_needed, replace=False).tolist()
                future_board = community_cards + future_cards
                treys_board = self._convert_to_treys_format(future_board)
                
                # Evaluate final hands
                final_opp_rank = self.evaluator.evaluate(treys_board, treys_opp)
                final_my_rank = self.evaluator.evaluate(treys_board, treys_hole)
                
                # Check if situation changed
                if ahead_status:  # Was ahead
                    if final_my_rank > final_opp_rank:  # Now behind
                        behind_improved += 1
                else:  # Was behind
                    if final_my_rank < final_opp_rank:  # Now ahead
                        ahead_improved += 1
        
        # Calculate potentials
        total_ahead = ahead_count * future_board_combinations
        total_behind = behind_count * future_board_combinations
        
        positive_potential = ahead_improved / total_behind if total_behind > 0 else 0
        negative_potential = behind_improved / total_ahead if total_ahead > 0 else 0
        
        # Calculate equity
        equity = (ahead_count + (positive_potential * behind_count)) / (ahead_count + behind_count)
        
        return positive_potential, negative_potential, equity
    
    def calc_effective_hand_strength(self, hole_cards, community_cards):
        """
        Calculate Effective Hand Strength (EHS) combining current strength and potential.
        
        EHS = HandStrength + (1 - HandStrength) * PositivePotential
        
        Args:
            hole_cards (list): List of hole cards in poker engine format
            community_cards (list): List of community cards in poker engine format
            
        Returns:
            float: Effective Hand Strength value between 0-1
        """
        # Get current hand strength
        current_strength, _ = self.calc_hand_strength(hole_cards, community_cards)
        
        # If we're at the river, no need to calculate potential
        if len(community_cards) >= 5:
            return current_strength
            
        # Calculate potential
        pp, np, equity = self.calc_hand_potential(hole_cards, community_cards)
        
        # Calculate EHS
        ehs = current_strength + (1 - current_strength) * pp
        
        return ehs
    
    def make_decision(self, hole_cards, community_cards, pot_odds):
        """
        Suggest a poker decision based on hand strength and pot odds.
        
        Args:
            hole_cards (list): List of hole cards in poker engine format
            community_cards (list): List of community cards in poker engine format
            pot_odds (float): Current pot odds (call amount / total pot after call)
            
        Returns:
            tuple: (action, confidence, reasoning)
                - action: Suggested action ('fold', 'call', 'raise')
                - confidence: Confidence in the decision (0-1)
                - reasoning: Explanation for the decision
        """
        # Calculate effective hand strength
        ehs = self.calc_effective_hand_strength(hole_cards, community_cards)
        _, hand_class = self.calc_hand_strength(hole_cards, community_cards)
        
        # Get stage
        if not community_cards:
            stage = "preflop"
        elif len(community_cards) == 3:
            stage = "flop"
        elif len(community_cards) == 4:
            stage = "turn"
        else:
            stage = "river"
        
        # Compare pot odds to hand strength
        if ehs >= pot_odds * 1.5:  # Much better than pot odds - raise
            action = "raise"
            confidence = min(1.0, ehs / pot_odds - 0.5)
            reasoning = f"Strong {hand_class} with EHS {ehs:.2f}, significantly better than pot odds {pot_odds:.2f}"
        elif ehs >= pot_odds:  # Better than pot odds - call
            action = "call"
            confidence = min(1.0, ehs / pot_odds)
            reasoning = f"{hand_class} with EHS {ehs:.2f} justifies a call with pot odds {pot_odds:.2f}"
        else:  # Worse than pot odds - fold
            action = "fold"
            confidence = min(1.0, 1.0 - (ehs / pot_odds))
            reasoning = f"Weak {hand_class} with EHS {ehs:.2f} doesn't justify calling with pot odds {pot_odds:.2f}"
        
        # Special preflop adjustments
        if stage == "preflop":
            if ehs > 0.6:  # Premium preflop hand
                action = "raise"
                confidence = ehs
                reasoning = f"Premium preflop hand ({hand_class}) with strength {ehs:.2f}"
            elif ehs > 0.45 and action == "fold":  # Speculative hand but too good to fold
                action = "call"
                confidence = ehs
                reasoning = f"Speculative preflop hand ({hand_class}) with strength {ehs:.2f}, worth seeing the flop"
        
        return action, confidence, reasoning
    
    def estimate_win_probability(self, hole_cards, community_cards, num_opponents=1, num_simulations=1000):
        """
        Estimate win probability through Monte Carlo simulation.
        
        Args:
            hole_cards (list): List of hole cards in poker engine format
            community_cards (list): List of community cards in poker engine format
            num_opponents (int): Number of opponents to simulate
            num_simulations (int): Number of simulations to run
            
        Returns:
            float: Estimated probability of winning (0-1)
        """
        wins = 0
        ties = 0
        
        # Convert to treys format
        treys_hole = self._convert_to_treys_format(hole_cards)
        treys_community = self._convert_to_treys_format(community_cards)
        
        # Create deck without known cards
        all_cards = [
            '2S', '3S', '4S', '5S', '6S', '7S', '8S', '9S', 'TS', 'JS', 'QS', 'KS', 'AS',
            '2H', '3H', '4H', '5H', '6H', '7H', '8H', '9H', 'TH', 'JH', 'QH', 'KH', 'AH',
            '2D', '3D', '4D', '5D', '6D', '7D', '8D', '9D', 'TD', 'JD', 'QD', 'KD', 'AD',
            '2C', '3C', '4C', '5C', '6C', '7C', '8C', '9C', 'TC', 'JC', 'QC', 'KC', 'AC'
        ]
        
        # Remove known cards
        for card in hole_cards + community_cards:
            if card in all_cards:
                all_cards.remove(card)
        
        # Calculate how many more community cards needed
        remaining_cards_needed = 5 - len(community_cards)
        
        # Run simulations
        for _ in range(num_simulations):
            # Shuffle remaining cards
            np.random.shuffle(all_cards)
            remaining_deck = all_cards.copy()
            
            # Deal opponent hands
            opponent_hands = []
            for _ in range(num_opponents):
                opponent_hole = [remaining_deck.pop(), remaining_deck.pop()]
                opponent_hands.append(self._convert_to_treys_format(opponent_hole))
            
            # Deal remaining community cards
            additional_community = []
            for _ in range(remaining_cards_needed):
                additional_community.append(self._convert_to_treys_format([remaining_deck.pop()])[0])
            
            # Complete the board
            complete_board = treys_community + additional_community
            
            # Evaluate all hands
            my_rank = self.evaluator.evaluate(complete_board, treys_hole)
            opponent_ranks = [self.evaluator.evaluate(complete_board, opp_hand) for opp_hand in opponent_hands]
            
            # Determine winner (lower rank is better in treys)
            best_opponent = min(opponent_ranks) if opponent_ranks else float('inf')
            
            if my_rank < best_opponent:
                wins += 1
            elif my_rank == best_opponent:
                ties += 0.5  # Count ties as half a win
        
        return (wins + ties) / num_simulations
    
    def generate_starting_hand_rankings(self, num_simulations=10000):
        """
        Generate rankings for all 169 starting hands in Texas Hold'em.
        This is a time-consuming operation but results can be cached.
        
        Args:
            num_simulations (int): Number of simulations per hand
            
        Returns:
            dict: Dictionary of hand rankings
        """
        # All possible ranks
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        
        # Generate all 169 starting hands (13 pairs, 78 suited, 78 unsuited)
        hand_rankings = {}
        self.logger.info("Generating starting hand rankings - this may take a while...")
        
        # Generate all possible 2-card combinations
        for i, rank1 in enumerate(ranks):
            for j, rank2 in enumerate(ranks[i:], i):  # Start from i to avoid duplicates
                # Pair
                if rank1 == rank2:
                    hand = [f'H{rank1}', f'D{rank1}']
                    win_prob = self.estimate_win_probability(hand, [], num_opponents=1, num_simulations=num_simulations)
                    key = f"{rank1}{rank1}"
                    hand_rankings[key] = win_prob
                    self.logger.info(f"Generated ranking for {key}: {win_prob:.4f}")
                else:
                    # Suited
                    suited_hand = [f'H{rank1}', f'H{rank2}']
                    suited_win_prob = self.estimate_win_probability(suited_hand, [], num_opponents=1, num_simulations=num_simulations)
                    suited_key = f"{rank1}{rank2}s"
                    hand_rankings[suited_key] = suited_win_prob
                    self.logger.info(f"Generated ranking for {suited_key}: {suited_win_prob:.4f}")
                    
                    # Unsuited
                    unsuited_hand = [f'H{rank1}', f'D{rank2}']
                    unsuited_win_prob = self.estimate_win_probability(unsuited_hand, [], num_opponents=1, num_simulations=num_simulations)
                    unsuited_key = f"{rank1}{rank2}o"
                    hand_rankings[unsuited_key] = unsuited_win_prob
                    self.logger.info(f"Generated ranking for {unsuited_key}: {unsuited_win_prob:.4f}")
        
        return hand_rankings