# ai/Abel/utils/decision_evaluator.py
import numpy as np
import json
import os
from datetime import datetime

class DecisionEvaluator:
    """
    Evaluates the quality of poker decisions and provides analytics.
    """
    def __init__(self, log_dir="logs/decisions"):
        self.log_dir = log_dir
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.decisions_file = os.path.join(log_dir, f"decision_eval_{self.session_id}.json")
        
        # Create log directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        self.decisions = []
        self.interesting_hands = []
    
    def evaluate_decision(self, player_id, hand_id, street, hole_cards, community_cards, 
                          pot_size, valid_actions, chosen_action, hand_strength=None, 
                          action_values=None, is_bluff=False):
        """
        Evaluate a poker decision and record metrics.
        
        Args:
            player_id: ID of the player making the decision
            hand_id: ID of the current hand
            street: Current street (preflop, flop, turn, river)
            hole_cards: Player's hole cards
            community_cards: Community cards
            pot_size: Current pot size
            valid_actions: List of valid actions
            chosen_action: Action selected by player
            hand_strength: Evaluated hand strength (0-1)
            action_values: Q-values or evaluation for each action
            is_bluff: Whether this action was classified as a bluff
        
        Returns:
            Dictionary containing decision evaluation metrics
        """
        # Calculate hand strength if not provided
        if hand_strength is None:
            hand_strength = self._quick_hand_strength(hole_cards, community_cards)
        
        # Create decision record
        decision = {
            'player_id': player_id,
            'hand_id': hand_id,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'street': street,
            'hole_cards': hole_cards,
            'community_cards': community_cards,
            'pot_size': pot_size,
            'valid_actions': valid_actions,
            'chosen_action': chosen_action,
            'hand_strength': hand_strength,
            'is_bluff': is_bluff
        }
        
        # Add action values if available
        if action_values is not None:
            decision['action_values'] = action_values
        
        # Evaluate decision quality (simplified heuristic)
        decision_quality = self._evaluate_quality(chosen_action, hand_strength, pot_size, is_bluff)
        decision['quality_score'] = decision_quality
        
        # Flag interesting decisions
        if self._is_interesting_decision(decision):
            decision['interesting'] = True
            self.interesting_hands.append(hand_id)
        
        # Store decision
        self.decisions.append(decision)
        
        # Save periodically
        if len(self.decisions) % 100 == 0:
            self._save_to_file()
        
        return decision
    
    def _quick_hand_strength(self, hole_cards, community_cards):
        """
        Simplified poker hand strength evaluation (0-1 scale).
        This is a very basic approximation - for real analysis, use a proper poker hand evaluator.
        """
        # Count high cards (10, J, Q, K, A)
        high_cards = 0
        for card in hole_cards:
            if card[0] in ['T', 'J', 'Q', 'K', 'A']:
                high_cards += 1
        
        # Check for pairs in hole cards
        has_pair = hole_cards[0][0] == hole_cards[1][0] if len(hole_cards) >= 2 else False
        
        # Very simplified strength calculation
        strength = 0.3  # Base strength
        
        # Adjust for high cards
        strength += high_cards * 0.15
        
        # Adjust for pairs
        if has_pair:
            strength += 0.3
        
        # Normalize to 0-1
        return min(1.0, max(0.0, strength))
    
    def _evaluate_quality(self, action, hand_strength, pot_size, is_bluff):
        """
        Simple heuristic to evaluate decision quality.
        Returns a score from 0 (poor) to 1 (excellent).
        """
        # Base quality score
        quality = 0.5
        
        # Strong hand heuristics
        if hand_strength > 0.7:
            if action == 'fold':
                quality -= 0.4  # Bad to fold strong hands
            elif action == 'raise':
                quality += 0.3  # Good to raise strong hands
        
        # Weak hand heuristics
        elif hand_strength < 0.3:
            if action == 'raise' and not is_bluff:
                quality -= 0.3  # Bad to raise weak hands unless bluffing
            elif action == 'fold':
                quality += 0.2  # Often good to fold weak hands
        
        # Bluffing quality
        if is_bluff:
            if hand_strength > 0.5:
                quality -= 0.2  # Bad to bluff with decent hands
            elif pot_size > 100:
                quality += 0.2  # Bluffing for big pots can be good
        
        # Normalize quality score
        return min(1.0, max(0.0, quality))
    
    def _is_interesting_decision(self, decision):
        """
        Determine if a decision is particularly interesting for analysis.
        """
        # Bluffs are always interesting
        if decision.get('is_bluff', False):
            return True
        
        # Folding strong hands is interesting
        if decision['chosen_action'] == 'fold' and decision['hand_strength'] > 0.7:
            return True
        
        # Raising very weak hands (not marked as bluffs) is interesting
        if decision['chosen_action'] == 'raise' and decision['hand_strength'] < 0.2 and not decision.get('is_bluff', False):
            return True
        
        # Large pot decisions are interesting
        if decision['pot_size'] > 200:
            return True
        
        return False
    
    def _save_to_file(self):
        """Save current decisions data to file."""
        with open(self.decisions_file, 'w') as f:
            json.dump({
                'decisions': self.decisions,
                'interesting_hands': list(set(self.interesting_hands))
            }, f, indent=2)
    
    def save(self):
        """Manually save decisions data to file."""
        self._save_to_file()
    
    def get_decisions(self):
        """Return all recorded decisions."""
        return self.decisions
    
    def get_interesting_hands(self):
        """Return list of hand IDs flagged as interesting."""
        return list(set(self.interesting_hands))
    
    def get_decision_stats(self):
        """
        Return statistics about decisions.
        """
        if not self.decisions:
            return {}
        
        stats = {
            'total_decisions': len(self.decisions),
            'actions': {
                'fold': 0,
                'call': 0,
                'raise': 0
            },
            'bluff_count': 0,
            'avg_quality': 0,
            'by_street': {
                'preflop': {'count': 0, 'avg_quality': 0},
                'flop': {'count': 0, 'avg_quality': 0},
                'turn': {'count': 0, 'avg_quality': 0},
                'river': {'count': 0, 'avg_quality': 0}
            }
        }
        
        # Calculate statistics
        total_quality = 0
        street_quality = {'preflop': 0, 'flop': 0, 'turn': 0, 'river': 0}
        
        for decision in self.decisions:
            # Count actions
            action = decision.get('chosen_action')
            if action in stats['actions']:
                stats['actions'][action] += 1
            
            # Count bluffs
            if decision.get('is_bluff', False):
                stats['bluff_count'] += 1
            
            # Sum quality scores
            quality = decision.get('quality_score', 0)
            total_quality += quality
            
            # By street
            street = decision.get('street', 'preflop')
            if street in stats['by_street']:
                stats['by_street'][street]['count'] += 1
                street_quality[street] += quality
        
        # Calculate averages
        stats['avg_quality'] = total_quality / len(self.decisions)
        
        for street in street_quality:
            if stats['by_street'][street]['count'] > 0:
                stats['by_street'][street]['avg_quality'] = street_quality[street] / stats['by_street'][street]['count']
        
        # Calculate action percentages
        for action in stats['actions']:
            stats['actions'][f'{action}_pct'] = stats['actions'][action] / len(self.decisions)
        
        # Bluff percentage
        stats['bluff_percentage'] = stats['bluff_count'] / len(self.decisions)
        
        return stats