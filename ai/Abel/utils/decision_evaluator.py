import numpy as np
import json
import os
from datetime import datetime

class DecisionEvaluator:
    def __init__(self, log_dir="logs/decisions"):
        self.log_dir = log_dir
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.decisions_file = os.path.join(log_dir, f"decision_eval_{self.session_id}.json")

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.decisions = []
        self.interesting_hands = []

        def record_decision(self, decision):
            """
            Allows direct logging of decision dictionaries.
            Useful when full evaluation is done outside or beforehand.
            """
            self.decisions.append(decision)

            if decision.get('is_bluff', False):
                self.interesting_hands.append(decision.get('hand_id', f"unknown-{len(self.decisions)}"))

            # Optional auto-save
            if len(self.decisions) % 100 == 0:
                self._save_to_file()

            print(f"📥 Recorded decision: {decision['chosen_action']} | Strength: {decision['hand_strength']}")


        # Optional auto-save every 100 decisions
        if len(self.decisions) % 100 == 0:
            self._save_to_file()


    def evaluate_decision(self, player_id, hand_id, street, hole_cards, community_cards, 
                          pot_size, valid_actions, chosen_action, hand_strength=None, 
                          action_values=None, is_bluff=False):
        if hand_strength is None:
            hand_strength = self._quick_hand_strength(hole_cards, community_cards)

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

        if action_values is not None:
            decision['action_values'] = action_values

        decision_quality = self._evaluate_quality(chosen_action, hand_strength, pot_size, is_bluff)
        decision['quality_score'] = decision_quality

        if self._is_interesting_decision(decision):
            decision['interesting'] = True
            self.interesting_hands.append(hand_id)

        self.decisions.append(decision)

        if len(self.decisions) % 100 == 0:
            self._save_to_file()

        return decision

    def _quick_hand_strength(self, hole_cards, community_cards):
        high_cards = 0
        for card in hole_cards:
            if card[0] in ['T', 'J', 'Q', 'K', 'A']:
                high_cards += 1
        has_pair = hole_cards[0][0] == hole_cards[1][0] if len(hole_cards) >= 2 else False
        strength = 0.3 + high_cards * 0.15
        if has_pair:
            strength += 0.3
        return min(1.0, max(0.0, strength))

    def _evaluate_quality(self, action, hand_strength, pot_size, is_bluff):
        quality = 0.5
        if hand_strength > 0.7:
            if action == 'fold':
                quality -= 0.4
            elif action == 'raise':
                quality += 0.3
        elif hand_strength < 0.3:
            if action == 'raise' and not is_bluff:
                quality -= 0.3
            elif action == 'fold':
                quality += 0.2
        if is_bluff:
            if hand_strength > 0.5:
                quality -= 0.2
            elif pot_size > 100:
                quality += 0.2
        return min(1.0, max(0.0, quality))

    def _is_interesting_decision(self, decision):
        if decision.get('is_bluff', False):
            return True
        if decision['chosen_action'] == 'fold' and decision['hand_strength'] > 0.7:
            return True
        if decision['chosen_action'] == 'raise' and decision['hand_strength'] < 0.2 and not decision.get('is_bluff', False):
            return True
        if decision['pot_size'] > 200:
            return True
        return False

    def _save_to_file(self):
        with open(self.decisions_file, 'w') as f:
            json.dump({
                'decisions': self.decisions,
                'interesting_hands': list(set(self.interesting_hands))
            }, f, indent=2)

    def save(self):
        self._save_to_file()

    def get_decisions(self):
        return self.decisions

    def get_interesting_hands(self):
        return list(set(self.interesting_hands))

    def get_decision_stats(self):
        if not self.decisions:
            return {}

        stats = {
            'total_decisions': len(self.decisions),
            'actions': {'fold': 0, 'call': 0, 'raise': 0},
            'bluff_count': 0,
            'avg_quality': 0,
            'by_street': {
                'preflop': {'count': 0, 'avg_quality': 0},
                'flop': {'count': 0, 'avg_quality': 0},
                'turn': {'count': 0, 'avg_quality': 0},
                'river': {'count': 0, 'avg_quality': 0}
            }
        }

        total_quality = 0
        street_quality = {'preflop': 0, 'flop': 0, 'turn': 0, 'river': 0}

        for decision in self.decisions:
            action = decision.get('chosen_action')
            if action in stats['actions']:
                stats['actions'][action] += 1
            if decision.get('is_bluff', False):
                stats['bluff_count'] += 1
            quality = decision.get('quality_score', 0)
            total_quality += quality
            street = decision.get('street', 'preflop')
            if street in stats['by_street']:
                stats['by_street'][street]['count'] += 1
                street_quality[street] += quality

        stats['avg_quality'] = total_quality / len(self.decisions)

        for street in street_quality:
            count = stats['by_street'][street]['count']
            if count > 0:
                stats['by_street'][street]['avg_quality'] = street_quality[street] / count

        for action in stats['actions']:
            stats['actions'][f'{action}_pct'] = stats['actions'][action] / len(self.decisions)

        stats['bluff_percentage'] = stats['bluff_count'] / len(self.decisions)

        return stats

    def reset(self):
        """
        Resets internal decision buffer and saves current session.
        """
        self._save_to_file()
        self.decisions = []
        self.interesting_hands = []
