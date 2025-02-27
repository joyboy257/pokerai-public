from ai.Kane.states.PokerState import State

class PostflopState(State):
    def enter_state(self, player, round_state):
        print("Entering Postflop State")

    def decide_action(self, player, valid_actions, hole_card, round_state):
        community_cards = round_state.get("community_card", [])
        hand_strength, hand_class = player.evaluate_hand_strength(hole_card, community_cards)
        
        # Basic Postflop Strategy
        if hand_strength > 0.7:  # Strong hand
            action = "raise"
            amount = valid_actions[2]['amount']['max']  # Max raise
        elif hand_strength > 0.4:  # Medium hand
            action = "call"
            amount = valid_actions[1]['amount']  # Match the bet
        else:  # Weak hand
            action = "fold"
            amount = 0

        return action, amount

    def next_state(self, round_state):
        community_cards = round_state.get("community_card", [])
        if len(community_cards) == 4:
            from ai.Kane.states.TurnState import TurnState
            return TurnState()
        return self
