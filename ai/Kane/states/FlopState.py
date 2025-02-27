from ai.Kane.states.PokerState import State
import logging

class FlopState(State):
    def enter_state(self, player, round_state):
        logging.info("Entering Flop State")
        print("Entering Flop State")

    def decide_action(self, player, valid_actions, hole_card, round_state):
        community_cards = round_state.get("community_card", [])
        hand_strength, hand_class = player.evaluate_hand_strength(hole_card, community_cards)

        if hand_strength > 0.7:  # Strong hand after flop
            action = "raise"
            amount = valid_actions[2]['amount']['max']  # Max raise
        elif hand_strength > 0.4:  # Medium hand
            action = "call"
            amount = valid_actions[1]['amount']  # Match the bet
        else:  # Weak hand
            action = "fold"
            amount = 0

        logging.info(
            f"Flop Decision: Hole={hole_card}, Community={community_cards}, "
            f"Hand Class={hand_class}, Hand Strength={hand_strength:.2f}, "
            f"Action={action}, Amount={amount}"
        )
        
        return action, amount

    def next_state(self, round_state):
        community_cards = round_state.get("community_card", [])
        if len(community_cards) == 3:
            return "Flop"
        elif len(community_cards) == 4:
            from ai.Kane.states.TurnState import TurnState
            return TurnState()
        elif len(community_cards) == 5:
            from ai.Kane.states.RiverState import RiverState
            return RiverState()
        else:
            return None
