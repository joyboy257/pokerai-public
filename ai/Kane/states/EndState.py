from ai.Kane.states.PokerState import State

class EndState(State):
    def enter_state(self, player, round_state):
        print("Entering End State")

    def decide_action(self, player, valid_actions, hole_card, round_state):
        return "fold", 0  # End of the round, fold to end the game

    def next_state(self, round_state):
        from ai.Kane.states.PreflopState import PreflopState
        return PreflopState()  # Reset to Preflop for new round
