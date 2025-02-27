from abc import ABC, abstractmethod

class State(ABC):  # Renamed from PokerState to State
    """
    Abstract Base Class for Poker States.
    All states must implement this interface.
    """

    @abstractmethod
    def enter_state(self, player, round_state):
        pass

    @abstractmethod
    def decide_action(self, player, valid_actions, hole_card, round_state):
        pass

    @abstractmethod
    def next_state(self, round_state):
        pass
