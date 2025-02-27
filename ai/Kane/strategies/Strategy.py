from abc import ABC, abstractmethod

class Strategy(ABC):
    """
    Abstract Base Class for Strategies.
    All strategies must implement this interface.
    """

    @abstractmethod
    def decide_action(self, player, valid_actions, hole_card, round_state):
        """
        Decide the action to take.
        Must be implemented by all derived strategies.
        """
        pass
