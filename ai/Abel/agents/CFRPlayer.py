import numpy as np
import random
from collections import defaultdict

class CFRPlayer:
    def __init__(self, actions=["fold", "call", "raise"]):
        self.actions = actions
        self.regrets = defaultdict(lambda: np.zeros(len(actions)))
        self.strategy = defaultdict(lambda: np.ones(len(actions)) / len(actions))  # Uniform strategy
        self.strategy_sum = defaultdict(lambda: np.zeros(len(actions)))
        self.opp_action_history = []  # Stores opponent betting patterns

    def get_strategy(self, state_key):
        """ Compute strategy using regret matching """
        regrets = self.regrets[state_key]
        positive_regrets = np.maximum(regrets, 0)
        normalizing_sum = np.sum(positive_regrets)

        if normalizing_sum > 0:
            self.strategy[state_key] = positive_regrets / normalizing_sum
        else:
            self.strategy[state_key] = np.ones(len(self.actions)) / len(self.actions)  # Uniform strategy

        return self.strategy[state_key]

    def update_regrets(self, state_key, action_idx, utility):
        """ Update regret values based on counterfactual utility """
        strategy = self.get_strategy(state_key)
        for i in range(len(self.actions)):
            regret = utility if i == action_idx else 0
            self.regrets[state_key][i] += regret - strategy[i] * utility
            self.strategy_sum[state_key][i] += strategy[i]

    def select_action(self, state_key, valid_actions):
        """ Select action using computed strategy probabilities """
        strategy = self.get_strategy(state_key)
        action_idx = np.random.choice(len(valid_actions), p=strategy[:len(valid_actions)])
        return valid_actions[action_idx]

    def get_average_strategy(self, state_key):
        """ Return the averaged strategy across training iterations """
        normalizing_sum = np.sum(self.strategy_sum[state_key])
        if normalizing_sum > 0:
            return self.strategy_sum[state_key] / normalizing_sum
        else:
            return np.ones(len(self.actions)) / len(self.actions)  # Uniform

    def track_opponent_action(self, action):
        """ Store opponent action history for bluffing decisions """
        self.opp_action_history.append(action)

