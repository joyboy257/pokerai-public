import os
import numpy as np
import random
import logging
from collections import deque
from pypokerengine.players import BasePokerPlayer
from ai.Abel.models.DQN import DQN
from ai.Abel.agents.CFRPlayer import CFRPlayer  # Import CFR module
from ai.Abel.utils.state_encoder import encode_state  # Import state encoding

class RLBasedPlayer(BasePokerPlayer):
    def __init__(self, state_size, action_size, model_path="models/abel_model.h5"):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 128
        self.memory = deque(maxlen=50000)
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.target_model.model.set_weights(self.model.model.get_weights())
        self.model_path = model_path
        self.cfr_player = CFRPlayer()  # CFR for bluffing
        self.opponent_history = []  # Tracks opponent actions

        logging.info("Initializing RLBasedPlayer...")
        self.load_model()

    def save_model(self):
        """ Saves the model to the specified file. """
        self.model.model.save(self.model_path)
        logging.info(f"Model saved at {self.model_path}")

    def load_model(self):
        """ Loads the model from the specified file if available. """
        if os.path.exists(self.model_path):
            self.model.model.load_weights(self.model_path)
            self.target_model.model.load_weights(self.model_path)
            logging.info(f"Model loaded successfully from {self.model_path}")
        else:
            logging.info("No saved model found. Training from scratch.")

    def remember(self, state, action, reward, next_state, done):
        """ Stores experience in memory buffer for replay. """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, valid_actions):
        """ Chooses an action using an epsilon-greedy policy (DQN). """
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_actions)
        state = np.reshape(state, [1, self.state_size])
        q_values = self.model.model.predict(state, verbose=0)[0]
        action_idx = np.argmax(q_values[:len(valid_actions)])
        return valid_actions[action_idx]

    def replay(self):
        """ Trains the model using experience replay. """
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(
                    self.target_model.model.predict(np.reshape(next_state, [1, self.state_size]), verbose=0)[0]
                )
            state = np.reshape(state, [1, self.state_size])
            target_f = self.model.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.model.fit(state, target_f, epochs=1, verbose=0)
        
        # Update target network
        self.target_model.model.set_weights(self.model.model.get_weights())

    def adjust_epsilon(self):
        """ Reduces exploration rate over time. """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def declare_action(self, valid_actions, hole_card, round_state):
        """ Uses CFR for bluffing and DQN for tactical play. """
        logging.info(f"Valid actions: {valid_actions}")
        state = encode_state(hole_card, round_state)  # Proper game state encoding
        state_key = str(round_state)  # Simplified state representation
        action_idx = None

        # **CFR for bluffing in early rounds**
        if len(round_state["community_card"]) <= 3:
            chosen_action = self.cfr_player.select_action(state_key, [a["action"] for a in valid_actions])
        else:
            # **DQN for postflop play**
            chosen_action = self.act(state, [a["action"] for a in valid_actions])
            action_idx = [a["action"] for a in valid_actions].index(chosen_action)

        amount = 0
        for action in valid_actions:
            if action["action"] == chosen_action:
                amount = action.get("amount", 0)
                if isinstance(amount, dict):  # Handle min-max raises
                    amount = amount["min"]
                break

        # **Track opponent actions for strategic adaptation**
        self.opponent_history.append(round_state["action_histories"])

        # **Update CFR regrets**
        if action_idx is not None:
            self.cfr_player.update_regrets(state_key, action_idx, reward=0)  # Reward to be calculated after round

        logging.info(f"Abel chose action: {chosen_action} with amount {amount}")
        return chosen_action, amount
