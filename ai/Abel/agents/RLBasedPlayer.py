# ai/Abel/agents/RLBasedPlayer.py
import os
import numpy as np
import random
import logging
from collections import deque
from pypokerengine.players import BasePokerPlayer
from ai.Abel.models.DQN import DQN

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
        """ Chooses an action using an epsilon-greedy policy. """
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
        """ Implements required method for PyPokerEngine. """
        logging.info(f"Valid actions: {valid_actions}")
        state = np.zeros(self.state_size)  # Placeholder for now, proper encoding needed
        chosen_action = self.act(state, [action["action"] for action in valid_actions])
        amount = 0
        for action in valid_actions:
            if action["action"] == chosen_action:
                amount = action.get("amount", 0)
                if isinstance(amount, dict):  # Handle min-max raises
                    amount = amount["min"]
                break
        logging.info(f"Abel chose action: {chosen_action} with amount {amount}")
        return chosen_action, amount

    def receive_game_start_message(self, game_info):
        """ Handles game start event (needed for PyPokerEngine). """
        logging.info(f"Game started with info: {game_info}")

    def receive_round_start_message(self, round_count, hole_card, seats):
        """ Handles round start event (needed for PyPokerEngine). """
        logging.info(f"Round {round_count} started with hole cards: {hole_card}")

    def receive_street_start_message(self, street, round_state):
        """ Handles new betting street event (needed for PyPokerEngine). """
        logging.info(f"New street: {street}")

    def receive_game_update_message(self, action, round_state):
        """ Handles game update event (needed for PyPokerEngine). """
        logging.info(f"Game updated with action: {action}")

    def receive_round_result_message(self, winners, hand_info, round_state):
        """ Handles round result event (needed for PyPokerEngine). """
        logging.info(f"Round ended. Winners: {winners}")
