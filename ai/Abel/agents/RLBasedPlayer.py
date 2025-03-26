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
    def __init__(self, state_size, action_size, model_path="models/abel_model.h5",
                 gamma=0.95, epsilon_decay=0.999, batch_size=128, buffer_size=50000, lr=0.0005):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01
        self.batch_size = batch_size
        self.memory = deque(maxlen=buffer_size)
        self.model = DQN(state_size, action_size, learning_rate=lr)
        self.target_model = DQN(state_size, action_size, learning_rate=lr)
        self.target_model.model.set_weights(self.model.model.get_weights())
        self.model_path = model_path
        self.cfr_player = CFRPlayer()  # CFR for bluffing
        self.opponent_history = []  # Tracks opponent actions
        self.bluff_success_tracker = {"total_bluffs": 0, "successful_bluffs": 0}  # Tracks bluffing efficiency

        logging.info("Initializing RLBasedPlayer...")
        self.load_model()

    def save_model(self):
        self.model.model.save(self.model_path)
        logging.info(f"Model saved at {self.model_path}")

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model.model.load_weights(self.model_path)
            self.target_model.model.load_weights(self.model_path)
            logging.info(f"Model loaded successfully from {self.model_path}")
        else:
            logging.info("No saved model found. Training from scratch.")

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, valid_actions):
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_actions)
        state = np.reshape(state, [1, self.state_size])
        q_values = self.model.model.predict(state, verbose=0)[0]
        action_idx = np.argmax(q_values[:len(valid_actions)])
        return valid_actions[action_idx]

    def replay(self):
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
        self.target_model.model.set_weights(self.model.model.get_weights())

    def adjust_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def should_bluff(self, opponent_history):
        if len(opponent_history) < 10:
            return np.random.rand() < 0.2
        total_raises = sum(1 for action in opponent_history if "raise" in action)
        total_folds = sum(1 for action in opponent_history if "fold" in action)
        total_calls = sum(1 for action in opponent_history if "call" in action)
        if total_folds > total_calls:
            return np.random.rand() < 0.5
        if total_raises > total_calls:
            return np.random.rand() < 0.15
        return np.random.rand() < 0.3

    def declare_action(self, valid_actions, hole_card, round_state):
        logging.info(f"Valid actions: {valid_actions}")
        state = encode_state(hole_card, round_state)
        action_idx = None
        is_bluff = False
        if len(round_state["community_card"]) <= 3 and self.should_bluff(self.opponent_history):
            chosen_action = "raise"
            is_bluff = True
        else:
            chosen_action = self.act(state, [a["action"] for a in valid_actions])
            action_idx = [a["action"] for a in valid_actions].index(chosen_action)

        amount = 0
        for action in valid_actions:
            if action["action"] == chosen_action:
                amount = action.get("amount", 0)
                if isinstance(amount, dict):
                    amount = amount["min"]
                    is_bluff = True

        self.opponent_history.append(round_state["action_histories"])
        if is_bluff:
            logging.info(f"Abel BLUFFED by raising {amount} on {round_state['street']}")
            self.bluff_success_tracker["total_bluffs"] += 1
        logging.info(f"Abel chose action: {chosen_action} with amount {amount}")
        return chosen_action, amount

    def receive_game_start_message(self, game_info):
        logging.info(f"Game started with info: {game_info}")
        self.total_players = game_info["player_num"]

    def receive_round_start_message(self, round_count, hole_card, seats):
        logging.info(f"Round {round_count} started. Hole cards: {hole_card}")

    def receive_street_start_message(self, street, round_state):
        logging.info(f"New betting street: {street}")

    def receive_game_update_message(self, action, round_state):
        logging.info(f"Game update: {action}")

    def receive_round_result_message(self, winners, hand_info, round_state):
        logging.info(f"Round ended. Winners: {winners}")

    def get_bluff_success_rate(self):
        total_bluffs = self.bluff_success_tracker["total_bluffs"]
        successful_bluffs = self.bluff_success_tracker["successful_bluffs"]
        return (successful_bluffs / total_bluffs) * 100 if total_bluffs > 0 else 0.0
