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
        self.bluff_success_tracker = {"total_bluffs": 0, "successful_bluffs": 0}  # Tracks bluffing efficiency

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

    def should_bluff(self, opponent_history):
        """ Determines if bluffing is a good decision based on opponent behavior. """
        if len(opponent_history) < 10:  # Not enough data yet
            return np.random.rand() < 0.2  # 20% chance of bluffing initially

        # Analyze opponent behavior
        total_raises = sum(1 for action in opponent_history if "raise" in action)
        total_folds = sum(1 for action in opponent_history if "fold" in action)
        total_calls = sum(1 for action in opponent_history if "call" in action)

        # If opponent folds often, bluff more
        if total_folds > total_calls:
            return np.random.rand() < 0.5  # Bluff 50% of the time

        # If opponent is aggressive, don't bluff as often
        if total_raises > total_calls:
            return np.random.rand() < 0.15  # Bluff 15% of the time

        return np.random.rand() < 0.3  # Default bluffing chance

    def declare_action(self, valid_actions, hole_card, round_state):
        """ Uses CFR for bluffing and DQN for tactical play. """
        logging.info(f"Valid actions: {valid_actions}")
        state = encode_state(hole_card, round_state)  # Proper game state encoding
        state_key = str(round_state)  # Simplified state representation
        action_idx = None
        is_bluff = False

        # **CFR for bluffing in early rounds, but only bluff if opponent is likely to fold**
        if len(round_state["community_card"]) <= 3 and self.should_bluff(self.opponent_history):
            chosen_action = "raise"
            is_bluff = True
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
                    is_bluff = True  # Raising with a weak hand is bluffing

        # **Track opponent actions for strategic adaptation**
        self.opponent_history.append(round_state["action_histories"])

        # **Log bluffing actions**
        if is_bluff:
            logging.info(f"Abel BLUFFED by raising {amount} on {round_state['street']}")
            self.bluff_success_tracker["total_bluffs"] += 1  # Track bluff count

        logging.info(f"Abel chose action: {chosen_action} with amount {amount}")
        return chosen_action, amount

    # Required Methods for PyPokerEngine Compatibility
    def receive_game_start_message(self, game_info):
        """ Handles game start event (needed for PyPokerEngine). """
        logging.info(f"Game started with info: {game_info}")
        self.total_players = game_info["player_num"]  # Ensure Abel knows the number of players

    def receive_round_start_message(self, round_count, hole_card, seats):
        """ Handles round start event. """
        logging.info(f"Round {round_count} started. Hole cards: {hole_card}")

    def receive_street_start_message(self, street, round_state):
        """ Handles new betting street event. """
        logging.info(f"New betting street: {street}")

    def receive_game_update_message(self, action, round_state):
        """ Handles game update event (opponent actions). """
        logging.info(f"Game update: {action}")

    def receive_round_result_message(self, winners, hand_info, round_state):
        """ Handles round results and resets any necessary values. """
        logging.info(f"Round ended. Winners: {winners}")

    def get_bluff_success_rate(self):
        """ Returns the percentage of successful bluffs. """
        total_bluffs = self.bluff_success_tracker["total_bluffs"]
        successful_bluffs = self.bluff_success_tracker["successful_bluffs"]
        return (successful_bluffs / total_bluffs) * 100 if total_bluffs > 0 else 0.0
