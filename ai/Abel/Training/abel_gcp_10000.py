import os
import tensorflow as tf
import numpy as np
import random
import logging
from collections import deque
from pypokerengine.api.game import setup_config, start_poker
from pypokerengine.players import BasePokerPlayer

# ✅ Enable GPU for TensorFlow
logging.info("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Setup TensorBoard
log_dir = "models/logs"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Initialize Logging for Debugging
logging.basicConfig(filename="abel_gcp_training.log", level=logging.INFO, format="%(asctime)s - %(message)s")

# 🎲 Simple Rule-Based Opponent (Fix for NotImplementedError)
class RuleBasedPlayer(BasePokerPlayer):
    def declare_action(self, valid_actions, hole_card, round_state):
        for action in valid_actions:
            if action['action'] == 'call':
                return 'call', action['amount']
            elif action['action'] == 'check':
                return 'check', action['amount']
        return 'fold', 0

    def receive_game_start_message(self, game_info):
        print(f"[Opponent] Game started with info: {game_info}")

    def receive_round_start_message(self, round_count, hole_card, seats):
        print(f"[Opponent] Round {round_count} started with hole cards: {hole_card}")

    def receive_street_start_message(self, street, round_state):
        print(f"[Opponent] New street: {street}")

    def receive_game_update_message(self, action, round_state):
        print(f"[Opponent] Game updated with action: {action}")

    def receive_round_result_message(self, winners, hand_info, round_state):
        print(f"[Opponent] Round ended. Winners: {winners}")


# ♠️ Reinforcement Learning Poker Player (Abel)
# ✅ Inherit from RuleBasedPlayer instead of BasePokerPlayer
class RLBasedPlayer(RuleBasedPlayer):
    def __init__(self, state_size, action_size, model_path="models/abel_model.h5"):
        super().__init__()
        logging.info(f"Initializing RLBasedPlayer with state_size={state_size}, action_size={action_size}")

        # Hyperparameters
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.learning_rate = 0.0005
        self.batch_size = 128
        self.memory = deque(maxlen=50000)
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_model.set_weights(self.model.get_weights())
        self.last_state = np.zeros(state_size)
        self.last_action = None
        self.model_path = model_path
        self.checkpoint_path = "models/checkpoints/abel_checkpoint.h5"
        self.load_model()  # Load previous model if available

    def _build_model(self):
        """ Builds the deep learning model for Q-learning. """
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation="relu", input_dim=self.state_size),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(self.action_size, activation="linear"),
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss="mse")
        return model

    def save_model(self):
        """ Saves the model to file. """
        self.model.save(self.model_path)
        self.model.save(self.checkpoint_path)
        logging.info("Model checkpoint saved successfully.")

    def load_model(self):
        """ Loads the model from file if available. """
        if os.path.exists(self.model_path):
            self.model = tf.keras.models.load_model(self.model_path)
            self.target_model = tf.keras.models.load_model(self.model_path)
            logging.info("Model loaded successfully.")
        else:
            logging.info("No previous model found. Training from scratch.")

    def remember(self, state, action, reward, next_state, done):
        """ Stores experience in memory buffer for replay. """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, valid_actions):
        """ Chooses an action using an epsilon-greedy policy. """
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_actions)
        state = np.reshape(state, [1, self.state_size])
        q_values = self.model.predict(state, verbose=0)[0]
        action_idx = np.argmax(q_values[:len(valid_actions)])
        chosen_action = valid_actions[action_idx]
        logging.info(f"Abel chose action: {chosen_action} | Q-values: {q_values}")
        return chosen_action

    def replay(self):
        """ Trains the model using experience replay. """
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.target_model.predict(np.reshape(next_state, [1, self.state_size]), verbose=0)[0])
            state = np.reshape(state, [1, self.state_size])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0, callbacks=[tensorboard_callback])
        self.target_model.set_weights(self.model.get_weights())

    def adjust_epsilon(self):
        """ Reduces exploration rate over time. """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# 🎲 Training Execution
def train_abel_gcp(num_games=10000):
    state_size = 6
    action_size = 3
    abel = RLBasedPlayer(state_size, action_size)
    win_count = 0

    for game_number in range(num_games):
        config = setup_config(max_round=5, initial_stack=1000, small_blind_amount=10)
        config.register_player(name="Abel", algorithm=abel)
        config.register_player(name="Opponent", algorithm=RuleBasedPlayer())
        game_result = start_poker(config, verbose=0)

        if game_result["players"][0]["name"] == "Abel" and game_result["players"][0]["stack"] > game_result["players"][1]["stack"]:
            win_count += 1
            logging.info(f"Win Count: {win_count}")

        if game_number % 100 == 0:
            abel.save_model()

    # 🔥 NEW: Log and Display Total Games and Win Rate
    total_games = num_games
    win_rate = (win_count / total_games) * 100
    print(f"\n=== Training Summary ===")
    print(f"Total Games Played: {total_games}")
    print(f"Total Wins: {win_count}")
    print(f"Win Rate: {win_rate:.2f}%")
    logging.info(f"Total Games Played: {total_games}")
    logging.info(f"Total Wins: {win_count}")
    logging.info(f"Win Rate: {win_rate:.2f}%")

abel = RLBasedPlayer(state_size=6, action_size=3)
abel.load_model()
train_abel_gcp(1000000)
