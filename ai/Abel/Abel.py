import random
import numpy as np
import tensorflow as tf
from collections import deque
from pypokerengine.players import BasePokerPlayer


class RLBasedPlayer(BasePokerPlayer):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995  # Decay rate for epsilon
        self.epsilon_min = 0.1  # Minimum epsilon value
        self.learning_rate = 0.001
        self.batch_size = 32
        self.memory = deque(maxlen=2000)  # Replay buffer

        # Build the neural network
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')  # Output layer
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss='mse')
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, valid_actions):
        """
        Selects an action using epsilon-greedy strategy.
        """
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_actions)  # Explore
        state = np.reshape(state, [1, self.state_size])
        q_values = self.model.predict(state, verbose=0)
        action_idx = np.argmax(q_values[0][:len(valid_actions)])
        return valid_actions[action_idx]  # Exploit

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        # Sample a mini-batch from memory
        minibatch = random.sample(self.memory, self.batch_size)

        for state, action_idx, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = np.reshape(next_state, [1, self.state_size])
                target += self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            state = np.reshape(state, [1, self.state_size])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action_idx] = target  # Use action index directly

            self.model.fit(state, target_f, epochs=1, verbose=0)


    def adjust_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def calculate_reward(self, is_win, action, pot_size, chips_change, pot_odds, opponent_folded):
        """
        Calculate rewards based on outcomes and actions.
        """
        if is_win:
            return max(1.0, pot_size / 1000)  # Big win, big reward
        elif action == "fold" and pot_odds < 0.2:
            return 0.1  # Good fold based on low pot odds
        elif action == "fold":
            return -0.5  # Penalize unnecessary folds
        elif action == "raise" and opponent_folded:
            return 0.5  # Reward successful bluff
        else:
            return -1.0  # Penalize significant losses


    def encode_state(self, hole_card, round_state):
        """
        Custom state encoding for the DQN.
        """
        community_cards = round_state.get("community_card", [])
        pot = round_state.get("pot", {}).get("main", {}).get("amount", 0)
        player_stack = round_state["seats"][round_state["next_player"]]["stack"]
        opponent_stack = round_state["seats"][1 - round_state["next_player"]]["stack"]
        street = round_state.get("street", "preflop")
        position = round_state["next_player"]
        pot_odds = pot / max(1, player_stack)

        # Encode state as a normalized vector
        state = np.array([
            len(hole_card),  # Number of hole cards
            len(community_cards),  # Number of community cards
            pot / 1000,  # Normalize pot size
            player_stack / 1000,  # Normalize player stack
            opponent_stack / 1000,  # Normalize opponent stack
            1 if street == "preflop" else 0,
            1 if street == "flop" else 0,
            1 if street == "turn" else 0,
            1 if street == "river" else 0,
            position,  # Player's position
            pot_odds  # Pot odds
        ])
        
        # Log encoded state to a file
        with open("encoded_states_log.txt", "a") as f:
            f.write(f"Encoded State: {state.tolist()}, Shape: {state.shape}\n")

        return state

    def declare_action(self, valid_actions, hole_card, round_state):
        """
        Declare the action based on the DQN policy.
        """
        state = self.encode_state(hole_card, round_state)
        action_idx = self.act(state, list(range(len(valid_actions))))  # Use indices for actions
        action = valid_actions[action_idx]['action']
        self.last_state = state
        self.last_action = action_idx  # Store index, not string
        return action, 0  # Placeholder amount for now


    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        """
        Update Q-values based on round results.
        """
        is_win = self.uuid in winners
        pot_size = round_state.get("pot", {}).get("main", {}).get("amount", 0)
        chips_change = round_state["seats"][round_state["next_player"]]["stack"] - self.last_state[2]
        opponent_folded = any(player["state"] == "folded" for player in round_state["seats"])
        pot_odds = pot_size / max(1, self.last_state[2])

        reward = self.calculate_reward(is_win, self.last_action, pot_size, chips_change, pot_odds, opponent_folded)
        next_state = self.encode_state([], round_state)  # Empty hole card at end of round
        self.remember(self.last_state, self.last_action, reward, next_state, True)
        self.replay()
        self.adjust_epsilon()
