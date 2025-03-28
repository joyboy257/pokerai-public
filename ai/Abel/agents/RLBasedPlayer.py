import os
import numpy as np
import random
import logging
from collections import deque
from pypokerengine.players import BasePokerPlayer
from ai.Abel.models.DQN import DQN
from ai.Abel.agents.CFRPlayer import CFRPlayer
from ai.Abel.utils.state_encoder import encode_state
from ai.Abel.utils.decision_evaluator import DecisionEvaluator

class RLBasedPlayer(BasePokerPlayer):
    def __init__(self, state_size, action_size, model_path="models/abel_model.h5",
                 gamma=0.95, epsilon_decay=0.999, batch_size=128, buffer_size=50000, lr=0.0005,
                 decision_evaluator=None):
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
        self.cfr_player = CFRPlayer()
        self.opponent_history = []
        self.bluff_success_tracker = {"total_bluffs": 0, "successful_bluffs": 0}
        self.raise_amounts = []
        self.justified_pot_odds_calls = 0
        self.unjustified_pot_odds_calls = 0
        self.decision_evaluator = decision_evaluator
        logging.info("Initializing RLBasedPlayer...")
        self.load_model()

    def save_model(self):
        self.model.model.save(self.model_path)
        logging.info(f"Model saved at {self.model_path}")
        if self.decision_evaluator:
            self.decision_evaluator.save()

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
            return None  # Not enough samples yet
        
        minibatch = random.sample(self.memory, self.batch_size)
        total_loss = 0

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(
                    self.target_model.model.predict(np.reshape(next_state, [1, self.state_size]), verbose=0)[0]
                )
            state = np.reshape(state, [1, self.state_size])
            target_f = self.model.model.predict(state, verbose=0)
            target_f[0][action] = target
            history = self.model.model.fit(state, target_f, epochs=1, verbose=0)
            total_loss += history.history["loss"][0]

        self.target_model.model.set_weights(self.model.model.get_weights())
        avg_loss = total_loss / self.batch_size
        return avg_loss


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
        try:
            state = encode_state(hole_card, round_state)
        except Exception as e:
            logging.error(f"[ERROR] Failed to encode state: {e}")
            return "fold", 0  # fallback safe action
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

        self.opponent_history.append(round_state.get("action_histories", {}).get(self.uuid, {}))
        if chosen_action == "raise":
            self.raise_amounts.append(amount)
        if is_bluff:
            logging.info(f"Abel BLUFFED by raising {amount} on {round_state['street']}")
            self.bluff_success_tracker["total_bluffs"] += 1
        logging.info(f"Abel chose action: {chosen_action} with amount {amount}")

        if self.decision_evaluator:
            try:
                hand_id = round_state.get("round_count", 0)
                player_id = self.uuid
                street = round_state.get("street", "preflop")
                community = round_state.get("community_card", [])
                pot_amount = round_state.get("pot", {}).get("main", {}).get("amount", 0)
                hand_strength = state[-1]

                logging.info("[DecisionEval] About to evaluate decision")
                self.decision_evaluator.evaluate_decision(
                    player_id=player_id,
                    hand_id=hand_id,
                    street=street,
                    hole_cards=hole_card,
                    community_cards=community,
                    pot_size=pot_amount,
                    valid_actions=[a["action"] for a in valid_actions],
                    chosen_action=chosen_action,
                    hand_strength=hand_strength,
                    is_bluff=is_bluff
                )
                logging.info("[DecisionEval] Decision evaluation complete")
            except Exception as e:
                logging.warning(f"Decision evaluator logging failed: {e}")

        logging.info(f"Returning action: {chosen_action}, amount: {amount}")
        return chosen_action, amount

    def get_action_frequencies(self):
        counts = {"fold": 0, "call": 0, "raise": 0}
        for hist in self.opponent_history:
            for street in hist.values():
                for action in street:
                    act = action.get("action")
                    if act in counts:
                        counts[act] += 1
        return counts

    def get_fold_rate(self):
        counts = self.get_action_frequencies()
        total = sum(counts.values())
        return (counts["fold"] / total) * 100 if total > 0 else 0.0

    def get_aggression_factor(self):
        counts = self.get_action_frequencies()
        calls = counts["call"]
        raises = counts["raise"]
        return (raises / calls) if calls > 0 else raises

    def get_average_raise_size(self):
        return np.mean(self.raise_amounts) if self.raise_amounts else 0.0

    def get_pot_odds_call_justification(self):
        return {
            "justified": self.justified_pot_odds_calls,
            "unjustified": self.unjustified_pot_odds_calls
        }

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
        if self.uuid in winners:
            last_bluffed = self.bluff_success_tracker["total_bluffs"]
            if last_bluffed > self.bluff_success_tracker["successful_bluffs"]:
                self.bluff_success_tracker["successful_bluffs"] += 1

    def get_bluff_success_rate(self):
        total_bluffs = self.bluff_success_tracker["total_bluffs"]
        successful_bluffs = self.bluff_success_tracker["successful_bluffs"]
        return (successful_bluffs / total_bluffs) * 100 if total_bluffs > 0 else 0.0
