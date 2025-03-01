# ai/Abel/models/DQN.py
import tensorflow as tf

class DQN:
    def __init__(self, state_size, action_size, learning_rate=0.0005):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.model = self.build_model()

    def build_model(self):
        """ Builds the deep learning model for Q-learning. """
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation="relu", input_dim=self.state_size),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(self.action_size, activation="linear"),
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss="mse")
        return model

    def save_model(self, path):
        """ Saves the model to file. """
        self.model.save(path)

    def load_model(self, path):
        """ Loads the model from file if available. """
        self.model = tf.keras.models.load_model(path)
