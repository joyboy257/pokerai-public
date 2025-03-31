import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pypokerengine.api.game import setup_config, start_poker
from ai.Abel.Abel import RLBasedPlayer

def visualize_learning_curve():
    """Display Abel's learning progress visualization"""
    # Create sample learning data for demonstration
    iterations = np.arange(0, 5000, 500)
    win_rates = [10, 25, 35, 40, 43, 44, 45, 46, 46.5, 47]
    
    plt.figure(figsize=(10, 5))
    plt.plot(iterations, win_rates, 'b-', linewidth=2)
    plt.title('Abel Learning Progress')
    plt.xlabel('Training Iterations')
    plt.ylabel('Win Rate (%)')
    plt.grid(True, alpha=0.3)
    plt.savefig('abel_learning.png')
    
    print("Learning curve visualization saved as 'abel_learning.png'")

def visualize_q_values():
    """Visualize example Q-values for different actions"""
    states = ['Strong Hand', 'Medium Hand', 'Weak Hand']
    fold_values = [0.1, 0.5, 0.9]
    call_values = [0.3, 0.7, 0.4]
    raise_values = [0.9, 0.3, 0.1]
    
    x = np.arange(len(states))
    width = 0.25
    
    plt.figure(figsize=(10, 5))
    plt.bar(x - width, fold_values, width, label='Fold')
    plt.bar(x, call_values, width, label='Call')
    plt.bar(x + width, raise_values, width, label='Raise')
    
    plt.ylabel('Q-Values')
    plt.title('Abel Q-Values by Hand Strength')
    plt.xticks(x, states)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.savefig('abel_q_values.png')
    
    print("Q-values visualization saved as 'abel_q_values.png'")

def abel_demonstration():
    """Demonstrate Abel's reinforcement learning approach"""
    print("\n" + "="*50)
    print("ABEL AGENT DEMONSTRATION - REINFORCEMENT LEARNING APPROACH")
    print("="*50)
    
    # Initialize Abel with demo parameters
    state_size = 6
    action_size = 3
    abel = RLBasedPlayer(state_size, action_size)
    
    print("\nAbel's Neural Network Architecture:")
    print("- Input Layer: 6 dimensions (state representation)")
    print("- Hidden Layer 1: 128 neurons with ReLU activation")
    print("- Hidden Layer 2: 128 neurons with ReLU activation")
    print("- Output Layer: 3 neurons (Q-values for fold, call, raise)")
    
    print("\nState Representation Features:")
    print("1. Number of hole cards")
    print("2. Number of community cards")
    print("3. Normalized pot size")
    print("4. Normalized player stack")
    print("5. Normalized opponent stack")
    print("6. Pot odds (potential winnings to call ratio)")
    
    print("\nTraining Process:")
    print("- Self-play training with exploration-exploitation balance")
    print("- Experience replay buffer for efficient learning")
    print("- Target network for stable Q-learning updates")
    
    # Create visualizations
    visualize_learning_curve()
    visualize_q_values()
    
    print("\nReinforcement Learning Process:")
    print("1. Encode the poker state into numerical representation")
    print("2. Feed state through neural network to get action Q-values")
    print("3. Select action using epsilon-greedy policy")
    print("4. Learn from experience through Bellman updates")

if __name__ == "__main__":
    abel_demonstration()