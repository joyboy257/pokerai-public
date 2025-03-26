# scripts/abel_grid_search_runner.py
import itertools
import os
import logging
import argparse
from datetime import datetime
from ai.Abel.agents.RLBasedPlayer import RLBasedPlayer
from pypokerengine.api.game import setup_config, start_poker
from ai.Abel.utils.training_metrics import TrainingMetricsTracker

# ========================
# CLI Setup
# ========================
parser = argparse.ArgumentParser(
    description="Grid Search Runner: Tune Abel's hyperparameters via self-play and log performance metrics."
)
parser.add_argument('--games', type=int, default=2000, help='Number of games per experiment (default: 2000)')
parser.add_argument('--eval_every', type=int, default=500, help='Interval for interim evaluations (default: 500)')
parser.add_argument('--output_dir', type=str, default="logs/grid_search", help='Base directory for logs (default: logs/grid_search)')
args = parser.parse_args()

# ========================
# Grid Search Space
# ========================
grid = list(itertools.product(
    [0.99, 0.995, 0.999],  # epsilon_decay
    [0.001, 0.0005, 0.0001],  # learning_rate
    [0.90, 0.95, 0.99],  # gamma
    [64, 128],  # batch_size
    [50000, 100000]  # buffer_size
))

# ========================
# Grid Search Loop
# ========================
os.makedirs(args.output_dir, exist_ok=True)

for idx, (eps_decay, lr, gamma, batch_size, buffer_size) in enumerate(grid):
    experiment_name = f"exp_{idx+1}_eps{eps_decay}_lr{lr}_g{gamma}_b{batch_size}_buf{buffer_size}"
    log_dir = os.path.join(args.output_dir, experiment_name)
    os.makedirs(log_dir, exist_ok=True)

    # Initialize logging for this experiment
    logging.basicConfig(
        filename=os.path.join(log_dir, "experiment.log"),
        level=logging.INFO,
        format="%(asctime)s - %(message)s"
    )

    print(f"\n Running {experiment_name}")

    # Initialize Abel Players with hyperparameters
    state_size = 6
    action_size = 3
    abel_1 = RLBasedPlayer(state_size, action_size, model_path=f"{log_dir}/abel_model.h5",
                           epsilon_decay=eps_decay, lr=lr, gamma=gamma,
                           batch_size=batch_size, buffer_size=buffer_size)
    abel_2 = RLBasedPlayer(state_size, action_size, model_path=f"{log_dir}/abel_opponent_model.h5",
                           epsilon_decay=eps_decay, lr=lr, gamma=gamma,
                           batch_size=batch_size, buffer_size=buffer_size)

    metrics_tracker = TrainingMetricsTracker(log_dir=log_dir, experiment_name=experiment_name)

    # ========================
    # Training Loop
    # ========================
    win_count = 0
    total_rewards = 0

    for game_number in range(args.games):
        config = setup_config(max_round=5, initial_stack=1000, small_blind_amount=10)
        config.register_player(name="Abel_1", algorithm=abel_1)
        config.register_player(name="Abel_2", algorithm=abel_2)
        result = start_poker(config, verbose=0)

        abel_1_stack = next(p["stack"] for p in result["players"] if p["name"] == "Abel_1")
        abel_2_stack = next(p["stack"] for p in result["players"] if p["name"] == "Abel_2")

        if abel_1_stack > abel_2_stack:
            win_count += 1

        total_rewards += (abel_1_stack - 1000)

        if (game_number + 1) % args.eval_every == 0:
            interim_win_rate = (win_count / (game_number + 1)) * 100
            logging.info(f"📊 Intermediate @ Game {game_number + 1}: Win Rate = {interim_win_rate:.2f}%")
            metrics_tracker.log_iteration(game_number + 1, {
                "win_rate": interim_win_rate,
                "exploration_rate": abel_1.epsilon,
                "avg_reward": total_rewards / (game_number + 1),
                "opponent": "self"
            })

        abel_1.adjust_epsilon()
        abel_2.adjust_epsilon()

    # ========================
    # Final Metrics Logging
    # ========================
    final_win_rate = (win_count / args.games) * 100
    avg_reward = total_rewards / args.games

    metrics_tracker.log_custom({
        "final_self_play_win_rate": final_win_rate,
        "average_reward_per_game": avg_reward
    })
    metrics_tracker.generate_training_summary()

    logging.info(f" {experiment_name} completed: Win Rate = {final_win_rate:.2f}%, Avg Reward = {avg_reward:.2f}")

print("\n All grid search experiments completed!")
