# ai/Abel/utils/training_metrics.py
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import logging

class TrainingMetricsTracker:
    """
    Tracks and visualizes training metrics for Abel's learning process.
    
    This class maintains a record of metrics during Abel's training, including
    win rates, loss values, exploration rates, and other key indicators of
    learning progress. It provides tools for visualizing these metrics and
    saving the data for later analysis.
    """
    
    def __init__(self, log_dir="logs/training_metrics", experiment_name=None):
        """
        Initialize the training metrics tracker.
        
        Args:
            log_dir (str): Directory where metrics will be saved
            experiment_name (str, optional): Name for this training run
        """
        # Set up directories
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Generate experiment name if not provided
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"training_run_{timestamp}"
        
        self.experiment_name = experiment_name
        self.experiment_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Set up logging
        logging.basicConfig(
            filename=f"{self.experiment_dir}/metrics.log",
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )
        
        # Initialize metrics storage
        self.metrics = {
            "iterations": [],
            "win_rates": [],
            "loss_values": [],
            "exploration_rates": [],
            "avg_rewards": [],
            "bluff_success_rates": [],
            "fold_rates": [],
            "aggression_factors": [],
            "custom_metrics": {},
            "action_frequencies": [],          # List of dicts: {"fold": X, "call": Y, "raise": Z}
            "raise_sizes": [],                 # Optional: track avg raise sizes (if accessible)
            "pot_odds_calls": [],              # Dicts: {"justified": X, "unjustified": Y}
            "hand_strength_correlations": []   # Numeric values per iteration
        }
        
        # Track opponent types
        self.opponent_metrics = {}
        
        # Initialize timing
        self.start_time = datetime.now()
        logging.info(f"Started training metrics tracking: {self.experiment_name}")

    
    def log_iteration(self, iteration, metrics_dict):
        """
        Log metrics for a training iteration.
        
        Args:
            iteration (int): Current training iteration/episode
            metrics_dict (dict): Dictionary of metrics to log
        """
        # Add iteration number
        self.metrics["iterations"].append(iteration)
        
        # Add standard metrics if available
        for metric_name in ["win_rate", "loss_value", "exploration_rate", "avg_reward",
                           "bluff_success_rate", "fold_rate", "aggression_factor", "raise_size",  "hand_strength_correlation"]:
            plural_name = f"{metric_name}s"
            if metric_name in metrics_dict:
                self.metrics[plural_name].append(metrics_dict[metric_name])
            else:
                # Keep arrays the same length by adding None
                if len(self.metrics[plural_name]) < len(self.metrics["iterations"]):
                    self.metrics[plural_name].append(None)

        # Track action frequency metrics
        if "action_frequency" in metrics_dict:
            self.metrics["action_frequencies"].append(metrics_dict["action_frequency"])
        else:
            self.metrics["action_frequencies"].append(None)

        if "raise_sizes" in metrics_dict:
            self.metrics["raise_sizes"].append(metrics_dict["raise_sizes"])
        else:
            self.metrics["raise_sizes"].append(None)

        if "pot_odds_calls" in metrics_dict:
            self.metrics["pot_odds_calls"].append(metrics_dict["pot_odds_calls"])
        else:
            self.metrics["pot_odds_calls"].append(None)

        if "hand_strength_correlation" in metrics_dict:
            self.metrics["hand_strength_correlations"].append(metrics_dict["hand_strength_correlation"])
        else:
            self.metrics["hand_strength_correlations"].append(None)

        # Add opponent-specific metrics if available
        if "opponent" in metrics_dict:
            opponent = metrics_dict["opponent"]
            if opponent not in self.opponent_metrics:
                self.opponent_metrics[opponent] = {
                    "iterations": [],
                    "win_rates": []
                }
            
            self.opponent_metrics[opponent]["iterations"].append(iteration)
            if "win_rate" in metrics_dict:
                self.opponent_metrics[opponent]["win_rates"].append(metrics_dict["win_rate"])
        
        # Add any custom metrics
        for key, value in metrics_dict.items():
            if key not in ["win_rate", "loss_value", "exploration_rate", 
                          "avg_reward", "bluff_success_rate", "fold_rate", 
                          "aggression_factor", "opponent"]:
                if key not in self.metrics["custom_metrics"]:
                    self.metrics["custom_metrics"][key] = []
                
                self.metrics["custom_metrics"][key].append(value)
        
        # Log to file
        logging.info(f"Iteration {iteration}: {metrics_dict}")
        
        # Periodically save metrics
        if iteration % 100 == 0:
            self.save_metrics()
    
    def log_training_event(self, event_type, description):
        """
        Log a significant training event.
        
        Args:
            event_type (str): Type of event
            description (str): Description of the event
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logging.info(f"EVENT - {event_type}: {description}")
        
        # Could add event tracking in the future
    
    def plot_win_rate(self, save_path=None, window_size=10):
        """
        Plot the win rate over training iterations.
        
        Args:
            save_path (str, optional): Path to save the plot
            window_size (int): Window size for smoothing the curve
        """
        plt.figure(figsize=(10, 6))
        
        # Original win rates
        iterations = self.metrics["iterations"]
        win_rates = self.metrics["win_rates"]
        
        if win_rates:
            plt.plot(iterations, win_rates, 'b-', alpha=0.3, label='Win Rate')
            
            # Smoothed win rates
            if len(win_rates) >= window_size:
                smoothed_rates = pd.Series(win_rates).rolling(window=window_size).mean().values
                plt.plot(iterations, smoothed_rates, 'b-', linewidth=2, 
                         label=f'Win Rate ({window_size}-pt Moving Avg)')
        
        plt.title('Win Rate Over Training Time')
        plt.xlabel('Iterations')
        plt.ylabel('Win Rate (%)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        else:
            default_path = f"{self.experiment_dir}/win_rate_plot.png"
            plt.savefig(default_path)
        
        plt.close()
    
    def plot_opponent_comparison(self, save_path=None):
        """
        Plot win rates against different opponents.
        
        Args:
            save_path (str, optional): Path to save the plot
        """
        if not self.opponent_metrics:
            logging.warning("No opponent-specific metrics to plot")
            return
        
        plt.figure(figsize=(12, 6))
        
        for opponent, metrics in self.opponent_metrics.items():
            iterations = metrics["iterations"]
            win_rates = metrics["win_rates"]
            
            if win_rates:
                plt.plot(iterations, win_rates, '-', linewidth=2, label=f'vs {opponent}')
        
        plt.title('Win Rates Against Different Opponents')
        plt.xlabel('Iterations')
        plt.ylabel('Win Rate (%)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        else:
            default_path = f"{self.experiment_dir}/opponent_comparison.png"
            plt.savefig(default_path)
        
        plt.close()
    
    def plot_learning_metrics(self, save_path=None, window_size=10):
        """
        Plot multiple learning metrics together.
        
        Args:
            save_path (str, optional): Path to save the plot
            window_size (int): Window size for smoothing curves
        """
        if not self.metrics["loss_values"] and not self.metrics["exploration_rates"]:
            logging.warning("No learning metrics (loss, exploration) to plot")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Set up multiple subplots
        n_plots = 2  # Loss and exploration rate by default
        if self.metrics["avg_rewards"]:
            n_plots += 1
        
        plot_index = 1
        
        # Loss values
        if self.metrics["loss_values"]:
            plt.subplot(n_plots, 1, plot_index)
            plot_index += 1
            
            iterations = self.metrics["iterations"]
            loss_values = [l for l in self.metrics["loss_values"] if l is not None]
            iter_with_loss = [iterations[i] for i in range(len(iterations)) 
                              if i < len(self.metrics["loss_values"]) and self.metrics["loss_values"][i] is not None]
            
            if loss_values:
                plt.plot(iter_with_loss, loss_values, 'r-', alpha=0.3, label='Loss')
                
                # Smoothed loss
                if len(loss_values) >= window_size:
                    smoothed_loss = pd.Series(loss_values).rolling(window=window_size).mean().values
                    plt.plot(iter_with_loss, smoothed_loss, 'r-', linewidth=2, 
                             label=f'Loss ({window_size}-pt Moving Avg)')
            
            plt.title('Loss Values During Training')
            plt.ylabel('Loss')
            plt.grid(True, alpha=0.3)
            plt.legend()
        
        # Exploration rate
        if self.metrics["exploration_rates"]:
            plt.subplot(n_plots, 1, plot_index)
            plot_index += 1
            
            iterations = self.metrics["iterations"]
            exploration_rates = [e for e in self.metrics["exploration_rates"] if e is not None]
            iter_with_exp = [iterations[i] for i in range(len(iterations)) 
                             if i < len(self.metrics["exploration_rates"]) and self.metrics["exploration_rates"][i] is not None]
            
            if exploration_rates:
                plt.plot(iter_with_exp, exploration_rates, 'g-', linewidth=2, label='Exploration Rate (ε)')
            
            plt.title('Exploration Rate (Epsilon) During Training')
            plt.ylabel('Epsilon')
            plt.grid(True, alpha=0.3)
            plt.legend()
        
        # Average rewards
        if self.metrics["avg_rewards"]:
            plt.subplot(n_plots, 1, plot_index)
            
            iterations = self.metrics["iterations"]
            avg_rewards = [r for r in self.metrics["avg_rewards"] if r is not None]
            iter_with_rewards = [iterations[i] for i in range(len(iterations)) 
                                 if i < len(self.metrics["avg_rewards"]) and self.metrics["avg_rewards"][i] is not None]
            
            if avg_rewards:
                plt.plot(iter_with_rewards, avg_rewards, 'm-', alpha=0.3, label='Avg Reward')
                
                # Smoothed rewards
                if len(avg_rewards) >= window_size:
                    smoothed_rewards = pd.Series(avg_rewards).rolling(window=window_size).mean().values
                    plt.plot(iter_with_rewards, smoothed_rewards, 'm-', linewidth=2, 
                             label=f'Avg Reward ({window_size}-pt Moving Avg)')
            
            plt.title('Average Rewards During Training')
            plt.ylabel('Reward')
            plt.grid(True, alpha=0.3)
            plt.legend()
        
        plt.xlabel('Iterations')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            default_path = f"{self.experiment_dir}/learning_metrics.png"
            plt.savefig(default_path)
        
        plt.close()
    
    def plot_playing_style_metrics(self, save_path=None, window_size=10):
        """
        Plot metrics related to playing style.
        
        Args:
            save_path (str, optional): Path to save the plot
            window_size (int): Window size for smoothing curves
        """
        style_metrics = [
            ("bluff_success_rates", "Bluff Success Rate (%)", "b"),
            ("fold_rates", "Fold Rate (%)", "g"),
            ("aggression_factors", "Aggression Factor", "r")
        ]
        
        # Check if we have any of these metrics
        has_metrics = any(len(self.metrics[m[0]]) > 0 for m in style_metrics)
        if not has_metrics:
            logging.warning("No playing style metrics to plot")
            return
        
        plt.figure(figsize=(12, 10))
        
        for i, (metric_name, label, color) in enumerate(style_metrics, 1):
            if self.metrics[metric_name]:
                plt.subplot(len(style_metrics), 1, i)
                
                iterations = self.metrics["iterations"]
                values = [v for v in self.metrics[metric_name] if v is not None]
                iter_with_values = [iterations[i] for i in range(len(iterations)) 
                                    if i < len(self.metrics[metric_name]) and self.metrics[metric_name][i] is not None]
                
                if values:
                    plt.plot(iter_with_values, values, f'{color}-', alpha=0.3, label=label)
                    
                    # Smoothed values
                    if len(values) >= window_size:
                        smoothed_values = pd.Series(values).rolling(window=window_size).mean().values
                        plt.plot(iter_with_values, smoothed_values, f'{color}-', linewidth=2, 
                                 label=f'{label} ({window_size}-pt Moving Avg)')
                
                plt.title(f'{label} During Training')
                plt.ylabel(label)
                plt.grid(True, alpha=0.3)
                plt.legend()
        
        plt.xlabel('Iterations')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            default_path = f"{self.experiment_dir}/playing_style_metrics.png"
            plt.savefig(default_path)
        
        plt.close()
    
    def plot_custom_metric(self, metric_name, title=None, y_label=None, 
                          color='b', save_path=None, window_size=10):
        """
        Plot a custom metric.
        
        Args:
            metric_name (str): Name of the custom metric to plot
            title (str, optional): Plot title
            y_label (str, optional): Y-axis label
            color (str): Line color
            save_path (str, optional): Path to save the plot
            window_size (int): Window size for smoothing the curve
        """
        if metric_name not in self.metrics["custom_metrics"]:
            logging.warning(f"Custom metric '{metric_name}' not found")
            return
        
        plt.figure(figsize=(10, 6))
        
        iterations = self.metrics["iterations"][:len(self.metrics["custom_metrics"][metric_name])]
        values = self.metrics["custom_metrics"][metric_name]
        
        plt.plot(iterations, values, f'{color}-', alpha=0.3, label=metric_name)
        
        # Smoothed values
        if len(values) >= window_size:
            smoothed_values = pd.Series(values).rolling(window=window_size).mean().values
            plt.plot(iterations, smoothed_values, f'{color}-', linewidth=2, 
                     label=f'{metric_name} ({window_size}-pt Moving Avg)')
        
        plt.title(title or f'{metric_name} During Training')
        plt.xlabel('Iterations')
        plt.ylabel(y_label or metric_name)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        else:
            metric_filename = metric_name.replace(" ", "_").lower()
            default_path = f"{self.experiment_dir}/custom_{metric_filename}.png"
            plt.savefig(default_path)
        
        plt.close()
    
    def generate_training_summary(self, save_path=None):
        """
        Generate a summary of the training metrics.
        
        Args:
            save_path (str, optional): Path to save the summary
            
        Returns:
            dict: Summary statistics
        """
        # Calculate training duration
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        # Prepare summary statistics
        summary = {
            "experiment_name": self.experiment_name,
            "start_time": self.start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration": str(duration),
            "total_iterations": len(self.metrics["iterations"]),
            "metrics": {}
        }
        
        # Add summary stats for each metric
        for metric_name in ["win_rates", "loss_values", "exploration_rates", 
                           "avg_rewards", "bluff_success_rates", "fold_rates", 
                           "aggression_factors"]:
            values = [v for v in self.metrics[metric_name] if v is not None]
            
            if values:
                metric_summary = {
                    "min": min(values),
                    "max": max(values),
                    "mean": np.mean(values),
                    "median": np.median(values),
                    "std": np.std(values),
                    "first": values[0],
                    "last": values[-1]
                }
                
                summary["metrics"][metric_name] = metric_summary

        # For hand strength correlations (numeric)
        correlations = [v for v in self.metrics["hand_strength_correlations"] if v is not None]
        if correlations:
            summary["metrics"]["hand_strength_correlations"] = {
                "min": min(correlations),
                "max": max(correlations),
                "mean": np.mean(correlations),
                "median": np.median(correlations),
                "std": np.std(correlations),
                "first": correlations[0],
                "last": correlations[-1]
            }

        # For raise sizes (assuming float values)
        raise_sizes = [v for v in self.metrics["raise_sizes"] if v is not None]
        if raise_sizes:
            summary["metrics"]["raise_sizes"] = {
                "min": min(raise_sizes),
                "max": max(raise_sizes),
                "mean": np.mean(raise_sizes),
                "median": np.median(raise_sizes),
                "std": np.std(raise_sizes),
                "first": raise_sizes[0],
                "last": raise_sizes[-1]
            }

        # For pot odds calls (categorical summary)
        justified, unjustified = 0, 0
        for entry in self.metrics["pot_odds_calls"]:
            if entry:
                justified += entry.get("justified", 0)
                unjustified += entry.get("unjustified", 0)

        if justified + unjustified > 0:
            summary["metrics"]["pot_odds_calls"] = {
                "justified": justified,
                "unjustified": unjustified,
                "justification_rate": justified / (justified + unjustified)
            }

        # For action frequency (aggregate and average over time)
        action_totals = {"fold": 0, "call": 0, "raise": 0}
        action_counts = 0

        for entry in self.metrics["action_frequencies"]:
            if entry:
                for act in action_totals:
                    action_totals[act] += entry.get(act, 0)
                action_counts += 1

        if action_counts > 0:
            summary["metrics"]["action_frequencies"] = {
                act: {
                    "total": action_totals[act],
                    "average_per_iteration": action_totals[act] / action_counts
                }
                for act in action_totals
            }

        # Add opponent-specific summaries
        if self.opponent_metrics:
            summary["opponent_metrics"] = {}
            
            for opponent, metrics in self.opponent_metrics.items():
                if metrics["win_rates"]:
                    opp_summary = {
                        "min_win_rate": min(metrics["win_rates"]),
                        "max_win_rate": max(metrics["win_rates"]),
                        "mean_win_rate": np.mean(metrics["win_rates"]),
                        "last_win_rate": metrics["win_rates"][-1]
                    }
                    summary["opponent_metrics"][opponent] = opp_summary
        
        # Add custom metrics
        if self.metrics["custom_metrics"]:
            summary["custom_metrics"] = {}
            
            for metric_name, values in self.metrics["custom_metrics"].items():
                if values:
                    custom_summary = {
                        "min": min(values),
                        "max": max(values),
                        "mean": np.mean(values),
                        "median": np.median(values),
                        "std": np.std(values),
                        "first": values[0],
                        "last": values[-1]
                    }
                    
                    summary["custom_metrics"][metric_name] = custom_summary
        
        # Save summary to file
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(summary, f, indent=2)
        else:
            default_path = f"{self.experiment_dir}/training_summary.json"
            with open(default_path, 'w') as f:
                json.dump(summary, f, indent=2)
        
        # Also save a human-readable version
        txt_path = save_path.replace('.json', '.txt') if save_path else f"{self.experiment_dir}/training_summary.txt"
        with open(txt_path, 'w') as f:
            f.write(f"TRAINING SUMMARY: {self.experiment_name}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Start time: {summary['start_time']}\n")
            f.write(f"End time: {summary['end_time']}\n")
            f.write(f"Duration: {summary['duration']}\n")
            f.write(f"Total iterations: {summary['total_iterations']}\n\n")
            
            f.write("METRICS:\n")
            for metric_name, stats in summary["metrics"].items():
                f.write(f"  {metric_name}:\n")
                for stat_name, value in stats.items():
                    f.write(f"    {stat_name}: {value:.4f}\n")
                f.write("\n")
            
            if "opponent_metrics" in summary:
                f.write("OPPONENT METRICS:\n")
                for opponent, stats in summary["opponent_metrics"].items():
                    f.write(f"  vs {opponent}:\n")
                    for stat_name, value in stats.items():
                        f.write(f"    {stat_name}: {value:.4f}\n")
                    f.write("\n")
            
            if "custom_metrics" in summary:
                f.write("CUSTOM METRICS:\n")
                for metric_name, stats in summary["custom_metrics"].items():
                    f.write(f"  {metric_name}:\n")
                    for stat_name, value in stats.items():
                        f.write(f"    {stat_name}: {value:.4f}\n")
                    f.write("\n")
        
        return summary
    
    def save_metrics(self):
        """
        Save the current metrics to disk.
        """
        metrics_path = f"{self.experiment_dir}/metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        logging.info(f"Saved metrics to {metrics_path}")
    
    def load_metrics(self, metrics_path):
        """
        Load metrics from a saved file.
        
        Args:
            metrics_path (str): Path to the saved metrics file
            
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        if not os.path.exists(metrics_path):
            logging.error(f"Metrics file not found: {metrics_path}")
            return False
        
        try:
            with open(metrics_path, 'r') as f:
                self.metrics = json.load(f)
            
            logging.info(f"Loaded metrics from {metrics_path}")
            return True
        except Exception as e:
            logging.error(f"Error loading metrics: {e}")
            return False
    
    def plot_all_metrics(self, output_dir=None):
        """
        Generate all available plots.
        
        Args:
            output_dir (str, optional): Directory to save plots
        """
        if output_dir is None:
            output_dir = f"{self.experiment_dir}/plots"
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Win rate plot
        self.plot_win_rate(save_path=f"{output_dir}/win_rate.png")
        
        # Opponent comparison
        if self.opponent_metrics:
            self.plot_opponent_comparison(save_path=f"{output_dir}/opponent_comparison.png")
        
        # Learning metrics
        self.plot_learning_metrics(save_path=f"{output_dir}/learning_metrics.png")
        
        # Playing style metrics
        self.plot_playing_style_metrics(save_path=f"{output_dir}/playing_style.png")
        
        # Custom metrics
        for metric_name in self.metrics["custom_metrics"]:
            metric_filename = metric_name.replace(" ", "_").lower()
            self.plot_custom_metric(
                metric_name=metric_name,
                save_path=f"{output_dir}/custom_{metric_filename}.png"
            )
        
        # Generate summary
        self.generate_training_summary(save_path=f"{output_dir}/summary.json")
        
        logging.info(f"Generated all plots in {output_dir}")

# Example usage
if __name__ == "__main__":
    # Initialize tracker
    tracker = TrainingMetricsTracker(experiment_name="abel_self_play_demo")
    
    # Simulate some training iterations
    import random
    
    for i in range(1000):
        # Simulate metrics
        win_rate = 40 + i * 0.02 + random.normalvariate(0, 3)
        loss_value = 1.5 - i * 0.001 + random.normalvariate(0, 0.1)
        exploration_rate = max(0.1, 1.0 - i * 0.001)
        bluff_rate = 20 + i * 0.01 + random.normalvariate(0, 2)
        
        # Log metrics
        tracker.log_iteration(i, {
            "win_rate": win_rate,
            "loss_value": loss_value,
            "exploration_rate": exploration_rate,
            "bluff_success_rate": bluff_rate,
            "opponent": "self" if i < 500 else "random",
            "hand_strength_correlation": 0.5 + i * 0.0003
        })
        
        # Log significant events
        if i == 0:
            tracker.log_training_event("start", "Training started with epsilon = 1.0")
        elif i == 500:
            tracker.log_training_event("opponent_change", "Switched from self-play to random opponent")
        elif i == 999:
            tracker.log_training_event("complete", "Training completed successfully")
    
    # Generate plots and summary
    tracker.plot_all_metrics()
    print("Metrics tracking demo complete. Check the logs/training_metrics directory for results.")